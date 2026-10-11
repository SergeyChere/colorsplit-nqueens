// SPDX-License-Identifier: MIT
/*
 * ColorSplit — exact counting of n-queens solutions, CPU version (C, POSIX threads)
 * =================================================================================
 *
 * Computes Q(n), the number of ways to place n mutually non-attacking queens on an
 * n x n board (OEIS A000170), exactly and without listing the solutions one by one.
 *
 * Idea
 *   Color the board like a chessboard: square (r, c) is white when r + c is even.
 *   Queens on squares of different colors never share a diagonal, so every solution
 *   is a placement of queens on white squares in some row set R and column set C,
 *   combined freely with a placement of queens on black squares in the remaining rows
 *   ~R and columns ~C:
 *
 *       Q(n) = Σ_R Σ_C  W_R(C) · B_~R(~C)
 *
 *   where W_R(C) is the number of non-attacking placements on white squares that use
 *   exactly the rows R and the columns C, and B_~R(~C) is the same for black squares.
 *   For each R the program enumerates the white placements and tabulates them by
 *   their column set C; then it enumerates the black placements on the rows ~R and,
 *   for each one, adds the table entry of the complementary column set.
 *
 * Techniques
 *   - Parity law: R can hold the white queens of a solution only if it contains as
 *     many even rows as odd rows (one more even row when n is odd); other R are
 *     skipped.
 *   - Board symmetries on row sets: only the smallest R of each symmetry orbit is
 *     processed, and its result is multiplied by the orbit size.
 *   - Weighted transposition: transposing a solution swaps the roles of R and C, so
 *     only pairs with inv(C) >= inv(R) are counted, with weight λ = 2 if inv(C) >
 *     inv(R), λ = 1 if equal and λ = 0 otherwise, where inv() is an invariant of
 *     row/column sets. Branches that can only reach λ = 0 are cut early.
 *   - Compact table: for a fixed R every column set C has a fixed number of even and
 *     of odd columns, so the table is indexed by the combinatorial ranks of these two
 *     halves (4-byte cells).
 *   - Half-key pruning: the probing enumeration first fixes the odd-column half of
 *     the key it needs and abandons the branch when no table key has that half.
 *   - Center-out row order: rows are placed from the middle of the board outwards.
 *   - Choice of table side: by default the color with more rows goes into the table
 *     and the other color probes it (option -side).
 *   - Nested left-right mirror for odd n (the mirror keeps square colors when n is
 *     odd): the first queen is placed only in the left half (weight 2), or in the
 *     center column with the second queen in the left half (weight 2).
 *   - 4096 independent chunks of row sets (2^n chunks for n < 12), shared out among
 *     the threads. Every finished chunk is appended to a checkpoint file, so an
 *     interrupted run resumes where it stopped, and chunk ranges or parts can run on
 *     different machines.
 *   - Time accounting across restarts, and a daemon mode.
 *
 * Build
 *     gcc -O3 -march=native -pthread -o colorsplit colorsplit.c
 *   On AMD Zen 2 and older add -DSOFT_PEXT: these CPUs execute the BMI2 instruction
 *   PEXT in slow microcode, and a small lookup table is faster.
 *
 * Usage
 *     ./colorsplit 1 18                  check n = 1..18 against OEIS A000170
 *     ./colorsplit 23 23 -t 16           n = 23 on 16 threads (default: all CPUs)
 *     ./colorsplit 25 25 -daemon         run in the background (log colorsplit_n25-25.log)
 *     kill $(cat colorsplit.pid)         stop it cleanly; the same command resumes it
 *     ./colorsplit 27 27 -part 0/4       part 0 of 4 (run parts 1/4..3/4 elsewhere)
 *     cat colorsplit_n27_part*of4.ckpt > colorsplit_n27.ckpt
 *     ./colorsplit 27 27 -status         combine the parts and print the total
 *
 * Options (after the two required arguments n_from n_to, the range of board sizes)
 *     -t N                    number of worker threads (default: number of online CPUs)
 *     -p S                    progress line on stderr every S seconds, 0 = off (default 60)
 *     -ckpt 0|1               checkpoint file off/on (default: on for n >= 20); a rerun resumes
 *     -from A                 only chunks A and above (default 0)
 *     -to B                   only chunks below B (default: all 4096)
 *     -part I/K               only part I of K, i.e. chunks c with c % K == I (I = 0..K-1)
 *     -status                 compute nothing: report how many chunks the checkpoint file
 *                             holds, and the total if it is complete (n < 20: add -ckpt 1)
 *     -side white|black|auto  color whose placements go into the table (default auto)
 *     -daemon                 run in the background: output to colorsplit_n<from>-<to>.log,
 *                             PID to colorsplit.pid
 *   Unknown options are ignored. SIGINT and SIGTERM stop the run cleanly: finished
 *   chunks are already in the checkpoint file, the time file is updated, and chunks in
 *   progress are recomputed on resume. SIGHUP is ignored.
 *
 * Output
 *   stdout: one line per n, e.g. "n=18: 666090624 OK | 4.321 s (0.1 min)", where
 *   OK / MISMATCH compares with OEIS A000170 (known up to n = 27); with checkpointing
 *   a second line gives the time over all sessions and the sum of the chunk times.
 *   stderr: resume, start and progress messages.
 *
 * Memory per thread: n=23 ~5 MB, n=25 ~20 MB, n=27 ~70 MB, n=28 ~140 MB.
 * Limits: n <= 31 (bit masks and table indices). All sums are signed 64-bit, which is
 * enough up to n = 28; Q(29) is expected to exceed 2^63.
 *
 * Files (in the current directory)
 *   colorsplit_n<n>.ckpt, or colorsplit_n<n>_part<i>of<k>.ckpt with -part i/k (k > 1):
 *     one line per finished chunk, in order of completion:
 *         <chunk> <sum> <milliseconds> ok
 *     chunk = chunk index (0..4095), sum = the chunk's weighted contribution to Q(n),
 *     milliseconds = its compute time. When a file is loaded, the first line for a
 *     chunk wins and malformed lines are ignored. Every line is appended by opening
 *     the file, writing, fsync() and closing it again, so a cloud-synced folder (for
 *     example Google Drive mounted in Colab, which saves a file only when it is
 *     closed) always holds the current progress. If a run was killed in the middle of
 *     a line, the next run first ends that torn line with a newline, so it is ignored.
 *   colorsplit_n<n>.time, or colorsplit_n<n>_part<i>of<k>.time:
 *         <seconds> <sessions>
 *     total wall-clock time of all runs so far and their number; rewritten (via a
 *     .tmp file and rename) every 30 s, when stopped and at the end.
 *   colorsplit_n<from>-<to>.log and colorsplit.pid: daemon mode only.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/stat.h>
#if defined(__BMI2__) && !defined(SOFT_PEXT)
#include <immintrin.h>
#endif

typedef uint64_t u64; typedef int64_t i64;

/* Known values Q(0..27), OEIS A000170, used to check the results. */
static const i64 REF[] = {1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512,
    95815104, 666090624, 4968057848LL, 39029188884LL, 314666222712LL, 2691008701644LL, 24233937684440LL,
    227514171973736LL, 2207893435808352LL, 22317699616364044LL, 234907967154122528LL};
#define NREF ((int)(sizeof(REF) / sizeof(REF[0])))

/* Verification builds (used by tests/check_reductions.sh, never for normal runs):
 *   -DNO_EARLY_CUT  keep branches whose pairs all have weight λ = 0 (same chunk sums, slower)
 *   -DNO_MIRROR     odd n: no left-right mirror inside a row set (same chunk sums, slower)
 *   -DNO_HALF_KEY   no half-key pruning in the probe (same chunk sums, slower)
 *   -DCHECK_CELLS   compare every table cell of weight λ > 0 with a plain enumeration that
 *                   uses neither the mirror nor the early cut (small n; runs on one thread)
 *   -DCELL_STATS    print the largest table cell and the largest W_R(C) of each n */
#ifdef NO_EARLY_CUT
#define EARLY_CUT 0
#else
#define EARLY_CUT 1
#endif
#ifdef NO_MIRROR
#define MIRROR 0
#else
#define MIRROR 1
#endif
#ifdef NO_HALF_KEY
#define HALF_KEY 0
#else
#define HALF_KEY 1
#endif

/* =====================================================================================
 * Board geometry and lookup tables (set by initBoard for the current n)
 * =====================================================================================
 * Bit c of a column set is column c; row sets use the same layout (bit r = row r).
 *   n, h               board size and n / 2 (for odd n, h is the center column)
 *   even, oddBM        n is even / n is odd (oddBM: use the left-right mirror, which
 *                      keeps square colors only when n is odd)
 *   SIDE               color of the table: 0 white, 1 black, 2 auto (option -side)
 *   full               all n columns (or rows)
 *   oddCols, evenCols  odd and even columns (or rows)
 *   white[r], black[r] columns of the white and black squares of row r
 *   ORD                rows in center-out order
 */
static int n, h, even, oddBM, SIDE = 2;
static u64 full, oddCols, evenCols, white[64], black[64];
static int ORD[64];
static int nE, nO;                       /* number of even and of odd columns */
static uint32_t rankTab[1 << 16];        /* rank of a mask among the masks with the same number of ones */
static uint64_t binomT[33][33];          /* binomial coefficients C(a, b) */
static uint32_t EVB[256], ODB[256];      /* software bit extraction, used when PEXT is not */

/* Even-column bits of x, packed into consecutive low bits. */
static inline uint32_t cE(u64 x) {
#if defined(__BMI2__) && !defined(SOFT_PEXT)
    return (uint32_t)_pext_u64(x, evenCols);
#else
    uint32_t r = 0;
    for (int i = 0; i < 4; i++) r |= EVB[(x >> (8 * i)) & 255] << (4 * i);
    return r;
#endif
}
/* Odd-column bits of x, packed into consecutive low bits. */
static inline uint32_t cO(u64 x) {
#if defined(__BMI2__) && !defined(SOFT_PEXT)
    return (uint32_t)_pext_u64(x, oddCols);
#else
    uint32_t r = 0;
    for (int i = 0; i < 4; i++) r |= ODB[(x >> (8 * i)) & 255] << (4 * i);
    return r;
#endif
}

/* Reverse the low n bits (bit c <-> bit n-1-c): mirror a column set left-right, or a
 * row set top-bottom. */
static inline u64 rev(u64 x) {
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
    return __builtin_bswap64(x) >> (64 - n);
}
static inline u64 umin(u64 a, u64 b) { return a < b ? a : b; }
static inline u64 umax(u64 a, u64 b) { return a > b ? a : b; }
/* Canonical form of a set of odd columns: s or its complement within the odd columns,
 * whichever is smaller. */
static inline u64 hOdd(u64 s) { return umin(s, oddCols ^ s); }

/* =====================================================================================
 * Transposition invariant
 * =====================================================================================
 * Transposing a solution keeps square colors and swaps its white row set R with its
 * white column set C, so the pairs (R, C) and (C, R) contribute equally. inv() takes
 * the same value on every set of a symmetry orbit used in chunkSum, and one parity
 * class of S alone gives its high part inv(S) >> n (n odd) or an upper bound on it
 * (n even):
 *   n odd:  the mirror keeps column parity; key(S) = odd part << n | even part, and
 *           inv(S) = min(key(S), key(rev(S))), so the high part is min(odd part,
 *           rev(odd part));
 *   n even: the mirror swaps column parity; u = hOdd(odd part of S),
 *           v = hOdd(rev(even part of S)), inv(S) = min(u, v) << n | max(u, v), so
 *           the high part is at most u and at most v.
 */
static inline u64 inv(u64 S) {
    if (even) {
        u64 u = hOdd(S & oddCols), v = hOdd(rev(S & evenCols));
        return umin(u, v) << n | umax(u, v);
    }
    u64 a = (S & oddCols) << n | (S & evenCols), rs = rev(S), b = (rs & oddCols) << n | (rs & evenCols);
    return umin(a, b);
}
/* Upper bound on the high part inv(C) >> n (exact for odd n) that is already known when
 * the rows of the first parity class are placed (a square's color ties the parity of
 * its row to that of its column). If it is below inv(R) >> n, every completion has
 * λ = 0 and the branch can be cut.
 *   partBuild: white placements in the table, cols = their columns (C = cols);
 *   pbBlack:   black placements in the table, cols = their columns (C = ~cols). */
static inline u64 partBuild(u64 cols) {
    return even ? hOdd(rev(cols & evenCols)) : umin(cols & oddCols, rev(cols & oddCols));
}
static inline u64 pbBlack(u64 cols) {
    u64 cOdd = oddCols & ~(cols & oddCols);
    return even ? hOdd(cOdd) : umin(cOdd, rev(cOdd) & oddCols);
}

/* =====================================================================================
 * Per-thread state and the compact table
 * =====================================================================================
 * For the current row set R a thread first builds the table (the placements of one
 * color, counted by their column set, the "key"), then probes it with the placements
 * of the other color. Fields:
 *   nr, rows[]   rows of the current enumeration, in order: first the rows of one
 *                parity class, then those of the other, each class center-out
 *   bsplit       (build) index where the second parity class starts; the columns of
 *                the first class are fixed there and transposition pruning applies
 *   split        (probe) the same for the probe; the odd half of the needed key is
 *                fixed there and half-key pruning applies
 *   gen          generation number of the current R, so halfStamp never needs clearing
 *   buildTarget  1: the placement also stands for its mirror image (adds 2); 2: adds 1
 *   ksz          number of keys in keyList
 */
typedef struct {
    int nr, split, bsplit, gen, buildTarget, ksz;
    i64 acc;               /* probe sum */
    u64 IR, IRfirst;       /* inv(R) and its high part inv(R) >> n */
    const u64 *colMask;    /* allowed columns of each row: white[] or black[] */
    int rows[64];
    uint32_t *tab;         /* counts, indexed by idxOf(key) */
    u64 *keyList;          /* keys whose cell is non-zero */
    int *halfStamp;        /* halfStamp[odd half] == gen: some key in the table has this odd half */
    uint32_t mulO;         /* number of possible odd halves (stride of the even-half rank) */
} W;

/* Table index of key k: (rank of its even half, rank of its odd half). For a fixed R
 * all keys have the same number of even and of odd columns, so the index is dense. */
static inline uint32_t idxOf(const W *w, u64 k) { return rankTab[cE(k)] * w->mulO + rankTab[cO(k)]; }

/* Count a finished table-side placement with column set `key`. For odd n the key is
 * stored in canonical form min(key, rev(key)). */
static inline void addW(W *w, u64 key) {
    u64 k = key;
    if (oddBM) { u64 r = rev(key); if (r < k) k = r; }
    uint32_t i = idxOf(w, k), v = w->tab[i];
    if (v == 0) w->keyList[w->ksz++] = k;
    w->tab[i] = v + (w->buildTarget == 1 ? 2 : 1);
}
/* Number of table-side placements with column set exactly `key` (times λ). For odd n
 * a canonical cell holds the counts of key and rev(key) together; the two are equal,
 * so the cell is halved unless the key is its own mirror image. */
static inline i64 getW(W *w, u64 key) {
    if (!oddBM) return w->tab[idxOf(w, key)];
    u64 rk = rev(key), k = rk < key ? rk : key;
    uint32_t v = w->tab[idxOf(w, k)];
    return key == rk ? (i64)v : (i64)(v >> 1);
}
/* Key a probe placement with columns `cols` needs from a white table: the complement.
 * For even n the probe enumerates the black placements as their mirror images on white
 * squares (the mirror swaps colors when n is even), so the mirror is undone here. */
static inline u64 keyOf(u64 cols) { return even ? full & ~rev(cols) : full & ~cols; }

/* =====================================================================================
 * Normal side: white placements in the table, black placements probe it
 * =====================================================================================
 * Recursive enumeration over rows[i..nr-1]: cols = columns in use, e1 and d2 = the two
 * diagonal directions in use (bit c - r + n - 1 and bit r + c for a queen at (r, c)).
 */
static void build(W *w, int i, u64 cols, u64 e1, u64 d2) {
    if (i == w->nr) { addW(w, cols); return; }
    /* first parity class placed: cut the branch if every completion has λ = 0 */
    if (EARLY_CUT && i == w->bsplit && i > 0 && partBuild(cols) < w->IRfirst) return;
    int r = w->rows[i];
    u64 av = w->colMask[r] & ~cols & ~((e1 >> (n - 1 - r)) | (d2 >> r));   /* free squares of row r */
    if (i == w->nr - 1 && i != w->bsplit - 1) {                          /* last row: count directly */
        for (; av; av &= av - 1) addW(w, cols | (av & -av));
        return;
    }
    while (av) {
        u64 b = av & -av;
        av ^= b;
        int c = __builtin_ctzll(b);
        build(w, i + 1, cols | b, e1 | 1ULL << (c - r + n - 1), d2 | 1ULL << (r + c));
    }
}
/* Probe: enumerate the black placements on the rows ~R and add to w->acc the table
 * entry of the complementary column set of each. For odd n the first row uses the
 * left-right mirror: a first queen in the left half counts twice; a first queen in the
 * center counts once if it is the only row, otherwise the second queen is restricted
 * to the left half and counts twice. */
static void probe(W *w, int i, u64 cols, u64 e1, u64 d2) {
    if (HALF_KEY && i == w->split && i > 0 && i < w->nr) {          /* half-key pruning */
        u64 kOdd = oddCols & ~(even ? rev(cols) : cols);             /* odd half of the key needed */
        if (w->halfStamp[cO(kOdd)] != w->gen) return;
    }
    if (i == w->nr) { w->acc += getW(w, keyOf(cols)); return; }
    int r = w->rows[i];
    u64 av = w->colMask[r] & ~cols & ~((e1 >> (n - 1 - r)) | (d2 >> r));
    if (i == 0 && oddBM) {
        i64 before = w->acc;
        w->acc = 0;
        for (u64 z = av & ((1ULL << h) - 1); z; z &= z - 1) {        /* first queen in the left half */
            u64 b = z & -z;
            int c = __builtin_ctzll(b);
            probe(w, 1, b, 1ULL << (c - r + n - 1), 1ULL << (r + c));
        }
        i64 left = w->acc;
        w->acc = 0;
        if ((av >> h) & 1) {                                           /* first queen in the center */
            u64 cols = 1ULL << h, e1 = 1ULL << (h - r + n - 1), d2 = 1ULL << (r + h);
            if (w->nr == 1) {                                          /* its own mirror image: once */
                probe(w, 1, cols, e1, d2);
                left = 2 * left + w->acc;
                w->acc = before + left;
                return;
            }
            /* this branch jumps straight to level 2, so do the level-1 half-key check here */
            if (!(HALF_KEY && w->split == 1 && 1 < w->nr && w->halfStamp[cO(oddCols & ~cols)] != w->gen)) {
                int r1 = w->rows[1];
                u64 a1 = w->colMask[r1] & ~cols & ~((e1 >> (n - 1 - r1)) | (d2 >> r1)) & ((1ULL << h) - 1);
                for (; a1; a1 &= a1 - 1) {                             /* second queen: left half only */
                    u64 b = a1 & -a1;
                    int c = __builtin_ctzll(b);
                    probe(w, 2, cols | b, e1 | 1ULL << (c - r1 + n - 1), d2 | 1ULL << (r1 + c));
                }
            }
        }
        w->acc = before + 2 * left + 2 * w->acc;
        return;
    }
    if (i == w->nr - 1) {                                              /* last row: look up directly */
        i64 s = 0;
        for (; av; av &= av - 1) s += getW(w, keyOf(cols | (av & -av)));
        w->acc += s;
        return;
    }
    while (av) {
        u64 b = av & -av;
        av ^= b;
        int c = __builtin_ctzll(b);
        probe(w, i + 1, cols | b, e1 | 1ULL << (c - r + n - 1), d2 | 1ULL << (r + c));
    }
}

/* =====================================================================================
 * Reverse side: black placements in the table, white placements probe it
 * =====================================================================================
 * Same structure as build() and probe(); the table keys are black column sets, and a
 * white placement with columns cols needs the key full & ~cols.
 */
static void buildB(W *w, int i, u64 cols, u64 e1, u64 d2) {
    if (i == w->nr) { addW(w, cols); return; }
    if (EARLY_CUT && i == w->bsplit && i > 0 && pbBlack(cols) < w->IRfirst) return;
    int r = w->rows[i];
    u64 av = w->colMask[r] & ~cols & ~((e1 >> (n - 1 - r)) | (d2 >> r));
    if (i == w->nr - 1 && i != w->bsplit - 1) {
        for (; av; av &= av - 1) addW(w, cols | (av & -av));
        return;
    }
    while (av) {
        u64 b = av & -av;
        av ^= b;
        int c = __builtin_ctzll(b);
        buildB(w, i + 1, cols | b, e1 | 1ULL << (c - r + n - 1), d2 | 1ULL << (r + c));
    }
}
static void probeW(W *w, int i, u64 cols, u64 e1, u64 d2) {
    /* half-key pruning: at level split all odd rows are placed, so cols is the odd half */
    if (HALF_KEY && i == w->split && i > 0 && i < w->nr && w->halfStamp[cO(cols)] != w->gen) return;
    if (i == w->nr) { w->acc += getW(w, full & ~cols); return; }
    int r = w->rows[i];
    u64 av = w->colMask[r] & ~cols & ~((e1 >> (n - 1 - r)) | (d2 >> r));
    if (i == 0 && oddBM) {                                             /* odd n: mirror, as in probe() */
        i64 before = w->acc;
        w->acc = 0;
        for (u64 z = av & ((1ULL << h) - 1); z; z &= z - 1) {
            u64 b = z & -z;
            int c = __builtin_ctzll(b);
            probeW(w, 1, b, 1ULL << (c - r + n - 1), 1ULL << (r + c));
        }
        i64 left = w->acc;
        w->acc = 0;
        if ((av >> h) & 1) {
            u64 cols = 1ULL << h, e1 = 1ULL << (h - r + n - 1), d2 = 1ULL << (r + h);
            if (w->nr == 1) {
                probeW(w, 1, cols, e1, d2);
                left = 2 * left + w->acc;
                w->acc = before + left;
                return;
            }
            if (!(HALF_KEY && w->split == 1 && 1 < w->nr && w->halfStamp[cO(cols)] != w->gen)) {
                int r1 = w->rows[1];
                u64 a1 = w->colMask[r1] & ~cols & ~((e1 >> (n - 1 - r1)) | (d2 >> r1)) & ((1ULL << h) - 1);
                for (; a1; a1 &= a1 - 1) {
                    u64 b = a1 & -a1;
                    int c = __builtin_ctzll(b);
                    probeW(w, 2, cols | b, e1 | 1ULL << (c - r1 + n - 1), d2 | 1ULL << (r1 + c));
                }
            }
        }
        w->acc = before + 2 * left + 2 * w->acc;
        return;
    }
    if (i == w->nr - 1) {
        i64 s = 0;
        for (; av; av &= av - 1) s += getW(w, full & ~(cols | (av & -av)));
        w->acc += s;
        return;
    }
    while (av) {
        u64 b = av & -av;
        av ^= b;
        int c = __builtin_ctzll(b);
        probeW(w, i + 1, cols | b, e1 | 1ULL << (c - r + n - 1), d2 | 1ULL << (r + c));
    }
}

/* =====================================================================================
 * Table preparation and the sum for one row set R
 * ===================================================================================== */

/* c * λ, refusing values that could overflow a 32-bit table cell. */
static inline uint32_t mul(uint32_t c, int lam) {
    if (c > (1u << 30)) { fprintf(stderr, "table counter overflow\n"); exit(2); }
    return c * lam;
}
/* Weighted transposition: multiply every cell by λ = 2 if inv(C) > inv(R), 1 if
 * inv(C) == inv(R); remove the keys with λ = 0 (inv(C) < inv(R)) from the table. */
#ifdef CELL_STATS
static _Atomic unsigned long long statMaxCell, statMaxW;    /* largest stored cell, largest W_R(C) */
static void statMax(_Atomic unsigned long long *m, unsigned long long v) {
    unsigned long long o = atomic_load(m);
    while (v > o && !atomic_compare_exchange_weak(m, &o, v)) {}
}
#endif
static void filterTR(W *w, int blackTable) {
    int m = 0;
    for (int t = 0; t < w->ksz; t++) {
        u64 k = w->keyList[t]; u64 C = blackTable ? (full & ~k) : k;   /* C = white column set */
        u64 ic = inv(C); int lam = ic > w->IR ? 2 : ic == w->IR ? 1 : 0;
        uint32_t i = idxOf(w, k);
#ifdef CELL_STATS
        if (lam) {   /* for odd n a canonical cell holds the counts of key and rev(key) together */
            statMax(&statMaxW, (oddBM && rev(k) != k) ? w->tab[i] >> 1 : w->tab[i]);
            statMax(&statMaxCell, (unsigned long long)w->tab[i] * lam);
        }
#endif
        if (lam == 0) w->tab[i] = 0; else { w->tab[i] = mul(w->tab[i], lam); w->keyList[m++] = k; }
    }
    w->ksz = m;
}
/* Zero the cells in use (cheaper than clearing the whole table). */
static inline void clearTab(W *w) { for (int t = 0; t < w->ksz; t++) w->tab[idxOf(w, w->keyList[t])] = 0; }

/* Build the table for the current R (blackTable: black placements, else white). For odd
 * n the first row uses the left-right mirror as in probe(): a first queen in the left
 * half is counted with weight 2; a first queen in the center with weight 1 if it is the
 * only row, otherwise the second queen goes into the left half only, with weight 2. */
static void buildFirstRow(W *w, int blackTable) {
    void (*bf)(W *, int, u64, u64, u64) = blackTable ? buildB : build;
    if (w->nr == 0) { w->buildTarget = 2; addW(w, 0); return; }       /* no rows: one empty placement */
    if (oddBM) {
        int r0 = w->rows[0]; u64 av = w->colMask[r0];
        for (u64 z = av & ((1ULL << h) - 1); z; z &= z - 1) {        /* first queen in the left half */
            u64 b = z & -z;
            int c = __builtin_ctzll(b);
            w->buildTarget = 1; bf(w, 1, b, 1ULL << (c - r0 + n - 1), 1ULL << (r0 + c));
        }
        if ((av >> h) & 1) {                                           /* first queen in the center */
            u64 cols = 1ULL << h, e1 = 1ULL << (h - r0 + n - 1), d2 = 1ULL << (r0 + h);
            if (w->nr == 1) { w->buildTarget = 2; addW(w, cols); }      /* a lone center queen is its own mirror image */
            else if (!(EARLY_CUT && w->bsplit == 1 && (blackTable ? pbBlack(cols) : partBuild(cols)) < w->IRfirst)) {
                int r1 = w->rows[1];
                u64 a1 = w->colMask[r1] & ~cols & ~((e1 >> (n - 1 - r1)) | (d2 >> r1)) & ((1ULL << h) - 1);
                w->buildTarget = 1;                                     /* second queen: left half only, weight 2 */
                for (; a1; a1 &= a1 - 1) {
                    u64 b = a1 & -a1;
                    int c = __builtin_ctzll(b);
                    bf(w, 2, cols | b, e1 | 1ULL << (c - r1 + n - 1), d2 | 1ULL << (r1 + c));
                }
            }
        }
    } else { w->buildTarget = 2; bf(w, 0, 0, 0, 0); }
}

#ifdef CHECK_CELLS
/* Plain enumeration of the table-side placements on w->rows (no mirror, no early cut),
 * counted by column set; then every key of weight λ > 0 is compared with getW(). */
static uint32_t *exactCnt; static long long checkedCells, badCells;
static void exactRec(W *w, int i, u64 cols, u64 e1, u64 d2) {
    if (i == w->nr) { exactCnt[cols]++; return; }
    int r = w->rows[i];
    for (u64 av = w->colMask[r] & ~cols & ~((e1 >> (n - 1 - r)) | (d2 >> r)); av; av &= av - 1) {
        u64 b = av & -av; int c = __builtin_ctzll(b);
        exactRec(w, i + 1, cols | b, e1 | 1ULL << (c - r + n - 1), d2 | 1ULL << (r + c));
    }
}
static void checkCells(W *w, int blackTable) {
    memset(exactCnt, 0, sizeof(uint32_t) << n);
    exactRec(w, 0, 0, 0, 0);
    for (u64 K = 0; K <= full; K++) {
        if (__builtin_popcountll(K) != w->nr) continue;
        if (inv(blackTable ? (full & ~K) : K) < w->IR) continue;     /* λ = 0: removed anyway */
        i64 got = exactCnt[K] ? getW(w, K) : 0;
        checkedCells++;
        if ((i64)exactCnt[K] != got && badCells++ < 10)
            fprintf(stderr, "  cell mismatch: n=%d key=%llx exact=%u table=%lld\n", n, (unsigned long long)K, exactCnt[K], (long long)got);
    }
    for (int t = 0; t < w->ksz; t++) {                               /* no key without a placement */
        u64 k = w->keyList[t];
        if (!exactCnt[k] && !(oddBM && exactCnt[rev(k)]) && badCells++ < 10)
            fprintf(stderr, "  extra key: n=%d key=%llx\n", n, (unsigned long long)k);
    }
}
#define CHECK_TABLE(w, b) checkCells(w, b)
#else
#define CHECK_TABLE(w, b) ((void)0)
#endif

/* Contribution of one row set R, with the white placements in the table:
 *     Σ_C λ(C) · W_R(C) · B_~R(~C). */
static i64 pairSum(W *w, u64 R) {
    /* table: white placements on the rows R; first the rows whose white squares lie in
       the columns that partBuild() needs (even rows for even n, odd rows for odd n),
       then the others */
    w->colMask = white; w->nr = 0; w->IR = inv(R); w->IRfirst = w->IR >> n; int f = even ? 0 : 1;
    for (int j = 0; j < n; j++) {
        int r = ORD[j];
        if ((r & 1) == f && ((R >> r) & 1)) w->rows[w->nr++] = r;
    }
    w->bsplit = w->nr;
    for (int j = 0; j < n; j++) {
        int r = ORD[j];
        if ((r & 1) != f && ((R >> r) & 1)) w->rows[w->nr++] = r;
    }
    /* white queens in odd rows stand in odd columns: every key has popcount(R & odd rows) odd columns */
    w->gen++; w->ksz = 0; w->mulO = (uint32_t)binomT[nO][__builtin_popcountll(R & oddCols)];
    buildFirstRow(w, 0);
    CHECK_TABLE(w, 0);
    filterTR(w, 0);
    if (w->ksz == 0) return 0;                       /* nothing left after the λ filter; table is clear */
    /* mark the odd halves present in the table (for odd n also those of the mirrored keys) */
    for (int t = 0; t < w->ksz; t++) {
        u64 k = w->keyList[t];
        w->halfStamp[cO(k)] = w->gen;
        if (oddBM) w->halfStamp[cO(rev(k))] = w->gen;
    }
    /* probe: black placements on the rows ~R (for even n their mirror images on white
       squares, see keyOf); even rows first, as their queens fix the odd half of the key */
    u64 comp = full ^ R; w->colMask = even ? white : black; w->nr = 0;
    for (int j = 0; j < n; j++) {
        int r = ORD[j];
        if ((r & 1) == 0 && ((comp >> r) & 1)) w->rows[w->nr++] = r;
    }
    w->split = w->nr;
    for (int j = 0; j < n; j++) {
        int r = ORD[j];
        if ((r & 1) == 1 && ((comp >> r) & 1)) w->rows[w->nr++] = r;
    }
    w->acc = 0; if (w->nr == 0) w->acc = getW(w, keyOf(0)); else probe(w, 0, 0, 0, 0);
    clearTab(w); return w->acc;
}
/* The same sum with the black placements in the table and the white ones probing it. */
static i64 pairSumSwap(W *w, u64 R) {
    /* table: black placements on the rows ~R; even rows first (their black squares are in
       the odd columns, which pbBlack() needs) */
    u64 comp = full ^ R; w->IR = inv(R); w->IRfirst = w->IR >> n;
    w->colMask = black; w->nr = 0;
    for (int j = 0; j < n; j++) {
        int r = ORD[j];
        if ((r & 1) == 0 && ((comp >> r) & 1)) w->rows[w->nr++] = r;
    }
    w->bsplit = w->nr;
    for (int j = 0; j < n; j++) {
        int r = ORD[j];
        if ((r & 1) == 1 && ((comp >> r) & 1)) w->rows[w->nr++] = r;
    }
    /* black queens in even rows stand in odd columns: every key has popcount(~R & even rows) odd columns */
    w->gen++; w->ksz = 0; w->mulO = (uint32_t)binomT[nO][__builtin_popcountll(comp & evenCols)];
    buildFirstRow(w, 1);
    CHECK_TABLE(w, 1);
    filterTR(w, 1);
    if (w->ksz == 0) return 0;
    /* mark the odd halves of the white column sets that match a table key */
    for (int t = 0; t < w->ksz; t++) {
        u64 k = w->keyList[t];
        w->halfStamp[cO(oddCols & ~k)] = w->gen;
        if (oddBM) w->halfStamp[cO(oddCols & ~rev(k))] = w->gen;
    }
    /* probe: white placements on the rows R; odd rows first (their white squares are in
       odd columns, which fix the odd half of the key) */
    w->colMask = white; w->nr = 0;
    for (int j = 0; j < n; j++) {
        int r = ORD[j];
        if ((r & 1) == 1 && ((R >> r) & 1)) w->rows[w->nr++] = r;
    }
    w->split = w->nr;
    for (int j = 0; j < n; j++) {
        int r = ORD[j];
        if ((r & 1) == 0 && ((R >> r) & 1)) w->rows[w->nr++] = r;
    }
    w->acc = 0; if (w->nr == 0) w->acc = getW(w, full); else probeW(w, 0, 0, 0, 0);
    clearTab(w); return w->acc;
}

/* Sum over the row sets R = lo..hi-1 of one chunk (R = the rows of the white queens).
 *   Parity law: white queens in even rows stand in even columns and those in odd rows
 *   in odd columns; counting the rows and columns of each parity shows that R holds as
 *   many even rows as odd rows (one more even row when n is odd).
 *   Symmetries: for even n the mirrors swap colors, so the left-right mirror maps R to
 *   ~R, the top-bottom mirror to ~rev(R) and the half turn to rev(R); for odd n they
 *   keep colors and the orbit is just {R, rev(R)}. R is processed only if it is the
 *   smallest of its orbit, with weight = orbit size.
 *   Side: SIDE = 2 (auto) puts the color with more rows into the table. */
static i64 chunkSum(W *w, u64 lo, u64 hi) {
    u64 evenRows = evenCols; i64 total = 0;
    for (u64 R = lo; R < hi; R++) {
        int re = __builtin_popcountll(R & evenRows), ro = __builtin_popcountll(R & ~evenRows & full);
        if (re != ro + (even ? 0 : 1)) continue;                       /* parity law */
        u64 comp = full ^ R, fl = rev(R), cf = full ^ fl; int weight;
        if (even) {
            if (R > comp || R > fl || R > cf) continue;                 /* not the smallest of its orbit */
            weight = 1 + (comp != R) + (fl != R && fl != comp) + (cf != R && cf != comp && cf != fl);
        }
        else { if (R > fl) continue; weight = fl == R ? 1 : 2; }
        int k = __builtin_popcountll(R);
        int swap = SIDE == 2 ? (n - k > k) : SIDE == 1;                /* black placements into the table? */
        total += weight * (swap ? pairSumSwap(w, R) : pairSum(w, R));
    }
    return total;
}

/* =====================================================================================
 * Threads, chunks, checkpoint and time files
 * ===================================================================================== */
static int nChunks, *todo, nTodo; static u64 chunkSize;   /* chunks; indices still to do; row sets per chunk */
static atomic_int nextIdx, finished; static _Atomic i64 sumAll;   /* next todo entry; chunks done; running total */
static const char *ckPath;                                 /* checkpoint file, NULL = no checkpointing */
static pthread_mutex_t ckLock = PTHREAD_MUTEX_INITIALIZER; /* serializes appends to the checkpoint file */
static i64 *doneVal; static char *doneFlag; static long *doneMs;   /* per chunk: sum, done, milliseconds */
static volatile sig_atomic_t stopFlag = 0;                 /* set by SIGINT / SIGTERM */
static void onStop(int sig) { (void)sig; stopFlag = 1; }

/* Time file "<seconds> <sessions>": wall-clock time and number of all previous runs. */
static char timeFile[160]; static double prevWall = 0; static int prevSessions = 0;
static void loadTime(void) {
    prevWall = 0; prevSessions = 0;
    FILE *f = fopen(timeFile, "r");
    if (f) {
        if (fscanf(f, "%lf %d", &prevWall, &prevSessions) != 2) { prevWall = 0; prevSessions = 0; }
        fclose(f);
    }
}
/* Write the time file atomically: write <name>.tmp, then rename it over the old file. */
static void saveTime(double total, int sessions) {
    char tmp[200]; snprintf(tmp, sizeof tmp, "%s.tmp", timeFile);
    FILE *f = fopen(tmp, "w"); if (!f) return;
    fprintf(f, "%.3f %d\n", total, sessions);
    fclose(f);
    rename(tmp, timeFile);
}
/* Human-readable duration: minutes below 1 hour, hours below 48 hours, days above. */
static void fmtDur(double s, char *out, size_t sz) {
    if (s < 3600) snprintf(out, sz, "%.1f min", s / 60);
    else if (s < 172800) snprintf(out, sz, "%.2f h", s / 3600);
    else snprintf(out, sz, "%.2f days", s / 86400);
}
/* Local time as HH:MM:SS. */
static void stamp(char *out, size_t sz) {
    time_t t = time(NULL); struct tm tmv;
    localtime_r(&t, &tmv); strftime(out, sz, "%H:%M:%S", &tmv);
}
/* Monotonic clock, seconds. */
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

/* Worker thread: takes chunks from the todo list until none is left; adds each result
 * to the total and appends it to the checkpoint file. */
static void *worker(void *arg) {
    (void)arg;
    W w; memset(&w, 0, sizeof w);
    /* the largest table any R can need: C(nE, nE/2) * C(nO, nO/2) cells */
    size_t sz = (size_t)(binomT[nE][nE / 2] * binomT[nO][nO / 2]);
    w.tab = calloc(sz + 1, 4); w.keyList = calloc(sz + 1, 8); w.halfStamp = calloc((size_t)1 << nO, 4);
    if (!w.tab || !w.halfStamp || !w.keyList) { fprintf(stderr, "out of memory for the per-thread tables\n"); exit(3); }
    for (int j; (j = atomic_fetch_add(&nextIdx, 1)) < nTodo; ) {
        int c = todo[j]; double t0 = now();
        i64 v = chunkSum(&w, (u64)c * chunkSize, (u64)(c + 1) * chunkSize);
        long ms = (long)((now() - t0) * 1000);
        atomic_fetch_add(&sumAll, v); doneVal[c] = v; doneMs[c] = ms; doneFlag[c] = 1; atomic_fetch_add(&finished, 1);
        if (ckPath) {
            /* Open, append, fsync and close for every line: cloud-synced file systems
               (e.g. Google Drive mounted in Colab) save a file only when it is closed. */
            pthread_mutex_lock(&ckLock);
            FILE *f = fopen(ckPath, "a");
            if (f) {
                fprintf(f, "%d %lld %ld ok\n", c, (long long)v, ms);
                fflush(f);
                fsync(fileno(f));
                fclose(f);
            } else {
                fprintf(stderr, "  warning: cannot open %s, chunk %d not saved (it will be recomputed on resume)\n",
                        ckPath, c);
            }
            pthread_mutex_unlock(&ckLock);
        }
    }
    free(w.tab); free(w.halfStamp); free(w.keyList); return NULL;
}

/* Set up the board geometry and lookup tables for board size nn. */
static void initBoard(int nn) {
    n = nn; h = n / 2; full = (n == 64) ? ~0ULL : ((1ULL << n) - 1); even = (n & 1) == 0; oddBM = MIRROR && !even;
    oddCols = 0; for (int c = 1; c < n; c += 2) oddCols |= 1ULL << c; evenCols = full ^ oddCols;
    for (int r = 0; r < n; r++) {
        u64 m0 = 0, m1 = 0;
        for (int c = 0; c < n; c++) if (((r + c) & 1) == 0) m0 |= 1ULL << c; else m1 |= 1ULL << c;
        white[r] = m0; black[r] = m1;
    }
    nE = (n + 1) / 2; nO = n / 2;
    for (int a = 0; a <= 32; a++) {
        binomT[a][0] = 1;
        for (int b = 1; b <= a; b++) binomT[a][b] = binomT[a - 1][b - 1] + (b <= a - 1 ? binomT[a - 1][b] : 0);
    }
    /* rank of each 16-bit mask among the masks with the same number of ones, in increasing order */
    { uint32_t cnt[17] = {0}; for (uint32_t m = 0; m < (1u << 16); m++) rankTab[m] = cnt[__builtin_popcount(m)]++; }
    /* bits 0, 2, 4, 6 (EVB) and 1, 3, 5, 7 (ODB) of a byte, packed into 4 bits */
    for (int x = 0; x < 256; x++) {
        uint32_t e = 0, o = 0;
        for (int j = 0; j < 4; j++) { e |= ((x >> (2 * j)) & 1u) << j; o |= ((x >> (2 * j + 1)) & 1u) << j; }
        EVB[x] = e; ODB[x] = o;
    }
    /* rows sorted by distance from the middle, ties: smaller row first */
    double mid = (n - 1) / 2.0; for (int r = 0; r < n; r++) ORD[r] = r;
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) {
        double da = ORD[a] > mid ? ORD[a] - mid : mid - ORD[a], db = ORD[b] > mid ? ORD[b] - mid : mid - ORD[b];
        if (db < da || (db == da && ORD[b] < ORD[a])) { int t = ORD[a]; ORD[a] = ORD[b]; ORD[b] = t; }
    }
}

/* =====================================================================================
 * Command line, daemon mode, main loop over n
 * ===================================================================================== */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s n_from n_to [options]\n"
            "  -t N                    worker threads (default: number of online CPUs)\n"
            "  -p S                    progress line every S seconds, 0 = off (default 60)\n"
            "  -ckpt 0|1               checkpoint file off/on (default: on for n >= 20)\n"
            "  -from A -to B           only chunks A..B-1 of the 4096\n"
            "  -part I/K               only part I of K: chunks c with c %% K == I (I = 0..K-1)\n"
            "  -status                 compute nothing, report what the checkpoint file holds\n"
            "  -side white|black|auto  color whose placements go into the table (default auto)\n"
            "  -daemon                 run in the background: output to colorsplit_n<from>-<to>.log,\n"
            "                          PID to colorsplit.pid; stop with: kill $(cat colorsplit.pid)\n"
            "example: %s 1 18          (checks n = 1..18 against OEIS A000170)\n",
            argv[0], argv[0]);
        return 1;
    }
    int daemonMode = 0;
    int n1 = atoi(argv[1]), n2 = atoi(argv[2]);
    int threads = (int)sysconf(_SC_NPROCESSORS_ONLN);   /* -t */
    int prog = 60;                                      /* -p: progress interval in seconds, 0 = off */
    int ckOpt = -1;                                     /* -ckpt: -1 = default (on for n >= 20) */
    int from = 0, to = -1;                              /* -from, -to: chunk range; to = -1: all */
    int pI = 0, pK = 1;                                 /* -part I/K */
    int status = 0;                                     /* -status */
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prog = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ckpt") && i + 1 < argc) ckOpt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-from") && i + 1 < argc) from = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-to") && i + 1 < argc) to = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-part") && i + 1 < argc) { sscanf(argv[++i], "%d/%d", &pI, &pK); }
        else if (!strcmp(argv[i], "-status")) status = 1;
        else if (!strcmp(argv[i], "-daemon")) daemonMode = 1;
        else if (!strcmp(argv[i], "-side") && i + 1 < argc) {
            i++; SIDE = !strcmp(argv[i], "white") ? 0 : !strcmp(argv[i], "black") ? 1 : 2;
        }
    }
#ifdef CHECK_CELLS
    threads = 1;                                     /* the check uses one shared buffer */
#endif
    /* sanity checks of the options */
    if (n1 < 1) n1 = 1;
    if (threads < 1) threads = 1;
    if (from < 0) from = 0;
    if (pK < 1 || pI < 0 || pI >= pK) {
        fprintf(stderr, "bad -part %d/%d: use I/K with K >= 1 and 0 <= I < K\n", pI, pK);
        return 1;
    }
    if (daemonMode) {
        char logName[96]; snprintf(logName, sizeof logName, "colorsplit_n%d-%d.log", n1, n2);
        pid_t pid = fork(); if (pid < 0) { perror("fork"); return 1; }
        if (pid > 0) {                                  /* parent: report and exit */
            printf("started in the background: PID %d\n"
                   "  progress: tail -f %s\n"
                   "  stop:     kill $(cat colorsplit.pid)\n", pid, logName);
            return 0;
        }
        /* child: new session, stdin from /dev/null, stdout and stderr appended to the log */
        setsid();
        int dn = open("/dev/null", O_RDONLY); if (dn >= 0) dup2(dn, 0);
        int lf = open(logName, O_WRONLY | O_CREAT | O_APPEND, 0644); if (lf >= 0) { dup2(lf, 1); dup2(lf, 2); }
        FILE *pf = fopen("colorsplit.pid", "w"); if (pf) { fprintf(pf, "%d\n", (int)getpid()); fclose(pf); }
    }
    /* SIGINT/SIGTERM: stop cleanly (handled in the main loop); SIGHUP: keep running */
    signal(SIGHUP, SIG_IGN); signal(SIGINT, onStop); signal(SIGTERM, onStop);
    setvbuf(stdout, NULL, _IOLBF, 0);
    {
        char ts[16], partInfo[48] = "";
        stamp(ts, sizeof ts);
        if (pK > 1) snprintf(partInfo, sizeof partInfo, ", part %d/%d", pI, pK);
        printf("[%s] colorsplit (C, ColorSplit, threads: %d%s)\n", ts, threads, partInfo);
        fflush(stdout);
    }
    for (int nn = n1; nn <= n2; nn++) {
        if (nn > 31) { printf("n > 31 is not supported\n"); break; }
        if (nn >= 29)
            fprintf(stderr, "  note: Q(%d) may exceed 2^63, so the total printed here can overflow; the chunk sums\n"
                            "  in the checkpoint file stay exact, add them with: python3 tools/verify.py merge ...\n", nn);
        if (nn == 1) { printf("n= 1: 1 OK\n"); continue; }
        initBoard(nn);
#ifdef CHECK_CELLS
        free(exactCnt); exactCnt = calloc((size_t)1 << n, sizeof(uint32_t)); checkedCells = badCells = 0;
#endif
#ifdef CELL_STATS
        atomic_store(&statMaxCell, 0); atomic_store(&statMaxW, 0);
#endif
        /* chunk c holds the row sets R = c * chunkSize .. (c + 1) * chunkSize - 1 */
        int chunkBits = n > 12 ? n - 12 : 0; nChunks = (int)((1ULL << n) >> chunkBits); chunkSize = 1ULL << chunkBits;
        int useCk = ckOpt < 0 ? (n >= 20) : ckOpt;
        doneVal = calloc(nChunks, sizeof(i64)); doneFlag = calloc(nChunks, 1); doneMs = calloc(nChunks, sizeof(long));
        char fname[128];
        if (pK > 1) snprintf(fname, sizeof fname, "colorsplit_n%d_part%dof%d.ckpt", n, pI, pK);
        else snprintf(fname, sizeof fname, "colorsplit_n%d.ckpt", n);
        /* time file: the checkpoint file name with ".ckpt" replaced by ".time" */
        snprintf(timeFile, sizeof timeFile, "%.*s.time", (int)(strlen(fname) - 5), fname); prevWall = 0; prevSessions = 0;
        if (useCk) loadTime();
        int sessions = prevSessions + 1;
        /* resume: load the chunks already in the checkpoint file */
        int loaded = 0;
        if (useCk) {
            FILE *in = fopen(fname, "r");
            if (in) {
                char line[256];
                while (fgets(line, sizeof line, in)) {
                    int c; long long v; long ms; char okw[8] = "";
                    int m = sscanf(line, "%d %lld %ld %7s", &c, &v, &ms, okw);
                    if (m == 4 && !strcmp(okw, "ok") && c >= 0 && c < nChunks && !doneFlag[c]) {
                        doneFlag[c] = 1; doneVal[c] = v; doneMs[c] = ms; loaded++;
                    } else if (m == 3) {
                        /* meant for "c v ok" lines of older versions; note that such a line
                           stops sscanf above at m == 2, so it is in fact skipped */
                        char w3[8];
                        if (sscanf(line, "%d %lld %7s", &c, &v, w3) == 3 && !strcmp(w3, "ok")
                            && c >= 0 && c < nChunks && !doneFlag[c]) {
                            doneFlag[c] = 1; doneVal[c] = v; loaded++;
                        }
                    }
                }
                fclose(in);
                if (loaded) fprintf(stderr, "  resuming: %d chunk(s) already done (%s)\n", loaded, fname);
            }
        }
        /* this run's chunks: from..hi-1, and with -part only those with c % pK == pI */
        int hi = to < 0 || to > nChunks ? nChunks : to; if (status) { from = hi = 0; }
        todo = malloc(sizeof(int) * nChunks); nTodo = 0; int mine = 0; i64 base = 0;
        for (int c = from; c < hi; c++)
            if (c % pK == pI) { mine++; if (doneFlag[c]) base += doneVal[c]; else todo[nTodo++] = c; }
        atomic_store(&nextIdx, 0); atomic_store(&finished, mine - nTodo); atomic_store(&sumAll, base);
        /* create the checkpoint file now; if it cannot be opened, run without checkpointing.
           The workers reopen it for every line they append. If an earlier run was killed in
           the middle of a line, the file does not end with a newline: add one, so that the
           torn line stays a separate (ignored) line instead of merging with the next one. */
        ckPath = NULL;
        if (useCk) {
            FILE *f = fopen(fname, "a+");
            if (f) {
                if (fseek(f, -1, SEEK_END) == 0 && fgetc(f) != '\n') {
                    fseek(f, 0, SEEK_END); fputc('\n', f); fflush(f); fsync(fileno(f));
                }
                fclose(f); ckPath = fname;
            }
            else fprintf(stderr, "  warning: cannot open %s, running without checkpointing\n", fname);
        }
        int th = threads < nTodo ? threads : (nTodo > 0 ? nTodo : 1);
        double t0 = now();
        if (prog > 0 && n >= 19 && nTodo > 0) fprintf(stderr, "  start n=%d, threads: %d, chunks: %d\n", n, th, nTodo);
        pthread_t *tid = malloc(sizeof(pthread_t) * th);
        for (int i = 0; i < th && nTodo > 0; i++) pthread_create(&tid[i], NULL, worker, NULL);
        if (nTodo > 0) {
            double last = now(), lastSave = now();
            while (atomic_load(&finished) < mine) {
                usleep(200000);
                if (stopFlag) {                         /* finished chunks are already in the checkpoint file */
                    if (useCk) saveTime(prevWall + (now() - t0), sessions);
                    char ts[16]; stamp(ts, sizeof ts);
                    fprintf(stderr, "  [%s] stopped: progress and time saved, run the same command again to resume\n", ts);
                    fflush(stderr); _exit(0);
                }
                if (useCk && now() - lastSave >= 30) { saveTime(prevWall + (now() - t0), sessions); lastSave = now(); }
                if (prog > 0 && now() - last >= prog) {
                    last = now(); i64 s2 = atomic_load(&sumAll);
                    char ts[16], d1[32], d2[32]; stamp(ts, sizeof ts);
                    fmtDur(now() - t0, d1, sizeof d1); fmtDur(prevWall + (now() - t0), d2, sizeof d2);
                    fprintf(stderr, "  [%s] n=%d | chunks %d/%d | this run %s, all sessions %s | sum %lld",
                            ts, n, atomic_load(&finished), mine, d1, d2, (long long)s2);
                    if (pK == 1 && n < NREF) fprintf(stderr, " (%.1f%% of Q(n))", 100.0 * s2 / REF[n]);
                    fprintf(stderr, "\n"); fflush(stderr);
                }
            }
            for (int i = 0; i < th; i++) pthread_join(tid[i], NULL);
        }
        free(tid); ckPath = NULL;
        double secs = now() - t0; i64 ans = atomic_load(&sumAll);
        if (useCk && nTodo > 0) saveTime(prevWall + secs, sessions);
        int ranged = !(from == 0 && hi == nChunks);
        /* a chunk range was requested: report it, and the total once the file is complete */
        if (ranged) {
            int nd = 0; i64 all = 0;
            for (int c = 0; c < nChunks; c++) if (doneFlag[c]) { nd++; all += doneVal[c]; }
            printf("  chunks %d..%d: sum %lld | done in file: %d of %d\n", from, hi - 1, (long long)ans, nd, nChunks);
            if (nd == nChunks) { ans = all; ranged = 0; printf("  ALL CHUNKS DONE, full sum:\n"); }
        }
        if (!ranged) {
            const char *chk = pK > 1 ? "(part)" : n < NREF ? (ans == REF[n] ? "OK" : "MISMATCH") : "(no reference value)";
            printf("n=%2d: %lld %s | %.3f s (%.1f min)\n", n, (long long)ans, chk, secs, secs / 60);
            if (useCk) {
                double totMs = 0; for (int c = 0; c < nChunks; c++) totMs += doneMs[c];
                char d1[32], d2[32];
                fmtDur(prevWall + (nTodo > 0 ? secs : 0), d1, sizeof d1); fmtDur(totMs / 1000, d2, sizeof d2);
                printf("     time over all sessions: %s (sessions: %d) | sum of chunk times: %s "
                       "(thread time; equals vCPU time for cost estimates when threads = vCPUs)\n", d1, sessions, d2);
            }
        }
#ifdef CHECK_CELLS
        printf("     cells n=%d: %lld compared, %lld wrong\n", n, checkedCells, badCells);
#endif
#ifdef CELL_STATS
        printf("     cells n=%d: largest stored cell %llu, largest W_R(C) %llu\n", n,
               (unsigned long long)atomic_load(&statMaxCell), (unsigned long long)atomic_load(&statMaxW));
#endif
        fflush(stdout); free(todo); free(doneVal); free(doneFlag); free(doneMs);
    }
    return 0;
}
