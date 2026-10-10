// SPDX-License-Identifier: MIT
/*
 * ColorSplit — exact counting of n-queens solutions, GPU version (CUDA)
 *
 * WHAT IT COMPUTES
 *   Q(n), the number of ways to place n non-attacking queens on an n x n board (OEIS A000170),
 *   as an exact integer. Known values up to n = 27 are built in and every finished n is checked
 *   against them ("OK" / "MISMATCH"); for larger n the total is checked for divisibility by 4.
 *
 * THE IDEA
 *   Color the board like a chessboard: square (r, c) is white when r + c is even. Every diagonal
 *   has a single color, so queens of different colors never share a diagonal and interact only
 *   through rows and columns. Every solution is therefore a non-attacking set of white queens on
 *   a set R of rows using a set C of columns plus a non-attacking set of black queens on the
 *   remaining rows ~R using the remaining columns ~C:
 *       Q(n) = sum over R, sum over C of  W_R(C) * B_~R(~C),
 *   where W_R(C) (resp. B_~R(~C)) is the number of white (black) placements on those rows that
 *   use exactly those columns. For each row set R the placements of one color are tabulated by
 *   their column set in a compact table indexed by combinatorial rank, and every placement of the
 *   other color looks up the complementary column set.
 *
 * REDUCTIONS
 *   - Parity law: the even columns are filled by white queens on even rows and black queens on odd
 *     rows, so R has as many even rows as odd rows for even n (one more for odd n); other row sets
 *     are skipped.
 *   - Symmetries of row sets: one representative R per symmetry class, counted with weight w.
 *   - Transposition (r, c) -> (c, r) turns a white placement on (R, C) into one on (C, R): each
 *     table entry gets a weight λ = 2, 1 or 0 by comparing symmetry invariants of C and R, and
 *     branches that can only produce λ = 0 are pruned early.
 *   - Rows are searched from the center of the board outwards.
 *   - Odd n: column mirror c -> n-1-c (first queen in the left half or the center column, nested
 *     mirror on the second row when the first queen is in the center).
 *   - Half keys: a probe branch is cut as soon as the odd-column half of the column set it needs
 *     is known and does not occur in the table.
 *
 * GPU DESIGN
 *   - The host enumerates the canonical row sets R and groups them into 4096 chunks by the top
 *     12 bits of R (2^n chunks for n <= 12). One kernel launch processes a batch of chunks
 *     (-batch) and returns one 64-bit sum per chunk; the checkpoint is written after each batch.
 *   - Each thread block takes one row set at a time from a global counter (dynamic assignment), the
 *     heavier row sets first (sorted by an estimate of their cost), so that small batches lose little
 *     to blocks that finish early. The threads of the block share the depth-PFX prefixes of the
 *     search tree: all threads walk the prefixes
 *     in the same order; a thread searches the subtree of the prefix whose number equals its
 *     ticket from a shared counter, then takes the next ticket (dynamic distribution). The
 *     depth-first search keeps only the untried columns and the chosen column per level (compact
 *     stack); occupied columns and diagonals are updated incrementally.
 *   - Steps per row set, separated by __syncthreads(): build the table (32-bit counters in global
 *     memory, atomicAdd; the first touch of a cell appends its key to a key list), apply the
 *     transposition weights λ, mark the half keys in a shared-memory bitmap, probe with the other
 *     color, add w * (block sum) to the chunk total, clear the touched cells.
 *   - Table side: by default (-side auto) the color with more rows goes into the table (white on a
 *     tie); -side white|black forces one side. Per-chunk sums do not depend on the side, so a run
 *     with a forced side is an independent check of chunks computed before.
 *   - Overflow guard: if a table counter could exceed 32 bits (including λ = 2), the program prints
 *     an error and exits with code 3 without saving the current batch.
 *
 * BUILD
 *   nvcc -O3 -arch=native -o colorsplit_gpu colorsplit.cu
 *   Explicit architecture instead of native:
 *     -arch=sm_86    RTX A6000, RTX 3090, A40
 *     -arch=sm_89    L4, RTX 4090, RTX 6000 Ada
 *     -arch=sm_120   RTX 5090, RTX PRO 6000 Blackwell (needs CUDA >= 12.8)
 *   -DPFX=N changes the prefix depth (default 4).
 *   CPU emulation for testing without a GPU (runs each block step for all "threads" in turn; slow):
 *     g++ -O2 -x c++ -DEMULATE -o colorsplit_emu colorsplit.cu
 *
 * USAGE
 *   colorsplit_gpu n_from n_to [options]
 *   ./colorsplit_gpu 4 20                         check n = 4..20 against the known values
 *   ./colorsplit_gpu 27 27 -dir /content/drive/MyDrive/nq
 *                                                 long run with the checkpoint on Google Drive;
 *                                                 run the same command again to resume
 *   ./colorsplit_gpu 27 27 -from 0 -to 1024       only chunks 0..1023 (e.g. to split the work)
 *   ./colorsplit_gpu 26 26 -side black -chunks sample.txt
 *                                                 recompute the chunks listed in sample.txt with a
 *                                                 forced table side and compare their sums
 *
 * OPTIONS
 *   -t N                    threads per block (default 512)
 *   -b N                    thread blocks (default 8 per SM), always capped to fit 70% of free GPU memory
 *   -ckpt 0|1               checkpoint off / on (default: on for n >= 20); a rerun resumes from it
 *   -dir DIR                directory for the checkpoint and time files (default: current directory)
 *   -from a -to b           compute only chunks a..b-1 (default: all chunks)
 *   -batch N                chunks per kernel launch (default 32); the checkpoint is saved after each batch.
 *                           For n >= 28 use -batch 8 or less: the first batches of Q(28) take hours each
 *   -chunks FILE            compute only the chunks listed in FILE (whitespace-separated numbers); overrides -from/-to
 *   -side white|black|auto  color that goes into the table (default auto); white/black use separate files
 *   Unknown options are ignored. A value other than white or black for -side means auto.
 *
 * MEMORY PER BLOCK
 *   Global memory: table + key list = 2 * 4 bytes * S cells, S = C(nE, nE/2) * C(nO, nO/2) + 1,
 *   where nE = (n+1)/2 and nO = n/2 are the numbers of even and odd columns (integer division):
 *     n = 24: 6.8 MB     n = 25: 12.7 MB    n = 26: 23.6 MB    n = 27: 47.1 MB
 *     n = 28: 94.2 MB    n = 29: 176.7 MB   n = 30: 331.3 MB   n = 31: 662.5 MB
 *   The number of blocks is reduced so that all blocks together use at most 70% of the free GPU
 *   memory. Shared memory: about 4.4 KB per block (block state and half-key bitmap).
 *
 * FILES (in -dir)
 *   colorsplit_n<n>.ckpt            checkpoint; with -side white / black: colorsplit_n<n>_sideW.ckpt /
 *                                   colorsplit_n<n>_sideB.ckpt. One line per finished chunk:
 *                                       <chunk> <sum> <milliseconds> ok
 *                                   (<milliseconds> = time of its batch / chunks in the batch).
 *                                   Rewritten completely after each batch (<file>.tmp, fsync,
 *                                   rename); on a restart the chunks found in it are skipped. Same
 *                                   format as the CPU versions, so CPU and GPU runs can share one
 *                                   file. colorsplit_v2.cu writes the same file with identical
 *                                   per-chunk sums, so its runs can be continued with this program.
 *   colorsplit_n<n>.time            (same suffixes)  <seconds> <sessions>: accumulated wall-clock
 *                                   time and number of runs.
 *
 * LIMITS
 *   n <= 31 (32-bit masks; the half-key bitmap holds 2^15 bits). The grand total is an unsigned
 *   64-bit sum, enough for n <= 28; Q(29) (about 3*10^19) exceeds 2^64, so for n >= 29 the printed
 *   total wraps around and the per-chunk sums of the checkpoint file must be added with arbitrary
 *   precision (the divisibility-by-4 check still holds, as 2^64 is divisible by 4).
 *
 * EXIT CODES
 *   0 done; 1 usage error, unreadable chunk list or CUDA error; 3 table counter overflow guard.
 *
 * HISTORY
 *   Prototype nqueens_gpu.cu. Version 2 (colorsplit_v2.cu, used for the Q(25)..Q(27) runs in the paper):
 *   checkpoint per chunk, accumulated time, choice of the table color, nested mirror for odd n.
 *   Version 3: -side, bit compression by shifts instead of lookup tables, compact search stack,
 *   dynamic distribution of subtrees within a block, checkpoint rewritten and closed after each batch.
 *   Version 4 (originally nqueens_v4.cu): overflow guard on the table counters, -chunks.
 *   Version 5 (this file): row sets are handed to the blocks dynamically, heavy ones first, so that a
 *   batch ends with all blocks busy and small batches (-batch 4 to 8, for frequent checkpoints on the
 *   largest boards) cost little; the CUDA versions are printed at the start of the run.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <unistd.h>
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned long long ull;

#ifdef EMULATE
  /* CPU emulation: the CUDA keywords expand to nothing, and the intrinsics and atomics used by the
     device code get plain sequential versions; main() runs each block step for all "threads" in turn. */
  #define __global__
  #define __device__
  #define __host__
  #define __forceinline__ inline
  static inline u32 atomicAdd(u32 *p, u32 v) { u32 o = *p; *p = o + v; return o; }
  static inline int atomicAdd(int *p, int v) { int o = *p; *p = o + v; return o; }
  static inline ull atomicAdd(ull *p, ull v) { ull o = *p; *p = o + v; return o; }
  static inline u32 atomicOr(u32 *p, u32 v) { u32 o = *p; *p = o | v; return o; }
  static inline u32 __brev(u32 x) {   /* reverse the order of the 32 bits */
      x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
      x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
      x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
      return __builtin_bswap32(x);
  }
  static inline int __popc(u32 x) { return __builtin_popcount(x); }
  static inline int __ffs(u32 x) { return __builtin_ffs(x); }
#endif

/* Known values Q(0..27) (OEIS A000170), used to check the results. */
static const ull REF[] = {1ULL, 1ULL, 0ULL, 0ULL, 2ULL, 10ULL, 4ULL, 40ULL, 92ULL, 352ULL, 724ULL, 2680ULL, 14200ULL,
    73712ULL, 365596ULL, 2279184ULL, 14772512ULL, 95815104ULL, 666090624ULL, 4968057848ULL, 39029188884ULL,
    314666222712ULL, 2691008701644ULL, 24233937684440ULL, 227514171973736ULL, 2207893435808352ULL,
    22317699616364044ULL, 234907967154122528ULL};
#define NREF 28
#ifndef PFX
#define PFX 4          /* depth of the search-tree prefixes shared out among the threads of a block (-DPFX=N) */
#endif
#define SENT 0xFFFFFFFFu   /* key-list entry removed by the transposition filter */

/* Board parameters shared by host and device (constant memory on the GPU). */
struct G {
    /* h = n/2; even: n is even; oddBM: n is odd (column mirror in use); nE, nO: numbers of even / odd columns */
    int n, h, even, oddBM, nE, nO;
    /* full: all n columns; oddCols / evenCols: odd / even columns (also used as masks of odd / even rows);
       white[r] / black[r]: the white / black squares of row r */
    u32 full, oddCols, evenCols, white[32], black[32];
    int ORD[32];       /* search order: rows sorted from the center outwards */
    /* EVB / ODB: byte tables for bit compression (still filled in, unused since unzip() replaced them);
       binomNO[k] = C(nO, k) */
    u32 EVB[256], ODB[256], binomNO[17];
    int sideMode;      /* 0: white queens go into the table, 1: black queens, 2: automatic */
};
#ifdef EMULATE
static G cg;
static int dErr = 0;
#else
__constant__ G cg;
__device__ int dErr = 0;   /* set to 1 when a table counter may overflow */
__device__ int dNextR = 0; /* next row set of the batch to hand out (reset before every launch) */
#endif

/* State of a thread block for its current row set (shared memory). */
struct Ctx {
    u32 R;                   /* rows of the white queens */
    int w, swap, cidx;       /* symmetry weight of R; swap = 1: black queens go into the table; chunk index in the batch */
    u64 IR, IRfirst;         /* transposition invariant inv(R) and its high part inv(R) >> n */
    u32 mulO;                /* number of possible odd halves of a key: multiplier of the compact index */
    int nrB, bsplit, nrP, split;   /* table side: number of rows, start of its second row parity; same for the probe side */
    int rowsB[32], rowsP[32];      /* rows of the table side and of the probe side, in search order */
    int keyCount, alive, nextB, nextP;   /* keys in the key list; 1 if a key survived the filter; ticket counters */
    ull acc;                 /* sum of the block for R (before the weight w) */
    u32 stamp[1025];         /* half-key bitmap: 2^(n/2) bits */
};

/* ---------- bit helpers ---------- */
__device__ __forceinline__ u32 rev(u32 x) { return __brev(x) >> (32 - cg.n); }   /* mirror a set of columns or rows: i -> n-1-i */
__device__ __forceinline__ int ctz(u32 x) { return __ffs(x) - 1; }               /* index of the lowest set bit */
__device__ __forceinline__ u64 umin(u64 a, u64 b) { return a < b ? a : b; }
__device__ __forceinline__ u64 umax(u64 a, u64 b) { return a > b ? a : b; }
/* the smaller of a set of odd columns and its complement within the odd columns */
__device__ __forceinline__ u64 hOdd(u32 s) { u32 t = cg.oddCols ^ s; return s < t ? s : t; }
/* Symmetry invariant of a row set or column set; the transposition weight compares inv(C) with inv(R).
   Its high part (>> n) is bounded from above as soon as one parity half of the set is known, which
   allows pruning before the set is complete (partBuild, pbBlack). */
__device__ u64 inv(u32 S) {
    if (cg.even) {
        u64 u = hOdd(S & cg.oddCols), v = hOdd(rev(S & cg.evenCols));
        return umin(u, v) << cg.n | umax(u, v);
    }
    u64 a = ((u64)(S & cg.oddCols) << cg.n) | (S & cg.evenCols);
    u32 rs = rev(S);
    u64 b = ((u64)(rs & cg.oddCols) << cg.n) | (rs & cg.evenCols);
    return umin(a, b);
}
/* white queens in the table: upper bound of the high part of inv(C) (exact for odd n), known once the rows
   of the first parity are placed; if it is below inv(R) >> n, then inv(C) < inv(R) and λ = 0 */
__device__ __forceinline__ u64 partBuild(u32 cols) {
    if (cg.even) return hOdd(rev(cols & cg.evenCols));
    u32 a = cols & cg.oddCols, b = rev(a);
    return a < b ? a : b;
}
/* bits at even positions -> consecutive low bits (like pext), with shifts only, no memory access */
__device__ __forceinline__ u32 unzip(u32 x) {
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4)) & 0x00FF00FFu;
    x = (x | (x >> 8)) & 0x0000FFFFu;
    return x; }
__device__ __forceinline__ u32 cE(u32 x) { return unzip(x); }        /* even-column half of a key, compressed */
__device__ __forceinline__ u32 cO(u32 x) { return unzip(x >> 1); }   /* odd-column half of a key, compressed */
/* compact table index: rank of the even half * number of possible odd halves + rank of the odd half */
__device__ __forceinline__ u32 idxOf(u32 k, u32 mulO, const u32 *rankTab) { return rankTab[cE(k)] * mulO + rankTab[cO(k)]; }
/* white queens in the table: key (white column set) completing a probe placement on columns cols
   (for even n the probe placement is mirrored, hence rev) */
__device__ __forceinline__ u32 keyOf(u32 cols) { return cg.even ? cg.full & ~rev(cols) : cg.full & ~cols; }
/* black queens in the table: same bound as partBuild; cols are black columns, the free odd columns are white ones */
__device__ __forceinline__ u64 pbBlack(u32 cols) {
    u32 cOdd = cg.oddCols & ~(cols & cg.oddCols);
    if (cg.even) return hOdd(cOdd);
    u32 r = rev(cOdd) & cg.oddCols;
    return cOdd < r ? cOdd : r; }

/* ---------- steps for one row set R (all threads of the block take part) ---------- */
/* thread 0: set up the block state for row set R */
__device__ void setupR(Ctx &s, u32 R, int w, int cidx) {
    s.R = R; s.w = w; s.cidx = cidx; s.IR = inv(R); s.IRfirst = s.IR >> cg.n;
    int k = __popc(R);
    s.swap = cg.sideMode == 2 ? (cg.n - k > k) : (cg.sideMode == 1);  /* auto: the color with more rows goes into the table */
    u32 comp = cg.full ^ R;
    if (!s.swap) {                                                    /* white queens (rows R) -> table, black queens probe */
        s.mulO = cg.binomNO[__popc(R & cg.oddCols)];
        int f = cg.even ? 0 : 1;
        s.nrB = 0;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == f && ((R >> r) & 1)) s.rowsB[s.nrB++] = r; }
        s.bsplit = s.nrB;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) != f && ((R >> r) & 1)) s.rowsB[s.nrB++] = r; }
        s.nrP = 0;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 0 && ((comp >> r) & 1)) s.rowsP[s.nrP++] = r; }
        s.split = s.nrP;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 1 && ((comp >> r) & 1)) s.rowsP[s.nrP++] = r; }
    } else {                                                          /* black queens (other rows) -> table, white queens probe */
        s.mulO = cg.binomNO[__popc(comp & cg.evenCols)];              /* black queens on even rows stand on odd columns */
        s.nrB = 0;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 0 && ((comp >> r) & 1)) s.rowsB[s.nrB++] = r; }
        s.bsplit = s.nrB;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 1 && ((comp >> r) & 1)) s.rowsB[s.nrB++] = r; }
        s.nrP = 0;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 1 && ((R >> r) & 1)) s.rowsP[s.nrP++] = r; }
        s.split = s.nrP;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 0 && ((R >> r) & 1)) s.rowsP[s.nrP++] = r; }
    }
    s.keyCount = 0; s.alive = 0; s.acc = 0; s.nextB = 0; s.nextP = 0;
}
/* add inc to the table cell of key; the thread that touches a cell first appends the key to the key list */
__device__ __forceinline__ void insertKey(Ctx &s, u32 *tab, u32 *keys, const u32 *rankTab, u32 key, u32 inc) {
    u32 k = key;
    if (cg.oddBM) { u32 rk = rev(key); if (rk < k) k = rk; }         /* odd n: a key and its mirror share one cell */
    u32 i = idxOf(k, s.mulO, rankTab);
    u32 old = atomicAdd(&tab[i], inc);
    if (old >= 0x7FFFFFF0u) dErr = 1;                                 /* stay below 2^31: still fits in 32 bits after λ = 2 */
    if (old == 0) { int pos = atomicAdd(&s.keyCount, 1); keys[pos] = k; }
}
/* Table (white queens on R, or black queens on the other rows): compact search stack + dynamic distribution
 * of the depth-P subtrees. All threads enumerate the depth-P prefixes in the same order; a thread takes a
 * ticket from the shared counter, processes the prefix with that number and, when its subtree is done,
 * takes the next ticket. */
__device__ void buildR(Ctx &s, u32 *tab, u32 *keys, const u32 *rankTab, int tid, int nth) {
    int nr = s.nrB;
    if (nr == 0) { if (tid == 0) insertKey(s, tab, keys, rankTab, 0, 1); return; }
    const int n = cg.n;
    int P = nr < PFX ? nr : PFX;
    const u32 *cm = s.swap ? cg.black : cg.white;
    u32 avs[32];             /* avs[d]: columns still to try at depth d */
    unsigned char ch[32];    /* ch[d]: column of the queen placed at depth d */
    u32 cols = 0;            /* occupied columns */
    u64 e1 = 0, d2 = 0;      /* occupied diagonals: bit c - r + n - 1 and bit r + c */
    u32 inc = 1;             /* weight of the current placement (2 when its mirror image is implied) */
    int center = 0;          /* odd n: the first queen is in the center column */
    int ticket = atomicAdd(&s.nextB, 1), idx = 0, inPrefix = 0;
    avs[0] = cm[s.rowsB[0]];
    if (cg.oddBM) avs[0] &= (2u << cg.h) - 1;                                      /* odd n: left half and center */
    int d = 0;
    while (true) {
        if (avs[d] == 0) {
            if (d == 0) break;
            d--;                                                                    /* backtrack: remove the queen of depth d */
            int c = ch[d], r = s.rowsB[d];
            cols ^= 1u << c; e1 ^= 1ULL << (c - r + n - 1); d2 ^= 1ULL << (r + c);
            if (inPrefix && d < P) { inPrefix = 0; ticket = atomicAdd(&s.nextB, 1); }   /* subtree done: new ticket */
            continue;
        }
        u32 b = avs[d] & (0u - avs[d]);                                             /* next column to try */
        avs[d] ^= b;
        int c = ctz(b), r = s.rowsB[d];
        u32 nc = cols | b;
        u64 ne = e1 | 1ULL << (c - r + n - 1), nd = d2 | 1ULL << (r + c);
        if (d == 0) { center = cg.oddBM && c == cg.h; inc = (cg.oddBM && c < cg.h) ? 2 : 1; }
        int d1 = d + 1;
        if (d1 == P) {                                                              /* prefix of depth P: is it ours? */
            int mine = (idx == ticket);
            idx++;
            if (!mine) continue;
            if (d1 == nr || (d1 == s.bsplit && (s.swap ? pbBlack(nc) : partBuild(nc)) < s.IRfirst)) {
                if (d1 == nr) insertKey(s, tab, keys, rankTab, nc, inc);
                ticket = atomicAdd(&s.nextB, 1);
                continue; }
            inPrefix = 1; }
        else {
            if (d1 == nr) { insertKey(s, tab, keys, rankTab, nc, inc); continue; }
            if (d1 == s.bsplit && (s.swap ? pbBlack(nc) : partBuild(nc)) < s.IRfirst) continue;  /* transposition pruning */
        }
        cols = nc; e1 = ne; d2 = nd; ch[d] = (unsigned char)c;                      /* go one level deeper */
        int rr = s.rowsB[d1];
        avs[d1] = cm[rr] & ~cols & ~(u32)((e1 >> (n - 1 - rr)) | (d2 >> rr));
        if (d1 == 1 && center) { avs[1] &= (1u << cg.h) - 1; inc = 2; }            /* nested mirror */
        d = d1;
    }
}
/* transposition weights λ: drop keys with inv(C) < inv(R), double those with inv(C) > inv(R) */
__device__ void filterR(Ctx &s, u32 *tab, u32 *keys, const u32 *rankTab, int tid, int nth) {
    for (int t = tid; t < s.keyCount; t += nth) {
        u32 k = keys[t];
        u64 ic = inv(s.swap ? (cg.full & ~k) : k);                                  /* invariant of the white column set */
        u32 i = idxOf(k, s.mulO, rankTab);
        if (ic < s.IR) { tab[i] = 0; keys[t] = SENT; }
        else if (ic > s.IR) { if (tab[i] >= 0x80000000u) dErr = 1; tab[i] *= 2; }
    }
}
/* mark the odd half of the white column set of every surviving key in the bitmap (odd n: also that of the
   mirrored key) */
__device__ void stampR(Ctx &s, u32 *keys, int tid, int nth) {
    for (int t = tid; t < s.keyCount; t += nth) {
        u32 k = keys[t];
        if (k == SENT) continue;
        u32 a = s.swap ? cO(cg.oddCols & ~k) : cO(k);
        atomicOr(&s.stamp[a >> 5], 1u << (a & 31));
        if (cg.oddBM) {
            u32 b = s.swap ? cO(cg.oddCols & ~rev(k)) : cO(rev(k));
            atomicOr(&s.stamp[b >> 5], 1u << (b & 31)); }
        s.alive = 1;
    }
}
/* table value for column set key (odd n: the cell is shared with the mirrored key and holds both counts) */
__device__ __forceinline__ ull getW(Ctx &s, const u32 *tab, const u32 *rankTab, u32 key) {
    if (!cg.oddBM) return tab[idxOf(key, s.mulO, rankTab)];
    u32 rk = rev(key), k = rk < key ? rk : key;
    u32 v = tab[idxOf(k, s.mulO, rankTab)];
    return key == rk ? v : (v >> 1);
}
/* probe: enumerate the other color on its rows and add up the table values of the complementary column
   sets; same scheme as buildR (compact stack, dynamic distribution of subtrees) */
__device__ void probeR(Ctx &s, const u32 *tab, const u32 *rankTab, int tid, int nth) {
    if (!s.alive) return;
    int nr = s.nrP;
    const int n = cg.n;
    if (nr == 0) { if (tid == 0) atomicAdd(&s.acc, getW(s, tab, rankTab, s.swap ? cg.full : keyOf(0))); return; }
    const u32 *cm = s.swap ? cg.white : (cg.even ? cg.white : cg.black);   /* even n, no swap: black queens enumerated mirrored */
    int P = nr < PFX ? nr : PFX;
    u32 avs[32];
    unsigned char ch[32];
    u32 cols = 0;
    u64 e1 = 0, d2 = 0;
    ull wcur = 1, local = 0;  /* weight of the current placement; partial sum of this thread */
    int center = 0;
    int ticket = atomicAdd(&s.nextP, 1), idx = 0, inPrefix = 0;
    avs[0] = cm[s.rowsP[0]];
    if (cg.oddBM) avs[0] &= (2u << cg.h) - 1;
    int d = 0;
    while (true) {
        if (avs[d] == 0) {
            if (d == 0) break;
            d--;
            int c = ch[d], r = s.rowsP[d];
            cols ^= 1u << c; e1 ^= 1ULL << (c - r + n - 1); d2 ^= 1ULL << (r + c);
            if (inPrefix && d < P) { inPrefix = 0; ticket = atomicAdd(&s.nextP, 1); }
            continue;
        }
        u32 b = avs[d] & (0u - avs[d]);
        avs[d] ^= b;
        int c = ctz(b), r = s.rowsP[d];
        u32 nc = cols | b;
        u64 ne = e1 | 1ULL << (c - r + n - 1), nd = d2 | 1ULL << (r + c);
        if (d == 0) { center = cg.oddBM && c == cg.h; wcur = (cg.oddBM && c < cg.h) ? 2 : 1; }
        int d1 = d + 1;
        int dead = 0, leaf = (d1 == nr);
        if (d1 == s.split && d1 < nr) {
            u32 a = s.swap ? cO(nc) : cO(cg.oddCols & ~(cg.even ? rev(nc) : nc));
            if (!((s.stamp[a >> 5] >> (a & 31)) & 1)) dead = 1; }                    /* half-key pruning */
        if (d1 == P) {
            int mine = (idx == ticket);
            idx++;
            if (!mine) continue;
            if (dead || leaf) {
                if (leaf && !dead) local += wcur * getW(s, tab, rankTab, s.swap ? (cg.full & ~nc) : keyOf(nc));
                ticket = atomicAdd(&s.nextP, 1);
                continue; }
            inPrefix = 1; }
        else {
            if (dead) continue;
            if (leaf) { local += wcur * getW(s, tab, rankTab, s.swap ? (cg.full & ~nc) : keyOf(nc)); continue; }
        }
        cols = nc; e1 = ne; d2 = nd; ch[d] = (unsigned char)c;
        int rr = s.rowsP[d1];
        avs[d1] = cm[rr] & ~cols & ~(u32)((e1 >> (n - 1 - rr)) | (d2 >> rr));
        if (d1 == 1 && center) { avs[1] &= (1u << cg.h) - 1; wcur = 2; }           /* nested mirror */
        d = d1;
    }
    if (local) atomicAdd(&s.acc, local);
}
/* reset the touched table cells and the bitmap for the next row set */
__device__ void clearR(Ctx &s, u32 *tab, u32 *keys, const u32 *rankTab, int tid, int nth) {
    for (int t = tid; t < s.keyCount; t += nth) { u32 k = keys[t]; if (k != SENT) tab[idxOf(k, s.mulO, rankTab)] = 0; }
    int words = ((1 << cg.nO) >> 5) + 1;
    for (int t = tid; t < words; t += nth) s.stamp[t] = 0;
}

#ifndef EMULATE
/* One block per row set at a time, taken from the global counter dNextR. Rs / Ws / Cs: row sets of the batch,
   their symmetry weights and chunk indices within the batch; tabAll / keysAll: S cells per block; total: one sum per chunk of the batch. */
__global__ void kern(const u32 *Rs, const unsigned char *Ws, const int *Cs, int nR,
                     u32 *tabAll, u32 *keysAll, size_t S, const u32 *rankTab, ull *total) {
    __shared__ Ctx s;
    int tid = threadIdx.x, nth = blockDim.x;
    u32 *tab = tabAll + (size_t)blockIdx.x * S, *keys = keysAll + (size_t)blockIdx.x * S;
    for (int t = tid; t < 1025; t += nth) s.stamp[t] = 0;
    __syncthreads();
    __shared__ int jNext;
    for (;;) {
        /* the next row set of the batch, handed out dynamically: a block that finishes early takes the
           next one, so all blocks finish at about the same time even when row sets differ in cost */
        if (tid == 0) { int j = atomicAdd(&dNextR, 1); jNext = j; if (j < nR) setupR(s, Rs[j], Ws[j], Cs[j]); }
        __syncthreads();
        if (jNext >= nR) break;
        buildR(s, tab, keys, rankTab, tid, nth);
        __syncthreads(); filterR(s, tab, keys, rankTab, tid, nth);
        __syncthreads(); stampR(s, keys, tid, nth);
        __syncthreads(); probeR(s, tab, rankTab, tid, nth);
        __syncthreads(); if (tid == 0 && s.acc) atomicAdd(&total[s.cidx], (ull)s.w * s.acc);
        clearR(s, tab, keys, rankTab, tid, nth);
        __syncthreads();
    }
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#endif

/* ---------- host-side preparation ---------- */
static G hg;
/* mirror a row set: r -> n-1-r */
static u32 hrev(u32 x) { u32 r = 0; for (int i = 0; i < hg.n; i++) if ((x >> i) & 1) r |= 1u << (hg.n - 1 - i); return r; }
static void initBoard(int n) {
    memset(&hg, 0, sizeof hg);
    hg.n = n; hg.h = n / 2; hg.even = (n & 1) == 0; hg.oddBM = !hg.even; hg.nE = (n + 1) / 2; hg.nO = n / 2;
    hg.full = n == 32 ? 0xFFFFFFFFu : ((1u << n) - 1);
    for (int c = 1; c < n; c += 2) hg.oddCols |= 1u << c;
    hg.evenCols = hg.full ^ hg.oddCols;
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) { if (((r + c) & 1) == 0) hg.white[r] |= 1u << c; else hg.black[r] |= 1u << c; }
    /* search order: by distance from the center row, ties broken by the smaller row number */
    for (int r = 0; r < n; r++) hg.ORD[r] = r;
    double mid = (n - 1) / 2.0;
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) {
        double da = hg.ORD[a] > mid ? hg.ORD[a] - mid : mid - hg.ORD[a],
               db = hg.ORD[b] > mid ? hg.ORD[b] - mid : mid - hg.ORD[b];
        if (db < da || (db == da && hg.ORD[b] < hg.ORD[a])) { int t = hg.ORD[a]; hg.ORD[a] = hg.ORD[b]; hg.ORD[b] = t; } }
    for (int x = 0; x < 256; x++) {
        u32 e = 0, o = 0;
        for (int j = 0; j < 4; j++) { e |= ((x >> (2 * j)) & 1u) << j; o |= ((x >> (2 * j + 1)) & 1u) << j; }
        hg.EVB[x] = e; hg.ODB[x] = o; }
    for (int k = 0; k <= 16; k++) {
        u64 b = 1;
        if (k > hg.nO) b = 0; else for (int i = 0; i < k; i++) b = b * (hg.nO - i) / (i + 1);
        hg.binomNO[k] = (u32)b; }
}
static u64 binom(int a, int b) {
    if (b < 0 || b > a) return 0;
    u64 r = 1;
    for (int i = 0; i < b; i++) r = r * (a - i) / (i + 1);
    return r; }

static double nowSec() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void stampNow(char *out, size_t sz) {   /* current local time as HH:MM:SS */
    time_t t = time(NULL); struct tm tmv; localtime_r(&t, &tmv); strftime(out, sz, "%H:%M:%S", &tmv); }
static void fmtDur(double s, char *out, size_t sz) {
    if (s < 3600) snprintf(out, sz, "%.1f min", s / 60);
    else if (s < 172800) snprintf(out, sz, "%.2f h", s / 3600);
    else snprintf(out, sz, "%.2f days", s / 86400); }

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s n_from n_to [-t threads] [-b blocks] [-ckpt 0|1] [-dir DIR] [-from a] [-to b] [-batch N]"
                        " [-chunks FILE] [-side white|black|auto]\n", argv[0]);
        return 1; }
    int n1 = atoi(argv[1]), n2 = atoi(argv[2]),   /* range of n */
        nth = 512,              /* -t: threads per block */
        blocksArg = 0,          /* -b: thread blocks, 0 = automatic */
        ckOpt = -1,             /* -ckpt: -1 = automatic (on for n >= 20) */
        from = 0, to = -1,      /* -from / -to: chunk range, -1 = up to the last chunk */
        batchArg = 32,          /* -batch: chunks per kernel launch */
        sideMode = 2;           /* -side: 0 white, 1 black, 2 auto */
    const char *dir = ".";              /* -dir */
    const char *chunkFile = NULL;       /* -chunks */
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) nth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) blocksArg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ckpt") && i + 1 < argc) ckOpt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-dir") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "-from") && i + 1 < argc) from = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-to") && i + 1 < argc) to = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-batch") && i + 1 < argc) batchArg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-chunks") && i + 1 < argc) chunkFile = argv[++i];
        else if (!strcmp(argv[i], "-side") && i + 1 < argc) {
            i++; sideMode = !strcmp(argv[i], "white") ? 0 : !strcmp(argv[i], "black") ? 1 : 2; }
    }
    /* rankTab[m]: rank of m among the 16-bit values with the same number of set bits (combinatorial rank) */
    static u32 rankTab[1 << 16];
    { u32 cnt[17] = {0}; for (u32 m = 0; m < (1u << 16); m++) rankTab[m] = cnt[__builtin_popcount(m)]++; }
#ifndef EMULATE
    int dev = 0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s, %d SMs, %.1f GB\n", prop.name, prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);
    {   /* versions, for the run log */
        int drv = 0, rt = 0;
        cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
        printf("CUDA: driver supports %d.%d, runtime %d.%d, compiled with %d.%d\n", drv / 1000, (drv % 1000) / 10,
               rt / 1000, (rt % 1000) / 10, CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    }
    u32 *dRank;
    CK(cudaMalloc(&dRank, sizeof rankTab));
    CK(cudaMemcpy(dRank, rankTab, sizeof rankTab, cudaMemcpyHostToDevice));
#else
    printf("CPU emulation mode (EMULATE), threads per block: %d\n", nth);
#endif
    printf("table side: %s, threads per block %d, chunks per launch %d, prefix depth %d\n",
           sideMode == 0 ? "white" : sideMode == 1 ? "black" : "auto", nth, batchArg, PFX);
    for (int n = n1; n <= n2; n++) {
        if (n == 1) { printf("n= 1: 1 OK\n"); continue; }
        if (n > 31) { printf("n > 31 is not supported\n"); break; }
        initBoard(n); hg.sideMode = sideMode;
        /* canonical row sets of the white queens, grouped into chunks by their top 12 bits */
        int chunkBits = n > 12 ? n - 12 : 0;
        int nChunks = (int)((1ULL << n) >> chunkBits);
        std::vector<u32> Rs;               /* row sets */
        std::vector<unsigned char> Ws;     /* their symmetry weights */
        std::vector<int> off(nChunks + 1, 0);   /* chunk c holds Rs[off[c]] .. Rs[off[c+1]-1] */
        u32 evenRows = hg.evenCols;
        for (u64 R = 0; R <= hg.full; R++) {
            /* parity law: #even rows of R = #odd rows of R (+1 for odd n) */
            int re = __builtin_popcount((u32)R & evenRows), ro = __builtin_popcount((u32)R & ~evenRows & hg.full);
            if (re != ro + (hg.even ? 0 : 1)) continue;
            /* symmetries: keep the smallest set of its class; weight = number of distinct sets in the class.
               Even n: complement (color swap) and row mirror; odd n: row mirror only. */
            u32 r = (u32)R, comp = hg.full ^ r, fl = hrev(r), cf = hg.full ^ fl;
            int w;
            if (hg.even) {
                if (r > comp || r > fl || r > cf) continue;
                w = 1 + (comp != r) + (fl != r && fl != comp) + (cf != r && cf != comp && cf != fl); }
            else { if (r > fl) continue; w = fl == r ? 1 : 2; }
            Rs.push_back(r); Ws.push_back((unsigned char)w); off[(r >> chunkBits) + 1]++;
        }
        for (int c = 0; c < nChunks; c++) off[c + 1] += off[c];
        size_t S = (size_t)(binom(hg.nE, hg.nE / 2) * binom(hg.nO, hg.nO / 2)) + 1;   /* table cells per block */
        /* checkpoint file (same format as the CPU versions) and time file */
        int useCk = ckOpt < 0 ? (n >= 20) : ckOpt;
        const char *sfx = sideMode == 0 ? "_sideW" : sideMode == 1 ? "_sideB" : "";
        char fname[512], tname[512];
        snprintf(fname, sizeof fname, "%s/colorsplit_n%d%s.ckpt", dir, n, sfx);
        snprintf(tname, sizeof tname, "%s/colorsplit_n%d%s.time", dir, n, sfx);
        std::vector<char> doneF(nChunks, 0);       /* chunk finished */
        std::vector<ull> doneV(nChunks, 0);        /* its sum */
        std::vector<long> doneMsLoad(nChunks, 0);  /* its time in ms */
        int loaded = 0;
        if (useCk) {
            FILE *in = fopen(fname, "r");
            if (in) {
                char line[256];
                while (fgets(line, sizeof line, in)) {
                    int c; long long v; long ms; char okw[8] = "";
                    int m = sscanf(line, "%d %lld %ld %7s", &c, &v, &ms, okw);
                    if (m == 4 && !strcmp(okw, "ok") && c >= 0 && c < nChunks && !doneF[c]) {
                        doneF[c] = 1; doneV[c] = (ull)v; doneMsLoad[c] = ms; loaded++; } }
                fclose(in);
                if (loaded) printf("  resuming: %d chunks already done (%s)\n", loaded, fname); } }
        double prevWall = 0;
        int prevSess = 0;
        if (useCk) {
            FILE *tf = fopen(tname, "r");
            if (tf) { if (fscanf(tf, "%lf %d", &prevWall, &prevSess) != 2) { prevWall = 0; prevSess = 0; } fclose(tf); } }
        /* chunks to compute: -chunks list, or the range -from .. -to; chunks already in the checkpoint are skipped */
        int hiC = (to < 0 || to > nChunks) ? nChunks : to;
        std::vector<int> todo;
        int mine = 0;
        ull base = 0;
        std::vector<char> want(nChunks, 0);
        if (chunkFile) {
            FILE *lf = fopen(chunkFile, "r");
            if (!lf) { printf("cannot open the chunk list file: %s\n", chunkFile); return 1; }
            int c, cnt = 0;
            while (fscanf(lf, "%d", &c) == 1) { if (c >= 0 && c < nChunks && !want[c]) { want[c] = 1; cnt++; } }
            fclose(lf);
            printf("  chunk list: %d chunks from %s\n", cnt, chunkFile);
            from = 0; hiC = nChunks; }
        else for (int c = from; c < hiC; c++) want[c] = 1;
        for (int c = 0; c < nChunks; c++) if (want[c]) { mine++; if (doneF[c]) base += doneV[c]; else todo.push_back(c); }
        std::vector<long> doneMsV(doneMsLoad);
        double t0 = nowSec(), lastPrint = t0;
        ull sum = base;
        int finished = mine - (int)todo.size();
#ifndef EMULATE
        CK(cudaMemcpyToSymbol(cg, &hg, sizeof hg));
        size_t freeB, totB;
        CK(cudaMemGetInfo(&freeB, &totB));
        /* blocks: 8 per SM unless -b is given, at most what fits in 70% of the free memory (table + key list) */
        int blocks = blocksArg > 0 ? blocksArg : prop.multiProcessorCount * 8;
        size_t maxByMem = (size_t)((freeB * 0.7) / (2 * 4 * S));
        if ((size_t)blocks > maxByMem) blocks = (int)maxByMem;
        if (blocks < 1) blocks = 1;
        /* the largest number of row sets in one batch sizes the input buffers */
        size_t maxBatchR = 0;
        for (size_t i = 0; i < todo.size(); i += batchArg) {
            size_t cnt = 0;
            for (size_t j = i; j < i + batchArg && j < todo.size(); j++) cnt += off[todo[j] + 1] - off[todo[j]];
            if (cnt > maxBatchR) maxBatchR = cnt; }
        if (maxBatchR == 0) maxBatchR = 1;
        u32 *dR, *dTab, *dKeys; unsigned char *dW; int *dC; ull *dTot;
        CK(cudaMalloc(&dR, maxBatchR * 4)); CK(cudaMalloc(&dW, maxBatchR)); CK(cudaMalloc(&dC, maxBatchR * 4));
        CK(cudaMalloc(&dTot, (size_t)batchArg * 8));
        CK(cudaMalloc(&dTab, (size_t)blocks * S * 4)); CK(cudaMalloc(&dKeys, (size_t)blocks * S * 4));
        CK(cudaMemset(dTab, 0, (size_t)blocks * S * 4));
#else
        cg = hg;
        std::vector<u32> tab(S, 0), keys(S, 0);
        Ctx *s = new Ctx();
        memset(s->stamp, 0, sizeof s->stamp);
#endif
        for (size_t i = 0; i < todo.size(); i += batchArg) {
            /* one batch: the row sets of nb chunks in one kernel launch */
            int nb = (int)std::min((size_t)batchArg, todo.size() - i);
            std::vector<u32> bR; std::vector<unsigned char> bW; std::vector<int> bC; std::vector<ull> tot(nb, 0);
            for (int q = 0; q < nb; q++) {
                int c = todo[i + q];
                for (int k = off[c]; k < off[c + 1]; k++) { bR.push_back(Rs[k]); bW.push_back(Ws[k]); bC.push_back(q); } }
            {   /* heavy row sets first, so that no block is left with a big one at the end of the launch: the cost
                   is largest when R holds about n/2 rows, and a little larger below n/2 than above (stable sort,
                   so equal keys keep their order; the sums do not depend on the order) */
                std::vector<int> ord(bR.size());
                for (size_t k = 0; k < ord.size(); k++) ord[k] = (int)k;
                auto costKey = [&](int k) { int p = __builtin_popcount(bR[k]); return 2 * std::abs(2 * p - n) + (2 * p > n); };
                std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return costKey(a) < costKey(b); });
                std::vector<u32> sR(bR.size()); std::vector<unsigned char> sW(bR.size()); std::vector<int> sC(bR.size());
                for (size_t k = 0; k < ord.size(); k++) { sR[k] = bR[ord[k]]; sW[k] = bW[ord[k]]; sC[k] = bC[ord[k]]; }
                bR.swap(sR); bW.swap(sW); bC.swap(sC);
            }
            double tb = nowSec();
            if (!bR.empty()) {
#ifndef EMULATE
                CK(cudaMemcpy(dR, bR.data(), bR.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dW, bW.data(), bW.size(), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dC, bC.data(), bC.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemset(dTot, 0, (size_t)nb * 8));
                { int z = 0; CK(cudaMemcpyToSymbol(dNextR, &z, sizeof z)); }
                int g = blocks < (int)bR.size() ? blocks : (int)bR.size();
                kern<<<g, nth>>>(dR, dW, dC, (int)bR.size(), dTab, dKeys, S, dRank, dTot);
                CK(cudaGetLastError());
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(tot.data(), dTot, (size_t)nb * 8, cudaMemcpyDeviceToHost));
                {
                    int e = 0;
                    CK(cudaMemcpyFromSymbol(&e, dErr, sizeof e));
                    if (e) {
                        printf("ERROR: a table counter may overflow (n=%d, chunks %d..). This batch was NOT saved."
                               " A version with 64-bit counters is needed.\n", n, todo[i]);
                        fflush(stdout);
                        return 3; } }
#else
                for (size_t j = 0; j < bR.size(); j++) {
                    setupR(*s, bR[j], bW[j], bC[j]);
                    for (int t = 0; t < nth; t++) buildR(*s, tab.data(), keys.data(), rankTab, t, nth);
                    for (int t = 0; t < nth; t++) filterR(*s, tab.data(), keys.data(), rankTab, t, nth);
                    for (int t = 0; t < nth; t++) stampR(*s, keys.data(), t, nth);
                    for (int t = 0; t < nth; t++) probeR(*s, tab.data(), rankTab, t, nth);
                    if (s->acc) tot[s->cidx] += (ull)s->w * s->acc;
                    for (int t = 0; t < nth; t++) clearR(*s, tab.data(), keys.data(), rankTab, t, nth);
                }
                if (dErr) {
                    printf("ERROR: a table counter may overflow (n=%d, chunks %d..). This batch was NOT saved.\n", n, todo[i]);
                    return 3; }
#endif
            }
            long msEach = (long)((nowSec() - tb) * 1000 / nb);
            for (int q = 0; q < nb; q++) {
                int c = todo[i + q]; doneF[c] = 1; doneV[c] = tot[q]; doneMsV[c] = msEach; sum += tot[q]; }
            if (useCk) {   /* rewrite the whole checkpoint and close it after each batch (Google Drive saves a file on close) */
                char tmp[520];
                snprintf(tmp, sizeof tmp, "%s.tmp", fname);
                FILE *cf = fopen(tmp, "w");
                if (cf) {
                    for (int c = 0; c < nChunks; c++) if (doneF[c]) fprintf(cf, "%d %llu %ld ok\n", c, doneV[c], doneMsV[c]);
                    fflush(cf); fsync(fileno(cf)); fclose(cf); rename(tmp, fname); }
                else fprintf(stderr, "  WARNING: could not write %s\n", tmp);
            }
            finished += nb;
            double el = nowSec() - t0;
            if (useCk) {   /* time file: accumulated wall-clock time and number of runs */
                char tmp[520];
                snprintf(tmp, sizeof tmp, "%s.tmp", tname);
                FILE *tf = fopen(tmp, "w");
                if (tf) { fprintf(tf, "%.3f %d\n", prevWall + el, prevSess + 1); fclose(tf); rename(tmp, tname); } }
            if (nowSec() - lastPrint >= 30 || i + nb >= todo.size()) {   /* progress line every 30 s and at the end */
                lastPrint = nowSec();
                char ts[16], d1[32], d2[32];
                stampNow(ts, sizeof ts); fmtDur(el, d1, sizeof d1); fmtDur(prevWall + el, d2, sizeof d2);
                printf("  [%s] n=%d | chunks %d/%d | this run %s, total %s | sum %llu", ts, n, finished, mine, d1, d2, sum);
                if (n < NREF && REF[n]) printf(" (%.1f%% of the known value)", 100.0 * (double)sum / (double)REF[n]);
                printf("\n"); fflush(stdout); }
        }
#ifndef EMULATE
        CK(cudaFree(dR)); CK(cudaFree(dW)); CK(cudaFree(dC));
        CK(cudaFree(dTot)); CK(cudaFree(dTab)); CK(cudaFree(dKeys));
#else
        delete s;
#endif
        double sec = nowSec() - t0;
        int all = 1;
        ull total = 0;
        for (int c = 0; c < nChunks; c++) { if (!doneF[c]) all = 0; total += doneV[c]; }
        if (!all) {
            int nd = 0;
            for (int c = 0; c < nChunks; c++) nd += doneF[c];
            if (chunkFile) printf("  listed chunks: %d, their sum %llu | chunks done: %d of %d\n", mine, sum, nd, nChunks);
            else printf("  chunks %d..%d: sum %llu | chunks done: %d of %d; run again to finish\n",
                        from, hiC - 1, sum, nd, nChunks);
            continue; }
        const char *chk = n < NREF ? (total == REF[n] ? "OK" : "MISMATCH")
                                   : (total % 4 == 0 ? "(no reference value; divisible by 4, Nielsen check passed)"
                                                     : "(no reference value; NOT divisible by 4: ERROR!)");
        char d1[32];
        fmtDur(prevWall + (todo.empty() ? 0 : sec), d1, sizeof d1);
        printf("n=%2d: %llu %s | %.3f s this run | total %s | row sets %zu | table %zu cells per block\n",
               n, total, chk, sec, d1, Rs.size(), S);
        fflush(stdout);
    }
    return 0;
}
