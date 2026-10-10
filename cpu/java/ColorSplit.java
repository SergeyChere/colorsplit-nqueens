// SPDX-License-Identifier: MIT
/**
 * ColorSplit -- exact counting of n-queens solutions, CPU version (Java).
 *
 * WHAT IT COMPUTES
 *   Q(n), the number of ways to place n mutually non-attacking queens on an n x n board
 *   (OEIS A000170). The count is exact; for n <= 27 it is checked against the OEIS values
 *   stored in the program.
 *
 * IDEA
 *   Color the board like a chessboard: square (r, c) is white if r + c is even, black otherwise.
 *   Every diagonal has a single color, so a white queen and a black queen never share a
 *   diagonal; the two colors interact only through rows and columns. If the white queens occupy
 *   the row set R and the column set C, the black queens occupy exactly the other rows ~R and
 *   columns ~C, so
 *       Q(n) = sum over R and C of W_R(C) * B_~R(~C),
 *   where W_R(C) is the number of ways to put one white queen in every row of R, using exactly
 *   the columns C, with no two on a diagonal, and B is the same for black. For each R, one color
 *   is enumerated into a table indexed by column set; then the other color is enumerated, and
 *   each of its complete placements adds the table entry of the complementary column set (one
 *   array read). The cost is about sqrt(Q(n)) * 2^n, against about Q(n) for backtracking.
 *
 * TECHNIQUES (approximate speedup in parentheses)
 *   - Parity law: in every solution the number of white queens on even rows equals the number
 *     on odd rows (plus 1 for odd n); all other row sets R are skipped (x4).
 *   - Symmetries on row sets: the board's reflections and its 180-degree rotation map a solution
 *     with white rows R to one with white rows rev(R) (R upside down) or, for even n, where a
 *     reflection swaps the two colors, ~R or ~rev(R). Such row sets contribute equally, so one
 *     set per orbit is computed and weighted by the orbit size. For odd n the left-right mirror
 *     keeps colors and rows; it halves the enumeration inside each R instead (x4 together).
 *   - Weighted transposition: transposing a solution swaps R and C, so the terms (R, C) and
 *     (C, R) are equal. Each column set C gets a weight lambda in {0, 1, 2} from comparing the
 *     symmetry-invariant keys inv(C) and inv(R), and branches with lambda = 0 are cut as soon
 *     as half of inv(C) is known (x2 for odd n, x1.4 for even n).
 *   - Half-key pruning: the second color is placed first on the rows of one parity, which fixes
 *     the odd-column half of the lookup key; the branch stops if no table key has that half.
 *   - Packed table cells: one long per cell, (generation stamp << 32) | count, so the
 *     2^n-entry table is never cleared between row sets; a new stamp empties it.
 *   - Center-out row order: rows are enumerated from the middle of the board outwards.
 *   - Choice of tabulated color: by default the color with more rows in R goes into the table.
 *   - Threads with private tables. The 2^n row sets are split into 4096 chunks (2^n chunks of
 *     one row set each when n < 12); each finished chunk is appended to a checkpoint file, so a
 *     run can be interrupted and resumed, computed as chunk ranges at different times, or split
 *     into parts for several machines.
 *
 * HOW TO RUN (Java 11+; the source file is run directly, without a compilation step)
 *   java [-Dname=value ...] ColorSplit.java <n_from> <n_to>
 *   java -Xmx2g ColorSplit.java 1 18                  check n = 1..18 against OEIS
 *   java -Xmx8g -Dthreads=8 ColorSplit.java 23 23     one large board
 *   The -D options must come before the file name. For each n, stdout gets one line
 *   "n=<n>: <Q(n)> <check> | <seconds> s (<minutes> min)", where <check> is OK,
 *   "MISMATCH, expected <value>", "(no reference)" or "(part i/k)". Progress goes to stderr.
 *
 * OPTIONS (Java system properties)
 *   -Dthreads=N         worker threads (default: number of logical CPUs); lowered if -Xmx is too small
 *   -Dckpt=true|false   checkpoint file (default: true for n >= 20); a rerun skips chunks already in it
 *   -Dfrom=a            first chunk to compute (default 0)
 *   -Dto=b              compute only chunks a..b-1 (default: all chunks); the full total is printed
 *                       once the checkpoint file holds every chunk
 *   -Dpart=i/k          compute only chunks c with c mod k = i (default 0/1); Q(n) = sum of the k parts
 *   -Dstatus=true       compute nothing; report how many chunks the checkpoint file holds
 *                       (needs the checkpoint, i.e. n >= 20 or -Dckpt=true)
 *   -Dprogress=S        progress line on stderr every S seconds (default 60; 0 turns it off)
 *   -Dside=white|black  color that goes into the table (default: the color with more rows, white on ties)
 *   -Dnotr=true         turn off the weighted transposition (default false)
 *   -Dcenter=false      enumerate rows in increasing order instead of center-out (default true)
 *
 * MEMORY
 *   16 * 2^n bytes per thread (one long[2^n] and two int[2^n] arrays): n = 20: 16 MB,
 *   n = 23: 128 MB, n = 25: 512 MB, n = 27: 2 GB. The number of threads is reduced so that
 *   their tables fit in 85% of the Java heap (-Xmx).
 *
 * CHECKPOINT FILE
 *   colorsplit_n<n>.ckpt in the working directory; with -Dpart=i/k (k > 1) the name is
 *   colorsplit_n<n>_part<i>of<k>.ckpt, and with -Dnotr=true "_notr" is added before ".ckpt"
 *   (per-chunk sums differ without transposition). Each finished chunk appends one line and
 *   flushes it:
 *       <chunk> <sum> <milliseconds> ok
 *   When the file is read, only lines of 3 or 4 fields ending in "ok" are used (chunk and sum
 *   are the first two fields); other lines are ignored, and their chunks are computed again.
 *
 * LIMIT ON n
 *   The program rejects n > 31. In practice n <= 30, because every table has 2^n entries and a
 *   Java array holds fewer than 2^31 elements; also, the 64-bit signed total can hold Q(n) only
 *   up to n = 28.
 */
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class ColorSplit {
    // ---------- board-size parameters (set by count() for each n, read-only while threads run) ----------
    //   n          board size
    //   h          n / 2; for odd n, the center column
    //   full       mask of all n columns (or rows)
    //   oddCols    mask of the odd columns; evenCols: the even ones (the same masks serve for rows)
    //   even       n is even
    //   oddBM      n is odd: the left-right mirror keeps square colors, so the table stores one key
    //              per mirror pair and the first row is enumerated only up to the center
    static int n, h;
    static long full, oddCols, evenCols;
    static boolean even, oddBM;
    static final boolean TR = !Boolean.getBoolean("notr"); // weighted transposition (-Dnotr=true turns it off)
    // color that goes into the table: 0 = white, 1 = black, 2 = automatic (the color with more rows)
    static final int SIDE = "white".equals(System.getProperty("side")) ? 0
            : "black".equals(System.getProperty("side")) ? 1 : 2;
    static final boolean CENTER = !"false".equals(System.getProperty("center"));
    static int[] ORD; // row order: from the center outwards (or increasing with -Dcenter=false)

    /** Mirrors the low n bits: column c <-> n-1-c (for a row set: row r <-> n-1-r). */
    static long rev(long x) { return Long.reverse(x) >>> (64 - n); }

    /** Canonical form of a set of odd columns under complementing it within the odd columns. */
    static long hOdd(long s) { return Math.min(s, oddCols ^ s); }

    /**
     * Symmetry-invariant key of a row or column set S, used for the transposition weights.
     * Even n: u = odd columns of S, v = even columns of S mirrored onto the odd ones, both in
     * canonical form; the key is the unordered pair {u, v} packed as (min << n) | max, which
     * does not change under S -> ~S or S -> rev(S). Odd n: S is repacked with its odd columns
     * in the high half, and the key is the smaller packing of S and rev(S).
     */
    static long inv(long S) {
        if (even) { long u = hOdd(S & oddCols), v = hOdd(rev(S & evenCols)); return Math.min(u, v) << n | Math.max(u, v); }
        long a = (S & oddCols) << n | (S & evenCols), rs = rev(S), b = (rs & oddCols) << n | (rs & evenCols);
        return Math.min(a, b);
    }

    /**
     * Early transposition bound for the white table. Once the white queens of one row parity
     * are placed (even rows, which use the even columns, for even n; odd rows, which use the odd
     * columns, for odd n), this value is >= the high half of inv(C) for every completion C
     * (equal for odd n). If it is already below the high half of inv(R), lambda = 0 for the
     * whole branch.
     */
    static long partBuild(long cols) { return even ? hOdd(rev(cols & evenCols)) : Math.min(cols & oddCols, rev(cols & oddCols)); }
    static long[] white = new long[64], black = new long[64]; // white[r] / black[r]: columns of that color in row r

    /** State of one worker thread: its own tables and enumeration stack. */
    static final class Worker {
        //   nr           number of rows in the current enumeration; rows[0..nr) lists them in visiting order
        //   split        index in rows[] at which the probe checks the half key
        //   bsplit       index in rows[] at which the table build checks the transposition bound
        //   gen          current generation stamp of tab[] and halfStamp[]
        //   buildTarget  1 while the first queen is in the left half (odd n: each placement also
        //                stands for its mirror image and adds 2), otherwise 2 (adds 1)
        //   ksz          number of keys in keyList
        //   acc          running sum of the probe
        //   IR, IRfirst  inv(R) and its high half
        int nr, split, bsplit, gen = 1, buildTarget, ksz; long acc, IR, IRfirst;
        long[] colMask; int[] rows = new int[64]; // colMask: white[] or black[], the color being enumerated
        // tab: table cells (format below); halfStamp[x] == gen: x is an odd-column half that the probe
        // can still match; keyList: table keys written in the current generation
        long[] tab; int[] halfStamp, keyList;
        Worker() {
            int sz = 1 << n; halfStamp = new int[sz]; keyList = new int[sz]; tab = new long[sz];
        }
        // ---------- table W[C] with direct addressing and generation stamps ----------
        // tab[k] = (generation << 32) | value; a cell with an older stamp counts as empty.
        // Even n: value = W(k). Odd n: k = min(C, mirror of C), and the value is
        //   V = 2 * (placements with the first queen in the left half) + (placements with it in the center);
        //   W(C) = V if C is a palindrome, V / 2 otherwise.
        void addW(long key) {
            int k = oddBM ? (int) Math.min(key, rev(key)) : (int) key; long v = tab[k];
            if ((int) (v >>> 32) != gen) { tab[k] = (long) gen << 32 | (buildTarget == 1 ? 2 : 1); keyList[ksz++] = k; }
            else tab[k] = v + (buildTarget == 1 ? 2 : 1);
        }
        long getW(long key) {
            if (!oddBM) { long v = tab[(int) key]; return (int) (v >>> 32) == gen ? (int) v : 0; }
            long rk = rev(key); long v = tab[(int) Math.min(key, rk)];
            if ((int) (v >>> 32) != gen) return 0;
            return key == rk ? (int) v : (int) v >> 1;
        }

        // ---------- build: white placements on rows rows[0..nr) go into the table ----------
        // cols: used columns; e1: used diagonals (bit c - r + n - 1); d2: used anti-diagonals (bit r + c).
        // Diagonals are stored in board coordinates because the rows are not visited top to bottom.
        void build(int i, long cols, long e1, long d2) {
            if (i == nr) { addW(cols); return; }
            if (TR && i == bsplit && i > 0 && partBuild(cols) < IRfirst) return; // lambda = 0 for the whole branch
            int r = rows[i];
            long av = colMask[r] & ~cols & ~((e1 >>> (n - 1 - r)) | (d2 >>> r)); // free squares of row r
            // last row: add the completed placements directly, without another call
            if (i == nr - 1 && i != bsplit - 1) { for (; av != 0; av &= av - 1) addW(cols | (av & -av)); return; }
            while (av != 0) {
                long b = av & -av; av ^= b; int c = Long.numberOfTrailingZeros(b);
                build(i + 1, cols | b, e1 | 1L << (c - r + n - 1), d2 | 1L << (r + c));
            }
        }

        // ---------- probe: the second part on rows ~R looks up its partner in the table ----------
        // Lookup key = the complementary column set. For even n, pairSum() enumerates the black part
        // as its left-right mirror image, which lies on white squares, so the columns are mirrored back.
        long key(long cols) { return even ? full & ~rev(cols) : full & ~cols; }
        void probe(int i, long cols, long e1, long d2) {
            if (i == split && i > 0 && i < nr) { // half-key pruning: the even rows are placed,
                long kOdd = oddCols & ~(even ? rev(cols) : cols); // so the odd half of the key is fixed
                if (halfStamp[(int) kOdd] != gen) return;
            }
            if (i == nr) { acc += getW(key(cols)); return; }
            int r = rows[i];
            long av = colMask[r] & ~cols & ~((e1 >>> (n - 1 - r)) | (d2 >>> r));
            if (i == 0 && !even) { // odd n, left-right mirror: left-half first queens count twice, the center once
                long before = acc; acc = 0;
                for (long z = av & ((1L << h) - 1); z != 0; z &= z - 1) {
                    long b = z & -z; int c = Long.numberOfTrailingZeros(b);
                    probe(1, b, 1L << (c - r + n - 1), 1L << (r + c));
                }
                long left = acc; acc = 0;
                if ((av >> h & 1) != 0) probe(1, 1L << h, 1L << (h - r + n - 1), 1L << (r + h));
                acc = before + 2 * left + acc;
                return;
            }
            // last row: look up the completed placements directly
            if (i == nr - 1) {
                long s = 0;
                for (; av != 0; av &= av - 1) s += getW(key(cols | (av & -av)));
                acc += s;
                return;
            }
            while (av != 0) {
                long b = av & -av; av ^= b; int c = Long.numberOfTrailingZeros(b);
                probe(i + 1, cols | b, e1 | 1L << (c - r + n - 1), d2 | 1L << (r + c));
            }
        }

        void setColor(int color) { colMask = color == 0 ? white : black; }
        /** Starts a new generation (empties the table); clears the arrays only when the stamp wraps. */
        void newGen() {
            if (++gen == Integer.MAX_VALUE) {
                java.util.Arrays.fill(tab, 0); java.util.Arrays.fill(halfStamp, 0);
                gen = 1;
            }
        }

        /** Sum over C of lambda * W(R, C) * B(~R, ~C) for one row set R (white placements in the table). */
        long pairSum(long R) {
            // 1) white queens on rows R -> table
            setColor(0); nr = 0;
            if (TR) {
                // rows of parity f first: after them, partBuild() can bound inv(C)
                IR = inv(R); IRfirst = IR >>> n; int f = even ? 0 : 1;
                for (int r : ORD) if ((r & 1) == f && (R >> r & 1) != 0) rows[nr++] = r;
                bsplit = nr;
                for (int r : ORD) if ((r & 1) != f && (R >> r & 1) != 0) rows[nr++] = r;
            } else { // without transposition: rows in increasing order
                bsplit = -1;
                for (long z = R; z != 0; z &= z - 1) rows[nr++] = Long.numberOfTrailingZeros(z);
            }
            newGen(); ksz = 0;
            if (nr == 0) { buildTarget = 2; addW(0); } // empty placement
            else if (oddBM) { // odd n: first queen in the left half (counted twice) or in the center column (once)
                int r0 = rows[0]; long av = colMask[r0];
                for (long z = av & ((1L << h) - 1); z != 0; z &= z - 1) {
                    long b = z & -z; int c = Long.numberOfTrailingZeros(b);
                    buildTarget = 1; build(1, b, 1L << (c - r0 + n - 1), 1L << (r0 + c));
                }
                if ((av >> h & 1) != 0) { buildTarget = 2; build(1, 1L << h, 1L << (h - r0 + n - 1), 1L << (r0 + h)); }
            } else { buildTarget = 2; build(0, 0, 0, 0); }
            // 2) transposition: multiply every entry by lambda, drop the keys with lambda = 0
            if (TR) {
                int m = 0;
                for (int t = 0; t < ksz; t++) {
                    int k = keyList[t]; long ic = inv(k); int lam = ic > IR ? 2 : ic == IR ? 1 : 0;
                    if (lam == 0) tab[k] = 0;
                    else { long v = tab[k]; tab[k] = (v & 0xFFFFFFFF00000000L) | mul((int) v, lam); keyList[m++] = k; }
                }
                ksz = m;
            }
            if (ksz == 0) return 0;
            // mark the odd halves of the keys left in the table (both mirror orientations for odd n)
            for (int t = 0; t < ksz; t++) {
                long k = keyList[t];
                halfStamp[(int) (k & oddCols)] = gen;
                if (oddBM) halfStamp[(int) (rev(k) & oddCols)] = gen;
            }
            // 3) second part on rows ~R: even rows first (after them the odd half of the key is known).
            //    Odd n enumerates black squares; even n enumerates the mirror image, on white squares.
            long comp = full ^ R;
            setColor(even ? 0 : 1); nr = 0;
            for (int r : ORD) if ((r & 1) == 0 && (comp >> r & 1) != 0) rows[nr++] = r;
            split = nr;
            for (int r : ORD) if ((r & 1) == 1 && (comp >> r & 1) != 0) rows[nr++] = r;
            acc = 0;
            if (nr == 0) acc = getW(key(0)); else probe(0, 0, 0, 0);
            return acc;
        }

        // ============ swapped sides: black placements go into the table, white ones probe ============
        boolean tBlack; // unused
        /** Like partBuild() for the black table: after the even rows, the black queens fix the odd columns D_odd. */
        long pbBlack(long cols) {
            long cOdd = oddCols & ~(cols & oddCols); // odd white columns C_odd = complement of D_odd
            return even ? hOdd(cOdd) : Math.min(cOdd, rev(cOdd) & oddCols);
        }
        /** Same as build(), with the transposition bound of the black table. */
        void buildB(int i, long cols, long e1, long d2) {
            if (i == nr) { addW(cols); return; }
            if (TR && i == bsplit && i > 0 && pbBlack(cols) < IRfirst) return;
            int r = rows[i]; long av = colMask[r] & ~cols & ~((e1 >>> (n - 1 - r)) | (d2 >>> r));
            if (i == nr - 1 && i != bsplit - 1) { for (; av != 0; av &= av - 1) addW(cols | (av & -av)); return; }
            while (av != 0) {
                long b = av & -av; av ^= b; int c = Long.numberOfTrailingZeros(b);
                buildB(i + 1, cols | b, e1 | 1L << (c - r + n - 1), d2 | 1L << (r + c));
            }
        }
        /** Same as probe() for white queens on rows R; the lookup key is the black column set ~C. */
        void probeW(int i, long cols, long e1, long d2) {
            // half-key pruning: after the odd rows, the white queens have fixed the odd columns C_odd
            if (i == split && i > 0 && i < nr && halfStamp[(int) (cols & oddCols)] != gen) return;
            if (i == nr) { acc += getW(full & ~cols); return; }
            int r = rows[i]; long av = colMask[r] & ~cols & ~((e1 >>> (n - 1 - r)) | (d2 >>> r));
            if (i == 0 && !even) { // odd n, left-right mirror
                long before = acc; acc = 0;
                for (long z = av & ((1L << h) - 1); z != 0; z &= z - 1) {
                    long b = z & -z; int c = Long.numberOfTrailingZeros(b);
                    probeW(1, b, 1L << (c - r + n - 1), 1L << (r + c));
                }
                long left = acc; acc = 0;
                if ((av >> h & 1) != 0) probeW(1, 1L << h, 1L << (h - r + n - 1), 1L << (r + h));
                acc = before + 2 * left + acc;
                return;
            }
            if (i == nr - 1) {
                long s = 0;
                for (; av != 0; av &= av - 1) s += getW(full & ~(cols | (av & -av)));
                acc += s;
                return;
            }
            while (av != 0) {
                long b = av & -av; av ^= b; int c = Long.numberOfTrailingZeros(b);
                probeW(i + 1, cols | b, e1 | 1L << (c - r + n - 1), d2 | 1L << (r + c));
            }
        }
        /** Same sum as pairSum(R), with black placements on rows ~R in the table. */
        long pairSumSwap(long R) {
            long comp = full ^ R; IR = inv(R); IRfirst = IR >>> n;
            // 1) table: black queens on rows ~R, even rows first (black queens there use the odd columns)
            setColor(1); nr = 0;
            for (int r : ORD) if ((r & 1) == 0 && (comp >> r & 1) != 0) rows[nr++] = r;
            bsplit = nr;
            for (int r : ORD) if ((r & 1) == 1 && (comp >> r & 1) != 0) rows[nr++] = r;
            newGen(); ksz = 0;
            if (nr == 0) { buildTarget = 2; addW(0); }
            else if (oddBM) {
                int r0 = rows[0]; long av = colMask[r0];
                for (long z = av & ((1L << h) - 1); z != 0; z &= z - 1) {
                    long b = z & -z; int c = Long.numberOfTrailingZeros(b);
                    buildTarget = 1; buildB(1, b, 1L << (c - r0 + n - 1), 1L << (r0 + c));
                }
                if ((av >> h & 1) != 0) { buildTarget = 2; buildB(1, 1L << h, 1L << (h - r0 + n - 1), 1L << (r0 + h)); }
            } else { buildTarget = 2; buildB(0, 0, 0, 0); }
            // 2) transposition: lambda is computed from the white column set C = ~D
            if (TR) {
                int m = 0;
                for (int t = 0; t < ksz; t++) {
                    int k = keyList[t]; long ic = inv(full & ~(long) k); int lam = ic > IR ? 2 : ic == IR ? 1 : 0;
                    if (lam == 0) tab[k] = 0;
                    else { long v = tab[k]; tab[k] = (v & 0xFFFFFFFF00000000L) | mul((int) v, lam); keyList[m++] = k; }
                }
                ksz = m;
            }
            if (ksz == 0) return 0;
            // mark the odd half of the matching white column set C = ~D (both mirror orientations for odd n)
            for (int t = 0; t < ksz; t++) {
                long k = keyList[t];
                halfStamp[(int) (oddCols & ~k)] = gen;
                if (oddBM) halfStamp[(int) (oddCols & ~rev(k))] = gen;
            }
            // 3) probe: white queens on rows R, odd rows first (white queens there use the odd columns)
            setColor(0); nr = 0;
            for (int r : ORD) if ((r & 1) == 1 && (R >> r & 1) != 0) rows[nr++] = r;
            split = nr;
            for (int r : ORD) if ((r & 1) == 0 && (R >> r & 1) != 0) rows[nr++] = r;
            acc = 0;
            if (nr == 0) acc = getW(full); else probeW(0, 0, 0, 0);
            return acc;
        }
        /** count * lambda, guarding the 32-bit count field of a table cell against overflow. */
        int mul(int c, int lam) {
            if (c > (1 << 30)) throw new ArithmeticException("table counter overflow");
            return c * lam;
        }

    }

    static final long[] REF = {1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512,
        95815104, 666090624, 4968057848L, 39029188884L, 314666222712L, 2691008701644L, 24233937684440L,
        227514171973736L, 2207893435808352L, 22317699616364044L, 234907967154122528L}; // OEIS A000170, Q(0..27)

    /** Sum over the row sets R in [lo, hi), using the symmetries of row sets. */
    static long chunkSum(Worker w, long lo, long hi) {
        long evenRows = evenCols, total = 0;
        for (long R = lo; R < hi; R++) {
            // parity law: the even columns hold the white queens of the even rows and the black queens
            // of the odd rows; the even rows hold the white and the black queens of the even rows. Both
            // counts are ceil(n/2), so black queens sit on as many odd as even rows: re - ro = n mod 2
            // (re, ro = white queens on even / odd rows)
            int re = Long.bitCount(R & evenRows), ro = Long.bitCount(R & ~evenRows);
            if (re != ro + (even ? 0 : 1)) continue;
            // keep the smallest set of each orbit, weighted by the orbit size
            long comp = full ^ R, fl = rev(R), cf = full ^ fl; int weight;
            if (even) { // orbit {R, ~R, flip R, ~flip R}
                if (R > comp || R > fl || R > cf) continue;
                weight = 1 + (comp != R ? 1 : 0) + (fl != R && fl != comp ? 1 : 0) + (cf != R && cf != comp && cf != fl ? 1 : 0);
            } else { if (R > fl) continue; weight = fl == R ? 1 : 2; } // orbit {R, flip R}
            int k = Long.bitCount(R);
            boolean swap = SIDE == 2 ? (n - k > k) : SIDE == 1; // black into the table if it has more rows
            total += weight * (swap ? w.pairSumSwap(R) : w.pairSum(R));
        }
        return total;
    }

    // state of the last count() call, for the range / status report in main()
    static Map<Integer, Long> rangeDone; static int rangeN, rangeFrom, rangeTo;

    static long count(int nn, int threads, boolean ckpt, int partI, int partK, double prog) throws Exception {
        n = nn; h = n / 2; full = (1L << n) - 1; even = (n & 1) == 0; oddBM = !even;
        if (n == 1) return 1;
        if (n > 31) throw new IllegalArgumentException("n <= 31");
        { // row order: by distance from the middle of the board (ties: lower row first), or increasing
            Integer[] o = new Integer[n]; for (int r = 0; r < n; r++) o[r] = r; final double mid = (n - 1) / 2.0;
            if (CENTER) Arrays.sort(o, (x, y) -> {
                int c = Double.compare(Math.abs(x - mid), Math.abs(y - mid));
                return c != 0 ? c : Integer.compare(x, y);
            });
            ORD = new int[n]; for (int r = 0; r < n; r++) ORD[r] = o[r];
        }
        oddCols = 0; for (int c = 1; c < n; c += 2) oddCols |= 1L << c;
        evenCols = full ^ oddCols;
        for (int r = 0; r < n; r++) {
            long m0 = 0, m1 = 0;
            for (int c = 0; c < n; c++) if (((r + c) & 1) == 0) m0 |= 1L << c; else m1 |= 1L << c;
            white[r] = m0; black[r] = m1;
        }
        // memory: 16 bytes per table entry per thread; use at most 85% of the heap
        long per = (1L << n) * 4 * 4, max = Runtime.getRuntime().maxMemory();
        int fit = (int) Math.max(1, (long) (max * 0.85) / per);
        if (threads > fit) {
            System.err.printf("  memory is enough for only %d threads (%d MB per thread, -Xmx %d MB), reducing%n",
                fit, per >> 20, max >> 20);
            threads = fit;
        }
        int chunkBits = Math.max(0, n - 12);           // 4096 chunks (for n < 12: 2^n chunks of one row set)
        int nChunks = (int) ((1L << n) >> chunkBits);
        long chunk = 1L << chunkBits;
        // checkpoint file
        Path file = Paths.get(String.format("colorsplit_n%d%s%s.ckpt", n,
            partK > 1 ? "_part" + partI + "of" + partK : "", TR ? "" : "_notr"));
        Map<Integer, Long> done = new ConcurrentHashMap<>();
        if (ckpt && Files.exists(file)) {
            for (String line : Files.readAllLines(file)) {
                // "<chunk> <sum> [<ms>] ok"; anything else, e.g. a line cut off by a crash, is skipped
                String[] p = line.trim().split("\\s+");
                if (p.length >= 3 && p.length <= 4 && p[p.length - 1].equals("ok"))
                    done.put(Integer.parseInt(p[0]), Long.parseLong(p[1]));
            }
            System.err.printf("  resuming: %d chunks already done (%s)%n", done.size(), file);
        }
        if (ckpt && Files.exists(file) && Files.size(file) > 0) {
            // a run killed in the middle of a line leaves no newline at the end of the file: add one,
            // so that the torn line stays a separate (ignored) line instead of merging with the next one
            try (RandomAccessFile raf = new RandomAccessFile(file.toFile(), "rw")) {
                raf.seek(raf.length() - 1);
                if (raf.read() != '\n') raf.write('\n');
            }
        }
        PrintWriter out = ckpt ? new PrintWriter(new FileWriter(file.toFile(), true)) : null; // append
        // work list: chunks of [from, to) that belong to this part and are not in the checkpoint yet
        List<Integer> todo = new ArrayList<>();
        int from = Integer.getInteger("from", 0), to = Math.min(Integer.getInteger("to", nChunks), nChunks);
        if (Boolean.getBoolean("status")) { from = to = 0; }
        for (int c = from; c < to; c++) if (c % partK == partI && !done.containsKey(c)) todo.add(c);
        int myTotal = 0; for (int c = from; c < to; c++) if (c % partK == partI) myTotal++;
        rangeDone = done; rangeN = nChunks; rangeFrom = from; rangeTo = to;
        AtomicInteger next = new AtomicInteger(0), finished = new AtomicInteger(myTotal - todo.size());
        AtomicLong sum = new AtomicLong(0);
        for (int c = from; c < to; c++) if (c % partK == partI && done.containsKey(c)) sum.addAndGet(done.get(c));
        long t0 = System.nanoTime();
        final int myTotalF = myTotal;
        ExecutorService ex = Executors.newFixedThreadPool(threads);
        List<Future<?>> fs = new ArrayList<>();
        for (int t = 0; t < threads; t++) fs.add(ex.submit(() -> {
            Worker w = new Worker(); // private tables
            for (int j; (j = next.getAndIncrement()) < todo.size(); ) {
                int c = todo.get(j);
                long tc = System.nanoTime();
                long v = chunkSum(w, c * chunk, (c + 1) * chunk);
                long ms = (System.nanoTime() - tc) / 1000000;
                sum.addAndGet(v); finished.incrementAndGet();
                done.put(c, v);
                if (out != null) synchronized (out) { out.println(c + " " + v + " " + ms + " ok"); out.flush(); }
            }
            return null;
        }));
        ex.shutdown();
        if (prog > 0 && n >= 19)
            System.err.printf("  [%s] start n=%d, threads %d, chunks %d%n",
                java.time.LocalTime.now().withNano(0), n, threads, todo.size());
        while (!ex.awaitTermination(prog > 0 ? (long) (prog * 1000) : 1000, TimeUnit.MILLISECONDS)) {
            if (prog > 0) {
                double min = (System.nanoTime() - t0) / 6e10; long s = sum.get();
                String ans = partK == 1 && n < REF.length ? String.format(" (%.1f%% of the answer)", 100.0 * s / REF[n]) : "";
                System.err.printf("  [%s] n=%d | chunks %d/%d | elapsed %.1f min | sum %,d%s%n",
                    java.time.LocalTime.now().withNano(0), n, finished.get(), myTotalF, min, s, ans);
            }
        }
        for (Future<?> f : fs) f.get(); // rethrows an exception from a worker
        if (out != null) out.close();
        return sum.get();
    }

    /** First chunk missing from the last count() call's results, or -1. */
    static int firstMissing() { for (int c = 0; c < rangeN; c++) if (!rangeDone.containsKey(c)) return c; return -1; }
    public static void main(String[] a) throws Exception {
        int threads = Integer.getInteger("threads", Runtime.getRuntime().availableProcessors());
        double prog = Double.parseDouble(System.getProperty("progress", "60"));
        String part = System.getProperty("part", "0/1"); // "i/k"
        int partI = Integer.parseInt(part.split("/")[0]), partK = Integer.parseInt(part.split("/")[1]);
        System.out.printf("ColorSplit (transposition %s, threads %d%s)%n",
            TR ? "on" : "off", threads, partK > 1 ? ", part " + part : "");
        for (int nn = Integer.parseInt(a[0]); nn <= Integer.parseInt(a[1]); nn++) {
            boolean ckpt = Boolean.parseBoolean(System.getProperty("ckpt", String.valueOf(nn >= 20)));
            long t0 = System.nanoTime(); long ans = count(nn, threads, ckpt, partI, partK, prog);
            double s = (System.nanoTime() - t0) / 1e9;
            boolean ranged = rangeFrom != 0 || rangeTo != rangeN; // -Dfrom / -Dto / -Dstatus
            if (ranged) {
                long all = 0; for (long v : rangeDone.values()) all += v;
                if (Boolean.getBoolean("status"))
                    System.out.printf("  n=%d: checkpoint file has %d of %d chunks%s%n", n, rangeDone.size(), rangeN,
                        rangeDone.size() == rangeN ? "" : String.format(", missing e.g. chunk %d", firstMissing()));
                else
                    System.out.printf("  chunks %d..%d: sum %,d | checkpoint file has %d of %d chunks%s%n",
                        rangeFrom, rangeTo - 1, ans, rangeDone.size(), rangeN,
                        rangeDone.size() == rangeN ? "" : String.format(", missing e.g. chunk %d", firstMissing()));
                // print the final result only once every chunk is done
                if (rangeDone.size() == rangeN) { ans = all; System.out.printf("  ALL CHUNKS DONE, total:%n"); }
                else continue;
            }
            String chk = partK > 1 ? "(part " + part + ")"
                : nn >= REF.length ? "(no reference)"
                : ans == REF[nn] ? "OK" : "MISMATCH, expected " + REF[nn];
            System.out.printf("n=%2d: %,d %s | %.3f s (%.1f min)%n", nn, ans, chk, s, s / 60);
            System.out.flush();
        }
    }
}
