// SPDX-License-Identifier: MIT
/**
 * BTSym -- symmetric bitmask backtracking for n-queens (CPU baseline, Java).
 *
 * WHAT IT COMPUTES
 *   Q(n), the number of ways to place n mutually non-attacking queens on an n x n board
 *   (OEIS A000170), using Takaken's bitmask backtracking with the 8 symmetries of the square
 *   (rotations and reflections). This is the CPU baseline against which ColorSplit is compared.
 *
 * METHOD
 *   Queens are placed row by row. The squares of the next row attacked by the queens placed so
 *   far are kept in three bitmasks: left diagonals (l), columns (d) and right diagonals (r).
 *   Each solution is generated in only one orientation, the canonical one, and is counted with
 *   the number of its distinct images under the 8 symmetries:
 *   - Corner case (bt1): the queen of row 0 is in the corner, column 0. Such a solution has no
 *     symmetry, so it stands for 8 solutions. Of it and its reflection in the main diagonal,
 *     only the one in which the queen of column 1 lies below row BOUND1 (the column of the row-1
 *     queen) is generated.
 *   - Non-corner case (bt2): the queen of row 0 is at column BOUND1 in the left half, and no
 *     edge queen may be closer than BOUND1 to a corner: the queens of the two edge columns lie
 *     in rows BOUND1..BOUND2 (BOUND2 = n-1-BOUND1), and the last-row queen avoids the BOUND1
 *     columns nearest to each corner. check() then compares the board with its rotations by
 *     90, 180 and 270 degrees, the only images whose first-row queen can also be at column
 *     BOUND1 (for a reflection this would need two queens on one diagonal). A board is dropped
 *     if a rotation is smaller, and otherwise counted as having 2 distinct images (symmetric
 *     under the 90-degree rotation), 4 (under the 180-degree rotation) or 8.
 *   Q(n) = 8 * C8 + 4 * C4 + 2 * C2; the values for n < 5 come from a table.
 *
 * HOW TO RUN (Java 11+; the source file is run directly)
 *   java [-Dreps=N] BTSym.java <n_from> <n_to>
 *   -Dreps=N   run each n N times and report the best time (default 1)
 *   Output per n: "n=<n>: <Q(n)> <check> | <seconds> s | <ns> ns/sol", where <check> is OK,
 *   "!!! MISMATCH (expected <value>)" or "(no reference)" (OEIS values are stored for n <= 27).
 *   The bitmasks are ints, so n <= 31; run time limits n much further in practice.
 */
public class BTSym {
    //   SIZE, SIZEE  n and n - 1 (the last row)
    //   MASK         all n columns; TOPBIT = column n-1
    //   SIDEMASK     the two edge columns
    //   LASTMASK     columns forbidden in the last row (closer than BOUND1 to a corner)
    //   ENDBIT       column BOUND2, where the last-row queen of a 180-degree-symmetric board stands
    //   BOUND1       column of the row-1 queen (corner case) or of the row-0 queen (non-corner case)
    //   BOUND2       n - 1 - BOUND1
    static int SIZE, SIZEE, MASK, TOPBIT, SIDEMASK, LASTMASK, ENDBIT, BOUND1, BOUND2;
    // B[y]: bit of the queen in row y; C8, C4, C2: canonical solutions with 8, 4 and 2 distinct images
    static int[] B = new int[64]; static long C8, C4, C2;

    /**
     * Classifies a complete board of the non-corner case. For each rotation that brings an edge
     * queen to column BOUND1 of row 0, the rotated board is compared with B row by row: if it is
     * smaller, B is not canonical and is dropped; if it is equal, B is symmetric under it.
     */
    static void check() {
        int own, you, bit, ptn;
        // 90 degrees: the queen at row BOUND2, column 0 goes to row 0, column BOUND1
        if (B[BOUND2] == 1) {
            for (ptn = 2, own = 1; own <= SIZEE; own++, ptn <<= 1) {
                // bit = queen of the rotated board in row own (scan stops once it exceeds B[own])
                bit = 1; for (you = SIZEE; B[you] != ptn && B[own] >= bit; you--) bit <<= 1;
                if (B[own] > bit) return; // the rotated board is smaller: B is not canonical
                if (B[own] < bit) break;  // B is smaller: not symmetric under this rotation
            }
            if (own > SIZEE) { C2++; return; } // identical: symmetric under the 90-degree rotation
        }
        // 180 degrees: the last-row queen at column BOUND2 goes to row 0, column BOUND1
        if (B[SIZEE] == ENDBIT) {
            for (you = SIZEE - 1, own = 1; own <= SIZEE; own++, you--) {
                bit = 1; for (ptn = TOPBIT; ptn != B[you] && B[own] >= bit; ptn >>= 1) bit <<= 1;
                if (B[own] > bit) return;
                if (B[own] < bit) break;
            }
            if (own > SIZEE) { C4++; return; } // symmetric under the 180-degree rotation
        }
        // 270 degrees: the queen at row BOUND1, column n-1 goes to row 0, column BOUND1
        if (B[BOUND1] == TOPBIT) {
            for (ptn = TOPBIT >> 1, own = 1; own <= SIZEE; own++, ptn >>= 1) {
                bit = 1; for (you = 0; B[you] != ptn && B[own] >= bit; you++) bit <<= 1;
                if (B[own] > bit) return;
                if (B[own] < bit) break;
            }
        }
        C8++; // no symmetry: 8 distinct images
    }
    /** Non-corner case, row y; l, d, r = squares of row y attacked along left diagonals, columns, right diagonals. */
    static void bt2(int y, int l, int d, int r) {
        int bm = MASK & ~(l | d | r), bit;
        if (y == SIZEE) { if (bm != 0 && (bm & LASTMASK) == 0) { B[y] = bm; check(); } return; } // last row
        if (y < BOUND1) bm &= ~SIDEMASK; // no edge-column queen above row BOUND1
        else if (y == BOUND2) {
            // no edge-column queen below row BOUND2: if both edge columns are still free, give up;
            // if one is free, this row must take it
            if ((d & SIDEMASK) == 0) return;
            if ((d & SIDEMASK) != SIDEMASK) bm &= SIDEMASK;
        }
        while (bm != 0) { bit = bm & -bm; bm ^= bit; B[y] = bit; bt2(y + 1, (l | bit) << 1, d | bit, (r | bit) >> 1); }
    }
    /** Corner case, row y: every solution found has 8 distinct images. */
    static void bt1(int y, int l, int d, int r) {
        int bm = MASK & ~(l | d | r), bit;
        if (y == SIZEE) { if (bm != 0) { B[y] = bm; C8++; } return; }
        if (y < BOUND1) bm &= ~2; // column 1 stays empty above row BOUND1 (excludes the diagonal reflection)
        while (bm != 0) { bit = bm & -bm; bm ^= bit; B[y] = bit; bt1(y + 1, (l | bit) << 1, d | bit, (r | bit) >> 1); }
    }
    static long count(int n) {
        long[] small = {1, 1, 0, 0, 2}; if (n < 5) return small[n];
        SIZE = n; SIZEE = n - 1; TOPBIT = 1 << SIZEE; MASK = (1 << n) - 1; C8 = C4 = C2 = 0;
        // corner case: queen at row 0, column 0; the row-1 queen at column BOUND1 = 2 .. n-2
        B[0] = 1;
        for (BOUND1 = 2; BOUND1 < SIZEE; BOUND1++) {
            int bit = 1 << BOUND1; B[1] = bit; bt1(2, (2 | bit) << 1, 1 | bit, bit >> 1);
        }
        // non-corner case: row-0 queen at column BOUND1 < BOUND2; for each next BOUND1,
        // LASTMASK grows by one column at both sides and ENDBIT follows BOUND2
        SIDEMASK = LASTMASK = TOPBIT | 1; ENDBIT = TOPBIT >> 1;
        for (BOUND1 = 1, BOUND2 = n - 2; BOUND1 < BOUND2; BOUND1++, BOUND2--) {
            int bit = 1 << BOUND1; B[0] = bit; bt2(1, bit << 1, bit, bit >> 1);
            LASTMASK |= LASTMASK >> 1 | LASTMASK << 1; ENDBIT >>= 1;
        }
        return C8 * 8 + C4 * 4 + C2 * 2;
    }
    public static void main(String[] a) {
        int reps = Integer.getInteger("reps", 1);
        for (int n = Integer.parseInt(a[0]); n <= Integer.parseInt(a[1]); n++) {
            double best = 1e99; long ans = 0;
            for (int t = 0; t < reps; t++) {
                long t0 = System.nanoTime(); ans = count(n); best = Math.min(best, (System.nanoTime() - t0) / 1e9);
            }
            System.out.printf("n=%2d: %,d %s | %.3f s | %.1f ns/sol%n",
                n, ans, Ref.ok(n, ans), best, best * 1e9 / Math.max(ans, 1));
        }
    }
}
/** OEIS A000170, Q(0..27), for checking results. */
class Ref {
    static final long[] R = {1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512,
        95815104, 666090624, 4968057848L, 39029188884L, 314666222712L, 2691008701644L, 24233937684440L,
        227514171973736L, 2207893435808352L, 22317699616364044L, 234907967154122528L};
    static String ok(int n, long v) {
        return n >= R.length ? "(no reference)" : v == R[n] ? "OK" : "!!! MISMATCH (expected " + R[n] + ")";
    }
}
