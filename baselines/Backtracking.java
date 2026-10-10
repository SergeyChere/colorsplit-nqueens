// SPDX-License-Identifier: MIT
/**
 * Backtracking -- two plain n-queens baselines without symmetry, CPU version (Java).
 *
 * WHAT IT COMPUTES
 *   Q(n), the number of ways to place n mutually non-attacking queens on an n x n board
 *   (OEIS A000170), by straightforward backtracking: one queen per row, every solution is
 *   visited once, no symmetry reduction. Two variants, for comparison with BTSym (symmetric
 *   bitmask backtracking) and ColorSplit:
 *     arrays  boolean arrays mark the used columns, diagonals and anti-diagonals; every row
 *             tries all n columns.
 *     bits    bitmasks of the attacked columns and diagonals; the free squares of a row are one
 *             AND-NOT, and only free squares are visited (lowest bit first).
 *
 * HOW TO RUN (Java 11+; the source file is run directly)
 *   java Backtracking.java <arrays|bits> <n_from> <n_to> [reps]
 *   e.g. java Backtracking.java bits 1 15
 *   For each n it prints the number of solutions, the OEIS check (OK or MISMATCH; OEIS values
 *   are stored for n <= 27), the best wall time of reps runs (default 1) in seconds, and that
 *   time per solution in nanoseconds:
 *     "n=<n>: <Q(n)> <check> | <seconds> s | <ns> ns/sol"
 *   The bitmasks are ints, so n <= 31; run time limits n much further in practice.
 */
public class Backtracking {
    static int n;
    static long sols;
    static boolean[] col, d1, d2; // used columns, diagonals (index r - c + n - 1), anti-diagonals (index r + c)

    /** arrays variant: place the queen of row r in every free column. */
    static void arr(int r) {
        if (r == n) { sols++; return; }
        for (int c = 0; c < n; c++) {
            if (col[c] || d1[r - c + n - 1] || d2[r + c]) continue;
            col[c] = d1[r - c + n - 1] = d2[r + c] = true;
            arr(r + 1);
            col[c] = d1[r - c + n - 1] = d2[r + c] = false;
        }
    }

    /**
     * bits variant: cols, dl, dr = columns of row r attacked along columns and the two diagonal
     * directions; the diagonal masks move by one column per row.
     */
    static void bits(int r, int cols, int dl, int dr) {
        if (r == n) { sols++; return; }
        int av = ((1 << n) - 1) & ~(cols | dl | dr); // free squares of row r
        while (av != 0) {
            int b = av & -av; // lowest free square
            av ^= b;
            bits(r + 1, cols | b, (dl | b) << 1, (dr | b) >> 1);
        }
    }

    /** Q(size) with the chosen variant. */
    static long count(int size, boolean useBits) {
        n = size;
        sols = 0;
        if (useBits) {
            bits(0, 0, 0, 0);
        } else {
            col = new boolean[n];
            d1 = new boolean[2 * n];
            d2 = new boolean[2 * n];
            arr(0);
        }
        return sols;
    }

    static final long[] REF = {1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512,
        95815104, 666090624, 4968057848L, 39029188884L, 314666222712L, 2691008701644L, 24233937684440L,
        227514171973736L, 2207893435808352L, 22317699616364044L, 234907967154122528L}; // OEIS A000170, Q(0..27)

    public static void main(String[] args) {
        boolean ok = (args.length == 3 || args.length == 4) && (args[0].equals("arrays") || args[0].equals("bits"));
        int from = 0, to = 0, reps = 1;
        if (ok) {
            try {
                from = Integer.parseInt(args[1]);
                to = Integer.parseInt(args[2]);
                if (args.length == 4) reps = Integer.parseInt(args[3]);
            } catch (NumberFormatException e) {
                ok = false;
            }
            ok = ok && from >= 0 && to <= 31 && reps >= 1;
        }
        if (!ok) {
            System.err.println("usage: java Backtracking.java <arrays|bits> <n_from> <n_to> [reps]   (0 <= n <= 31, reps >= 1)");
            System.exit(2);
        }
        boolean useBits = args[0].equals("bits");
        System.out.printf("Backtracking (%s, best of %d run%s)%n", args[0], reps, reps == 1 ? "" : "s");
        for (int size = from; size <= to; size++) {
            double best = 1e99;
            long ans = 0;
            for (int t = 0; t < reps; t++) {
                long t0 = System.nanoTime();
                ans = count(size, useBits);
                best = Math.min(best, (System.nanoTime() - t0) / 1e9);
            }
            String chk = size >= REF.length ? "(no reference)"
                : ans == REF[size] ? "OK" : "MISMATCH (expected " + REF[size] + ")";
            System.out.printf("n=%2d: %,d %s | %.3f s | %.1f ns/sol%n", size, ans, chk, best, best * 1e9 / Math.max(ans, 1));
        }
    }
}
