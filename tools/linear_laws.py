#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lemma 2 of the paper: for n = 9..32 the n-queens solutions satisfy no linear law beyond
the row and column sums.

The 0/1 incidence vectors of the solutions (one coordinate per square) are permutation
matrices, so they span at most the space of all permutation matrices, of dimension
(n - 1)^2 + 1 (affine dimension (n - 1)^2). The script computes the rank of a set of
solutions exactly modulo the prime 2^31 - 1. A rank modulo p never exceeds the rank over
the rationals, so a set that reaches (n - 1)^2 + 1 proves that all solutions together span
the full space: no weighting of the squares gives the same sum in every solution, except
combinations of row and column sums.

  python3 tools/linear_laws.py               n = 4..32 (all solutions up to n = 13,
                                              randomized backtracking above; about 1 min)
  python3 tools/linear_laws.py 9 16          a range of n

Needs numpy. For n <= 8 the rank is lower; at n = 8, for example, every solution has
exactly four queens on white squares.
"""
import random
import sys
import time

import numpy as np

P = 2147483647          # 2^31 - 1, prime
ALL_UP_TO = 13          # use every solution up to this n, a random sample above


def all_solutions(n):
    full = (1 << n) - 1
    out, sol = [], [0] * n

    def dfs(r, cols, d1, d2):
        if r == n:
            out.append(sol[:])
            return
        av = full & ~(cols | d1 | d2)
        while av:
            b = av & -av
            av ^= b
            sol[r] = b.bit_length() - 1
            dfs(r + 1, cols | b, ((d1 | b) << 1) & full, (d2 | b) >> 1)

    dfs(0, 0, 0, 0)
    return out


def random_solution(n, rng, limit=200000):
    """One solution by backtracking with a random column order; restarts after `limit` nodes."""
    full = (1 << n) - 1
    while True:
        sol, nodes = [0] * n, [0]

        def dfs(r, cols, d1, d2):
            if r == n:
                return True
            nodes[0] += 1
            if nodes[0] > limit:
                return False
            av = full & ~(cols | d1 | d2)
            cand = [c for c in range(n) if av >> c & 1]
            rng.shuffle(cand)
            for c in cand:
                b = 1 << c
                sol[r] = c
                if dfs(r + 1, cols | b, ((d1 | b) << 1) & full, (d2 | b) >> 1):
                    return True
                if nodes[0] > limit:
                    return False
            return False

        if dfs(0, 0, 0, 0):
            return sol


def rank_mod_p(m):
    """Rank of an integer matrix modulo P (Gaussian elimination)."""
    m = m.copy() % P
    rows, cols = m.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(m[r:, c])[0]
        if len(nz) == 0:
            continue
        p = r + nz[0]
        if p != r:
            m[[r, p]] = m[[p, r]]
        m[r] = (m[r] * pow(int(m[r, c]), P - 2, P)) % P
        below = m[r + 1:, c].copy()
        nzb = np.nonzero(below)[0]
        if len(nzb):
            idx = r + 1 + nzb
            m[idx] = (m[idx] - np.outer(below[nzb], m[r])) % P
        r += 1
    return r


def vectors(n, sols):
    v = np.zeros((len(sols), n * n), dtype=np.int64)
    for i, s in enumerate(sols):
        for row, c in enumerate(s):
            v[i, row * n + c] = 1
    return v


def check(n, rng):
    target = (n - 1) ** 2 + 1
    t0 = time.time()
    if n <= ALL_UP_TO:
        sols = all_solutions(n)
        how = "all solutions"
        rk = rank_mod_p(vectors(n, sols)) if sols else 0
    else:
        sols = [random_solution(n, rng) for _ in range(target + 60)]
        rk = rank_mod_p(vectors(n, sols))
        while rk < target and len(sols) < 20 * target:
            sols += [random_solution(n, rng) for _ in range(target)]
            rk = rank_mod_p(vectors(n, sols))
        how = "random solutions"
    verdict = "full: no extra linear law" if rk == target else "LOWER: extra linear laws"
    print(f"n={n:2d}  {len(sols):6d} {how:16s}  rank {rk:4d} of {target:4d}  "
          f"(affine dimension {max(rk - 1, 0)} of {(n - 1) ** 2})  {verdict}  [{time.time() - t0:.1f} s]",
          flush=True)
    return rk == target


def main():
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (4, 32)
    rng = random.Random(12345)
    results = {n: check(n, rng) for n in range(lo, hi + 1)}
    ok = all(full for n, full in results.items() if n >= 9)
    print("RESULT:", "full rank for every n >= 9 checked" if ok else "a board with n >= 9 has an extra linear law")


if __name__ == "__main__":
    main()
