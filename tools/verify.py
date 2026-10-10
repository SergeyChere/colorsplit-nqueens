#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
verify.py -- check, compare and merge ColorSplit checkpoint files.

A checkpoint file has one line per finished chunk:

    <chunk> <sum> <milliseconds> ok

The chunk sums do not depend on the implementation (C, Java, GPU), on the number of
threads or on which color goes into the table, so any chunk can be recomputed
independently and compared. All arithmetic here uses exact Python integers.

Commands

  total CKPT [--n N]
      Sum a checkpoint file and check it: every chunk present, no chunk with two
      different sums, divisible by 4 (n >= 6), equal to the known value if there is one.

  sample CKPT [--count 150] [--seed 1] [--out sample.txt]
      Pick random finished chunks with a nonzero sum, for an independent recomputation:
          python3 tools/verify.py sample colorsplit_n28.ckpt --count 150 --out sample28.txt
          ./colorsplit_gpu 28 28 -side white -chunks sample28.txt     (writes ..._sideW.ckpt)

  compare MAIN CHECK
      Compare two checkpoint files chunk by chunk (only the chunks present in both):
          python3 tools/verify.py compare colorsplit_n28.ckpt colorsplit_n28_sideW.ckpt

  merge OUT INPUT... [--n N] [--chunks K] [--expect Q]
      Merge files from several machines into OUT and check the total like `total`:
          python3 tools/verify.py merge colorsplit_n28.ckpt part0.ckpt part1.ckpt part2.ckpt

n is taken from a file name like colorsplit_n28.ckpt unless --n is given. The number
of chunks is 4096 (2^n for n < 12). Exit status: 0 when every check passes, 1 otherwise.
"""
import argparse
import os
import random
import re
import sys

# Q(n) for n = 0..27, OEIS A000170
KNOWN = [1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184,
         14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712,
         2691008701644, 24233937684440, 227514171973736, 2207893435808352,
         22317699616364044, 234907967154122528]


def fmt(x):
    """Integer with thin groups of three digits: 1 234 567."""
    return f"{x:,}".replace(",", " ")


def load(path):
    """Read a checkpoint file.

    Returns (chunks, conflicts, bad): chunks maps chunk -> (sum, ms) using the first
    line of each chunk; conflicts lists (chunk, first sum, other sum) for chunks that
    appear again with a different sum; bad counts lines that are not well formed
    (for example a line cut off by a crash).
    """
    chunks, conflicts, bad = {}, [], 0
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if len(p) != 4 or p[3] != "ok" or not all(re.fullmatch(r"-?\d+", x) for x in p[:3]):
                bad += 1
                continue
            c, v, ms = int(p[0]), int(p[1]), int(p[2])
            if c in chunks and chunks[c][0] != v:
                conflicts.append((c, chunks[c][0], v))
            chunks.setdefault(c, (v, ms))
    return chunks, conflicts, bad


def guess_n(paths):
    for path in paths:
        m = re.search(r"_n(\d+)", os.path.basename(path))
        if m:
            return int(m.group(1))
    return None


def n_chunks(n, given):
    if given:
        return given
    if n is not None and n < 12:
        return 1 << n
    return 4096


def check_total(chunks, n, nch, expect, problems):
    """Print the total and the checks; return the number of failed checks."""
    missing = [c for c in range(nch) if c not in chunks]
    extra = [c for c in chunks if not 0 <= c < nch]
    total = sum(v for v, _ in chunks.values())
    hours = sum(ms for _, ms in chunks.values()) / 3.6e6
    fails = problems
    print(f"chunks: {len(chunks)} of {nch}; compute time in the file: {hours:.2f} h")
    if missing:
        print(f"MISSING {len(missing)} chunk(s), for example: {missing[:20]}")
        fails += 1
    if extra:
        print(f"OUT OF RANGE chunk numbers: {sorted(extra)[:20]}")
        fails += 1
    print(f"sum: {fmt(total)}")
    if n is not None and n >= 6:
        ok = total % 4 == 0
        print("divisible by 4 (Nielsen):", "yes" if ok else "NO -- ERROR")
        fails += 0 if ok else 1
    if expect is None and n is not None and n < len(KNOWN):
        expect = KNOWN[n]
    if expect is not None:
        ok = total == expect
        print(f"expected {fmt(expect)}:", "MATCH" if ok else "DIFFERENT")
        fails += 0 if ok else 1
    return fails


def cmd_total(a):
    chunks, conflicts, bad = load(a.ckpt)
    n = a.n if a.n is not None else guess_n([a.ckpt])
    for c, x, y in conflicts:
        print(f"CONFLICT: chunk {c} appears with sums {x} and {y}")
    if bad:
        print(f"note: {bad} malformed line(s) ignored")
    fails = check_total(chunks, n, n_chunks(n, a.chunks), a.expect, len(conflicts))
    print("RESULT:", "all checks passed" if fails == 0 else "SOME CHECKS FAILED")
    sys.exit(0 if fails == 0 else 1)


def cmd_sample(a):
    chunks, _, _ = load(a.ckpt)
    pool = sorted(c for c, (v, _) in chunks.items() if v != 0)
    if not pool:
        sys.exit("no finished chunk with a nonzero sum in this file")
    random.seed(a.seed)
    k = min(a.count, len(pool))
    pick = sorted(random.sample(pool, k))
    with open(a.out, "w") as f:
        f.write("\n".join(map(str, pick)) + "\n")
    print(f"picked {k} of {len(pool)} finished chunks (seed {a.seed}) -> {a.out}")


def cmd_compare(a):
    main, c1, _ = load(a.main)
    check, c2, _ = load(a.check)
    for name, cf in ((a.main, c1), (a.check, c2)):
        for c, x, y in cf:
            print(f"WARNING: in {name} chunk {c} appears with sums {x} and {y}")
    common = sorted(set(main) & set(check))
    diff = [(c, main[c][0], check[c][0]) for c in common if main[c][0] != check[c][0]]
    print(f"chunks in both files: {len(common)}; equal: {len(common) - len(diff)}; different: {len(diff)}")
    for c, x, y in diff[:20]:
        print(f"  chunk {c}: {fmt(x)} in {a.main}, {fmt(y)} in {a.check}")
    if diff or c1 or c2:
        print("RESULT: MISMATCH -- recompute these chunks")
        sys.exit(1)
    if not common:
        print("RESULT: no chunk in common, nothing compared")
        sys.exit(1)
    print("RESULT: every compared chunk is identical")


def cmd_merge(a):
    if os.path.abspath(a.out) in [os.path.abspath(p) for p in a.inputs]:
        sys.exit("OUT must not be one of the input files")
    merged, problems = {}, 0
    for path in a.inputs:
        chunks, conflicts, bad = load(path)
        for c, x, y in conflicts:
            print(f"CONFLICT: {path}: chunk {c} with sums {x} and {y}")
            problems += 1
        for c, (v, ms) in chunks.items():
            if c in merged and merged[c][0] != v:
                print(f"CONFLICT: chunk {c}: {fmt(merged[c][0])} and {fmt(v)} in different files")
                problems += 1
            merged.setdefault(c, (v, ms))
        print(f"{path}: {len(chunks)} chunk(s)" + (f", {bad} malformed line(s) ignored" if bad else ""))
    with open(a.out, "w") as f:
        for c in sorted(merged):
            f.write(f"{c} {merged[c][0]} {merged[c][1]} ok\n")
    print(f"merged file: {a.out}")
    n = a.n if a.n is not None else guess_n([a.out] + a.inputs)
    fails = check_total(merged, n, n_chunks(n, a.chunks), a.expect, problems)
    print("RESULT:", "all checks passed" if fails == 0 else "SOME CHECKS FAILED")
    sys.exit(0 if fails == 0 else 1)


def main():
    ap = argparse.ArgumentParser(description="check, compare and merge ColorSplit checkpoint files")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("total", help="sum one checkpoint file and check it")
    t.add_argument("ckpt")
    s = sub.add_parser("sample", help="pick random finished chunks for an independent recomputation")
    s.add_argument("ckpt")
    s.add_argument("--count", type=int, default=150)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", default="sample.txt")
    c = sub.add_parser("compare", help="compare two checkpoint files chunk by chunk")
    c.add_argument("main")
    c.add_argument("check")
    m = sub.add_parser("merge", help="merge checkpoint files from several machines")
    m.add_argument("out")
    m.add_argument("inputs", nargs="+")
    for p in (t, m):
        p.add_argument("--n", type=int, default=None, help="board size (default: from the file name)")
        p.add_argument("--chunks", type=int, default=None, help="number of chunks (default 4096)")
        p.add_argument("--expect", type=int, default=None, help="expected total (default: known Q(n))")
    a = ap.parse_args()
    {"total": cmd_total, "sample": cmd_sample, "compare": cmd_compare, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
