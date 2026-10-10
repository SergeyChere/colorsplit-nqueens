#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Quick self-test of the repository (a few minutes on a laptop, no GPU needed):
#   1. C version: Q(1..17) against OEIS A000170, with both table sides
#   2. C version: interrupted run, a line cut off by a crash, resume
#   3. Java version (if java is installed): Q(1..16)
#   4. GPU code emulated on the CPU: Q(1..13), and its per-chunk sums for n = 16 against the C version
#   5. tools/verify.py on the Q(27) checkpoint in results/
#
# Run from anywhere:  bash tests/run_tests.sh      (or: make test)
# Builds into bin/ if needed; works in a temporary directory that is removed at the end.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
FAILS=0

pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; FAILS=$((FAILS + 1)); }
# check_ok LOG FROM TO: the log has an "OK" line for every n in FROM..TO and no MISMATCH
check_ok() {
    local log="$1" a="$2" b="$3" n missing=0
    for ((n = a; n <= b; n++)); do
        grep -Eq "^n= *$n: .*OK" "$log" || missing=1
    done
    ! grep -q MISMATCH "$log" && [ "$missing" = 0 ]
}

echo "building"
mkdir -p "$BIN"
[ -x "$BIN/colorsplit" ] || gcc -O3 -march=native -pthread -o "$BIN/colorsplit" "$ROOT/cpu/colorsplit.c" || exit 1
[ -x "$BIN/colorsplit_emu" ] || g++ -O2 -x c++ -DEMULATE -o "$BIN/colorsplit_emu" "$ROOT/gpu/colorsplit.cu" || exit 1
cd "$WORK"

echo "1. C version, Q(1..17)"
"$BIN/colorsplit" 1 17 -p 0 > c.log 2>&1
check_ok c.log 1 17 && pass "default table side" || fail "default table side (see output above)"
for side in white black; do
    "$BIN/colorsplit" 1 15 -p 0 -side $side > c_$side.log 2>&1
    check_ok c_$side.log 1 15 && pass "-side $side" || fail "-side $side"
done

echo "2. C version, interrupted run and resume"
"$BIN/colorsplit" 16 16 -p 0 -ckpt 1 -to 1500 > r1.log 2>&1
printf '15' >> colorsplit_n16.ckpt                        # a line cut off by a crash
"$BIN/colorsplit" 16 16 -p 0 -ckpt 1 > r2.log 2>&1
if check_ok r2.log 16 16 && grep -q "resuming: 1500" r2.log && grep -qx '15' colorsplit_n16.ckpt; then
    pass "resume after 1500 of 4096 chunks, torn line ignored"
else
    fail "resume"
fi
mkdir -p cref && (cd cref && "$BIN/colorsplit" 16 16 -p 0 -ckpt 1 > /dev/null 2>&1)

echo "3. Java version"
if command -v java > /dev/null 2>&1; then
    java -Xmx1g -Dprogress=0 "$ROOT/cpu/java/ColorSplit.java" 1 16 > j.log 2>&1
    check_ok j.log 1 16 && pass "Q(1..16)" || fail "Java Q(1..16)"
else
    echo "  skip  (java not installed)"
fi

echo "4. GPU code in CPU emulation"
"$BIN/colorsplit_emu" 1 13 -t 32 -ckpt 0 > e.log 2>&1
check_ok e.log 1 13 && pass "Q(1..13)" || fail "emulation Q(1..13)"
mkdir -p eref
"$BIN/colorsplit_emu" 16 16 -t 32 -ckpt 1 -dir eref > e16.log 2>&1
python3 "$ROOT/tools/verify.py" compare cref/colorsplit_n16.ckpt eref/colorsplit_n16.ckpt > cmp.log 2>&1 \
    && grep -q "chunks in both files: 4096" cmp.log \
    && pass "n = 16: all 4096 chunk sums equal to the C version" || { cat cmp.log; fail "chunk sums C vs GPU emulation"; }

echo "5. tools"
python3 "$ROOT/tools/verify.py" total "$ROOT/results/colorsplit_n27.ckpt" > v27.log 2>&1 \
    && pass "results/colorsplit_n27.ckpt sums to Q(27)" || { cat v27.log; fail "verify.py total on Q(27)"; }
python3 "$ROOT/tools/verify.py" merge merged.ckpt cref/colorsplit_n16.ckpt eref/colorsplit_n16.ckpt > m.log 2>&1 \
    && pass "verify.py merge of two n = 16 files" || { cat m.log; fail "verify.py merge"; }

echo
if [ "$FAILS" = 0 ]; then echo "ALL TESTS PASSED"; else echo "$FAILS TEST(S) FAILED"; exit 1; fi
