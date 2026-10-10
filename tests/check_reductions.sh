#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Checks that the reductions of the algorithm leave the result unchanged (about two minutes):
#   1. Table cells: for every row set and every key of weight λ > 0, the table cell (read
#      through the mirror folding for odd n) equals the count found by a plain enumeration
#      without the mirror and without the early cut; n = 4..15.
#   2. Early cut switched off: identical 4096 chunk sums; n = 13..18.
#   3. Left-right mirror switched off (odd n): identical chunk sums; n = 13, 15, 17.
#   4. Largest table cell and largest W_R(C) for n = 10..19: the cells are 32-bit counters,
#      so this shows the margin before an overflow.
#
# Run from anywhere:  bash tests/check_reductions.sh      (or: make check-reductions)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/cpu/colorsplit.c"
VERIFY="python3 $ROOT/tools/verify.py"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
FAILS=0
pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; FAILS=$((FAILS + 1)); }

echo "building the normal version and four verification builds"
for v in "" NO_EARLY_CUT NO_MIRROR CHECK_CELLS CELL_STATS; do
    out="cs${v:+_$v}"
    gcc -O3 -march=native -pthread ${v:+-D$v} -o "$out" "$SRC" || exit 1
done

echo "1. table cells against a plain enumeration"
./cs_CHECK_CELLS 4 15 -p 0 > cells.log 2>&1
if grep -q "MISMATCH" cells.log || grep -Eq "cells n=[0-9]+: [0-9]+ compared, [1-9]" cells.log \
   || [ "$(grep -c "0 wrong" cells.log)" != 12 ]; then
    cat cells.log; fail "table cells"
else
    pass "n = 4..15: $(awk '/compared/ {s += $3} END {print s}' cells.log) cells compared, none wrong"
fi

mkdir -p base nocut nomirror
for N in 13 14 15 16 17 18; do
    (cd base && ../cs "$N" "$N" -ckpt 1 -p 0 > /dev/null 2>&1)
    (cd nocut && ../cs_NO_EARLY_CUT "$N" "$N" -ckpt 1 -p 0 > /dev/null 2>&1)
    if [ $((N % 2)) = 1 ]; then (cd nomirror && ../cs_NO_MIRROR "$N" "$N" -ckpt 1 -p 0 > /dev/null 2>&1); fi
done

echo "2. early cut switched off"
ok=1
for N in 13 14 15 16 17 18; do
    $VERIFY compare base/colorsplit_n$N.ckpt nocut/colorsplit_n$N.ckpt > c.log 2>&1 && grep -q "chunks in both files: 4096" c.log || { cat c.log; ok=0; }
done
[ $ok = 1 ] && pass "n = 13..18: all 4096 chunk sums identical" || fail "early cut"

echo "3. left-right mirror switched off (odd n)"
ok=1
for N in 13 15 17; do
    $VERIFY compare base/colorsplit_n$N.ckpt nomirror/colorsplit_n$N.ckpt > c.log 2>&1 && grep -q "chunks in both files: 4096" c.log || { cat c.log; ok=0; }
done
[ $ok = 1 ] && pass "n = 13, 15, 17: all 4096 chunk sums identical" || fail "mirror"

echo "4. largest table cells (32-bit counters overflow above 4294967295)"
./cs_CELL_STATS 10 19 -p 0 | grep "cells n=" | sed 's/^ */  /'

echo
if [ "$FAILS" = 0 ]; then echo "ALL CHECKS PASSED"; else echo "$FAILS CHECK(S) FAILED"; exit 1; fi
