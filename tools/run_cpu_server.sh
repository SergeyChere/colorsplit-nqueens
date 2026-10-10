#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Build the C version of ColorSplit on a fresh Linux server (Ubuntu/Debian) and compute a list of
# board sizes one after another on all cores.
#
#   git clone https://github.com/SergeyChere/colorsplit-nqueens.git && cd colorsplit-nqueens
#   tmux new -s nq                          # keeps the run alive when you disconnect
#   bash tools/run_cpu_server.sh 22 23 24   # default: 22 23 24
#
# Leave tmux without stopping the run: Ctrl+B, then D.   Come back: tmux attach -t nq
# Results: results_<hostname>.txt; full output: log_<n>.txt; progress: colorsplit_n<n>.ckpt.
# Stop with Ctrl+C at any time and run the same command again to continue where it stopped.
set -u
cd "$(dirname "$0")/.."

if ! command -v gcc > /dev/null 2>&1 || ! command -v make > /dev/null 2>&1; then
    echo "installing the compiler..."
    sudo apt-get update -y && sudo apt-get install -y build-essential tmux
fi

# AMD Zen, Zen+ and Zen 2 (CPU family 23) execute PEXT slowly: use the lookup-table version
EXTRA=""
if grep -q AuthenticAMD /proc/cpuinfo && grep -m1 "cpu family" /proc/cpuinfo | grep -qw 23; then
    EXTRA="-DSOFT_PEXT"
fi
make -B cpu CPU_EXTRA="$EXTRA" > /dev/null || { echo "build failed"; exit 1; }

BOARDS=${*:-22 23 24}
THREADS=$(nproc)
OUT="results_$(hostname).txt"
echo "machine: $(hostname), $THREADS threads, $(free -g | awk '/^Mem:/{print $2}') GB memory${EXTRA:+, built with $EXTRA}" | tee -a "$OUT"
for N in $BOARDS; do
    echo "=== n=$N: start $(date '+%F %T')" | tee -a "$OUT"
    ./bin/colorsplit "$N" "$N" -t "$THREADS" 2>&1 | tee "log_$N.txt" | grep --line-buffered -E "^n=|time over all" | tee -a "$OUT"
    echo "=== n=$N: end $(date '+%F %T')" | tee -a "$OUT"
done
echo "DONE. If this is a rented server, delete it in the provider's console: it is billed while it exists." | tee -a "$OUT"
