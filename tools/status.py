#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
status.py -- progress report for a long ColorSplit run, read from its checkpoint file.

    python3 tools/status.py colorsplit_n28.ckpt
    python3 tools/status.py /content/drive/MyDrive/nq/colorsplit_n28.ckpt --units-per-hour 8.9
    watch -n 300 python3 tools/status.py colorsplit_n28.ckpt        (refresh every 5 minutes)

Shows whether a colorsplit program is running, the GPU load (if nvidia-smi exists), how
many chunks are done, the pure compute time, the sum so far, an estimate of the work done
and of the time left, and how long ago the file was last saved. When all chunks are done
it prints the total and checks it (divisibility by 4, the known value if there is one).

The chunks are far from equal in cost, so the fraction of work done is estimated from
measured profiles: the per-chunk times of our Q(26) run for even n and of our Q(27) run
(results/colorsplit_n27.ckpt) for odd n. For a run split over several machines, point it
at the merged file (tools/verify.py merge) or read the estimate for one part as a rough
guide only.

Options: --n N (default: from the file name), --units-per-hour U (cloud credits per GPU
hour, e.g. 8.9 for a Colab G4; shows the expected cost of the rest of the run).
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = 4096
KNOWN = [1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184,
         14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712,
         2691008701644, 24233937684440, 227514171973736, 2207893435808352,
         22317699616364044, 234907967154122528]

# Q(26) on one GPU: (chunks finished in order 0, 1, 2, ..., seconds since the start).
# For even n all work is in chunks 0..2047; chunks 2048..4095 are empty.
PROFILE_EVEN = [(0, 0), (32, 1563), (64, 3510), (96, 5250), (128, 7761), (160, 9251), (192, 10695),
                (224, 12660), (256, 14473), (288, 15878), (320, 17917), (352, 19135), (384, 20927),
                (416, 22516), (448, 24000), (480, 25371), (512, 26663), (544, 27419), (576, 28170),
                (608, 29169), (640, 30120), (672, 30715), (704, 31177), (736, 31940), (768, 32523),
                (800, 33321), (832, 34116), (864, 34864), (896, 35588), (928, 36227), (960, 36725),
                (992, 37318), (1024, 37760), (1056, 38420), (1088, 39395), (1120, 39993), (1152, 40861),
                (1184, 41583), (1216, 42264), (1248, 42899), (1280, 43478), (1312, 43920), (1344, 44558),
                (1376, 44818), (1408, 45190), (1440, 45661), (1472, 46100), (1504, 46369), (1536, 46625),
                (1568, 46874), (1600, 47080), (1632, 47283), (1664, 47450), (1696, 47563), (1728, 47639),
                (1760, 47731), (1792, 47792), (1824, 47898), (1856, 47978), (1888, 48029), (1920, 48067),
                (1984, 48102), (4096, 48112)]


def fmt_int(x):
    return f"{x:,}".replace(",", " ")


def fmt_hours(h):
    return f"{h:.1f} h" if h < 48 else f"{h:.0f} h ({h / 24:.1f} days)"


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def weights_even():
    w = [0.0] * CHUNKS
    total = PROFILE_EVEN[-1][1]
    for (k0, t0), (k1, t1) in zip(PROFILE_EVEN, PROFILE_EVEN[1:]):
        for c in range(k0, k1):
            w[c] = (t1 - t0) / (k1 - k0) / total
    return w


def weights_odd():
    """Per-chunk times of the Q(27) run; uniform weights if the file is not found."""
    path = os.path.join(HERE, "..", "results", "colorsplit_n27.ckpt")
    ms = [0.0] * CHUNKS
    try:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) == 4 and p[3] == "ok" and 0 <= int(p[0]) < CHUNKS:
                    ms[int(p[0])] = float(p[2])
    except OSError:
        return [1.0 / CHUNKS] * CHUNKS, "uniform (results/colorsplit_n27.ckpt not found)"
    total = sum(ms)
    return [m / total for m in ms], "profile of our Q(27) run"


def main():
    ap = argparse.ArgumentParser(description="progress of a long ColorSplit run")
    ap.add_argument("ckpt", nargs="?", help="checkpoint file (default: the only colorsplit_n*.ckpt here)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--units-per-hour", type=float, default=0.0)
    a = ap.parse_args()

    path = a.ckpt
    if path is None:
        found = sorted(glob.glob("colorsplit_n*.ckpt"))
        if len(found) != 1:
            sys.exit("give the checkpoint file, e.g.: python3 tools/status.py colorsplit_n28.ckpt")
        path = found[0]
    n = a.n
    if n is None:
        m = re.search(r"_n(\d+)", os.path.basename(path))
        n = int(m.group(1)) if m else None

    # a run is recognized by its process (any binary name: colorsplit*, cs4, ...) on the GPU or on the CPU
    gpu_apps = [l.strip() for l in run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                        "--format=csv,noheader"]).splitlines() if l.strip()]
    procs = [l for l in run(["pgrep", "-af", r"colorsplit|/cs[0-9]|ColorSplit"]).splitlines()
             if "status.py" not in l and "pgrep" not in l]
    if gpu_apps:
        print("program: RUNNING on the GPU (" + "; ".join(gpu_apps) + ")")
    elif procs:
        print("program: RUNNING (" + procs[0][:80] + ")")
    else:
        print("program: NOT RUNNING")
    gpu = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"])
    if gpu:
        util, mem = [x.strip() for x in gpu.splitlines()[0].split(",")]
        print(f"GPU: load {util}, memory {mem}")

    if not os.path.exists(path):
        sys.exit(f"no checkpoint file yet: {path}\n(it is written after the first batch of chunks; on the largest boards the first\n"
                 f"batches are the slowest, so this can take hours: use -batch 8 or smaller for n >= 28)")
    done, conflicts = {}, 0
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 4 and p[3] == "ok":
                try:
                    c, v, ms = int(p[0]), int(p[1]), int(p[2])
                except ValueError:
                    continue
                if c in done and done[c][0] != v:
                    conflicts += 1
                done.setdefault(c, (v, ms))

    nch = (1 << n) if n is not None and n < 12 else CHUNKS
    n_done = len(done)
    s = sum(v for v, _ in done.values())
    hours = sum(ms for _, ms in done.values()) / 3.6e6
    print(f"board: n = {n}" if n is not None else "board: unknown n")
    print(f"done: {n_done} of {nch} chunks, {fmt_hours(hours)} of pure compute time")
    print(f"sum so far: {fmt_int(s)}" + (f"  ({100 * s / KNOWN[n]:.2f}% of the known Q({n}))"
                                         if n is not None and 0 < n < len(KNOWN) and KNOWN[n] else ""))
    if conflicts:
        print(f"WARNING: {conflicts} chunk(s) appear with different sums -- check with tools/verify.py")

    if n_done >= nch:
        print()
        print(f"ALL {nch} CHUNKS DONE.  Q({n}) = {fmt_int(s)}")
        if n is not None and n >= 6:
            print("divisible by 4 (Nielsen):", "yes" if s % 4 == 0 else "NO -- ERROR, do not publish")
        if n is not None and n < len(KNOWN):
            print("known value:", "MATCH" if s == KNOWN[n] else f"DIFFERENT (expected {fmt_int(KNOWN[n])})")
        print("next: python3 tools/verify.py total", path, "  and an independent recomputation of a sample")
    elif nch == CHUNKS and n is not None:
        w, src = (weights_even(), "profile of our Q(26) run") if n % 2 == 0 else weights_odd()
        frac = sum(w[c] for c in done if 0 <= c < CHUNKS)
        bar = int(round(frac * 40))
        print("[" + "#" * bar + "." * (40 - bar) + f"] about {100 * frac:.1f}% of the work ({src})")
        if frac > 0.005 and hours > 0:
            total = hours / frac
            left = total - hours
            print(f"estimate: {fmt_hours(total)} in total, {fmt_hours(left)} left")
            if a.units_per_hour > 0:
                print(f"          about {fmt_int(round(left * a.units_per_hour))} compute units for the rest")

    age = (time.time() - os.path.getmtime(path)) / 60
    print(f"last save: {age:.0f} min ago" + ("  <- long ago, is the run still going?" if age > 180 and n_done < nch else ""))


if __name__ == "__main__":
    main()
