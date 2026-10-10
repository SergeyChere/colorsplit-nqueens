# Results

Every value below agrees with OEIS A000170.

## Runs on one GPU

NVIDIA RTX PRO 6000 Blackwell Server Edition (188 SMs, 96 GB), Google Colab Pro+ (G4 runtime).
Software of the Colab runtime at the time of writing: NVIDIA driver 580.82.07, CUDA 13.0
(nvcc V13.0.88); the power limit shown by `nvidia-smi` was 600 W.

| n | Q(n) | Time | Program |
|---|---|---|---|
| 22 | 2,691,008,701,644 | 43 s | first GPU version (prototype) |
| 23 | 24,233,937,684,440 | 151.5 s | `colorsplit_v2.cu`, tuned (`-DPFX=4`, `-t 512 -batch 32`); 3.2 min before tuning |
| 24 | 227,514,171,973,736 | 21 min | first GPU version (prototype) |
| 25 | 2,207,893,435,808,352 | 6,289 s (1.75 h) | `colorsplit_v2.cu`, untuned settings |
| 26 | 22,317,699,616,364,044 | 48,112 s (13.4 h) | `colorsplit_v2.cu`, tuned |
| 27 | 234,907,967,154,122,528 | 62.2 h of computation | `colorsplit_v2.cu`, tuned |

`gpu/colorsplit.cu` (version 4) on the same kind of GPU, 9 October 2026: Q(4) to Q(23) all equal
to the known values; Q(22) in 20.5 s and Q(23) in 79.7 s, against 156.8 s for `colorsplit_v2.cu`
in the same session. On chunks 1344 to 1375 of Q(27) it gave the same 32 sums as
`colorsplit_n27.ckpt` and took 10.7 min, against 693 s for these chunks in the Q(27) run.
Version 5 (dynamic assignment of row sets to blocks, heaviest first), same day: Q(4) to Q(23) all
equal to the known values (Q(23) in 86.5 s); the same 32 chunks of Q(27) with `-batch 8` gave the
same sums in 10.6 min, so checkpoints every 8 chunks cost nothing.

Q(27) ran in five sessions over 2.74 days of wall-clock time, because the Colab sessions ended
several times; each session resumed from the checkpoint file. The 3.6 hours lost in the first
interruption (before the checkpoint writing was made robust) are not counted in the 62.2 hours.

[`colorsplit_n27.ckpt`](colorsplit_n27.ckpt) is the checkpoint file of that run: one line per
chunk, `<chunk> <sum> <milliseconds> ok`, where the milliseconds are the time of the chunk's batch
divided by the 32 chunks of the batch. Check it with

```
python3 tools/verify.py total results/colorsplit_n27.ckpt
```

which adds the 4096 sums with exact integers (234,907,967,154,122,528, divisible by 4, equal to
the known value). Any chunk can be recomputed with any version and compared, for example chunk 0
on a GPU:

```
echo 0 > one.txt
./colorsplit_gpu 27 27 -side black -chunks one.txt        # chunk 0: 260013399880
```

## Runs on rented CPU servers

| n | Time | Program, machine |
|---|---|---|
| 22 | | Java version, DigitalOcean |
| 23 | 4,343 s (72 min) | C version, 8 vCPUs, DigitalOcean |
| 24 | 79,162 s (22.0 h) | Java version, 8 vCPUs, DigitalOcean |
| 25 | 120,592 s (33.5 h) | Java version, DigitalOcean |

## One CPU core

Intel Xeon at 2.1 GHz, one thread, best of repeated runs; `~` marks extrapolated values that were
not run.

| n | Bitmask backtracking | BTSym (8 symmetries) | ColorSplit |
|---|---|---|---|
| 16 | 10.1 s | 2.35 s | 0.76 s |
| 17 | 71 s | 16.0 s | 1.94 s |
| 18 | ~8 min | 115 s | 12.4 s |
| 19 | ~1 h | ~14 min | 47 s |
| 20 | ~8 h | ~1.9 h | 5.9 min |
| 21 | — | ~15 h | 21.1 min |

## Checks of the reductions

`make check-reductions` (script [`../tests/check_reductions.sh`](../tests/check_reductions.sh))
builds the C version with compile-time switches that turn single reductions off, and checks:

- every table cell of weight λ > 0, read through the mirror folding for odd n, equals the count of
  a plain enumeration without the mirror and without the early cut (the script runs n = 4 to 15;
  we also ran n = 16: 22.3 million cells in all, no difference);
- with the early cut switched off, all 4096 chunk sums stay identical (script: n = 13 to 18; we
  also ran n = 19);
- with the left-right mirror switched off, all 4096 chunk sums stay identical for odd n (script:
  n = 13, 15, 17; we also ran n = 19);
- without the weighted transposition, the Java version (`-Dnotr=true`) still returns Q(1) to Q(16).

Largest table cell (after the weights λ and, for odd n, the mirror folding) and largest number of
placements W_R(C) behind one key. The cells are 32-bit counters, which overflow only above
4,294,967,295.

| n | largest cell | largest W_R(C) |
|---|---|---|
| 14 | 20 | 20 |
| 15 | 80 | 24 |
| 16 | 36 | 20 |
| 17 | 124 | 60 |
| 18 | 104 | 54 |
| 19 | 280 | 72 |
| 20 | 164 | 94 |
| 21 | 728 | 272 |

The maximum grows by a factor of 2 to 3 every two steps of n, so it should stay below 10^5 at
n = 28. An overflow could only lower a total (a counter wraps to a smaller value), and all totals
for n ≤ 27 equal the known values. `gpu/colorsplit.cu` also stops with exit code 3 if a counter
could exceed 2^31.
