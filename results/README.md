# Results

Every value below up to n = 27 agrees with OEIS A000170. Q(28) was not known before:

**Q(28) = 2,564,269,152,506,723,832**

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
| 28 | 2,564,269,152,506,723,832 | 436.4 h of computation | `colorsplit.cu` (version 5), `-batch 8` |

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

## Q(28)

Computed with `gpu/colorsplit.cu` (version 5) on one RTX PRO 6000 Blackwell in Google Colab,
`-batch 8`, in October 2026: 436.4 hours of computation (18.2 days), in several Colab sessions,
each resuming from the checkpoint file. Q(27) to Q(28) took 7.0 times longer, in line with the
factor of 6 to 8 we measured on steps to even n.

[`colorsplit_n28.ckpt`](colorsplit_n28.ckpt) is the checkpoint file of that run, in the same
format as the Q(27) file (the milliseconds are the time of a batch divided by its 8 chunks).
For even n every row set lies in chunks 0 to 2047, so chunks 2048 to 4095 have the sum 0.

```
python3 tools/verify.py total results/colorsplit_n28.ckpt
sha256sum results/colorsplit_n28.ckpt   # 7289113b36701dd699d30f58f15069a4238049fec892e759171af5d013a81757
```

Checks:

- all 4096 chunks present, each once, each marked `ok`; the exact sum is the value above;
- divisible by 4 (Nielsen's theorem);
- the ratio Q(n)/Q(n−1) was 9.70, 10.11 and 10.53 for n = 25, 26 and 27; for n = 28 it is 10.92,
  which continues this steady rise of about 0.4 per step;
- the total is 3.6 times below the largest signed 64-bit integer, and the largest chunk sum is
  6.0 · 10^15, so no 64-bit sum can have overflowed; the 32-bit table counters were guarded too,
  since version 5 stops with an error if a counter could exceed 2^31, and the run never stopped
  for that reason;
- independent recomputation of a random sample of 50 of the 2048 nonzero chunks:
  `python3 tools/verify.py sample results/colorsplit_n28.ckpt --count 100 --seed 1921585467`
  draws 100 chunks, and the sample is every second one of them, starting with the first (chunks
  39, 63, 70, …, 2011). The seed is the first 8 hex digits of the file's sha256 (`7289113b`), so
  the sample is fixed by the result itself and anyone can draw it again. Each sampled chunk was
  recomputed twice, with the table forced to white (`-side white`) and to black (`-side black`).
  The main run chose the color with more rows, so every row set in the sample also went through
  the code path the main run did not use: **RESULT PENDING**.

Any chunk can be recomputed and compared, for example chunk 0:

```
echo 0 > one.txt
./colorsplit_gpu 28 28 -side black -chunks one.txt        # chunk 0: 7780397743116
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
- with the half-key pruning switched off, all 4096 chunk sums stay identical (script: n = 13 to 18;
  we also ran n = 19);
- without the weighted transposition, the Java version (`-Dnotr=true`) still returns Q(1) to Q(16).

What the half-key pruning saves: with it switched off (`-DNO_HALF_KEY`) the C version took 6.2 s
instead of 5.2 s at n = 18, 26.0 s instead of 18.8 s at n = 19 and 168 s instead of 143 s at
n = 20 (2 threads, the same machine for both builds, best of two runs for n = 18 and 19): it saves
about 15% on even boards and 28% at n = 19.

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
could exceed 2^31; the Q(28) run never stopped for that reason.
