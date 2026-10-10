# GPU version

`colorsplit.cu` is the GPU version (CUDA). It computes the same chunk sums as the CPU versions and
writes the same checkpoint files, so a computation can move between CPU and GPU, or between GPUs.

`colorsplit_v2.cu` is the archival version that computed Q(25), Q(26) and Q(27) for the paper.
`colorsplit.cu` adds a dynamic distribution of the work inside a thread block and of the row sets
over the blocks (heavy row sets first), a smaller search stack, the `-side` and `-chunks` options
and a guard against overflow of the 32-bit table counters. Both give identical chunk sums (checked in CPU emulation up to n = 18). On the G4 GPU of
Google Colab, `colorsplit.cu` computes Q(22) in 20 s and Q(23) in 80 s, about twice as fast as
`colorsplit_v2.cu` (157 s for Q(23) in the same session). On large boards, where memory access
dominates, the gain is small: on 32 chunks of Q(27) (chunks 1344 to 1375) it reproduced the sums of
our Q(27) run and took about 640 s, against 693 s for `colorsplit_v2.cu`, about 8% faster. Handing the
row sets to the blocks dynamically lets it save a checkpoint every 8 chunks at the same speed: the
same 32 chunks took 10.6 min with `-batch 8`.

## Build

```
nvcc -O3 -arch=native -o colorsplit_gpu colorsplit.cu
```

Or from the repository root: `make gpu`. Our runs used the Colab G4 runtime (at the time of writing:
CUDA 13.0, nvcc 13.0.88, NVIDIA driver 580.82.07); the program prints the CUDA versions at the
start of every run, so the run log records them.

For a GPU other than the one in the machine use an explicit architecture: `-arch=sm_86` (RTX A6000,
RTX 3090, A40), `-arch=sm_89` (L4, RTX 4090, RTX 6000 Ada),
`-arch=sm_90` (H100), `-arch=sm_120` (RTX 5090, RTX PRO 6000 Blackwell; needs CUDA 12.8 or later).

The archival version, exactly as used for the record runs:

```
nvcc -O3 -arch=native -DPFX=4 -o colorsplit_v2 colorsplit_v2.cu
./colorsplit_v2 27 27 -t 512 -batch 32 -dir DIR
```

Without a GPU, the same source builds as a CPU emulation that runs every thread of a block in turn
(slow; for testing the logic): `g++ -O2 -x c++ -DEMULATE -o colorsplit_emu colorsplit.cu`.

## Run

```
./colorsplit_gpu 4 22                          # n = 4..22, each checked against OEIS A000170
./colorsplit_gpu 28 28 -dir /path/to/folder    # long run; the same command resumes it
```

| Option | Meaning |
|---|---|
| `-t N` | threads per block (default 512) |
| `-b N` | thread blocks (default 8 per SM), always capped to fit 70% of the free GPU memory |
| `-batch N` | chunks per kernel launch (default 32); the checkpoint is saved after each batch. For n ≥ 28 use `-batch 8` or less: there the first batches of 32 chunks would take many hours each |
| `-ckpt 0\|1` | checkpoint off / on (default: on for n ≥ 20) |
| `-dir DIR` | folder for the checkpoint and time files (default: the current folder) |
| `-from A -to B` | only the chunks A to B−1 |
| `-chunks FILE` | only the chunks listed in FILE (numbers separated by spaces or newlines) |
| `-side white\|black\|auto` | color that goes into the table (default auto: the color with more rows); a forced side writes separate files `..._sideW.ckpt` / `..._sideB.ckpt` |

Each finished n prints a line like

```
n=26: 22317699616364044 OK | 48112.232 s this run | total 13.36 h | row sets 2602198 | table 2944657 cells per block
```

and a progress line every 30 seconds goes to the screen. The checkpoint file
`colorsplit_n<n>.ckpt` is rewritten in full after every batch (through a temporary file, flushed
and renamed), so it is always complete on disk, also on a cloud-synced folder. Exit code 3 means
that a table counter could overflow; the batch is not saved and the program stops (not expected
for n ≤ 28).

GPU memory per thread block: table and key list of 8 bytes per cell, about 94 MB at n = 28
(47 MB at n = 27). The number of blocks is reduced automatically when memory is short.

## Step by step on Google Colab

These are the steps used for Q(26) and Q(27) on Colab Pro+ with the G4 runtime (NVIDIA RTX PRO
6000 Blackwell Server Edition).

1. **Runtime.** Runtime → Change runtime type → choose a GPU (G4 is the fastest we used).
2. **Code on Google Drive, once.** Create a folder, for example `MyDrive/colorsplit`, and upload
   `gpu/colorsplit.cu` and the `tools` folder into it (or clone this repository there).
3. **Every session: mount Google Drive** in a code cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. **Every session: compile into `/content`.** Files on Google Drive cannot be executed, so the
   program is built on the local disk of the session:
   ```
   !nvidia-smi
   !nvcc -O3 -arch=native -o /content/colorsplit_gpu /content/drive/MyDrive/colorsplit/colorsplit.cu
   ```
5. **Smoke test** (first time on a GPU, or after changing the code): every line must end with OK.
   ```
   !/content/colorsplit_gpu 4 23 -ckpt 0
   ```
   For reference, on G4 `colorsplit.cu` takes 80 s for n = 23 and the archival version 157 s.
6. **Start the long run** in a code cell, with the checkpoint folder on Drive and a copy of the
   output in a log file (with the driver and CUDA versions of the session):
   ```
   !mkdir -p /content/drive/MyDrive/colorsplit/n28
   !nvidia-smi | tee -a /content/drive/MyDrive/colorsplit/n28/run.log
   !/content/colorsplit_gpu 28 28 -batch 8 -dir /content/drive/MyDrive/colorsplit/n28 2>&1 | tee -a /content/drive/MyDrive/colorsplit/n28/run.log
   ```
7. **Watch it from the Terminal** (the terminal button of the Colab window), not from new code
   cells, which would queue behind the running cell:
   ```
   python3 /content/drive/MyDrive/colorsplit/tools/status.py /content/drive/MyDrive/colorsplit/n28/colorsplit_n28.ckpt --units-per-hour 8.9
   ```
   It shows the chunks done, the pure compute time, the time and compute units left
   (8.9 units per hour was the G4 rate we saw in autumn 2026; check the current one).
8. **After a disconnection** or at the end of a session: open the notebook again and repeat steps
   3, 4 and 6 with the same command. The program skips the chunks already in the checkpoint file;
   at most the batch in progress (32 chunks) is computed again.

Rules that protected our runs:

- Never run two sessions on the same folder at the same time: both would rewrite the same
  checkpoint file.
- Download a copy of the checkpoint file now and then.
- Do not edit the checkpoint file while a run is going.

## Several GPUs

The chunks are independent, so several GPUs (several sessions, accounts or providers) can share a
board. Give each GPU its own list and its own folder. For even n all the work lies in chunks 0 to
2047, so interleave the lists rather than cutting the range into blocks:

```
for i in 0 1 2 3; do seq $i 4 4095 > part$i.txt; done
./colorsplit_gpu 28 28 -chunks part0.txt -dir n28_part0      # on GPU 0, and so on
python3 tools/verify.py merge colorsplit_n28.ckpt n28_part*/colorsplit_n28.ckpt
```

## Checking a new result

The chunk sums do not depend on which color is tabulated, but the code paths do. To check a
finished run independently, recompute a random sample of chunks with the table forced to each
color and compare:

```
python3 tools/verify.py sample colorsplit_n28.ckpt --count 50 --out sample28.txt
./colorsplit_gpu 28 28 -side white -chunks sample28.txt -dir DIR    # writes colorsplit_n28_sideW.ckpt
./colorsplit_gpu 28 28 -side black -chunks sample28.txt -dir DIR    # writes colorsplit_n28_sideB.ckpt
python3 tools/verify.py compare DIR/colorsplit_n28.ckpt DIR/colorsplit_n28_sideW.ckpt
python3 tools/verify.py compare DIR/colorsplit_n28.ckpt DIR/colorsplit_n28_sideB.ckpt
```

With the default side the table holds the color with more rows; forcing each color in turn makes
every sampled row set run through the other code path in one of the two runs. For n = 28, 50
chunks cost about 2.5% of the full run per side. Finally `python3 tools/verify.py total
colorsplit_n28.ckpt` checks that all 4096 chunks are present and that the total is divisible by 4.
