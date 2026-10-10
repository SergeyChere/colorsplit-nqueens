# CPU versions

Two implementations of the same algorithm, with identical checkpoint files and chunk sums:

| | `colorsplit.c` (C, POSIX threads) | `java/ColorSplit.java` (Java 11+) |
|---|---|---|
| Speed | 15–30% faster than Java on large boards | baseline |
| Table | compact (ranks of the key halves) | 2^n cells per thread |
| Memory per thread | n = 25: ~20 MB, n = 27: ~70 MB, n = 28: ~140 MB | 16 · 2^n bytes: n = 25: 512 MB, n = 27: 2 GB |
| Build | `make` (or `gcc -O3 -march=native -pthread`) | none: `java ColorSplit.java ...` |

Both check every result for n ≤ 27 against OEIS A000170 and print `OK` or `MISMATCH`.

## C version

```
make                                   # from the repository root: bin/colorsplit
./bin/colorsplit 1 18                  # n = 1..18
./bin/colorsplit 23 23 -t 16           # n = 23 on 16 threads (default: all CPUs)
```

On AMD Zen 2 and older CPUs build with `make CPU_EXTRA=-DSOFT_PEXT`: these processors execute
the PEXT instruction in slow microcode, and a small lookup table is faster.

Options (after the two board sizes `n_from n_to`):

| Option | Meaning |
|---|---|
| `-t N` | worker threads (default: number of online CPUs) |
| `-p S` | progress line on stderr every S seconds, 0 = off (default 60) |
| `-ckpt 0\|1` | checkpoint file off/on (default: on for n ≥ 20); a rerun resumes |
| `-from A`, `-to B` | only the chunks A to B−1 of the 4096 |
| `-part I/K` | only the chunks c with c mod K = I, for K machines |
| `-status` | compute nothing; report what the checkpoint file holds and the total if complete |
| `-side white\|black\|auto` | color whose placements go into the table (default auto) |
| `-daemon` | run in the background: output to `colorsplit_n<from>-<to>.log`, PID to `colorsplit.pid` |

### A long run on a server

```
./bin/colorsplit 25 25 -daemon         # starts in the background and returns
tail -f colorsplit_n25-25.log          # progress (Ctrl+C stops watching, not the run)
kill $(cat colorsplit.pid)             # stops cleanly; the same command resumes later
```

A finished chunk is appended to `colorsplit_n25.ckpt` at once (the file is opened, written,
synced and closed for every line), so a crash or a stop loses only the chunks in progress.
`colorsplit_n25.time` keeps the wall-clock time over all sessions. On a fresh Ubuntu or Debian
server, [`../tools/run_cpu_server.sh`](../tools/run_cpu_server.sh) installs the compiler, builds
and runs a list of board sizes one after another.

### Several machines

Give each machine one part and its own working directory:

```
./bin/colorsplit 27 27 -part 0/4       # machine 0; parts 1/4, 2/4, 3/4 on the others
```

Each writes `colorsplit_n27_part<I>of4.ckpt`. Collect the files in one directory and merge them:

```
python3 tools/verify.py merge colorsplit_n27.ckpt colorsplit_n27_part*of4.ckpt
```

(or `cat colorsplit_n27_part*of4.ckpt > colorsplit_n27.ckpt` followed by
`./bin/colorsplit 27 27 -status`). Parts split by `-part` interleave the chunks, which matters for
even n: all the work then lies in chunks 0 to 2047, and a split by `-from`/`-to` ranges would give
some machines nothing to do.

## Java version

```
java -Xmx2g cpu/java/ColorSplit.java 1 18
java -Xmx8g -Dthreads=8 cpu/java/ColorSplit.java 23 23
```

Options are Java system properties and must come before the file name: `-Dthreads=N`,
`-Dckpt=true|false`, `-Dfrom=A`, `-Dto=B`, `-Dpart=I/K`, `-Dstatus=true`, `-Dprogress=S`,
`-Dside=white|black`, `-Dnotr=true` (no transposition) and `-Dcenter=false` (rows in natural
order); see the comment at the top of the file. The number of threads is reduced when their tables
do not fit in 85% of the heap given by `-Xmx`.

## Output

```
n=18: 666090624 OK | 4.321 s (0.1 min)
```

With a checkpoint file a second line gives the wall-clock time over all sessions and the sum of
the chunk times (thread time, which equals vCPU time when the number of threads equals the number
of vCPUs). The totals are signed 64-bit integers, enough up to n = 28; for larger n add the chunk
sums of the checkpoint file with `tools/verify.py`, which uses exact integers.

## Verification builds

Compile-time switches turn single reductions off or add checks (`tests/check_reductions.sh` uses
them): `-DNO_EARLY_CUT`, `-DNO_MIRROR` (both give the same chunk sums, only slower),
`-DCHECK_CELLS` (compares every table cell with a plain enumeration; small n, one thread) and
`-DCELL_STATS` (prints the largest table cell of each n).
