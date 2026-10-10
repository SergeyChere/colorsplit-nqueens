// SPDX-License-Identifier: MIT
/*
 * ColorSplit — exact counting of n-queens solutions, GPU version (CUDA), version 2 (ARCHIVAL)
 *
 * ARCHIVAL COPY. This is the exact code that was used for the runs Q(25), Q(26) and Q(27) in the paper:
 * static round-robin distribution of the search-tree prefixes among the threads of a block. The robust
 * checkpoint writing (the file is rewritten, flushed and closed after every batch) was added during the
 * Q(27) run; the counting code is the same for all three runs. In this copy (original file name
 * nqueens_v2fix.cu) only the comments and the program messages were translated to English; the code and
 * its layout are unchanged.
 *
 * The record runs were built with
 *     nvcc -O3 -arch=native -DPFX=4 -o colorsplit_v2 colorsplit_v2.cu
 * and run with the options  -t 512 -batch 32  (the defaults below are 128 and 64; PFX defaults to 3).
 *
 * This version has no -side and no -chunks option and no overflow guard on the 32-bit table counters.
 * For new computations use colorsplit.cu, which computes the same per-chunk sums, writes the same
 * checkpoint files (with its default -side auto) and adds those features.
 *
 * WHAT IT COMPUTES
 *   Q(n), the number of placements of n non-attacking queens on an n x n board (OEIS A000170), exactly.
 *   Squares with r + c even are white, the others black. Every diagonal has a single color, so white and
 *   black queens interact only through rows and columns: Q(n) = sum over R, C of W_R(C) * B_~R(~C), where
 *   W_R(C) counts the white placements on the row set R with column set C and B the black placements on
 *   the remaining rows and columns. Each GPU block takes a row set R, builds the table of one color
 *   (compact numbering of the keys), applies the transposition filter, marks the key halves, then
 *   enumerates the other color and looks up the matching entries. Within a block the work is divided
 *   among the threads by the prefixes of the search tree.
 *   Implemented: parity law, symmetries of row sets, transposition (weights λ), rows from the center,
 *   mirror of the first row for odd n (plus nested mirror), compact table, pruning by half keys, choice of
 *   the color that goes into the table (the color with more rows).
 *   History: based on nqueens_gpu.cu (prototype). Version 2 added checkpointing per chunk (file compatible
 *   with the CPU versions, so they can work on the same file), accumulated time over all runs, the choice of
 *   the table color and the nested mirror (odd n).
 *
 * BUILD
 *   GPU:                 nvcc -O3 -arch=native -DPFX=4 -o colorsplit_v2 colorsplit_v2.cu
 *   CPU check (no GPU):  g++ -O2 -x c++ -DEMULATE -DPFX=4 -o colorsplit_v2_emu colorsplit_v2.cu
 *
 * USAGE
 *   ./colorsplit_v2 10 24 -t 512 -batch 32
 *   Options:  -t N         threads per block (default 128)
 *             -b N         number of blocks (default: 8 per SM, capped to fit 70% of free GPU memory)
 *             -ckpt 0|1    save progress (default 1 for n >= 20); a rerun continues where it stopped
 *             -dir DIR     where to keep the progress files (in Colab: a folder on Google Drive)
 *             -from a -to b   only chunks a..b-1 of the 4096
 *             -batch N     chunks per kernel launch (default 64)
 *
 * FILES (in -dir)
 *   colorsplit_n<n>.ckpt   one line per finished chunk:  <chunk> <sum> <milliseconds> ok
 *   colorsplit_n<n>.time   <seconds> <sessions>  (accumulated wall-clock time, number of runs)
 *
 * Correct for n <= 31 (the printed grand total is a 64-bit sum, enough for n <= 28).
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <unistd.h>
typedef uint32_t u32; typedef uint64_t u64; typedef unsigned long long ull;

#ifdef EMULATE
  #define __global__
  #define __device__
  #define __host__
  #define __forceinline__ inline
  static inline u32 atomicAdd(u32 *p, u32 v) { u32 o = *p; *p = o + v; return o; }
  static inline int atomicAdd(int *p, int v) { int o = *p; *p = o + v; return o; }
  static inline ull atomicAdd(ull *p, ull v) { ull o = *p; *p = o + v; return o; }
  static inline u32 atomicOr(u32 *p, u32 v) { u32 o = *p; *p = o | v; return o; }
  static inline u32 __brev(u32 x) { x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1); x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
                                    x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4); return __builtin_bswap32(x); }
  static inline int __popc(u32 x) { return __builtin_popcount(x); }
  static inline int __ffs(u32 x) { return __builtin_ffs(x); }
#endif

static const ull REF[] = {1ULL, 1ULL, 0ULL, 0ULL, 2ULL, 10ULL, 4ULL, 40ULL, 92ULL, 352ULL, 724ULL, 2680ULL, 14200ULL, 73712ULL, 365596ULL, 2279184ULL,
    14772512ULL, 95815104ULL, 666090624ULL, 4968057848ULL, 39029188884ULL, 314666222712ULL, 2691008701644ULL, 24233937684440ULL,
    227514171973736ULL, 2207893435808352ULL, 22317699616364044ULL, 234907967154122528ULL};
#define NREF 28
#ifndef PFX
#define PFX 3          /* prefix depth by which the work is divided among the threads of a block; build with -DPFX=4 */
#endif
#define SENT 0xFFFFFFFFu

struct G {             /* board parameters */
    int n, h, even, oddBM, nE, nO;
    u32 full, oddCols, evenCols, white[32], black[32];
    int ORD[32];
    u32 EVB[256], ODB[256], binomNO[17];
};
#ifdef EMULATE
static G cg;
#else
__constant__ G cg;
#endif

struct Ctx {           /* state shared by the threads of a block (shared memory) */
    u32 R; int w, swap, cidx; u64 IR, IRfirst; u32 mulO;
    int nrB, bsplit, nrP, split; int rowsB[32], rowsP[32];
    int keyCount, alive; ull acc;
    u32 stamp[1025];   /* key-half marks: 2^(n/2) bits */
};

__device__ __forceinline__ u32 rev(u32 x) { return __brev(x) >> (32 - cg.n); }
__device__ __forceinline__ int ctz(u32 x) { return __ffs(x) - 1; }
__device__ __forceinline__ u64 umin(u64 a, u64 b) { return a < b ? a : b; }
__device__ __forceinline__ u64 umax(u64 a, u64 b) { return a > b ? a : b; }
__device__ __forceinline__ u64 hOdd(u32 s) { u32 t = cg.oddCols ^ s; return s < t ? s : t; }
__device__ u64 inv(u32 S) {
    if (cg.even) { u64 u = hOdd(S & cg.oddCols), v = hOdd(rev(S & cg.evenCols)); return umin(u, v) << cg.n | umax(u, v); }
    u64 a = ((u64)(S & cg.oddCols) << cg.n) | (S & cg.evenCols); u32 rs = rev(S); u64 b = ((u64)(rs & cg.oddCols) << cg.n) | (rs & cg.evenCols);
    return umin(a, b);
}
__device__ __forceinline__ u64 partBuild(u32 cols) {
    if (cg.even) return hOdd(rev(cols & cg.evenCols));
    u32 a = cols & cg.oddCols, b = rev(a); return a < b ? a : b;
}
__device__ __forceinline__ u32 cE(u32 x) { return cg.EVB[x & 255] | cg.EVB[(x >> 8) & 255] << 4 | cg.EVB[(x >> 16) & 255] << 8 | cg.EVB[(x >> 24) & 255] << 12; }
__device__ __forceinline__ u32 cO(u32 x) { return cg.ODB[x & 255] | cg.ODB[(x >> 8) & 255] << 4 | cg.ODB[(x >> 16) & 255] << 8 | cg.ODB[(x >> 24) & 255] << 12; }
__device__ __forceinline__ u32 idxOf(u32 k, u32 mulO, const u32 *rankTab) { return rankTab[cE(k)] * mulO + rankTab[cO(k)]; }
__device__ __forceinline__ u32 keyOf(u32 cols) { return cg.even ? cg.full & ~rev(cols) : cg.full & ~cols; }
__device__ __forceinline__ u64 pbBlack(u32 cols) { u32 cOdd = cg.oddCols & ~(cols & cg.oddCols); if (cg.even) return hOdd(cOdd); u32 r = rev(cOdd) & cg.oddCols; return cOdd < r ? cOdd : r; }

/* ---------- steps for processing one row set R ---------- */
__device__ void setupR(Ctx &s, u32 R, int w, int cidx) {
    s.R = R; s.w = w; s.cidx = cidx; s.IR = inv(R); s.IRfirst = s.IR >> cg.n;
    int k = __popc(R); s.swap = (cg.n - k > k);                              /* the color with more rows goes into the table */
    u32 comp = cg.full ^ R;
    if (!s.swap) {                                                            /* white queens into the table (rows R), black queens look up pairs */
        s.mulO = cg.binomNO[__popc(R & cg.oddCols)];
        int f = cg.even ? 0 : 1; s.nrB = 0;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == f && ((R >> r) & 1)) s.rowsB[s.nrB++] = r; }
        s.bsplit = s.nrB;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) != f && ((R >> r) & 1)) s.rowsB[s.nrB++] = r; }
        s.nrP = 0;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 0 && ((comp >> r) & 1)) s.rowsP[s.nrP++] = r; }
        s.split = s.nrP;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 1 && ((comp >> r) & 1)) s.rowsP[s.nrP++] = r; }
    } else {                                                                  /* black queens into the table (other rows), white queens look up pairs */
        s.mulO = cg.binomNO[__popc(comp & cg.evenCols)];                      /* black queens on even rows stand on odd columns */
        s.nrB = 0;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 0 && ((comp >> r) & 1)) s.rowsB[s.nrB++] = r; }
        s.bsplit = s.nrB;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 1 && ((comp >> r) & 1)) s.rowsB[s.nrB++] = r; }
        s.nrP = 0;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 1 && ((R >> r) & 1)) s.rowsP[s.nrP++] = r; }
        s.split = s.nrP;
        for (int j = 0; j < cg.n; j++) { int r = cg.ORD[j]; if ((r & 1) == 0 && ((R >> r) & 1)) s.rowsP[s.nrP++] = r; }
    }
    s.keyCount = 0; s.alive = 0; s.acc = 0;
}
__device__ __forceinline__ void insertKey(Ctx &s, u32 *tab, u32 *keys, const u32 *rankTab, u32 key, u32 inc) {
    u32 k = key; if (cg.oddBM) { u32 rk = rev(key); if (rk < k) k = rk; }
    u32 i = idxOf(k, s.mulO, rankTab);
    u32 old = atomicAdd(&tab[i], inc);
    if (old == 0) { int pos = atomicAdd(&s.keyCount, 1); keys[pos] = k; }
}
/* white queens on rows R -> table (black queens on the other rows when swapped); threads split the branches by the number of the depth-P prefix */
__device__ void buildR(Ctx &s, u32 *tab, u32 *keys, const u32 *rankTab, int tid, int nth) {
    int nr = s.nrB;
    if (nr == 0) { if (tid == 0) insertKey(s, tab, keys, rankTab, 0, 1); return; }
    const int n = cg.n; int P = nr < PFX ? nr : PFX;
    const u32 *cm = s.swap ? cg.black : cg.white;
    u32 av[32], C[32]; u64 E1[32], D2[32]; u32 inc = 1; ull cnt = 0; int center = 0;
    int r0 = s.rowsB[0]; av[0] = cm[r0]; if (cg.oddBM) av[0] &= (2u << cg.h) - 1;          /* odd n: left half and center */
    C[0] = 0; E1[0] = 0; D2[0] = 0; int d = 0;
    while (d >= 0) {
        if (av[d] == 0) { d--; continue; }
        u32 b = av[d] & (0u - av[d]); av[d] ^= b; int c = ctz(b); int r = s.rowsB[d];
        u32 nc = C[d] | b; u64 ne = E1[d] | 1ULL << (c - r + n - 1), nd = D2[d] | 1ULL << (r + c);
        if (d == 0) { center = cg.oddBM && c == cg.h; inc = (cg.oddBM && c < cg.h) ? 2 : 1; }
        int d1 = d + 1;
        if (d1 == P) { cnt++; if ((int)((cnt - 1) % (ull)nth) != tid) continue; }
        if (d1 == nr) { insertKey(s, tab, keys, rankTab, nc, inc); continue; }
        if (d1 == s.bsplit && (s.swap ? pbBlack(nc) : partBuild(nc)) < s.IRfirst) continue;     /* transposition pruning */
        int rr = s.rowsB[d1]; C[d1] = nc; E1[d1] = ne; D2[d1] = nd;
        av[d1] = cm[rr] & ~nc & ~(u32)((ne >> (n - 1 - rr)) | (nd >> rr));
        if (d1 == 1 && center) { av[1] &= (1u << cg.h) - 1; inc = 2; }                         /* nested mirror */
        d = d1;
    }
}
__device__ void filterR(Ctx &s, u32 *tab, u32 *keys, const u32 *rankTab, int tid, int nth) {
    for (int t = tid; t < s.keyCount; t += nth) {
        u32 k = keys[t]; u64 ic = inv(s.swap ? (cg.full & ~k) : k); u32 i = idxOf(k, s.mulO, rankTab);
        if (ic < s.IR) { tab[i] = 0; keys[t] = SENT; } else if (ic > s.IR) tab[i] *= 2;
    }
}
__device__ void stampR(Ctx &s, u32 *keys, int tid, int nth) {
    for (int t = tid; t < s.keyCount; t += nth) {
        u32 k = keys[t]; if (k == SENT) continue;
        u32 a = s.swap ? cO(cg.oddCols & ~k) : cO(k); atomicOr(&s.stamp[a >> 5], 1u << (a & 31));
        if (cg.oddBM) { u32 b = s.swap ? cO(cg.oddCols & ~rev(k)) : cO(rev(k)); atomicOr(&s.stamp[b >> 5], 1u << (b & 31)); }
        s.alive = 1;
    }
}
__device__ __forceinline__ ull getW(Ctx &s, const u32 *tab, const u32 *rankTab, u32 key) {
    if (!cg.oddBM) return tab[idxOf(key, s.mulO, rankTab)];
    u32 rk = rev(key), k = rk < key ? rk : key; u32 v = tab[idxOf(k, s.mulO, rankTab)];
    return key == rk ? v : (v >> 1);
}
/* black queens on the other rows -> pair lookup (white queens on R when swapped); for even n they are enumerated mirrored, on white squares */
__device__ void probeR(Ctx &s, const u32 *tab, const u32 *rankTab, int tid, int nth) {
    if (!s.alive) return;
    int nr = s.nrP; const int n = cg.n;
    if (nr == 0) { if (tid == 0) atomicAdd(&s.acc, getW(s, tab, rankTab, s.swap ? cg.full : keyOf(0))); return; }
    const u32 *cm = s.swap ? cg.white : (cg.even ? cg.white : cg.black);
    int P = nr < PFX ? nr : PFX;
    u32 av[32], C[32]; u64 E1[32], D2[32]; ull wcur = 1, local = 0, cnt = 0; int center = 0;
    int r0 = s.rowsP[0]; av[0] = cm[r0]; if (cg.oddBM) av[0] &= (2u << cg.h) - 1;
    C[0] = 0; E1[0] = 0; D2[0] = 0; int d = 0;
    while (d >= 0) {
        if (av[d] == 0) { d--; continue; }
        u32 b = av[d] & (0u - av[d]); av[d] ^= b; int c = ctz(b); int r = s.rowsP[d];
        u32 nc = C[d] | b; u64 ne = E1[d] | 1ULL << (c - r + n - 1), nd = D2[d] | 1ULL << (r + c);
        if (d == 0) { center = cg.oddBM && c == cg.h; wcur = (cg.oddBM && c < cg.h) ? 2 : 1; }
        int d1 = d + 1;
        if (d1 == P) { cnt++; if ((int)((cnt - 1) % (ull)nth) != tid) continue; }
        if (d1 == s.split && d1 < nr) { u32 a = s.swap ? cO(nc) : cO(cg.oddCols & ~(cg.even ? rev(nc) : nc));
            if (!((s.stamp[a >> 5] >> (a & 31)) & 1)) continue; }                              /* half-key pruning */
        if (d1 == nr) { local += wcur * getW(s, tab, rankTab, s.swap ? (cg.full & ~nc) : keyOf(nc)); continue; }
        int rr = s.rowsP[d1]; C[d1] = nc; E1[d1] = ne; D2[d1] = nd;
        av[d1] = cm[rr] & ~nc & ~(u32)((ne >> (n - 1 - rr)) | (nd >> rr));
        if (d1 == 1 && center) { av[1] &= (1u << cg.h) - 1; wcur = 2; }                        /* nested mirror */
        d = d1;
    }
    if (local) atomicAdd(&s.acc, local);
}
__device__ void clearR(Ctx &s, u32 *tab, u32 *keys, const u32 *rankTab, int tid, int nth) {
    for (int t = tid; t < s.keyCount; t += nth) { u32 k = keys[t]; if (k != SENT) tab[idxOf(k, s.mulO, rankTab)] = 0; }
    int words = ((1 << cg.nO) >> 5) + 1;
    for (int t = tid; t < words; t += nth) s.stamp[t] = 0;
}

#ifndef EMULATE
__global__ void kern(const u32 *Rs, const unsigned char *Ws, const int *Cs, int nR, u32 *tabAll, u32 *keysAll, size_t S, const u32 *rankTab, ull *total) {
    __shared__ Ctx s; int tid = threadIdx.x, nth = blockDim.x;
    u32 *tab = tabAll + (size_t)blockIdx.x * S, *keys = keysAll + (size_t)blockIdx.x * S;
    for (int t = tid; t < 1025; t += nth) s.stamp[t] = 0;
    __syncthreads();
    for (int j = blockIdx.x; j < nR; j += gridDim.x) {
        if (tid == 0) setupR(s, Rs[j], Ws[j], Cs[j]);
        __syncthreads(); buildR(s, tab, keys, rankTab, tid, nth);
        __syncthreads(); filterR(s, tab, keys, rankTab, tid, nth);
        __syncthreads(); stampR(s, keys, tid, nth);
        __syncthreads(); probeR(s, tab, rankTab, tid, nth);
        __syncthreads(); if (tid == 0 && s.acc) atomicAdd(&total[s.cidx], (ull)s.w * s.acc);
        clearR(s, tab, keys, rankTab, tid, nth);
        __syncthreads();
    }
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#endif

/* ---------- host-side preparation ---------- */
static G hg;
static u32 hrev(u32 x) { u32 r = 0; for (int i = 0; i < hg.n; i++) if ((x >> i) & 1) r |= 1u << (hg.n - 1 - i); return r; }
static void initBoard(int n) {
    memset(&hg, 0, sizeof hg); hg.n = n; hg.h = n / 2; hg.even = (n & 1) == 0; hg.oddBM = !hg.even; hg.nE = (n + 1) / 2; hg.nO = n / 2;
    hg.full = n == 32 ? 0xFFFFFFFFu : ((1u << n) - 1);
    for (int c = 1; c < n; c += 2) hg.oddCols |= 1u << c; hg.evenCols = hg.full ^ hg.oddCols;
    for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) { if (((r + c) & 1) == 0) hg.white[r] |= 1u << c; else hg.black[r] |= 1u << c; }
    for (int r = 0; r < n; r++) hg.ORD[r] = r;
    double mid = (n - 1) / 2.0;
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) {
        double da = hg.ORD[a] > mid ? hg.ORD[a] - mid : mid - hg.ORD[a], db = hg.ORD[b] > mid ? hg.ORD[b] - mid : mid - hg.ORD[b];
        if (db < da || (db == da && hg.ORD[b] < hg.ORD[a])) { int t = hg.ORD[a]; hg.ORD[a] = hg.ORD[b]; hg.ORD[b] = t; } }
    for (int x = 0; x < 256; x++) { u32 e = 0, o = 0; for (int j = 0; j < 4; j++) { e |= ((x >> (2 * j)) & 1u) << j; o |= ((x >> (2 * j + 1)) & 1u) << j; } hg.EVB[x] = e; hg.ODB[x] = o; }
    for (int k = 0; k <= 16; k++) { u64 b = 1; if (k > hg.nO) b = 0; else for (int i = 0; i < k; i++) b = b * (hg.nO - i) / (i + 1); hg.binomNO[k] = (u32)b; }
}
static u64 binom(int a, int b) { if (b < 0 || b > a) return 0; u64 r = 1; for (int i = 0; i < b; i++) r = r * (a - i) / (i + 1); return r; }

static double nowSec() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void stampNow(char *out, size_t sz) { time_t t = time(NULL); struct tm tmv; localtime_r(&t, &tmv); strftime(out, sz, "%H:%M:%S", &tmv); }
static void fmtDur(double s, char *out, size_t sz) {
    if (s < 3600) snprintf(out, sz, "%.1f min", s / 60); else if (s < 172800) snprintf(out, sz, "%.2f h", s / 3600); else snprintf(out, sz, "%.2f days", s / 86400); }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s n_from n_to [-t threads] [-b blocks] [-ckpt 0|1] [-dir DIR] [-from a] [-to b] [-batch N]\n", argv[0]); return 1; }
    int n1 = atoi(argv[1]), n2 = atoi(argv[2]), nth = 128, blocksArg = 0, ckOpt = -1, from = 0, to = -1, batchArg = 64; const char *dir = ".";
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) nth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) blocksArg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ckpt") && i + 1 < argc) ckOpt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-dir") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "-from") && i + 1 < argc) from = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-to") && i + 1 < argc) to = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-batch") && i + 1 < argc) batchArg = atoi(argv[++i]);
    }
    static u32 rankTab[1 << 16]; { u32 cnt[17] = {0}; for (u32 m = 0; m < (1u << 16); m++) rankTab[m] = cnt[__builtin_popcount(m)]++; }
#ifndef EMULATE
    int dev = 0; cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s, %d SMs, %.1f GB\n", prop.name, prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);
    u32 *dRank; CK(cudaMalloc(&dRank, sizeof rankTab)); CK(cudaMemcpy(dRank, rankTab, sizeof rankTab, cudaMemcpyHostToDevice));
#else
    printf("CPU emulation mode (EMULATE), threads per block: %d\n", nth);
#endif
    for (int n = n1; n <= n2; n++) {
        if (n == 1) { printf("n= 1: 1 OK\n"); continue; }
        if (n > 31) { printf("n > 31 is not supported\n"); break; }
        initBoard(n);
        int chunkBits = n > 12 ? n - 12 : 0; int nChunks = (int)((1ULL << n) >> chunkBits);
        std::vector<u32> Rs; std::vector<unsigned char> Ws; std::vector<int> off(nChunks + 1, 0);
        u32 evenRows = hg.evenCols;
        for (u64 R = 0; R <= hg.full; R++) {
            int re = __builtin_popcount((u32)R & evenRows), ro = __builtin_popcount((u32)R & ~evenRows & hg.full);
            if (re != ro + (hg.even ? 0 : 1)) continue;
            u32 r = (u32)R, comp = hg.full ^ r, fl = hrev(r), cf = hg.full ^ fl; int w;
            if (hg.even) { if (r > comp || r > fl || r > cf) continue; w = 1 + (comp != r) + (fl != r && fl != comp) + (cf != r && cf != comp && cf != fl); }
            else { if (r > fl) continue; w = fl == r ? 1 : 2; }
            Rs.push_back(r); Ws.push_back((unsigned char)w); off[(r >> chunkBits) + 1]++;
        }
        for (int c = 0; c < nChunks; c++) off[c + 1] += off[c];
        size_t S = (size_t)(binom(hg.nE, hg.nE / 2) * binom(hg.nO, hg.nO / 2)) + 1;
        /* checkpoint file (same format as the CPU versions) and time file */
        int useCk = ckOpt < 0 ? (n >= 20) : ckOpt;
        char fname[512], tname[512]; snprintf(fname, sizeof fname, "%s/colorsplit_n%d.ckpt", dir, n); snprintf(tname, sizeof tname, "%s/colorsplit_n%d.time", dir, n);
        std::vector<char> doneF(nChunks, 0); std::vector<ull> doneV(nChunks, 0); std::vector<long> doneMsLoad(nChunks, 0); int loaded = 0;
        if (useCk) { FILE *in = fopen(fname, "r");
            if (in) { char line[256]; while (fgets(line, sizeof line, in)) { int c; long long v; long ms; char okw[8] = "";
                    int m = sscanf(line, "%d %lld %ld %7s", &c, &v, &ms, okw);
                    if (m == 4 && !strcmp(okw, "ok") && c >= 0 && c < nChunks && !doneF[c]) { doneF[c] = 1; doneV[c] = (ull)v; doneMsLoad[c] = ms; loaded++; } }
                fclose(in); if (loaded) printf("  resuming: %d chunks already done (%s)\n", loaded, fname); } }
        double prevWall = 0; int prevSess = 0;
        if (useCk) { FILE *tf = fopen(tname, "r"); if (tf) { if (fscanf(tf, "%lf %d", &prevWall, &prevSess) != 2) { prevWall = 0; prevSess = 0; } fclose(tf); } }
        int hiC = (to < 0 || to > nChunks) ? nChunks : to;
        std::vector<int> todo; int mine = 0; ull base = 0;
        for (int c = from; c < hiC; c++) { mine++; if (doneF[c]) base += doneV[c]; else todo.push_back(c); }
        std::vector<long> doneMsV(doneMsLoad);
        double t0 = nowSec(), lastPrint = t0; ull sum = base; int finished = mine - (int)todo.size();
#ifndef EMULATE
        CK(cudaMemcpyToSymbol(cg, &hg, sizeof hg));
        size_t freeB, totB; CK(cudaMemGetInfo(&freeB, &totB));
        int blocks = blocksArg > 0 ? blocksArg : prop.multiProcessorCount * 8;
        size_t maxByMem = (size_t)((freeB * 0.7) / (2 * 4 * S)); if ((size_t)blocks > maxByMem) blocks = (int)maxByMem; if (blocks < 1) blocks = 1;
        size_t maxBatchR = 0; for (size_t i = 0; i < todo.size(); i += batchArg) { size_t cnt = 0; for (size_t j = i; j < i + batchArg && j < todo.size(); j++) cnt += off[todo[j] + 1] - off[todo[j]]; if (cnt > maxBatchR) maxBatchR = cnt; }
        if (maxBatchR == 0) maxBatchR = 1;
        u32 *dR, *dTab, *dKeys; unsigned char *dW; int *dC; ull *dTot;
        CK(cudaMalloc(&dR, maxBatchR * 4)); CK(cudaMalloc(&dW, maxBatchR)); CK(cudaMalloc(&dC, maxBatchR * 4)); CK(cudaMalloc(&dTot, (size_t)batchArg * 8));
        CK(cudaMalloc(&dTab, (size_t)blocks * S * 4)); CK(cudaMalloc(&dKeys, (size_t)blocks * S * 4)); CK(cudaMemset(dTab, 0, (size_t)blocks * S * 4));
#else
        cg = hg; std::vector<u32> tab(S, 0), keys(S, 0); Ctx *s = new Ctx(); memset(s->stamp, 0, sizeof s->stamp);
#endif
        for (size_t i = 0; i < todo.size(); i += batchArg) {
            int nb = (int)std::min((size_t)batchArg, todo.size() - i);
            std::vector<u32> bR; std::vector<unsigned char> bW; std::vector<int> bC; std::vector<ull> tot(nb, 0);
            for (int q = 0; q < nb; q++) { int c = todo[i + q]; for (int k = off[c]; k < off[c + 1]; k++) { bR.push_back(Rs[k]); bW.push_back(Ws[k]); bC.push_back(q); } }
            double tb = nowSec();
            if (!bR.empty()) {
#ifndef EMULATE
                CK(cudaMemcpy(dR, bR.data(), bR.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dW, bW.data(), bW.size(), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dC, bC.data(), bC.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(dTot, 0, (size_t)nb * 8));
                int g = blocks < (int)bR.size() ? blocks : (int)bR.size();
                kern<<<g, nth>>>(dR, dW, dC, (int)bR.size(), dTab, dKeys, S, dRank, dTot);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(tot.data(), dTot, (size_t)nb * 8, cudaMemcpyDeviceToHost));
#else
                for (size_t j = 0; j < bR.size(); j++) {
                    setupR(*s, bR[j], bW[j], bC[j]);
                    for (int t = 0; t < nth; t++) buildR(*s, tab.data(), keys.data(), rankTab, t, nth);
                    for (int t = 0; t < nth; t++) filterR(*s, tab.data(), keys.data(), rankTab, t, nth);
                    for (int t = 0; t < nth; t++) stampR(*s, keys.data(), t, nth);
                    for (int t = 0; t < nth; t++) probeR(*s, tab.data(), rankTab, t, nth);
                    if (s->acc) tot[s->cidx] += (ull)s->w * s->acc;
                    for (int t = 0; t < nth; t++) clearR(*s, tab.data(), keys.data(), rankTab, t, nth);
                }
#endif
            }
            long msEach = (long)((nowSec() - tb) * 1000 / nb);
            for (int q = 0; q < nb; q++) { int c = todo[i + q]; doneF[c] = 1; doneV[c] = tot[q]; doneMsV[c] = msEach; sum += tot[q]; }
            if (useCk) {   /* the checkpoint file is rewritten completely and closed after each batch (Google Drive saves a file when it is closed) */
                char tmp[520]; snprintf(tmp, sizeof tmp, "%s.tmp", fname); FILE *cf = fopen(tmp, "w");
                if (cf) { for (int c = 0; c < nChunks; c++) if (doneF[c]) fprintf(cf, "%d %llu %ld ok\n", c, doneV[c], doneMsV[c]);
                    fflush(cf); fsync(fileno(cf)); fclose(cf); rename(tmp, fname); }
                else fprintf(stderr, "  WARNING: could not write %s\n", tmp);
            }
            finished += nb;
            double el = nowSec() - t0;
            if (useCk) { char tmp[520]; snprintf(tmp, sizeof tmp, "%s.tmp", tname); FILE *tf = fopen(tmp, "w"); if (tf) { fprintf(tf, "%.3f %d\n", prevWall + el, prevSess + 1); fclose(tf); rename(tmp, tname); } }
            if (nowSec() - lastPrint >= 30 || i + nb >= todo.size()) { lastPrint = nowSec(); char ts[16], d1[32], d2[32]; stampNow(ts, sizeof ts); fmtDur(el, d1, sizeof d1); fmtDur(prevWall + el, d2, sizeof d2);
                printf("  [%s] n=%d | chunks %d/%d | this run %s, total %s | sum %llu", ts, n, finished, mine, d1, d2, sum);
                if (n < NREF && REF[n]) printf(" (%.1f%% of the known value)", 100.0 * (double)sum / (double)REF[n]);
                printf("\n"); fflush(stdout); }
        }
#ifndef EMULATE
        CK(cudaFree(dR)); CK(cudaFree(dW)); CK(cudaFree(dC)); CK(cudaFree(dTot)); CK(cudaFree(dTab)); CK(cudaFree(dKeys));
#else
        delete s;
#endif
        double sec = nowSec() - t0;
        int all = 1; ull total = 0; for (int c = 0; c < nChunks; c++) { if (!doneF[c]) all = 0; total += doneV[c]; }
        if (!all) { int nd = 0; for (int c = 0; c < nChunks; c++) nd += doneF[c];
            printf("  chunks %d..%d: sum %llu | chunks done: %d of %d; run again to finish\n", from, hiC - 1, sum, nd, nChunks); continue; }
        const char *chk = n < NREF ? (total == REF[n] ? "OK" : "MISMATCH") : "(no reference value)";
        char d1[32]; fmtDur(prevWall + (todo.empty() ? 0 : sec), d1, sizeof d1);
        printf("n=%2d: %llu %s | %.3f s this run | total %s | row sets %zu | table %zu cells per block\n", n, total, chk, sec, d1, Rs.size(), S);
        fflush(stdout);
    }
    return 0;
}
