# How mantissa gets its speed

mantissa is a from-scratch C engine for low-precision neural networks. This is
the short tour of *why* it is fast — the handful of decisions that matter, each
with a measured number and an honest counter-story where there is one. The
exhaustive design log (every rejected alternative, every ULP envelope) lives in
[DESIGN.md](DESIGN.md); this page is the narrative.

The one-line thesis: **narrow storage, wide accumulate — and treat memory
bandwidth as the budget.**

## 1. The budget is memory, not FLOPs

A dense layer's forward pass (a GEMV) reads the whole weight matrix once and does
~2 FLOPs per weight. Once the matrix spills out of cache it is **memory-bandwidth
bound** — so halving the *stored* size of a weight is worth nearly as much as
halving the runtime, while every accumulator stays float32 so the error of a sum
over millions of terms stays bounded (the mixed-precision recipe, Micikevicius et
al. 2017). Storing narrow and accumulating wide is the whole engine in one
sentence.

**The honest counter-story:** narrow storage only becomes speed when the *read*
is cheap. bf16 is literally the top 16 bits of a float32, so its read is a single
shift and it edges out float32. `tekin8` (FP8 E4M3) is the cautionary case — a
1-byte format that is *still slower* than 4-byte float32, at roughly **half its
GEMV throughput in the same run** (M4, 2048², 28.8 vs 59.2 GFLOP/s). The reason
has moved once already:

- It began as a **conversion** problem. E4M3's unpack was a subnormal *branch*,
  which forced the compiler into a compare/select/lane-extract chain and left
  the conversion dominating the arithmetic. E4M3 reserves no inf/nan encoding
  here, so that read is now branch-free — place the exp:mantissa field where a
  float32 significand goes, multiply by 2^(254−bias), done. Worth **1.5–2.3×**
  across GEMV, batch GEMM and the dense backward (paired A/B, bit-identical over
  all 256 encodings).
- It is now a **SIMD** problem. float32, bf16 and fp16 each have a hand-written
  NEON kernel in `tk__dot4`; tekin8 does not, so it runs the portable loop. That,
  not the conversion, is what remains of the gap.

The lesson survives its own fix: byte count only pays if the hot path can *read*
those bytes cheaply — which is why the next two sections exist.

## 2. The dot-product core

`tk_dot` (`src/ops.c`) reads narrow inputs into **four independent float32
accumulators**. One accumulator would serialize on the FP adder's latency (a
loop-carried dependency); four keep several FMAs in flight for the out-of-order
engine. With `-ffp-contract=fast`, `restrict`, and `-O3`, the compiler emits
vectorized FMA.

**Register blocking.** `tk_linear_forward` doesn't call `tk_dot` per row — it
computes **four output rows at once** (`tk__dot4`). Each input element is loaded
and converted *once* and feeds four FMA chains, so the shared input is reused
instead of re-read per row. ~**1.3×** across every dtype, for free.

## 3. Cheap narrow reads: hot path vs cold path

- **narrow → float32 (hot)** runs for every weight of every forward pass →
  branchless `static inline` bit-ops in `dtypes.h` that inline into the kernel
  and vectorize.
- **float32 → narrow (cold)** runs once at load/quantize time → clear
  `frexpf`/`lroundf` code in `dtypes.c`, correctness over speed.


For fp16 the hot read is **hardware conversion** — one `FCVT`/`vcvt_f32_f16` on
arm64, `_mm_cvtph_ps` (F16C) on x86 — verified bit-exact against the software
converter across all 65536 patterns. That took the fp16 GEMV from ~3 GFLOP/s
(conversion-bound) to **~54 GFLOP/s**, ~18×, closing the "tekin8 lesson" for fp16.

**That second bullet was half wrong, and it cost real time.** It holds for the
narrow API, but `tk_linear_forward_f32` — the entry point every language binding
uses — round-trips *every weight through the storage grid on every pass* via
`tk_q`. There a libm-shaped converter is not cold at all: the `bl` per element
blocked vectorization outright, so the loop ran one element per iteration. fp16
and tekin8, the two formats whose converters were libm-shaped, now have inline
bit-arithmetic writers: **1.53×** and **2.28×** on that ladder (paired A/B),
bit-identical to the originals over all 2³² float patterns.

bf16 was measured and deliberately left alone — its converter was already
branchless bit arithmetic, and inlining it landed at 1.007× inside a 1.049× A/A
noise band. Nothing to buy.

The tempting wrong answer, recorded so nobody re-tries it: the one-instruction
hardware convert (`__fp16`, `_mm256_cvtps_ph`) is faster still but rounds ties to
**even** where the reference rounds ties **away from zero** — they differ on
0.39% of all float inputs. Adopting it would move every fp16 weight trajectory;
that is a numerics decision, not an optimization.

## 4. Explicit SIMD, one portable binary

The 4-row kernel has hand-written vector versions with **two accumulator chains
per row** (depth-8, to beat FMA *latency* rather than throughput):

(The before/after pairs below were taken with the earlier single-run bench
harness; `make bench` now reports medians of five, which shifts the absolute
GFLOP/s upward on this machine. Compare figures within a section, not across.)

| dtype | arm64 NEON (serial, M4) | how the narrow read is done |
|---|---|---|
| float32 | 31 → **39** GFLOP/s | `vld1q_f32` (bandwidth-bound at 4 B/el) |
| bfloat16 | 34 → **53** GFLOP/s | `vld1_u16` + `vshll_n_u16(v,16)` — the shift *is* the conversion |
| fp16 | ~3 → **~54** GFLOP/s | one `FCVTL` widen + hardware scalar read |
| bf16 on FEAT_BF16 | **~65** GFLOP/s | `BFMLALB/T` — exact f32 FMAs of bf16 products |

The **same kernels exist as AVX2 + FMA** on x86-64, compiled unconditionally via
a per-function `__attribute__((target("avx2,fma")))` and dispatched at runtime
with `__builtin_cpu_supports` — so a *single portable binary* uses AVX2 where the
CPU has it and falls back to the scalar loop on older chips, no build flags, no
fat binaries. CI medians (GitHub ubuntu runner): f32 GEMV 45.8, bf16 50.8, **fp16
63.2** GFLOP/s; fp16 batch GEMM hits **119** vs bf16's 72 (`VCVTPH2PS` widens
cheaper than the bf16 zext+shift there). FEAT_BF16 and AVX2 paths are picked by
runtime feature detection, never by a separate build.

## 5. Convolution is one big GEMM

Convolution is im2col + GEMM (Chellapilla et al. 2006). Two decisions carry it:

**A branch-free im2col fast path.** One interior test per patch: when the whole
patch is in-bounds (every `pad=0` patch, and interior patches under padding) the
copy is a straight contiguous `memcpy`-style run that vectorizes; only true
border patches take the bounds-checked slow path. ~**11–15%** faster forward on
the LeNet conv shapes, bit-identical output.

**The batch is a single GEMM, blocked BLIS-style** (Van Zee & van de Geijn 2015).
Above a work threshold, `C (out_c × n·oh·ow) = K · im2col` is packed into
register micro-panels: `K` packed once per call; the im2col columns packed
panel-by-panel (256 columns at a time, spanning samples) so the **full-batch
im2col matrix is never materialized** — a worker's working set is one small panel
(~590 KB at VGG shape) instead of an n× blow-up. An MR×NR register tile of `C`
accumulates the full `kdim` and stores once with bias+activation. The micro-kernel
reaches ~**86% of the core's float32 peak** (up from ~28% for the naïve
per-sample version); the same machinery runs the backward (`dK = dZ·im2colᵀ`) at
~78% of forward. Net: the conv rewrite was **1.23×** end-to-end (5.36 → 4.35 s).

**Packing only what the kernel holds.** Both pack sites used to buffer a whole
`NC`=256-column panel before the micro-kernel consumed it, but the kernel retires
one `NR`-column micro-panel at a time. Packing in bursts of `TK_CONV_PB` panels
inside the consuming loop sizes the scratch to what is actually live, per worker:
threaded conv **peak RSS −20.7%** (38.5 → 29.7 MB) and −41.3% on the conv test
binary, output byte-identical, no measurable time cost (all paired-A/B intervals
straddle 1.0). The saving scales with worker count, which is the right shape —
memory pressure is a threaded-run problem; at one thread it is under a megabyte.

## 6. The backward pass

**Dead-row skip.** A killed-ReLU row (`dz == 0`) contributes `w·0 == ±0` to the
input gradient — a no-op — so the full narrow-weight-row read is skipped in the
threaded `tk_linear_backward`. ~**18%** faster dense backward on a layer with
~50% dead rows, bit-identical result.

**The reduction wrinkle.** `dW`/`db` are per-row and disjoint (bitwise-identical
to serial), but `dx = Σ_o Wᵀ·dz` is a reduction over rows — every row feeds every
`dx[i]`. Splitting rows would race on `dx`, so each worker accumulates a private
`dx` and the partials are summed afterward; when `dx` is `NULL` (the first layer
needs no input gradient) the reduction disappears and the whole backward is
bitwise-identical to serial.

## 7. Threads: a persistent pool, and where it saturates

Output rows are independent, so layers split across a **persistent thread pool**
(`pool.c`) — workers created once and woken per call via a condition-variable
barrier, never `pthread_create` per call (fatal for a function called millions of
times). Small layers run serially below a work threshold so the pool never adds
overhead to the common case.

Measured ~**2.9× forward / 3.1× backward** on 10 cores (bf16). Sub-linear on
purpose, and the scaling curve is *not* monotone — this is the honest part:

| threads | 2048² (cache-resident) | 4096² (DRAM-bound) |
|:-:|:-:|:-:|
| 4 | **2.87× (peak)** | 2.44× |
| 6 | 2.42× | **3.12× (peak)** |
| 10 | 2.35× | 2.73× |

A cache-resident layer peaks at **4 threads — the M4's P-core count — then
regresses**: an equal-row split means the fork-join barrier waits on a slow E-core
straggler while the P-cores idle. A DRAM-bound layer peaks later (~6) because the
E-cores buy memory-level parallelism before bandwidth saturates. Either way the
ceiling is **bandwidth, not cores** — the same reason narrow storage matters in
the first place. Pin `MANTISSA_THREADS` to the P-core count for cache-resident
work.

## Reproduce

```sh
make bench          # dense GEMV/GEMM across dtypes
make benchconv      # conv forward/backward
make benchscale     # the thread-scaling curve + per-dispatch barrier latency
```

Numbers here are M4 (serial, interleaved medians) and GitHub CI (x86-64, AVX2)
as labelled; see [DESIGN.md](DESIGN.md) for the full method, the ULP-repro
envelope, and the alternatives that were measured and rejected.

**On believing any of these numbers.** Absolute wall-clock on a laptop is
dominated by DVFS, thermal state, background load, and — on Apple Silicon — which
core class the thread landed on, so timing A now and B later compares two
different machines. Speed claims above come from a paired design: both builds
timed alternately in randomised order, adjacent runs paired, and the *median of
the paired ratio* reported with a bootstrap CI, so drift slower than one pair
cancels. Every such claim is gated on an **A/A control** — the identical build
measured against itself — and anything inside that band is reported as no result.
That gate earns its keep: an in-process variant of the harness produced a 10%
*false* speedup between two byte-identical builds, and a non-paired sequential
run showed a 16% conv "regression" that vanished under pairing. Where a change
can be shown correct without timing at all — e.g. the stochastic-rounding fix,
whose generated assembly is byte-identical for float32/bf16/tekin32 — that proof
is preferred over any measurement.

## References

- Micikevicius et al., *Mixed Precision Training*, ICLR 2018 (arXiv:1710.03740).
- Chellapilla, Puri & Simard, *High Performance Convolutional Neural Networks for
  Document Processing*, 2006 (im2col + GEMM).
- Van Zee & van de Geijn, *BLIS: A Framework for Rapidly Instantiating BLAS
  Functionality*, ACM TOMS 2015.
