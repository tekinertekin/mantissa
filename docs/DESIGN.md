# Design notes

## 1. Numeric formats

All formats are IEEE-style `(-1)^sign · 1.mantissa · 2^(exp-bias)` with a
subnormal branch when the exponent field is zero and no-implicit-leading-one.
Only the bit budget differs.

| Format  | S | E | M | bias | max finite | min normal | mantissa digits |
|---------|:-:|:-:|:-:|:----:|-----------:|-----------:|:---------------:|
| float32 | 1 | 8 | 23 | 127 | 3.4e38 | 1.2e-38 | ~7.2 |
| fp16    | 1 | 5 | 10 | 15  | 65504  | 6.1e-5  | ~3.3 |
| bfloat16| 1 | 8 | 7  | 127 | 3.4e38 | 1.2e-38 | ~2.4 |
| tekin32 | 1 | 7 | 24 | 63  | 1.8e19 | 2.2e-19 | ~7.5 |
| tekin8  | 1 | 4 | 3  | 7   | 480    | 1.6e-2  | ~1.1 |
| e5m2    | 1 | 5 | 2  | 15  | 57344  | 6.1e-5  | ~0.9 |
| fp4     | 1 | 2 | 1  | 1   | 6      | 1.0     | ~0.3 |

References for the standard formats:
- **fp16 / mixed precision** — Micikevicius et al., *Mixed Precision Training*,
  ICLR 2018 (arXiv:1710.03740). IEEE 754-2008 binary16.
- **bfloat16** — Kalamkar et al., *A Study of BFLOAT16 for Deep Learning
  Training*, 2019 (arXiv:1905.12322). Same 8-bit exponent as float32, so it is
  literally the top 16 bits of a float32 — conversion is a shift.
- **fp8 E4M3 / E5M2** — Micikevicius et al., *FP8 Formats for Deep Learning*,
  2022 (arXiv:2209.05433), OCP FP8. E4M3 (tekin8) favors precision; E5M2 favors
  range and is used for gradients. tekin8 does not reserve the `S.1111.111` NaN
  slot, so its max finite is 480 rather than OCP's 448.
- **fp4 E2M1** — the element type of MXFP4/NVFP4 (OCP Microscaling v1.0, 2023;
  NVIDIA Blackwell). Eight magnitudes `{0,.5,1,1.5,2,3,4,6}`, no inf/nan; only
  usable inside a block-scaling scheme (roadmap), included here as the scalar
  primitive.

### Why bfloat16 beats fp16 for training

Both are 16 bits. fp16 spends them on precision (10 mantissa bits) with only 5
exponent bits, so its dynamic range is narrow (~6e-5 to 65504) and gradients
underflow without loss scaling. bfloat16 keeps float32's full 8-bit exponent
and spends only 7 bits on mantissa: far coarser, but the range never changes,
which is what makes it drop-in for training. The formats encode a genuine
range-vs-precision decision, not a quality ranking — that decision is the whole
subject of this library.

### tekin32 (1·7·24) — the design choice

The premise: a 32-bit float for neural networks does not need float32's ±3.4e38
range. Weights, activations, and loss-scaled gradients live comfortably inside
±1e18. Those unused exponent bits are pure waste.

tekin32 reallocates **one exponent bit into the mantissa**: 7 exponent bits
still cover ±1.8e19 (ample for any NN tensor), while 24 mantissa bits give a
25-bit significand — one bit *more* precise than float32. Consequences:

- Round-trips float32 **losslessly** for any value in `[2^-62, 2^63)`, i.e. the
  entire NN operating range (the 24th mantissa bit is zero-padded).
- Provides an extra bit of precision for values produced *inside* tekin32
  arithmetic or quantized down from float64 — useful as an accumulator or
  high-precision master-weight format.
- Costs range that NN workloads never use.

This is the bfloat16 philosophy inverted: bf16 assumes range is scarce and
sacrifices precision; tekin32 assumes range is abundant and buys precision back.
NVIDIA's TF32 (8 exp, 10 mantissa in a 32-bit lane) makes the opposite bet —
keep full range, cut precision to fp16 levels — which suits GPU matmul
throughput. tekin32 targets fidelity instead.

### tekin8 (1·4·3)

The 1-byte experiment: 4× smaller than float32. With 3 mantissa bits its
relative step is ~1/8, so it is only usable where quantization noise is
tolerable (post-training weight quantization, activations feeding a robust
layer). Included to make the memory-vs-accuracy trade visible at its extreme —
running the test suite across formats shows exactly where 3 mantissa bits stop
being enough.

## 2. Performance

### Store narrow, accumulate wide

Inputs are read as narrow types; every dot-product accumulator is float32
(`tk_dot`, `ops.c`). This is the mixed-precision recipe (Micikevicius 2017):
narrow storage cuts memory traffic, float32 accumulation keeps the error of a
sum over millions of terms bounded. Accumulating in the narrow type instead
would lose low-order bits catastrophically for large layers.

### Memory bandwidth is the budget

A dense layer's forward pass (GEMV) reads the whole weight matrix once and does
~2 FLOPs per weight — when the matrix exceeds cache it is **memory-bandwidth
bound**, so halving the storage size approaches halving the runtime. `make
bench` confirms the direction: bfloat16 (half the bytes, a one-shift read) edges
out float32.

But the benchmark also tells the honest counter-story: **tekin8 is slower**, not
4× faster, once the matrix is cache-resident — its E4M3→float conversion (a
subnormal branch, not SIMD-friendly) now dominates the arithmetic. The lesson is
real: narrow storage only converts to speed when the read is cheap (a shift) or
hardware-accelerated (F16C, AVX512-BF16, Blackwell FP8 tensor cores). This is
why the conversion hot path matters as much as the byte count.

### Conversion: hot path vs cold path

- **narrow → float32 (hot)**: runs for every weight, every forward pass. These
  are branchless `static inline` bit manipulations in `dtypes.h` so they inline
  into `tk_dot` and vectorize. bf16 is a single shift.
- **float32 → narrow (cold)**: runs once, when weights are quantized/loaded, so
  `dtypes.c` favors clear, correct code (`frexpf`/`lroundf`).

For fp16 that door is now walked through: the hot read uses **hardware
conversion** — a single `FCVT` on arm64 (base ISA, via `__fp16`; verified
bit-exact against the software converter across all 65536 patterns, sNaN
payloads excepted) and `_mm_cvtph_ps` on x86 builds compiled with F16C. The
software converter remains the portable fallback. This took the fp16 GEMV from
~3 GFLOP/s (conversion-bound: ~10 instructions + 2 branches per element) to
parity with bfloat16 — the tekin8 lesson above, closed for fp16. AVX512-BF16
remains future work for the x86 bf16 path.

### The dot-product loop

`tk_dot` uses four independent accumulators. A single accumulator serializes on
the floating-point adder's latency (a loop-carried dependency); four let the
out-of-order engine keep multiple FMAs in flight. Combined with
`-ffp-contract=fast` (fold `a*b + c` into one FMA), `restrict` (no
aliasing → free to vectorize), and `-O3 -funroll-loops`, the compiler emits
vectorized FMA. `-march=native` unlocks the full SIMD width locally.

**Register blocking (GEMV).** `tk_linear_forward` does not call `tk_dot`
per row; it computes **four output rows at once** (`tk__dot4`). Each `x[i]` is
loaded and converted once and feeds four independent FMA chains, so the FP units
stay busy and the shared input is reused instead of reloaded per row — a
single-row loop can do neither. This is ~1.3× across all dtypes.

**Explicit NEON.** On arm64, `tk__dot4` uses hand-written NEON kernels with
**two accumulator chains per row** (depth-8): a single 4-wide chain per row is
bound by FMA *latency*, not throughput, and splitting it measured ~1.55× on its
own (M4, interleaved runs). The portable blocked loop stays as the fallback on
every other target/dtype:
- **float32** loads with `vld1q_f32` — serial 31 → 39 GFLOP/s (+25%,
  bandwidth-bound at 4 B/element).
- **bfloat16** loads 4 values with `vld1_u16` and widens them in-register with
  `vshll_n_u16(v, 16)` (bf16 *is* the top 16 bits of a float32, so the shift is
  the conversion) — serial 34 → 53 GFLOP/s with depth-8.
- **fp16** loads 4 values and widens with a single `FCVTL`
  (`vcvt_f32_f16`, base A64) — with the hardware scalar read above, serial
  ~3 → ~54 GFLOP/s overall (~18×).
- **bfloat16 on FEAT_BF16 CPUs** (ARMv8.6: Apple M2+, Graviton3+) additionally
  dispatches to a `BFMLALB`/`BFMLALT` kernel — exact f32 FMAs of exact bf16
  products, two instructions per row per 8 elements — for serial ~65 GFLOP/s
  (1.9× total over the old depth-4 kernel; ~140 GFLOP/s at 10 threads). Runtime
  detection mirrors the AVX2 pattern: compiled via a per-function target
  attribute, called only when `sysctl`/`hwcap` reports the feature.

On **x86-64** the same kernels exist as **AVX2 + FMA** (`_mm256_fmadd_ps` into
four `__m256` accumulators; bfloat16 widened with `_mm256_cvtepu16_epi32` +
`_mm256_slli_epi32`). They are compiled unconditionally via a per-function
`__attribute__((target("avx2,fma")))` and dispatched at runtime with
`__builtin_cpu_supports`, so a single portable binary uses AVX2 where the CPU
has it and falls back to the scalar loop on older chips — no build flags, no
separate binaries. Authoritative CI numbers (GitHub ubuntu runner, medians of 5): f32 GEMV
45.8 GFLOP/s, bf16 50.8, fp16 63.2 with the runtime-dispatched AVX2+F16C
kernel — fp16 is the fastest storage dtype on that machine, and its batch
GEMM hits 119 GFLOP/s vs bf16's 72 (`VCVTPH2PS` widens cheaper than the
bf16 zext+shift sequence there). `tekin8`'s E4M3 unpack does not map to a single widening
instruction, so it stays on the portable path on both architectures.

**Multithreading.** A dense layer's output rows are independent, so
`tk_linear_forward` splits them across a **persistent thread pool** (`pool.c`):
a fixed set of workers created once (lazily) and woken per call via a
condition-variable barrier — never `pthread_create` per call, which would be
fatal for a function invoked millions of times. Each thread owns a contiguous
row range and writes its own slice of `y`, so there is no locking and every dot
product is still computed start-to-finish by one thread — no cross-thread
reduction ever happens. One honest caveat on reproducibility: each chunk runs
rows through the 4-row SIMD kernel and finishes its `<4` leftover rows with the
scalar kernel, whose reduction orders differ by ~1–2 ULP. Chunk boundaries move
with the thread count, so a row near a boundary can switch kernels between
`MANTISSA_THREADS` settings — results are **bitwise reproducible for a fixed
thread count**, but only ULP-stable across different ones (a determinism test
in `tests/test_dtypes.c` pins this envelope). Pin `MANTISSA_THREADS` when exact
run-to-run bit-reproducibility matters. Small layers run serially below a work
threshold so the pool never adds overhead to the common small-call case;
`MANTISSA_THREADS` overrides the worker count, and platforms without pthreads
compile a serial stub.

The **backward pass** parallelizes the same way, with one wrinkle: `dW` and `db`
are per-row (disjoint, bitwise-identical to serial), but `dx = Σ_o Wᵀ·dz` is a
*reduction* over rows — every row contributes to every `dx[i]`. Splitting rows
would race on `dx`. So when `dx` is requested, each worker accumulates into a
private `dx` buffer and the partials are summed afterwards; `dW`/`db` stay
bitwise-identical and `dx` differs only by float reduction order. When `dx` is
`NULL` (e.g. the first layer, which needs no input gradient) the reduction
disappears and the whole backward is bitwise-identical.

Measured on bfloat16 across 10 cores: forward ~2.9×, backward ~3.1×. Sub-linear
because GEMV's low arithmetic intensity (~2 FLOPs per byte) makes it
memory-bandwidth bound before it is compute bound — the classic reason narrow
storage matters.

**Where it saturates, and why (M4, 4 P-cores + 6 E-cores, bf16, interleaved
medians).** The scaling curve is not monotone in thread count, and the reason is
the heterogeneous cores under a static equal-split barrier:

| threads | 2048² (8 MB, cache-resident) | 4096² (32 MB, DRAM-bound) |
|:-------:|:----------------------------:|:-------------------------:|
| 1       | 1.00×                        | 1.00×                     |
| 2       | 1.76×                        | 1.67×                     |
| 4       | **2.87× (peak)**             | 2.44×                     |
| 6       | 2.42×                        | **3.12× (peak)**          |
| 8       | 2.44×                        | 2.91×                     |
| 10      | 2.35×                        | 2.73×                     |

A cache-resident layer peaks at **4 threads — the P-core count** — then
*regresses*: `tk_parallel_for` gives every thread an equal row range, so once the
6 slower E-cores join, the fork-join barrier waits on an E-core straggler while
the P-cores idle, and total throughput falls below the P-core-only peak. A
DRAM-bound layer (working set past the SLC) peaks later, at ~6 threads, because
there the E-cores buy *memory-level parallelism* — more outstanding loads against
shared DRAM bandwidth — before that too saturates. Either way the ceiling is
bandwidth, not cores: ~2.9–3.1× is the most this GEMV extracts from 10 cores, and
the online-CPU-count default (10) is slightly past the sweet spot for
cache-resident work. Pin `MANTISSA_THREADS` to the P-core count for the
cache-resident regime if a workload lives there. (`make benchscale` /
`benchscale-cross`, `bench/bench_scaling.c`, reproduce the curve, the crossover,
and the per-dispatch barrier latency — ~19 µs at T=10, ~7.5 µs at T=4.)

The multithread work threshold (`TK_MT_MIN_WORK = 1<<18`) was swept per dtype
against this barrier cost and kept as one conservative constant — see the
rejected list below.

### Activation dispatch: switch beats a function pointer

Intuition says an indirect `function pointer` avoids the cost of a `switch`.
The benchmark says otherwise: applying an activation element-wise, the inline
`switch` runs ~7× faster than a resolved pointer for `relu` on Apple Silicon
(the x86 CI runner shows ≤1.2× — the branch-predictor/vectorization economics
are platform-dependent). Reason — the
`switch` body inlines and the loop **vectorizes**; an indirect call per element
is opaque to the vectorizer and pays call overhead each iteration. So
`tk_activate` uses the `switch`, and `tk_act_resolve()` (the pointer table) is
kept only for genuinely pluggable dispatch (e.g. a caller-supplied activation).
A dispatch mechanism is only "faster" if you measure it in the loop it runs in.

### Optimizations measured and rejected

Several textbook optimizations were tried and dropped because the measurement
disagreed with the theory. Recorded here so they are not re-attempted.

- **Software prefetch** (`__builtin_prefetch`) in the GEMV loop — **0.55× (≈2×
  slower)**. GEMV streams weights sequentially, the case hardware prefetchers
  handle perfectly; manual prefetch only adds instructions and competes with the
  hardware unit.
- **Cache-line padding of the per-thread `dx` partials** to defeat false
  sharing — a ~1.6× win in an *isolated* contention microbenchmark, but **~10%
  slower in the real backward** across three variants (aligned+memset, and
  calloc with an aligned base). In the real kernel the boundary false-sharing is
  masked by the per-element weight read + FMA, while the padding adds allocation
  overhead. Kept the plain contiguous buffer.
- **IFUNC dispatch** for the SIMD kernel — rejected on portability: `ifunc` is an
  ELF feature and does not exist on macOS/Mach-O (a development and release
  target). The `__builtin_cpu_supports` check it would replace is a cached,
  perfectly-predicted invariant branch (~0 cost), so there was nothing to win.
- **Stack (VLA) scratch instead of `malloc`** in the f32 binding path —
  measured **~60% slower** (large fixed stack frame + a runtime-typed pointer
  hurt codegen), and that path is quantization-bound, not allocator-bound.
- **Manual branchless `dropout`/L1-sign and loop unswitching** — no change:
  `-O3` already emits `csel`/`cmov` for the data-dependent selects and hoists the
  loop-invariant branches.
- **Explicit NEON for the backward pass** — **~40% slower** for the `dx`
  reduction than the compiler's auto-vectorized scalar loop. Backward is
  store-bandwidth bound (it writes `dW`, 4×params bytes), so hand-vectorizing the
  arithmetic wins nothing and the extra shuffle/reduce instructions cost. Kept the
  scalar loop the compiler already vectorizes.
- **Depth-16 (four accumulator chains per row)** in the NEON GEMV — no gain
  over depth-8 (56.5 vs 56.3 GFLOP/s): two chains already cover the FMA
  latency-throughput product; more only burns registers.
- **Single-chain BFMLALB/T** (the naive form: both instructions accumulating
  into one register) — no gain over widen+FMA (~37 vs ~35 GFLOP/s): the two
  BFMLALs form one serial dependency chain, so it is latency-bound exactly like
  a depth-4 loop. The instruction only pays with split even/odd accumulator
  chains (see the kernel section above).
- **Sharing one range function between the f32 serial and pooled paths** (the
  clean refactor, and exactly how the narrow path is structured) — **~40%
  slower serial** (bf16 3.5 → 2.5 GFLOP/s). Passing the serial branch's
  `W`/`x`/`y` into an external pool worker from the same function makes the
  pointers *escape*, defeating no-alias analysis and de-vectorizing the serial
  quantization loop. The narrow path is immune only because it uses the
  hand-written NEON/AVX2 kernel rather than the auto-vectorizer. Fix shipped:
  the pristine serial loop lives in its own pool-free function and a thin
  wrapper dispatches — isolation wins over elegance.
- **Non-temporal (streaming) stores for `dW`** in the backward pass — the
  theory says a write-only 16 MB gradient stream should bypass the cache; the
  consumer analysis says the opposite: in the real training loop `tk_sgd_step`
  reads `dW` immediately after `tk_linear_backward`, so evicting it buys a DRAM
  round-trip. Measured (arm64 `stnp` via inline asm — clang silently drops
  `__builtin_nontemporal_store`'s hint on a single q register — against a
  shape-identical `stp` control, interleaved medians): **38% slower** on the
  backward alone (M4's LLC absorbs the write stream that NT forces out) and no
  better chained with the SGD consumer. A cautionary footnote: the first cold,
  non-interleaved run showed NT "faster" — DVFS lying exactly as the benchmark
  header warns.
- **Software-pipelining `tk_train_step_f32`** (fusing row *o*'s SGD update with
  row *o+1*'s forward dot so the store and load streams overlap) — **+6% measured
  (0.994 → 0.935 ms/pass), rejected on numerics**. The W-traffic theory was
  already void: a 8 KB row is L1-resident between its dot and its update, so
  DRAM traffic is 1R+1W either way and the win is only ILP overlap. The
  disqualifier: per-row *source* order can be preserved exactly, but the
  baseline's rounding on this path is not a function of source order —
  `-O3 -funroll-loops -ffp-contract=fast` contracts the same `s += w*x` into
  `fmla` in the 4-wide loop, *unfused* `fmul`+`fadd` in the 16-wide unrolled
  block, and `fmadd` in the scalar tail, and the fused loop shifts which
  variant runs for which `in_dim` (observed: 1-ULP `dz` divergence at
  `in_dim % 4 == 3`, bit-identical at 1024/2048). Any restructuring of this
  function moves weight trajectories at *some* shape; +6% does not buy that.
- **Spin-wait in the thread pool** instead of the condvar barrier — declined:
  it would burn CPU on idle workers, which directly contradicts the
  low-footprint / millions-of-small-calls goal (small layers stay serial and the
  pool sits idle most of the time). The condvar's wake latency is negligible
  against the per-GEMV work above the threshold.
- **Dynamic atomic-counter chunking** in the pool (fixed-grain chunks handed out
  from a shared `atomic_int` so fast P-cores drain more than slow E-cores) —
  measured a real but *narrow* win and dropped. At T=10 on the M4 it recovered
  the E-core straggler loss on large layers (2048²/4096²: ~6–10% over the static
  split, best grain ~256 rows: +25%), but (a) it *regressed* small layers ~5–15%
  (atomic contention + finer dispatch), (b) the auto-grain the pool can pick
  blindly (~8 chunks/thread) captured only part of the win — the peak needs
  size-dependent tuning, (c) it breaks the documented **fixed-thread-count
  bit-reproducibility** of the backward `dx` reduction: each worker's private
  partial would sum a race-dependent set of rows, so `dx` would vary run-to-run
  even at pinned `MANTISSA_THREADS`. Decisively, its best case (T=10, ~150
  GFLOP/s) is still *below* simply running the static split at T=4 (~189): the
  real lever for the P+E asymmetry is the thread count (see the scaling curve
  above), not a cleverer load balancer. Static equal chunks stay.
- **Per-dtype `TK_MT_MIN_WORK`** instead of one constant — swept and rejected in
  favor of the single conservative `1<<18`. The break-even where threading beats
  serial tracks the per-element cost: the fast SIMD dtypes (f32/fp16/bf16) break
  even *near* the current threshold at T=10 (the ~19 µs dispatch ≈ a 512² serial
  GEMV), while the slow portable-path dtypes (tekin8/e5m2/fp4, ~10× the per-
  element cost) break even at smaller layers but have large absolute runtimes
  regardless. One constant is therefore correct for the fast common case and
  merely *conservative* — never harmful — for the slow types; a dtype-varying
  threshold adds a moving constant across three consumers (`ops.c` ×2,
  `backprop.c`) for no common-case gain. Honesty over cleverness.
- **Row padding for the SIMD tail at odd `in_dim`** (pad each weight row to a
  multiple of the vector width so the kernel never runs a scalar tail) — the
  tail is not the cost, so this was dropped. bf16/fp16 GEMV at `in_dim` = 2049 /
  2050 / 2055 is within noise (±1–3%) of the aligned 2048/2056. float32 *does*
  show a real ~7–10% penalty at those widths, but it is **row-stride**-driven,
  not scalar-tail-driven: adding a 4-wide NEON tail step to the kernel did not
  help, and even `in_dim` = 2049 (a one-element tail) regresses — the penalty is
  the odd row stride mis-aligning the 4-row-blocked `vld1q_f32` loads and
  crossing cache lines. The only remedy is caller-side padded row *stride*, which
  the library cannot impose without copying the whole caller-owned `W` each call
  (far more than the 7% saved), it bites only float32 (not the narrow-storage
  headline), and it vanishes once the layer is DRAM-bandwidth bound. Callers who
  hit it can pad the stride themselves: the public API already accepts a padded
  `in_dim` with zeroed tail columns and returns bit-consistent results (verified
  in `bench/bench_layout.c`). `make benchlayout` reproduces the tail, padding,
  alignment, and cache-residency sweeps.
- **A caller-owned scratch context** (`tk_ctx`) to eliminate `tk_linear_forward_f32`'s
  per-call `malloc(xq)` and `tk_linear_backward`'s per-call `calloc` of the
  dx-partials — the churn is real (a `tk_linear_forward_f32` inference loop
  allocates once per call: 16,217 allocations over a 16k-call run, cut to 211 by
  the context) but the *time* is not. The `xq` malloc is 12–25 ns; against the
  call it rides on that is **~5% at a toy 16×16 layer and 0.5–4% at 64–256**
  (interleaved medians, M4, serial). The dx-partials `calloc` only fires above
  the multithreading work threshold (`out_dim·in_dim ≥ 2^18`), where each
  backward call is ~250 µs and a 20 KB `calloc` is <1%. Meanwhile the real
  inference win — holding weights **resident narrow** so the forward skips
  re-quantizing every weight *and* runs the SIMD kernel instead of the scalar
  f32 loop — is **6–35× faster** (`tk_quantize` once + `tk_linear_forward`,
  already shipped) and strictly dominates removing the malloc. A per-call scratch
  buffer is not worth a new lifetime-managed, non-thread-safe API surface for a
  sub-5% gain on a path the resident-narrow route already replaces. (The Python
  binding exposes that route as `Mantissa.prepare()` / `Prepared`.)

Two suggestions were already in place: `dW`/`dx` are computed in one pass over
the weights (no double read), and small layers skip the thread pool via a work
threshold.

## 3. Reuse across architectures

`tk_linear_forward` — `activation(W·x + bias)` with an optional (NULL-able)
bias — is deliberately the only "layer". It is the shared primitive of:

- **MLP / ANN**: stack it.
- **RNN / GRU / LSTM**: apply it per timestep; gates are just linear projections
  followed by sigmoid/tanh, both provided.
- **Transformer**: Q/K/V and feed-forward are linear projections; attention
  output projections are typically bias-free (hence the NULL bias path), and
  GELU (Hendrycks & Gimpel, 2016, arXiv:1606.08415) is provided.
- **CNN**: a convolution is a batch of dot products of a filter over input
  patches. `tk_dot` is that inner product already; a conv layer adds an
  `im2col`/patch iterator (or a direct sliding window) on top and reuses the
  same narrow-store/float-accumulate numerics — no new arithmetic. Planned.

Heterogeneous models are supported today: bias is a per-call NULL-able pointer
and the activation is a per-call argument, so each layer chooses independently
(see `examples/mlp_example.c`).

Building this one primitive well now means those models reuse it later instead
of reimplementing the numerics.

## 4. Numeric landscape (context for the format choices)

- **Microscaling (MX)** — OCP MX v1.0 (2023): a block of 32 elements shares one
  E8M0 scale, restoring the dynamic range that 4/6-bit elements lack. Element
  types MXFP8 (E4M3/E5M2), MXFP6, MXFP4 (E2M1). mantissa implements the element
  formats; a shared per-block scale is the next step.
- **NVFP4** — NVIDIA Blackwell (2024 hardware): 16-element blocks, an FP8 E4M3
  block scale plus a per-tensor FP32 scalar; LLMs pretrained at 4 bits per
  "Pretraining LLMs with NVFP4" (NVIDIA, arXiv:2509.25149, 2025).
- **Posit / takum** (Gustafson) — tapered-precision alternatives to IEEE
  floats; more precision near ±1 where zero-centered weights concentrate.
- **IEEE P3109** — emerging standard for ML arithmetic formats (2025).

## 5. Back-propagation

The backward pass keeps the forward core's philosophy: no autograd graph, an
explicit primitive the caller drives.

`tk_linear_backward` collapses the activation derivative into the output
gradient once (`dz = dy · act'(z)`), then computes `dW`, `db`, and `dx` in a
single pass over each weight row — `W`/`dW` accessed sequentially (cache
friendly) while `dx` stays hot across the output loop. Accumulation is float32,
matching the forward pass.

**Stochastic rounding** is the key low-precision-training idea (Gupta et al.,
2015, arXiv:1502.02551). Storing weights narrow, a plain round-to-nearest write
of `w - lr·g` discards any update below the type's ULP, so with a small learning
rate training silently stalls. SR rounds to the neighbouring grid point with
probability equal to the fractional distance, so an update of ⅓ ULP moves the
weight ⅓ of the time — correct in expectation. `tk_sr_from_float` does this in
pure bit arithmetic: the storage grid is exactly the float32 patterns whose low
`k = 23 − TK_MANT_BITS` bits are zero, so adding a uniform random k-bit tail to
the f32 pattern and truncating rounds up with probability equal to the
fractional distance — a mantissa carry lands exactly on the next binade's first
grid point, so the algebra (and the exact unbiasedness proof, in the source
comment) holds across binade boundaries; the final `TK_FROM_FLOAT` is exact.
No fdiv/floorf per weight, and the random tail is sign-adjusted so the up/down
decision — and therefore every seeded weight trajectory — is bit-identical to
the earlier floor-based implementation (verified exhaustively per exponent for
all 7 dtypes). Exact unbiasedness is pinned by the ±SR-mean test in
`tests/test_dtypes.c` across all 7 dtypes. This is what lets `make train` learn
XOR in bfloat16 without an fp32 master weight copy; on float32/tekin32 it is a
no-op. It is the same mechanism used for FP8 weight updates on
Hopper/Blackwell. Two codegen lessons live in the source: the 0/tiny/inf/nan
guard is one range test routed through a `noinline` helper, because a guard
that tail-merges into the hot path's `TK_FROM_FLOAT` call gets if-converted and
parks a csel (the RNG rollback) on the xorshift loop-carried chain — measured
−12% on bf16, whose SR loop is bound by exactly that chain.

The **plain (non-SR) bf16 update** has a NEON fast path on arm64: the scalar
loop pays a `tk_float_to_bf16` call per weight — the requantizing store, not
the FMA, dominates. The kernel widens with a shift, updates with a fused
`vfmsq` (the same single-rounding `w − lr·g` the scalar `fmsub` computes), and
narrows in-register with the identical RNE+NaN bit recipe (verified against
the scalar converter over all 2^32 patterns): 1.34 → 6.1 G weights/s (4.55×,
interleaved medians), trajectories bit-identical. L1/L2/SR fall back to the
scalar loop.

`make benchbp` quantifies it on the XOR run (4000 epochs, final loss): in the
1-byte tekin8, round-to-nearest stalls at 0.249 (never learns) while stochastic
rounding reaches 0.00008; in bfloat16, 0.011 vs 0.00009; in float32 the two are
identical, as expected.

**L1/L2** fold into the gradient at the optimizer step (`g += l2·w`,
`g += l1·sign(w)`); **dropout** is inverted (survivors scaled by `1/(1-rate)`),
with the mask reused on the backward pass. All are config-gated, off by default.

**Correctness** is not asserted, it is checked: `make testbp` compares analytic
gradients against central finite differences (`(L(w+h)−L(w−h))/2h`) at float32,
requiring <1e-2 relative error over all weight and input gradients for tanh,
sigmoid, relu, and gelu. Finite-difference gradient checking is the standard way
to prove a hand-written backward pass matches its forward pass.
