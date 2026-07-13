# Release notes

Version history for **mantissa**, newest first. Updated on every release;
benchmark numbers are from an Apple M-series laptop (clang `-O3`), a 2048×2048
dense layer (4.2M params) unless noted, and are indicative, not absolute.

---

## Unreleased — thread-pool & data-layout investigation (all-drop)

A measurement pass on two fronts — thread-pool load balancing and SIMD/cache
data layout — that shipped **no kernel change**: every candidate lost on the
bench. Recorded because a well-evidenced rejection is worth as much as a merge
(see DESIGN.md's rejected list for the full teardown).

**Added (tooling only)**
- `bench/bench_scaling.c` (`make benchscale` / `benchscale-cross`) — GEMV scaling
  across `MANTISSA_THREADS`, the serial-vs-threaded crossover that locates the
  `TK_MT_MIN_WORK` payoff, and per-dispatch barrier latency. Interleaved medians.
- `bench/bench_layout.c` (`make benchlayout`) — SIMD-tail cost vs `in_dim`,
  zero-padding as remedy, base-pointer alignment, and layer-stack cache
  residency.

**Measured and rejected**
- **Dynamic atomic-counter chunking** in the pool: a real but narrow win (T=10
  large layers +6–10%) that regressed small layers, needed untunable per-size
  grain, and broke fixed-thread-count reproducibility of the backward `dx`
  reduction. Its best case still trailed the plain static split at 4 threads —
  the P+E asymmetry is a thread-count lever, not a load-balancer one.
- **Per-dtype `TK_MT_MIN_WORK`**: the fast SIMD dtypes break even near the
  current `1<<18`; the slow portable dtypes only benefit below it but run long
  regardless. One conservative constant kept.
- **Row padding for the odd-`in_dim` SIMD tail**: bf16/fp16 tail cost is noise;
  float32's ~7–10% penalty at 2049/2050/2055 is row-stride-, not tail-, driven
  (a 4-wide kernel tail did not help) and only fixable by caller-side stride
  padding, which the API already permits.

**Documented**
- DESIGN.md scaling curve: bf16 GEMV peaks at ~4 threads (cache-resident) to ~6
  (DRAM-bound) on the M4's 4P+6E cores, ~2.9–3.1× — bandwidth-bound, and static
  chunking regresses past the P-core count as E-core stragglers hold the barrier.

---

## v0.1.12 — 2026-07-12  (tag `v0.1.12`)

First true GEMM: batch inference stops re-streaming the weight matrix.

**Added**
- `tk_linear_forward_batch(W, X, bias, Y, n_samples, out_dim, in_dim, act)` —
  forward for a whole batch of inputs. Each 4-row W block sweeps every sample
  before moving on, so W streams from memory **once per batch** instead of
  once per sample; threads own row ranges (disjoint Y columns, no locking).
  Same `tk__dot4` kernel — per-sample output is bit-identical to
  `tk_linear_forward` (pinned by a new test).
- Measured (2048×2048 bf16, batch 64, interleaved runs): serial
  60 → 64.8 GFLOP/s (+8% — one core was already near the BFMLAL compute
  ceiling), **10 threads 127.7 → 348.8 GFLOP/s (2.7×)**. The multithreaded
  GEMV was bandwidth-bound on ten cores re-streaming the same 8 MB of
  weights per sample; amortizing W across the batch turns those threads
  compute-bound. This is the primitive batch predict / MLP mini-batches
  build on.

---

## v0.1.11 — 2026-07-12  (tag `v0.1.11`)

One addition, driven by a real measurement from the sister
mantissa-perceptron project: its honest benchmark showed our fit time
~380× behind scikit-learn while running an *identical* algorithm — and
isolated the cause to per-sample Python→C crossings (~4–5 µs each,
~137k per fit), not the kernels.

**Added**
- `tk_train_epoch_f32(W, bias, X, targets, n_samples, out_dim, in_dim,
  act, lr)` — a whole epoch of sequential SGD in one call; the sample
  loop runs inside the library. Weight updates are bit-identical to
  per-sample `tk_train_step_f32` calls (pinned by a new test); returns
  the mean pre-update loss. Python binding: `Mantissa.train_epoch`.
- Measured from Python (1000×4 layer, 100 epochs, array('f') zero-copy):
  per-sample loop **269.2 ms** → epoch API **1.90 ms** = **141×**, final
  weights identical. That puts interpreted callers in compiled-loop
  territory while keeping the memory profile untouched.

---

## v0.1.10 — 2026-07-12  (tag `v0.1.10`)

NEON kernel overhaul, driven by a 20-lens optimization review (every idea
adversarially verified, then benchmarked before merging — several died on the
bench; see DESIGN.md's rejected list). Serial numbers below are medians of
interleaved before/after runs on an M4 (DVFS makes single runs lie).

**Optimized**
- **fp16 read path uses hardware conversion**: a single `FCVT` on arm64 (base
  ISA), `_mm_cvtph_ps` on x86 F16C builds; software converter stays as the
  portable fallback. Verified bit-exact against the old converter for all
  65536 bit patterns (sNaN payload quieting excepted). The fp16 GEMV was
  conversion-bound at ~3 GFLOP/s — with the new NEON `FCVTL` kernel below it
  now runs at bfloat16 speed.
- **NEON GEMV kernels rebuilt at depth-8** (two accumulator chains per row —
  one chain is FMA-latency bound), and **fp16 gets an explicit NEON kernel**
  for the first time. Serial 2048×2048: float32 31 → 39 GFLOP/s (+25%),
  bfloat16 34 → 53 (+55%), fp16 3 → ~54 (**~18×**).
- **BFMLALB/T kernel for bfloat16 on FEAT_BF16 CPUs** (Apple M2+, Graviton3+),
  runtime-dispatched like the AVX2 path: serial ~65 GFLOP/s (**1.9×** total),
  ~140 GFLOP/s at 10 threads (was ~93). Note for the record: the naive
  single-chain BFMLAL form measured *no* gain — the win only appears with
  split even/odd accumulator chains.
- `tk_train_step_f32` skips the weight-update pass for dead rows (`dz == 0`,
  e.g. inactive relu units): +3–6% on the mixed benchmark, up to the full
  update traffic on sparse-activation layers, and a `0*inf=NaN` edge case gone.
- Worker thread stacks capped at 512 KiB (glibc default is RLIMIT_STACK,
  commonly 8 MiB *per worker* — 63 workers held ~500 MB of address space).
  Workers only run shallow row kernels; macOS behavior unchanged.

**Verified**
- All 7 dtypes + gradient check pass; fp16 converter swept exhaustively;
  kernel outputs match scalar reference within reduction-order tolerance
  (tests from v0.1.9 pin this envelope).

---

## v0.1.9 — 2026-07-12  (tag `v0.1.9`)

The float32 API — the entry point every non-C binding calls — was serial and
scalar while all the fast machinery (register blocking, NEON/AVX2, the thread
pool) was reachable only from the narrow C API. This release closes that gap,
plus a multi-agent review pass over correctness, tests, and docs.

**Optimized**
- `tk_train_step_f32` forward dot: 4 independent accumulators (the `tk_dot`
  recipe) instead of one loop-carried FP add. **7.4 → ~14.6 GFLOP/s (~2×)** on
  the full training step.
- `tk_linear_forward_f32` now threads its output rows above `TK_MT_MIN_WORK`.
  bf16 **3.1 → 10.7 GFLOP/s**, float32 **9.2 → 33.6 GFLOP/s** at 10 threads;
  serial path unchanged, threaded output bit-identical to serial (rows are
  independent). Implementation note: the serial loop had to stay in its own
  pool-free function — the shared-helper refactor measured ~40% slower serial
  (pointer escape defeats no-alias analysis; see DESIGN.md).
- **Python binding zero-copy path**: pass float32 `numpy` arrays or `array('f')`
  to `linear_forward` / `train_step` and the buffer goes straight to C, mutated
  in place — no per-element boxing, no list round-trip. **~225× faster per
  `train_step`** at 512×512 vs the list path (which still works; numpy optional).

**Added**
- `tk_quantize(const float *src, tk_scalar_t *dst, int n)` — narrow weights
  once, then run repeated inference through the fast `tk_linear_forward`:
  **9.5× serial / ~20–26× threaded** over per-call `tk_linear_forward_f32`
  re-quantization (bf16 2048×2048), and the caller holds a 2-byte/param narrow
  copy instead of round-tripping 4-byte floats every call.
- Benchmark cases for the previously unmeasured f32 path: `[GEMV f32]`,
  `[GEMV f32 prepared]`, `[train_step_f32]`.
- Test coverage from the QA review: BCE loss (value/gradcheck/extremes),
  dropout (rate 0/1, mask-backward consistency, seeded determinism), L1+L2 SGD
  vs closed form, empty-batch guards, **stochastic-rounding unbiasedness**
  (mean of 40k SR write-backs matches the true value in all 7 formats — the
  Gupta et al. 2015 property, now pinned), non-finite conversions (NaN/Inf
  preserved by bf16/fp16/tekin32/e5m2; tekin8/fp4 clamp to ±480/±6 by design),
  `tk_linear_forward_f32` vs reference, and a 600×600 above-threshold layer
  exercising the pool + SIMD kernels.

**Fixed**
- `tk_loss` / `tk_train_step_f32` returned **NaN for an empty batch** (`n<=0`:
  `0 * inf`); now a clean `0.0f`.
- The multithread work threshold used `(long)out_dim*in_dim`, which can
  overflow on LLP64 (Windows); widened to `size_t`.

**Docs**
- fp16 was mis-attributed to Micikevicius et al. 2017 (that paper is the
  *training recipe*, not the format — fp16 is IEEE 754-2008 binary16); NVFP4
  paper year disambiguated (2025 paper, 2024 hardware).
- DESIGN.md's claim that threaded forward results are "bitwise identical to
  serial" was empirically false (chunk boundaries shift the 4-row/leftover
  kernel split): restated as bitwise-reproducible for a fixed
  `MANTISSA_THREADS`, ~ULP-stable across settings.
- README: gradcheck tolerance corrected (<1e-2, not <1e-3), the mixed-arch
  snippet's requantization requirement noted, the library smoke-test snippet
  actually calls `tk_dtype_name()` now, stale tekin8 SGD figure refreshed,
  fp4 "½ byte" footnoted as logical width (storage is 1 B/value today).

**Verified**
- All 7 dtypes + gradient check pass; threaded f32 output byte-compared equal
  to serial; Python list vs numpy paths produce identical loss and weights.

---

## v0.1.8 — 2026-07-10  (tag `v0.1.8`)

AVX2 GEMV kernels for x86-64 — the counterpart to the arm64 NEON path, so the
same acceleration applies on Linux/Windows servers.

**Added**
- AVX2 + FMA kernels for `tk__dot4` (float32 and bfloat16), 8-wide
  `_mm256_fmadd_ps` accumulators; bfloat16 widened in-register with
  `_mm256_cvtepu16_epi32` + `_mm256_slli_epi32`.
- **Runtime dispatch**: the kernel is compiled unconditionally via
  `__attribute__((target("avx2,fma")))` and called only when
  `__builtin_cpu_supports` reports AVX2+FMA, so one portable binary is fast on
  modern x86 and correct on older CPUs — no build flags, no fat binaries. Prior
  x86 releases ran the scalar fallback; they now auto-accelerate on capable CPUs.
- A "linear layer vs scalar reference" test that exercises the register-blocked
  SIMD kernel (out≥4, in≥8), so CI — which runs on x86 AVX2 hardware — validates
  the AVX2 path on every push.

**Verified**
- arm64 (NEON) unchanged: all 7 dtypes + gradient check pass.
- x86-64: AVX2 codegen compiles for float32 and bfloat16; the full suite passes
  under an x86 emulator on the scalar-fallback path, and CI exercises the live
  AVX2 path. `tekin8` stays on the portable path on both ISAs (its E4M3 unpack
  has no single widening instruction).

---

## v0.1.7 — 2026-07-10  (tag `v0.1.7`)

Code-review pass over the hot paths. Each suggestion was benchmarked before
being accepted; only the change that actually helped was kept.

**Optimized**
- `tk_linear_forward_f32` (the float32-facing binding entry point) now quantizes
  the input vector `x` **once** into a scratch buffer instead of re-quantizing it
  on every output row, removing the x-side round-trip from the inner loop
  (~half the per-element conversions in that path). Output is unchanged; the
  narrow-storage hot path `tk_linear_forward` never had this cost.

**Reviewed and *not* changed** (kept for the record, since "measure, don't
assume" cuts both ways):
- *Explicit NEON for the backward pass* — benchmarked and **rejected**: it
  matched the auto-vectorized `dW` path and was ~40% *slower* for the `dx`
  reduction, so the compiler's code was already better.
- *Thread pool overhead on tiny layers* — already handled: `TK_MT_MIN_WORK`
  keeps small layers fully serial (they never touch the pool). A spin-wait was
  declined: it would burn cores when idle, wrong for a general-purpose library.
- *"Cache thrash" / vectorization stall in the backward `dx += …`* — the inner
  loop is element-wise (not a reduction over `i`), and `dx` fits L1 for realistic
  `in_dim`, so the premise did not hold.
- *Branchless dropout mask / L1 sign* — `-O3` already lowers these data-dependent
  selects to branchless `csel`/`cmov`, so a manual rewrite changes nothing.

All 7 dtypes, the gradient check, and the examples pass, warning-free.

---

## v0.1.6 — 2026-07-10  (tag `v0.1.6`)

Multithread the **backward** pass too (v0.1.5 did the forward).

**Optimized**
- `tk_linear_backward` splits output rows across the thread pool. `dW`/`db` are
  per-row so they stay bitwise-identical to serial; `dx` is a reduction over
  rows, so when it is requested each worker accumulates into a private buffer
  and the partials are summed afterwards (dx then differs only by float
  reduction order). When `dx == NULL` (e.g. the first layer) the whole pass is
  bitwise-identical.

**Benchmarks** (bfloat16 backward GEMV, 2048×2048, 10-core Apple laptop)
| threads | ms/pass | GFLOP/s |
|---|:--:|:--:|
| 1  | 1.01 | 12.45 |
| 10 | 0.33 | **38.67** (3.1×) |

**Correctness:** verified `dW`/`db` bitwise-identical serial vs 10 threads, and
`dx` matching to reduction-order tolerance; all tests + gradient check pass.

---

## v0.1.5 — 2026-07-10  (tag `v0.1.5`)

Multithreaded GEMV — split the dense-layer output rows across CPU cores.

**Added**
- Persistent fork-join **thread pool** (`src/pool.c`): workers created once
  (lazily), woken per call by a condvar barrier — no per-call `pthread_create`.
  `tk_linear_forward` splits output rows across it above a work threshold; small
  layers stay serial so the pool never hurts the millions-of-small-calls path.
  `MANTISSA_THREADS` tunes the count; non-pthreads targets get a serial stub.

**Benchmarks** (bfloat16 forward GEMV, 2048×2048, 10-core Apple laptop)
| threads | GFLOP/s |
|---|:--:|
| 1  | ~29 |
| 10 | **~83** (2.9×) |

Sub-linear by design: GEMV does ~2 FLOPs/byte, so it hits the memory-bandwidth
wall before the compute wall (float32, twice the bytes, gains only ~1.2×).
**Correctness:** the multithreaded result is bitwise identical to serial (each
row is computed entirely by one thread — no reduction reordering); verified by
matching checksums and the full test suite.

---

## v0.1.4 — 2026-07-10  (tag `v0.1.4`)

NEON kernel for **bfloat16** (the default dtype) — the follow-up promised in
v0.1.3.

**Optimized**
- Explicit NEON GEMV for bfloat16 on arm64: 4 values loaded with `vld1_u16` and
  widened in-register via `vshll_n_u16(v, 16)` (that shift *is* the bf16→float32
  conversion), then FMA'd. Portable fallback unchanged elsewhere.

**Benchmarks** (forward GEMV, 2048×2048; indicative, laptop-noisy)
| dtype | baseline | v0.1.3 | v0.1.4 |
|---|:--:|:--:|:--:|
| bfloat16 | 8.57 | 10.70 | **~21** GFLOP/s |
| float32  | 7.97 | 14.29 | ~15 |
| tekin8   | 2.55 | 2.76  | ~3.2 |

bfloat16 is now the fastest dtype *and* the smallest: it moves half the bytes of
float32 for the same FMA work. All 7 forward dtypes + gradient check pass; the
bf16 NEON path is validated by the bf16 tests and the XOR trainer on arm64.

---

## v0.1.3 — 2026-07-10  (tag `v0.1.3`)

Matmul (GEMV) throughput — the actual bottleneck in a real network.

**Optimized**
- **Register-blocked forward pass** (`tk__dot4`): computes four output rows at
  once so each `x` element is loaded/converted once and feeds four independent
  FMA chains. ~1.3× across all dtypes, portable.
- **Explicit NEON kernel** for the float32 build on arm64 (`vfmaq_f32` +
  `vaddvq_f32`), with the portable path as the fallback on every other
  target/dtype. ~1.8× for float32.

**Benchmarks** (forward GEMV, 2048×2048)
| dtype | v0.1.2 GFLOP/s | v0.1.3 GFLOP/s | speedup |
|---|:--:|:--:|:--:|
| float32  | 7.97 | 14.29 | 1.8× (NEON) |
| bfloat16 | 8.57 | 10.70 | 1.25× |
| tekin8   | 2.55 | 2.76  | 1.08× |

Correctness re-verified: all 7 forward dtypes + the gradient check pass. The
NEON path is exercised by the float32 tests on arm64 (locally); CI runs on
x86_64 and covers the portable fallback. Next: a bf16-widening NEON kernel and
an AVX path.

---

## v0.1.2 — 2026-07-10  (tag `v0.1.2`)

Multi-language clients, Python training, README diagrams, and an
externally-reviewed optimization pass.

**Added**
- **Python back-prop**: `tk_train_step_f32` (one float32 call = forward + MSE +
  backward + SGD update for a dense layer), wrapped as `Mantissa.train_step`;
  `python/train_example.py`.
- **`clients/`** — forward + back-prop demos calling the downloaded library:
  C++ (`dlopen`), C# (P/Invoke), Java (JNA), JavaScript (Node/koffi), Rust
  (`libloading`). C++, Node, and Python verified locally.
- **README diagrams** (Mermaid): colorized call-flow architecture + a
  training-loop figure.

**Optimized** (reviewed suggestions kept only where safe / measured)
- Stochastic-rounding ULP computed by exponent bit arithmetic instead of
  `frexpf`/`ldexpf` — two fewer libm calls per weight, numerically identical.
- RNG state hoisted into a register across the SGD / dropout loops.
- GELU gradient reordered to reuse `z²` (FMA-friendly).
- Opt-in **`TK_FAST_MATH`** (default off): branchless rational `tanh`/`sigmoid`
  that vectorizes instead of calling libm.

**Benchmarks**
| pass (bf16) | ms/pass | GFLOP/s |
|---|:--:|:--:|
| forward  | 0.98 | 8.6 |
| backward | 0.98 | 12.8 |

| SGD weight update | M weights/s |
|---|:--:|
| float32 | 9379 |
| bfloat16 | 994 |
| tekin8 (stochastic rounding) | 321  *(was 298 in v0.1.1)* |

| sigmoid over 4M values | ms |
|---|:--:|
| exact (libm, default) | 4.42 |
| `TK_FAST_MATH=1` | 1.15  *(~3.8× faster, ~2% error)* |

Honest note: these micro-optimizations are modest because `-O3` already does
much of it, and activations are O(n) vs the layer's O(n²) matmul — so `TK_FAST_MATH`
only helps activation-bound workloads. The matmul (GEMV) remains the lever.

---

## v0.1.1 — 2026-07-10  (commit `c61231a`)

Performance framing, branchless activations, CI, and downloadable binaries.

**Added**
- **CI** (`ci.yml`): all 7 storage dtypes + the back-prop gradient check on
  every push; live badge in the README.
- **Release workflow** (`release.yml`): every `v*` tag builds the shared library
  on Linux/macOS/Windows and attaches them to the GitHub Release — download and
  use with no toolchain.
- Back-propagation benchmark (`bench/bench_backprop.c`).

**Changed**
- README rebuilt to lead with speed/memory; storage formats demoted to "the
  precision dial"; bit-level detail moved to `docs/DESIGN.md`.
- Branchless activation kernels, each picked by benchmark: `step` via sign-bit
  (~40% faster than compare), `relu` via `fmax`, `sign` via `copysign`.
- Prebuilt library removed from git (now shipped via releases).

**Benchmarks**
| GEMV forward | weight mem | ms/pass | GFLOP/s |
|---|:--:|:--:|:--:|
| float32  | 16 MB | 1.05 | 7.97 |
| bfloat16 |  8 MB | 0.98 | 8.57 |
| tekin8   |  4 MB | 3.29 | 2.55 |

| `relu` over 4M values | ms |
|---|:--:|
| before (ternary) | 1.16 |
| after (`fmax`, branchless) | 0.41 |

Stochastic rounding vs round-to-nearest — XOR final loss after 4000 epochs:

| dtype | round-to-nearest | stochastic |
|---|:--:|:--:|
| float32  | 0.00008 | 0.00008 |
| bfloat16 | 0.01090 | 0.00009 |
| tekin8   | 0.24862 *(stalled)* | 0.00008 *(converged)* |

---

## v0.1.0 — 2026-07-10  (commit `73bee96`)

First tagged release: the numeric core plus training.

**Added**
- Seven storage formats: `float32`, `fp16`, `bfloat16`, custom `tekin32`
  (1·7·24), `tekin8` (FP8 E4M3), `fp8_e5m2`, `fp4_e2m1` — selected via `DTYPE`,
  default `bfloat16`. Narrow storage, float32 accumulation.
- Dense-layer forward (`tk_linear_forward`, GEMV), activations
  (step/sign/relu/sigmoid/tanh/gelu), mixed per-layer bias & activation.
- **Back-propagation**: `tk_linear_backward`, `tk_loss` (MSE/BCE), `tk_sgd_step`,
  dropout, L1/L2, and **stochastic rounding** — all training options off by
  default. Validated by a finite-difference gradient check (`make testbp`) and
  by learning XOR in bfloat16 (`make train`).
- ctypes Python binding, benchmark, MIT license.
