# Release notes

Version history for **mantissa**, newest first. Updated on every release;
benchmark numbers are from an Apple M-series laptop (clang `-O3`), a 2048×2048
dense layer (4.2M params) unless noted, and are indicative, not absolute.

---

## v0.2.3 — 2026-07-14  (tag `v0.2.3`)

The audit release. Six independent read-only analysis lenses (training loop,
RAM, FFI seam, inference, numpy idioms, cross-repo architecture) were run
over the downstream family (perceptron / cnn / auto-encoder); everything
core-fixable landed here, the downstream items and verified-negatives are
recorded (agent notes, 2026-07-14). Plus two dedicated test passes.

**Fixed**
- **Session pointer memo now references arrays weakly.** The strong-ref
  design retained every array a Session ever saw — three lenses hit it
  independently (fresh inference chunk slices, per-batch noise arrays,
  reshape views): ~100 KB pinned per predict call, 1.8 GB over a 10-epoch
  denoise fit, unbounded. Two subtleties: numpy's `data_as()` embeds a
  strong backref inside the pointer it returns (memoized pointers are now
  built from the raw address; the caller's reference covers each call), and
  dead entries bulk-sweep past a threshold. Volatile arrays: all
  collectable; memo bounded (~5 entries after 4k fresh arrays); stable-
  buffer hit 186 ns; bit-identical results.
- **maxpool2d rejects non-int32 argmax** instead of writing the winners to
  a discarded boxed temp (found by the new binding contract tests — a
  later backward through that argmax would have scattered garbage).

**Added**
- **`tk_upsample2d_nearest_f32` / `_backward_f32`** — the upsample+conv
  decoder op (Odena, Dumoulin & Olah, 2016). The numpy forms measured far
  under copy speed at autoencoder shapes (fwd broadcast 4 vs 74 GB/s; bwd
  fused-sum 9×): C measures fwd 200→41 / 416→37 µs, bwd 739→21 /
  1415→30 µs (**up to 47×**; a denoise-decoder batch's upsample work drops
  2.77 → 0.13 ms). Oracle-verified at k=2,3,4; threaded over planes.
- **`tk_sgd_update_list_f32`** — one crossing updates a parameter LIST,
  bit-identical to per-tensor calls (measured downstream: 3,780 → 315
  crossings/fit; a 16-float bias costs more to cross than to compute).
  The Session variant memoizes pointer arrays per-list and re-verifies
  every tensor's identity on hits.
- **`conv2d_out_dim` computed in Python** — the four-int formula was 50% of
  ALL FFI crossings in a downstream fit (2 per conv/pool call). Mirrors C
  BIT-EXACTLY including truncate-toward-zero division in the degenerate
  band `-stride < in+2p-k < 0` (a 152-case grid cross-check caught floor
  vs trunc; the binding must agree with the kernel's internal oh/ow math).
  369 → 133 ns and no crossing.
- **`Z=None` documented for ANY activation** on the forward/inference path
  (the C guards always allowed it; verified bit-identical Y) — one
  output-sized store elided per layer when no backward follows.
- **`mse_loss` binding over `tk_loss`** — API completeness; the docstring
  is honest that a well-formed numpy MSE is not slower.
- **Tests**: degenerate conv shapes join the gradcheck table; new
  `make testedge` C contract suite (out_dim table, 1×1-conv ≡ dense-head
  cross-validation bit-for-bit, softmax ±1e4 stability, whole-input pool,
  fixed-thread determinism); new `python/test_binding.py` (50 checks:
  marshalling writebacks, Trainer/Session error contracts verbatim,
  Prepared's ULP-envelope promise, the weakref leak contract).

**Recorded, not shipped**: downstream adoption backlog (staging-buffer
inference, lazy dX scratch, cifar loader single-pass, float64 metric
temporaries, in-place noise; each sized in the audit notes), maxpool
backward one-pass zeroing (~20% of a small op), inference forward that
skips the Z store entirely, `tk_delta_fit`, and the verified negatives
(Prepared adoption would regress; perceptron's seam is already right).

**Downstream adoption (same day)**: mantissa-autoencoder v0.1.1 switched
its `Upsample2D` layer onto the new primitive (feature-detected via
`hasattr(backend, "upsample2d")`; the numpy oracle path stays the
fallback) — the measured numbers live in that repo's CHANGELOG and
benchmark README.

---

## v0.2.2 — 2026-07-13  (tag `v0.2.2`)

The conv-GEMM release: the v0.2.1 primitives kept their exact ABI and got a
real GEMM underneath. Plus a Session on the Python side. Downstream A/B
(mantissa-cnn minivgg/CIFAR-10 fit, identical machine state, median of 5):
**5.36 → 4.35 s (1.23×)**; LeNet-scale fits additionally −13% from Session.

**perf(conv)** — three staged rewrites of the f32 conv path (`src/conv.c`),
public signatures untouched:
- **Batch-whole GEMM with panel packing**: K packed once per call into
  MR-row micro-panels; im2col packed per 256-column panel *spanning
  samples* (a full-batch im2col matrix is still never materialized — the
  v0.2.1 rejection stands; panel packing gets the compute benefit at one
  panel's footprint per worker). Bias + activation fold into the tile
  store. Forward became bit-identical across thread counts.
- **NEON 6×16 outer-product micro-kernel** (24 accumulators; AVX2 6×16
  equivalent behind the ops.c-style runtime dispatch, portable-C tile
  fallback). In-situ `sample` profile: ~86% of a core's f32 peak. The
  flanking costs mattered as much as the kernel: per-element `if(Z)` +
  activation switch hoisted out of the store, pack rewritten as clipped
  contiguous runs.
- **Backward through the same machinery**: dK = dZ·im2col^T over column
  chunks with per-worker partials (bit-reproducible at fixed thread
  counts, as before); dX by sample ownership with per-tile run-based
  col2im scatter. Serial backward reaches ~78% of the new forward.

Measured (M4, benchconv, 50 reps; baseline remeasured same-day):

| shape (batch 16) | fwd 1T | fwd 10T | bwd 1T | bwd 10T |
|---|---|---|---|---|
| VGG 3→64@3×3 p1 | 2.28 → **0.85 ms** | 0.58 → **0.28** | 11.2 → **5.8** | 2.51 → **1.45** |
| VGG 64→64@3×3 p1 | 24.5 → **12.6** (96 GF) | 8.17 → **3.3** (363 GF) | 53.9 → **31.6** | 15.0 → **8.5** (285 GF) |

LeNet-scale shapes dispatch below a measured 2²⁴-MAC threshold to the
byte-identical per-sample path — interleaved A/B put every LeNet row
within ±2% (the small-shape wins that beat torch/TF downstream are
untouched).

**feat(python)** — **`Mantissa.session()` / `Session`**: the CNN-primitive
methods with identity-memoized pointers (a hit is a dict lookup + `is`),
for training loops whose buffers are allocated once and refilled in place.
Measured: pointer conversion was ~12% of a LeNet-5/MNIST fit; downstream
268 → 234 ms. One session per model; zero-copy is mandatory (raises rather
than silently copying).

**Measured and rejected** (recorded in DESIGN.md): NC=512 panels (+2% 1T,
−10-15% 10T — L2 contention), NC=128 (−10% 1T); 8×12 NEON tile — the
winner *flipped* with pack strategy (8×12 won pre-run-based packing, 6×16
wins after; both numbers recorded, 8×12 kept buildable); k-unroll ×2 (no
gain); JC sweep (flat).

**Known-unmeasured**: the AVX2 6×16 kernel mirrors proven ops.c patterns
but has not been executed on x86 hardware in this cycle; tile constants
are per-arch and adapt via `TK_CONV_MR/NR` when x86 numbers arrive. ASan
clean at 1/3/10 threads.

---

## v0.2.1 — 2026-07-13  (tag `v0.2.1`)

The CNN release — a minor-version bump because a whole primitive family
lands: 2-D convolution, max pooling, a batched dense head, fused
softmax-cross-entropy and a plain-f32 SGD step (`include/conv.h`,
`src/conv.c`), per the fixed mantissa ↔ mantissa-cnn engine contract. The
family is pure float32 end to end — the `_f32` training-path convention —
NCHW, batch outermost, all buffers caller-allocated.

**Added**
- **`tk_conv2d_forward_f32` / `tk_conv2d_backward_f32`** — im2col + GEMM
  (Chellapilla et al., 2006) with a 4-row register-blocked f32 dot kernel
  (NEON on arm64, portable scalar elsewhere; measured on the VGG-block GEMM:
  8.5 → 63.4 GFLOP/s over the single-row scalar loop). One sample's patch
  matrix at a time, heap scratch allocated once per call; threaded across
  batch samples above `TK_MT_MIN_WORK`. Backward: dK by im2col^T
  accumulation with per-worker partials (batch-summed, reproducible at fixed
  thread counts), dX by col2im scatter; `dz = dy * act'(z)`, the
  `tk_linear_backward` convention. Alloc-failure falls back to exact direct
  loops.
- **`tk_maxpool2d_f32` / `tk_maxpool2d_backward_f32`** — floor semantics
  (ragged edges dropped, documented), int32 argmax of each winner's
  plane-flat index; backward zeroes dX and scatters, overlaps accumulate.
- **`tk_linear_forward_batch_f32` / `tk_linear_backward_batch_f32`** — the
  CNN head: batched f32 dense layer on the same blocked kernel; backward is
  two race-free parallel passes (dW/db row-parallel, dX sample-parallel).
- **`tk_softmax_xent_f32`** — fused, max-subtracted; mean loss out,
  `dlogits = (softmax - onehot)/n` in one pass. **`tk_sgd_update_f32`** —
  `W -= lr*dW` (the f32 sibling of narrow-storage `tk_sgd_step`).
  **`tk_conv2d_out_dim`** — the shared output-size rule, exported.
- **Python bindings** for the whole family (zero-copy numpy float32/int32;
  `hasattr(tk, "conv2d_forward")` is the feature probe), plus
  `python/test_conv_binding.py` cross-checking every op against numpy
  references (conv vs a naive im2col; analytic einsum gradients).
- **`make testconv`** — finite-difference gradient checks (dK/db/dX,
  maxpool, dense batch, softmax-xent) over stride 1/2, pad 0/1/2, kh≠kw,
  non-square inputs, identity/tanh/relu (relu bias-nudged off its z==0
  kink, asserted); rel err < 1e-2, measured ≤ 1.6e-3. **`make benchconv`**
  — LeNet-5/VGG shapes, serial + threaded.

Measured (M4, clang -O3, float32 family, 50 reps; fwd/bwd ms per batch):

| shape (batch)                   | fwd 1T | fwd 10T | bwd 1T | bwd 10T |
|---------------------------------|-------:|--------:|-------:|--------:|
| LeNet C1 1×28×28→6@5×5 (32)     | 1.00 ms | 0.15 ms | 1.32 ms | 0.34 ms |
| LeNet C3 6×14×14→16@5×5 (32)    | 0.52 ms | 0.17 ms | 1.18 ms | 0.34 ms |
| VGG 3×32×32→64@3×3 p1 (16)      | 2.23 ms | 0.58 ms | 11.1 ms | 2.38 ms |
| VGG 64×32×32→64@3×3 p1 (16)     | 24.4 ms | 6.07 ms | 52.6 ms | 14.8 ms |

Peak: 199 GFLOP/s forward / 163 backward (VGG 64→64, threaded).

**Deliberately NOT done**
- **Narrow-dtype conv**: the family quantizes nothing. Storage-dtype conv is
  gated on the block-scaling design (per-block scales are what make 8/4-bit
  conv weights viable), same roadmap item as fp4 two-per-byte packing.
- **Whole-batch im2col**: n× the scratch footprint for no measured speed
  win over per-sample unrolling — the batch loop already streams K.
- **Pure-C 4-row GEMM block**: measured 13.8 GFLOP/s vs 63.4 for the NEON
  kernel on the same shape; clang would not keep eight independent chains
  live from scalar code, so the intrinsics body ships (ops.c precedent).

---

## v0.1.14 — 2026-07-13  (tag `v0.1.14`)

The FFI-was-the-bottleneck release. Profiling the sister project's per-epoch
loop (mantissa-perceptron, banknote 1030×4, M4) showed one `perceptron_epoch`
binding call costs ~9.8 µs — of which only ~3 µs is the C epoch. The other
~7 µs re-derived ctypes pointers for the same five unchanged buffers, every
epoch. The C core needed nothing; the binding did.

**Added**
- **`Mantissa.trainer()` / `Trainer`** — a pre-bound training session for the
  epoch-in-a-loop pattern: W/X/targets/bias pointers are derived once,
  per-epoch calls pass only `lr` and the visit order. Same C entry points,
  bit-identical weight trajectories (verified over shuffled epochs, both
  rules). Measured per-epoch call (interleaved medians, banknote 1030×4):
  `perceptron_epoch` 9.8 → 4.8 µs (**2.1×**), ordered SGD `train_epoch`
  19.5 → 14.0 µs. `Trainer.margins()` computes the post-epoch linear
  responses in one row-parallel GEMV, 11.8 → 5.8 µs over the equivalent
  `linear_forward` call.

**Measured and rejected** (same profiling pass, Python side)
- Batching all epochs' shuffle orders with `rng.permuted` (11.3 µs/epoch) or
  an in-place int32 shuffle (11.9 µs) — both LOSE to the naive per-epoch
  `rng.permutation(n).astype(int32)` (6.4 µs at n=1030): numpy's index-
  permutation fast path beats element shuffles, so the obvious form stays.
- C-kernel work: the measured epoch is ~3 ns/sample at d=4 — the C core was
  never the bottleneck here; no kernel change ships in this release.

---

## v0.1.13 — 2026-07-13  (tag `v0.1.13`)

The fridge-clearing release: five parallel investigation lanes closed —
three shipped, two buried with evidence. Plus pip wheels on every release
from here on.

**Added**
- **Prebuilt wheels + sdist on every release tag**: Linux x86-64
  (manylinux_2_28 + auditwheel), macOS arm64 (platform tag pinned —
  unpinned bdist_wheel over-claims universal2 with an arm64-only dylib),
  Windows via MinGW (non-blocking until CI proves it). Fixed the sdist
  being uncompilable (MANIFEST.in shipped src/*.c but not src/pool.h).
- **`Mantissa.prepare()` / `Prepared`** — resident narrow-weight inference
  from Python: quantize a layer once (half the weight bytes at bf16),
  narrow only x per call, run the SIMD kernel. 2.3x at 64x64, 7x at
  256x256, 16-35x at 1024x1024 over `linear_forward`; bit-identical to
  `tk_linear_forward`. One instance per thread.
- **x86-64 kernel parity**, now with authoritative CI numbers (GitHub
  ubuntu runner, medians of 5): two `__m256` FMA chains per row in
  `tk__dot4_avx2`, and a runtime-dispatched **AVX2+F16C fp16 kernel** so
  default portable builds get hardware fp16 conversion. Measured on real
  x86: f32 GEMV 45.8 GFLOP/s, bf16 50.8, **fp16 63.2 — the fastest
  storage dtype on that machine** (fp16 batch GEMM: 119 GFLOP/s vs bf16's
  72; VCVTPH2PS beats the bf16 zext+shift widen there). 10-12 of 16 YMM,
  zero spills — depth-16 would spill, which is where x86 stops vs
  arm64's 32 registers. The before/after deltas (+23% f32, 6.7x fp16)
  remain Rosetta-indicative — the old kernels never ran on CI.

**Thread-pool & data-layout investigation (all-drop on kernel code)** — every
candidate lost on the bench; recorded because a well-evidenced rejection is
worth as much as a merge (full teardown in DESIGN.md's rejected list).

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

The optimizer step gets the kernel treatment: integer stochastic rounding and
a NEON bf16 requantizing store. Seeded weight trajectories are bit-identical
to v0.1.12 everywhere — SR included.

**Changed**
- `tk_sr_from_float` — stochastic rounding in pure bit arithmetic: add a
  random k-bit tail below the storage type's mantissa width to the f32
  pattern, truncate onto the grid (Gupta et al., 2015). Replaces the
  fdiv/floorf/compare per weight; exact unbiasedness is argued algebraically
  in the source comment and pinned by the ±SR-mean test across all 7 dtypes.
  The tail is sign-adjusted so every up/down decision — and the consumed RNG
  stream — matches the old implementation bit-for-bit (verified over a dense
  sweep of every exponent, all 7 dtypes): seeded SR runs reproduce exactly.
  SR `[sgd_step]`, interleaved medians, 2048×2048: fp16 328 → 359, bf16
  578 → 584, tekin8/E4M3 259 → 284, fp4 169 → 258 M weights/s
  (+9%/+1%/+10%/+53% — the narrower the type, the more the removed float ops
  were the bottleneck; bf16's SR loop is xorshift-latency-bound, see
  DESIGN.md for the csel-on-the-RNG-chain war story).
- `tk_sgd_step` (plain SGD, bf16, arm64) — NEON fast path: widen by shift,
  fused `vfmsq` update, in-register RNE narrowing store (bit-identical to
  `tk_float_to_bf16` over all 2^32 patterns, exhaustively verified). Kills
  the per-weight converter call: **1.34 → 6.1 G weights/s (4.55×)**. L1/L2/SR
  use the scalar loop.

**Measured and rejected** (recorded in DESIGN.md):
- Non-temporal `stnp` stores for `dW`: 38% slower on backward alone, no gain
  chained with the `tk_sgd_step` consumer that re-reads `dW` immediately.
- Software-pipelining `tk_train_step_f32` (update row o fused with dot of row
  o+1): +6%, but clang's contraction choices already vary per unroll variant
  on this path, so any restructuring moves some trajectories by 1 ULP at some
  shapes — rejected to keep f32 training bit-stable.

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
