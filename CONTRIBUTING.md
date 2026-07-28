# Contributing to mantissa

Thanks for your interest in `mantissa` — a low-precision neural-network engine
written in C11, with a thin Python (ctypes) binding. Contributions of all
sizes are welcome: bug fixes, new low-precision formats, SIMD kernels for more
architectures, additional ops, benchmarks, examples, and docs.

**Anyone is welcome here** — no prior involvement needed. Open an issue to
report a bug, ask a question, or propose an idea, and open a pull request when
you have a change. First-time contributors are especially welcome.

## Getting started

```sh
git clone https://github.com/tekinertekin/mantissa
cd mantissa
make test        # build + run the test suite (default storage dtype = bfloat16)
```

No third-party dependencies — just a C11 compiler and `make`. The Python
binding needs only the standard library (`ctypes`); NumPy is optional and used
for zero-copy array passing.

## The precision system (read this first)

`mantissa` is a **mixed-precision** engine: weights and activations are *stored*
in a narrow format, while math is *accumulated* in float32 ("narrow storage,
wide accumulate"). The storage format is selected at build time via `TK_DTYPE`
(see `include/config.h`):

| `DTYPE=` | format | layout | bytes |
|---|---|---|---|
| 0 | float32 | 1-8-23 | 4 |
| 1 | fp16 | 1-5-10 | 2 |
| 2 | bfloat16 *(default)* | 1-8-7 | 2 |
| 3 | tekin32 | 1-7-24 | 4 |
| 4 | tekin8 (FP8 E4M3) | 1-4-3 | 1 |
| 5 | fp8 E5M2 | 1-5-2 | 1 |
| 6 | fp4 E2M1 (MXFP4) | 1-2-1 | 0.5 |

```sh
make DTYPE=4 test     # build + test with FP8 (E4M3) storage
make DTYPE=6 bench    # benchmark with FP4
```

If you add a new format, wire it through `include/config.h`, `include/dtypes.h`,
`src/dtypes.c`, and add a round-trip case to `tests/test_dtypes.c`.

## Project layout

```
include/    public headers (config.h, dtypes.h, ...)
src/        the C engine (dtypes.c, ops.c, conv.c, backprop.c, pool.c,
            activations.c, loss.c)
python/     ctypes binding + examples (mantissa/, *_example.py)
tests/      C tests (test_dtypes.c, test_conv.c, test_backprop.c, test_edges.c)
bench/      microbenchmarks (benchmark.c, bench_conv.c, bench_backprop.c,
            bench_layout.c, bench_scaling.c)
```

## Making a change

1. **Open an issue first** for anything non-trivial, so we can agree on the
   approach before you invest time. Small, obvious bug fixes can go straight to
   a PR.
2. Keep the diff focused — one logical change per PR.
3. **Add or update a test.** Correctness is checked bit-exactly where possible
   (`make test` must pass). If your change is a numerical fix, add a case that
   fails before and passes after.
4. Match the surrounding style: C11, 4-space indent, `tk_`-prefixed public
   symbols, comments only where the *why* isn't obvious (cite a paper/equation
   when relevant rather than narrating the code).
5. If you touch performance, include before/after numbers from `bench/`
   (`make DTYPE=<n> bench`), and confirm results stay bit-exact where expected.

## Submitting

- Run `make test` (and, for perf work, the relevant `bench/` target) before
  pushing.
- Open a PR describing **what** changed and **why**, and link the issue.
- Be prepared to explain any line when asked.

## AI-assisted contributions

Using AI to help write code is **completely fine**. But you are **responsible
for every single line** you submit — you must understand it, be able to explain
it, and stand behind its correctness. Please also mention in the PR that AI
helped, so reviewers have the full picture. Code that the author clearly doesn't
understand will be sent back regardless of how it was produced.

## Good places to start

- Look for issues labelled **`good first issue`**.
- New storage formats (e.g. other FP8/FP4 variants, integer types).
- SIMD kernels for architectures beyond NEON / AVX2-F16C.
- More ops or layer primitives, additional `bench/` coverage, worked examples.

## License

By contributing, you agree that your contributions are licensed under the
project's MIT License (see [LICENSE](LICENSE)).
