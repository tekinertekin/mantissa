#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include "dtypes.h"
#include "ops.h"
#include "backprop.h"   /* tk_rng, tk_sgd_step (SR unbiasedness test) */

static int failures = 0;

static void check_ok(int ok, const char *fmt, ...) {
    va_list ap;
    if (!ok) failures++;
    printf("  [%s] ", ok ? "OK" : "!!");
    va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n");
}

/* Relative tolerance: a K-mantissa-bit format can only pin a value to ~2^-K of
 * its magnitude, so a fixed absolute bound is the wrong test for large values
 * (e.g. tekin8's step near 100 is 8). `floor` covers values near zero. */
static void check(const char *name, float got, float want, float rel, float floor) {
    float err = fabsf(got - want);
    float tol = rel * fabsf(want) + floor;
    check_ok(err <= tol, "%-22s got=%.6g want=%.6g err=%.3g", name, got, want, err);
}

static void test_roundtrips(void) {
    /* Round-trip a few values through every format. Tolerances reflect each
     * format's precision, so this doubles as a spec check. */
    float xs[] = { 0.0f, 1.0f, -2.75f, 0.5f, 3.14159f, 100.0f, 0.01f };
    int   n    = (int)(sizeof(xs) / sizeof(xs[0]));

    printf("float32 round-trip (exact):\n");
    for (int i = 0; i < n; i++) check("f32", xs[i], xs[i], 0.0f, 0.0f);

    printf("fp16 round-trip (10-bit mantissa, rel ~2^-11):\n");
    for (int i = 0; i < n; i++)
        check("fp16", tk_fp16_to_float(tk_float_to_fp16(xs[i])), xs[i], 5e-4f, 1e-4f);

    printf("bfloat16 round-trip (7-bit mantissa, rel ~2^-8):\n");
    for (int i = 0; i < n; i++)
        check("bf16", tk_bf16_to_float(tk_float_to_bf16(xs[i])), xs[i], 4e-3f, 1e-4f);

    printf("tekin32 round-trip (24-bit mantissa, near-exact):\n");
    for (int i = 0; i < n; i++)
        check("t32", tk_t32_to_float(tk_float_to_t32(xs[i])), xs[i], 1e-6f, 1e-7f);

    printf("tekin8 (E4M3) round-trip (3-bit mantissa, rel ~2^-4):\n");
    for (int i = 0; i < n; i++)
        check("t8", tk_f8_to_float(tk_float_to_f8(xs[i])), xs[i], 0.07f, 1e-3f);

    printf("fp8_e5m2 round-trip (2-bit mantissa, wider range):\n");
    for (int i = 0; i < n; i++)
        check("e5m2", tk_e5m2_to_float(tk_float_to_e5m2(xs[i])), xs[i], 0.15f, 1e-3f);

    printf("fp4 (E2M1) round-trip on its exact grid {0,.5,1,1.5,2,3,4,6}:\n");
    float g[] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    for (int i = 0; i < (int)(sizeof(g) / sizeof(g[0])); i++)
        check("fp4", tk_fp4_to_float(tk_float_to_fp4(g[i])), g[i], 0.0f, 0.0f);
}

static void test_and_perceptron(void) {
    /* A 2-input AND perceptron built from the generic dense-layer primitive. */
    printf("\nAND perceptron via tk_linear_forward (step):\n");
    tk_scalar_t W[2]   = { TK_FROM_FLOAT(1.0f), TK_FROM_FLOAT(1.0f) };
    tk_scalar_t b[1]   = { TK_FROM_FLOAT(-1.5f) };
    float want[4]      = { 0, 0, 0, 1 };
    float in[4][2]     = { {0,0}, {0,1}, {1,0}, {1,1} };
    for (int i = 0; i < 4; i++) {
        tk_scalar_t x[2] = { TK_FROM_FLOAT(in[i][0]), TK_FROM_FLOAT(in[i][1]) };
        float y;
        tk_linear_forward(W, x, b, &y, 1, 2, TK_ACT_STEP);
        char lbl[16]; snprintf(lbl, sizeof lbl, "AND(%d,%d)", (int)in[i][0], (int)in[i][1]);
        check(lbl, y, want[i], 0.0f, 0.0f);
    }
}

static void test_simd_kernel(void) {
    /* Exercise the register-blocked SIMD kernel (out>=4, in>=8 hits tk__dot4's
     * 4-row NEON/AVX2 path) against a scalar reference. This is what validates
     * the vectorized kernels — e.g. the AVX2 path when CI runs on x86. I=44
     * covers every sub-loop of the two-chain AVX2 kernel: the 16-wide main loop
     * twice (chain accumulation across iterations), the 8-wide residual, and
     * the scalar tail — so a bug in any of the three is caught on real x86. */
    printf("\nlinear layer vs scalar reference (SIMD kernel check):\n");
    enum { O = 8, I = 44 };
    tk_scalar_t Wl[O * I], xl[I], bl[O];
    for (int k = 0; k < O * I; k++) Wl[k] = TK_FROM_FLOAT(((k % 9) - 4) * 0.1f);
    for (int k = 0; k < I; k++)     xl[k] = TK_FROM_FLOAT(((k % 5) - 2) * 0.2f);
    for (int k = 0; k < O; k++)     bl[k] = TK_FROM_FLOAT(0.05f);
    float y[O];
    tk_linear_forward(Wl, xl, bl, y, O, I, TK_ACT_RELU);
    for (int o = 0; o < O; o++) {
        float z = TK_TO_FLOAT(bl[o]);
        for (int i = 0; i < I; i++) z += TK_TO_FLOAT(Wl[o * I + i]) * TK_TO_FLOAT(xl[i]);
        float ref = z > 0.0f ? z : 0.0f;               /* relu */
        char lbl[16]; snprintf(lbl, sizeof lbl, "row%d", o);
        check(lbl, y[o], ref, 1e-4f, 1e-4f);           /* differ only by reduction order */
    }
}

static void test_nonfinite(void) {
    /* Non-finite handling. bf16/fp16/tekin32/e5m2 encode and preserve NaN/Inf;
     * tekin8 (E4M3) and fp4 have no Inf and clamp to their max finite (480, 6)
     * by design — see docs/DESIGN.md. */
    printf("\nnon-finite conversions:\n");
    check_ok(isnan(tk_fp16_to_float(tk_float_to_fp16(NAN)))
          && isinf(tk_fp16_to_float(tk_float_to_fp16(INFINITY)))
          && isinf(tk_fp16_to_float(tk_float_to_fp16(1e6f)))     /* >= 65520 -> inf */
          && tk_fp16_to_float(tk_float_to_fp16(-INFINITY)) < 0.0f,
          "fp16 nan/inf preserved, overflow -> inf");

    check_ok(isnan(tk_bf16_to_float(tk_float_to_bf16(NAN)))
          && isinf(tk_bf16_to_float(tk_float_to_bf16(INFINITY))),
          "bf16 nan/inf preserved");

    check_ok(isnan(tk_t32_to_float(tk_float_to_t32(NAN)))
          && isinf(tk_t32_to_float(tk_float_to_t32(INFINITY))),
          "tekin32 nan/inf preserved");

    check_ok(isnan(tk_e5m2_to_float(tk_float_to_e5m2(NAN)))
          && isinf(tk_e5m2_to_float(tk_float_to_e5m2(INFINITY)))
          && isinf(tk_e5m2_to_float(tk_float_to_e5m2(1e6f))),    /* >= 61440 -> inf */
          "e5m2 nan/inf preserved, overflow -> inf");

    check_ok(fabsf(tk_f8_to_float(tk_float_to_f8(NAN)))      == 480.0f
          && tk_f8_to_float(tk_float_to_f8(INFINITY))        == 480.0f
          && tk_f8_to_float(tk_float_to_f8(-INFINITY))       == -480.0f
          && tk_f8_to_float(tk_float_to_f8(1e6f))            == 480.0f,
          "tekin8 clamps non-finite/overflow to +-480");

    check_ok(fabsf(tk_fp4_to_float(tk_float_to_fp4(NAN)))    == 6.0f
          && tk_fp4_to_float(tk_float_to_fp4(INFINITY))      == 6.0f
          && tk_fp4_to_float(tk_float_to_fp4(-INFINITY))     == -6.0f
          && tk_fp4_to_float(tk_float_to_fp4(100.0f))        == 6.0f,
          "fp4 clamps non-finite/overflow to +-6");
}

static void test_forward_f32(void) {
    /* tk_linear_forward_f32 — the FFI entry point every language binding uses —
     * against a quantize-then-accumulate scalar reference. */
    printf("\ntk_linear_forward_f32 vs scalar reference:\n");
    enum { O = 5, I = 7 };
    float Wf[O * I], xf[I], bf[O], y[O];
    for (int k = 0; k < O * I; k++) Wf[k] = ((k % 9) - 4) * 0.11f;
    for (int k = 0; k < I; k++)     xf[k] = ((k % 5) - 2) * 0.21f;
    for (int k = 0; k < O; k++)     bf[k] = 0.05f * (float)(k - 2);
    tk_linear_forward_f32(Wf, xf, bf, y, O, I, TK_ACT_TANH);
    for (int o = 0; o < O; o++) {
        float z = TK_TO_FLOAT(TK_FROM_FLOAT(bf[o]));
        for (int i = 0; i < I; i++)
            z += TK_TO_FLOAT(TK_FROM_FLOAT(Wf[o * I + i])) *
                 TK_TO_FLOAT(TK_FROM_FLOAT(xf[i]));
        char lbl[16]; snprintf(lbl, sizeof lbl, "f32 row%d", o);
        check(lbl, y[o], tk_act_scalar(z, TK_ACT_TANH), 1e-4f, 1e-5f);
    }
}

static void test_threaded_layer(void) {
    /* Above the multithread threshold (out*in >= TK_MT_MIN_WORK) rows are split
     * across the pool; every row must still match a scalar reference to
     * reduction-order noise. Note: which kernel computes a row (4-row SIMD
     * block vs the scalar chunk-leftover) shifts with chunk boundaries, i.e.
     * with thread count — so results are reproducible only to ~ULP across
     * MANTISSA_THREADS settings, which is why this check is tolerance-based. */
    printf("\n600x600 layer vs scalar reference (threaded path):\n");
    enum { O = 600, I = 600 };
    tk_scalar_t *Wl = malloc(sizeof(tk_scalar_t) * O * I);
    tk_scalar_t *xl = malloc(sizeof(tk_scalar_t) * I);
    float       *y  = malloc(sizeof(float) * O);
    if (Wl && xl && y) {
        tk_rng r = tk_rng_seed(123);
        for (int k = 0; k < O * I; k++) Wl[k] = TK_FROM_FLOAT(tk_rng_f01(&r) - 0.5f);
        for (int k = 0; k < I; k++)     xl[k] = TK_FROM_FLOAT(tk_rng_f01(&r) - 0.5f);
        tk_linear_forward(Wl, xl, NULL, y, O, I, TK_ACT_IDENTITY);
        float max_abs = 0.0f;
        for (int o = 0; o < O; o++) {
            float z = 0.0f;
            for (int i = 0; i < I; i++)
                z += TK_TO_FLOAT(Wl[(size_t)o * I + i]) * TK_TO_FLOAT(xl[i]);
            float d = fabsf(y[o] - z);
            if (d > max_abs) max_abs = d;
        }
        check_ok(max_abs < 5e-3f, "max |threaded+simd - scalar| = %.2e", max_abs);
    }
    free(Wl); free(xl); free(y);
}

static void test_batch_forward(void) {
    /* Batch forward must match per-sample tk_linear_forward to reduction-order
     * noise (same kernel; only chunk boundaries can shift kernel selection). */
    printf("\ntk_linear_forward_batch vs per-sample:\n");
    enum { NS = 5, O = 600, I = 600 };
    tk_scalar_t *Wl = malloc(sizeof(tk_scalar_t) * O * I);
    tk_scalar_t *Xl = malloc(sizeof(tk_scalar_t) * NS * I);
    float *Yb = malloc(sizeof(float) * NS * O), *ys = malloc(sizeof(float) * O);
    if (Wl && Xl && Yb && ys) {
        tk_rng r = tk_rng_seed(77);
        for (int k = 0; k < O * I; k++)  Wl[k] = TK_FROM_FLOAT(tk_rng_f01(&r) - 0.5f);
        for (int k = 0; k < NS * I; k++) Xl[k] = TK_FROM_FLOAT(tk_rng_f01(&r) - 0.5f);
        tk_linear_forward_batch(Wl, Xl, NULL, Yb, NS, O, I, TK_ACT_TANH);
        float max_abs = 0.0f;
        for (int s = 0; s < NS; s++) {
            tk_linear_forward(Wl, Xl + (size_t)s * I, NULL, ys, O, I, TK_ACT_TANH);
            for (int o = 0; o < O; o++) {
                float d = fabsf(Yb[(size_t)s * O + o] - ys[o]);
                if (d > max_abs) max_abs = d;
            }
        }
        check_ok(max_abs < 5e-3f, "max |batch - per-sample| = %.2e", max_abs);
    }
    free(Wl); free(Xl); free(Yb); free(ys);
}

static void test_sr_unbiased(void) {
    /* Stochastic rounding is unbiased (Gupta et al., 2015): the mean of many
     * independent SR write-backs equals the true value, not the nearest grid
     * point. w = ±1.0 (exact in every format), one SGD step to ±1.2; the
     * negative mirror exercises SR's sign-magnitude rounding direction. */
    printf("\nstochastic rounding unbiasedness (+/-1.0 -> +/-1.2):\n");
    enum { TRIALS = 40000 };
    tk_optim opt = { 1.0f, 0.0f, 0.0f, 1 };
    tk_rng r = tk_rng_seed(2024);
    for (int sign = 1; sign >= -1; sign -= 2) {
        double acc = 0.0;
        for (int k = 0; k < TRIALS; k++) {
            tk_scalar_t w = TK_FROM_FLOAT(sign * 1.0f);
            float g = sign * -0.2f;
            tk_sgd_step(&w, &g, 1, &opt, &r);
            acc += (double)TK_TO_FLOAT(w);
        }
        check(sign > 0 ? "SR mean of 40k" : "SR mean of 40k (neg)",
              (float)(acc / TRIALS), sign * 1.2f, 0.0f, 5e-3f);
    }
}

/* test_sr_unbiased above only probes |w| >= 1.0, which is target-NORMAL in every
 * format -- the band where SR's float32-lattice construction is valid. Below the
 * format's smallest normal the representable grid turns uniform, and getting
 * that wrong is invisible to the test above: it made fp4's E[SR(0.75)] come out
 * 0.5 instead of 0.75, and collapsed every fp4 |v| <= 0.25 to exactly 0. These
 * two tests pin the low band, where fp4 (smallest normal 1.0) lives almost
 * entirely and where SR matters most, since that is the regime it exists for. */
static void test_sr_unbiased_subnormal(void) {
    printf("\nSR unbiasedness below the smallest normal (1.0 -> 0.9375):\n");
    enum { TRIALS = 200000 };
    tk_optim opt = { 1.0f, 0.0f, 0.0f, 1 };
    tk_rng r = tk_rng_seed(2024);
    double acc = 0.0;
    for (int k = 0; k < TRIALS; k++) {
        tk_scalar_t w = TK_FROM_FLOAT(1.0f);
        float g = 0.0625f;                      /* w - lr*g = 0.9375 exactly */
        tk_sgd_step(&w, &g, 1, &opt, &r);
        acc += (double)TK_TO_FLOAT(w);
    }
    /* Bound from the estimator, not from what happens to pass: the write-back is
     * Bernoulli on a grid of step s, so the mean's standard error is at most
     * s/(2*sqrt(TRIALS)). fp4 has the coarsest grid here (s = 0.5), giving
     * ~5.6e-4; 4e-3 is ~7 SE, loose enough never to flake and still ~16x tighter
     * than the 6.3e-2 bias the pre-fix code showed. */
    check("SR mean of 200k (0.9375)", (float)(acc / TRIALS), 0.9375f, 0.0f, 4e-3f);
}

/* The test above is anchored at 0.9375, which is subnormal only for fp4 (whose
 * smallest normal is 1.0). Every other narrow format keeps its subnormal band far
 * lower -- 2^-6 for tekin8, 2^-14 for e5m2/fp16 -- so that test cannot see a bug
 * there. This one is expressed in units of the format's OWN grid step: w starts
 * at two steps (exactly representable, m=2) and one SGD step lands it on 1.5
 * steps, which is never representable, so SR must split evenly between m=1 and
 * m=2 and average back to 1.5 steps. Pre-fix, tekin8 returned two steps instead
 * of 1.5 -- the fast path put its `down` candidate off the uniform grid and the
 * closing conversion re-rounded it after the coin had been flipped. */
#if TK_MANT_BITS < 23 && TK_SUB_SHIFT <= 30
static void test_sr_unbiased_own_subnormal(void) {
    printf("\nSR unbiasedness inside this format's own subnormal band:\n");
    enum { TRIALS = 200000 };
    const float step = ldexpf(1.0f, -TK_SUB_SHIFT);
    tk_optim opt = { 1.0f, 0.0f, 0.0f, 1 };
    tk_rng r = tk_rng_seed(31337);
    double acc = 0.0;
    for (int k = 0; k < TRIALS; k++) {
        tk_scalar_t w = TK_FROM_FLOAT(2.0f * step);
        float g = 0.5f * step;
        tk_sgd_step(&w, &g, 1, &opt, &r);
        acc += (double)TK_TO_FLOAT(w);
    }
    /* Same estimator bound as above, in grid units: SE <= step/(2*sqrt(TRIALS))
     * ~= 1.1e-3 steps, so 0.02 steps is ~18 SE and cannot flake, while the
     * pre-fix error was a full 0.5 steps -- 25x larger. */
    check("SR mean of 200k (1.5 steps)", (float)(acc / TRIALS), 1.5f * step,
          0.0f, 0.02f * step);
}
#endif

static void test_sr_accumulates(void) {
    /* config.h's claim for SR: an update too small to change the stored value
     * still accumulates in expectation, which is what lets a narrow type train
     * without an fp32 master copy. Round-to-nearest provably stalls; SR must not.
     * Uses a step of an eighth of the format's ULP, repeated 50 times. */
    printf("\nSR: sub-ULP updates accumulate where round-to-nearest stalls:\n");
#if TK_MANT_BITS > 20
    /* Not applicable to the float32-class formats. The probe needs w - lr*g to
     * still carry delta = 2^-(TK_MANT_BITS+3) when computed in float32, but the
     * spacing just below 1.0 is 2^-24, so delta survives only while
     * TK_MANT_BITS <= 21. Above that the update is lost to the ambient float
     * arithmetic before the storage grid is ever consulted -- and SR is a no-op
     * for these formats anyway (their grid is exact, see tk_sr_from_float's
     * TK_MANT_BITS >= 23 branch), so there is nothing here to test. */
    check_ok(1, "not applicable: TK_MANT_BITS=%d is float32-class, SR grid exact",
             TK_MANT_BITS);
#else
    const float ulp   = ldexpf(1.0f, -TK_MANT_BITS);
    const float delta = ulp / 8.0f;
    enum { STEPS = 50, TRIALS = 20000 };
    tk_optim rtn = { 1.0f, 0.0f, 0.0f, 0 }, sr = { 1.0f, 0.0f, 0.0f, 1 };

    tk_scalar_t w = TK_FROM_FLOAT(1.0f);
    float g = delta;
    for (int s = 0; s < STEPS; s++) tk_sgd_step(&w, &g, 1, &rtn, NULL);
    check_ok(TK_TO_FLOAT(w) == 1.0f,
             "round-to-nearest stalls after %d sub-ULP/8 steps (w=%.9g)",
             STEPS, TK_TO_FLOAT(w));

    tk_rng r = tk_rng_seed(2718);
    double acc = 0.0;
    for (int t = 0; t < TRIALS; t++) {
        tk_scalar_t wt = TK_FROM_FLOAT(1.0f);
        float gt = delta;
        for (int s = 0; s < STEPS; s++) tk_sgd_step(&wt, &gt, 1, &sr, &r);
        acc += (double)TK_TO_FLOAT(wt);
    }
    /* 50 steps of ulp/8 is 6.25 ULP of drift; allow a twentieth of a ULP, which
     * is well above the estimator's noise and far below the drift being checked. */
    check("SR mean after 50 steps", (float)(acc / TRIALS),
          1.0f - STEPS * delta, 0.0f, 0.05f * ulp);
#endif
}

int main(void) {
    printf("Active storage type: %s (%d bytes)\n\n", tk_dtype_name(), tk_scalar_size());

    test_roundtrips();
    test_and_perceptron();
    test_simd_kernel();
    test_nonfinite();
    test_forward_f32();
    test_threaded_layer();
    test_batch_forward();
    test_sr_unbiased();
    test_sr_unbiased_subnormal();
#if TK_MANT_BITS < 23 && TK_SUB_SHIFT <= 30
    test_sr_unbiased_own_subnormal();
#endif
    test_sr_accumulates();

    printf("\n%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
