#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dtypes.h"
#include "ops.h"
#include "backprop.h"   /* tk_rng, tk_sgd_step (SR unbiasedness test) */

static int failures = 0;
/* Relative tolerance: a K-mantissa-bit format can only pin a value to ~2^-K of
 * its magnitude, so a fixed absolute bound is the wrong test for large values
 * (e.g. tekin8's step near 100 is 8). `floor` covers values near zero. */
static void check(const char *name, float got, float want, float rel, float floor) {
    float err = fabsf(got - want);
    float tol = rel * fabsf(want) + floor;
    int ok = err <= tol;
    if (!ok) failures++;
    printf("  [%s] %-22s got=%.6g want=%.6g err=%.3g\n",
           ok ? "OK" : "!!", name, got, want, err);
}

int main(void) {
    printf("Active storage type: %s (%d bytes)\n\n", tk_dtype_name(), tk_scalar_size());

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

    /* Exercise the register-blocked SIMD kernel (out>=4, in>=8 hits tk__dot4's
     * 4-row NEON/AVX2 path) against a scalar reference. This is what validates
     * the vectorized kernels — e.g. the AVX2 path when CI runs on x86. */
    printf("\nlinear layer vs scalar reference (SIMD kernel check):\n");
    {
        enum { O = 8, I = 20 };
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

    /* Non-finite handling. bf16/fp16/tekin32/e5m2 encode and preserve NaN/Inf;
     * tekin8 (E4M3) and fp4 have no Inf and clamp to their max finite (480, 6)
     * by design — see docs/DESIGN.md. */
    printf("\nnon-finite conversions:\n");
    {
        int ok;
        ok = isnan(tk_fp16_to_float(tk_float_to_fp16(NAN)))
          && isinf(tk_fp16_to_float(tk_float_to_fp16(INFINITY)))
          && isinf(tk_fp16_to_float(tk_float_to_fp16(1e6f)))     /* >= 65520 -> inf */
          && tk_fp16_to_float(tk_float_to_fp16(-INFINITY)) < 0.0f;
        if (!ok) failures++;
        printf("  [%s] fp16 nan/inf preserved, overflow -> inf\n", ok ? "OK" : "!!");

        ok = isnan(tk_bf16_to_float(tk_float_to_bf16(NAN)))
          && isinf(tk_bf16_to_float(tk_float_to_bf16(INFINITY)));
        if (!ok) failures++;
        printf("  [%s] bf16 nan/inf preserved\n", ok ? "OK" : "!!");

        ok = isnan(tk_t32_to_float(tk_float_to_t32(NAN)))
          && isinf(tk_t32_to_float(tk_float_to_t32(INFINITY)));
        if (!ok) failures++;
        printf("  [%s] tekin32 nan/inf preserved\n", ok ? "OK" : "!!");

        ok = isnan(tk_e5m2_to_float(tk_float_to_e5m2(NAN)))
          && isinf(tk_e5m2_to_float(tk_float_to_e5m2(INFINITY)))
          && isinf(tk_e5m2_to_float(tk_float_to_e5m2(1e6f)));    /* >= 61440 -> inf */
        if (!ok) failures++;
        printf("  [%s] e5m2 nan/inf preserved, overflow -> inf\n", ok ? "OK" : "!!");

        ok = fabsf(tk_f8_to_float(tk_float_to_f8(NAN)))      == 480.0f
          && tk_f8_to_float(tk_float_to_f8(INFINITY))        == 480.0f
          && tk_f8_to_float(tk_float_to_f8(-INFINITY))       == -480.0f
          && tk_f8_to_float(tk_float_to_f8(1e6f))            == 480.0f;
        if (!ok) failures++;
        printf("  [%s] tekin8 clamps non-finite/overflow to +-480\n", ok ? "OK" : "!!");

        ok = fabsf(tk_fp4_to_float(tk_float_to_fp4(NAN)))    == 6.0f
          && tk_fp4_to_float(tk_float_to_fp4(INFINITY))      == 6.0f
          && tk_fp4_to_float(tk_float_to_fp4(-INFINITY))     == -6.0f
          && tk_fp4_to_float(tk_float_to_fp4(100.0f))        == 6.0f;
        if (!ok) failures++;
        printf("  [%s] fp4 clamps non-finite/overflow to +-6\n", ok ? "OK" : "!!");
    }

    /* tk_linear_forward_f32 — the FFI entry point every language binding uses —
     * against a quantize-then-accumulate scalar reference. */
    printf("\ntk_linear_forward_f32 vs scalar reference:\n");
    {
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

    /* Above the multithread threshold (out*in >= TK_MT_MIN_WORK) rows are split
     * across the pool; every row must still match a scalar reference to
     * reduction-order noise. Note: which kernel computes a row (4-row SIMD
     * block vs the scalar chunk-leftover) shifts with chunk boundaries, i.e.
     * with thread count — so results are reproducible only to ~ULP across
     * MANTISSA_THREADS settings, which is why this check is tolerance-based. */
    printf("\n600x600 layer vs scalar reference (threaded path):\n");
    {
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
            int ok = max_abs < 5e-3f;
            if (!ok) failures++;
            printf("  [%s] max |threaded+simd - scalar| = %.2e\n", ok ? "OK" : "!!", max_abs);
        }
        free(Wl); free(xl); free(y);
    }

    /* Stochastic rounding is unbiased (Gupta et al., 2015): the mean of many
     * independent SR write-backs equals the true value, not the nearest grid
     * point. w = 1.0 (exact in every format), one SGD step to 1.2. */
    printf("\nstochastic rounding unbiasedness (1.0 -> 1.2):\n");
    {
        enum { TRIALS = 40000 };
        tk_optim opt = { 1.0f, 0.0f, 0.0f, 1 };
        tk_rng r = tk_rng_seed(2024);
        float g = -0.2f;
        double acc = 0.0;
        for (int k = 0; k < TRIALS; k++) {
            tk_scalar_t w = TK_FROM_FLOAT(1.0f);
            tk_sgd_step(&w, &g, 1, &opt, &r);
            acc += (double)TK_TO_FLOAT(w);
        }
        check("SR mean of 40k", (float)(acc / TRIALS), 1.2f, 0.0f, 5e-3f);
    }

    printf("\n%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
