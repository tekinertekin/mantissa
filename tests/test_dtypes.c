#include <stdio.h>
#include <math.h>
#include "dtypes.h"
#include "ops.h"

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

    printf("\n%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
