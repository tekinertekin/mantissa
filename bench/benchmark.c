/* Micro-benchmarks for the compiled storage dtype.
 *
 *   1. Dense-layer (GEMV) throughput + memory footprint.
 *   2. Activation dispatch: per-element `switch` vs a resolved function pointer
 *      (the optimization discussed in DESIGN.md).
 *
 * Build & run per dtype:  make DTYPE=0 bench  /  make DTYPE=2 bench  ...
 * Timing uses CLOCK_MONOTONIC; a volatile sink stops the optimizer deleting the
 * work. Numbers are laptop-noisy -- run a few times. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ops.h"

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static volatile float g_sink = 0.0f;   /* keeps results "used" */

int main(void) {
    const int IN = 2048, OUT = 2048, REPS = 200;
    const long params = (long)IN * OUT;

    tk_scalar_t *W = malloc((size_t)params * sizeof(tk_scalar_t));
    tk_scalar_t *x = malloc((size_t)IN * sizeof(tk_scalar_t));
    tk_scalar_t *b = malloc((size_t)OUT * sizeof(tk_scalar_t));
    float       *y = malloc((size_t)OUT * sizeof(float));
    if (!W || !x || !b || !y) return 1;

    for (long i = 0; i < params; i++) W[i] = TK_FROM_FLOAT(((float)(i % 17) - 8.0f) * 0.05f);
    for (int i = 0; i < IN; i++)  x[i] = TK_FROM_FLOAT(((float)(i % 13) - 6.0f) * 0.1f);
    for (int i = 0; i < OUT; i++) b[i] = TK_FROM_FLOAT(0.01f);

    printf("=== mantissa benchmark  (dtype=%s, %d bytes/param) ===\n",
           tk_dtype_name(), tk_scalar_size());

    /* --- weight-matrix memory footprint --- */
    double mb = (double)params * tk_scalar_size() / (1024.0 * 1024.0);
    printf("layer %dx%d = %ld params\n", OUT, IN, params);
    printf("weight memory: %.2f MB   (float32 would be %.2f MB)\n",
           mb, (double)params * 4 / (1024.0 * 1024.0));
    printf("same weights at 1B params: %.2f GB | 7B params: %.2f GB\n",
           (double)tk_scalar_size(), (double)tk_scalar_size() * 7);

    /* --- 1. GEMV throughput --- */
    tk_linear_forward(W, x, b, y, OUT, IN, TK_ACT_RELU);   /* warm up */
    double t0 = now_s();
    for (int r = 0; r < REPS; r++) {
        tk_linear_forward(W, x, b, y, OUT, IN, TK_ACT_RELU);
        g_sink += y[r % OUT];
    }
    double dt = now_s() - t0;
    double flops = 2.0 * params * REPS;                    /* mul + add per weight */
    printf("\n[GEMV] %d passes in %.3f s -> %.3f ms/pass, %.2f GFLOP/s\n",
           REPS, dt, dt / REPS * 1e3, flops / dt / 1e9);

    /* --- 2. activation dispatch: switch (per element) vs resolved pointer --- */
    const int N = 1 << 22;                                 /* 4M elements */
    float *v = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) v[i] = ((float)(i % 200) - 100.0f) * 0.05f;

    for (int trial = 0; trial < 2; trial++) {
        tk_activation_t act = trial ? TK_ACT_SIGMOID : TK_ACT_RELU;
        const char *name = trial ? "sigmoid" : "relu";

        double s = now_s();
        for (int i = 0; i < N; i++) v[i] = tk_act_scalar(v[i], act);   /* per-element switch */
        double t_switch = now_s() - s;
        g_sink += v[0];

        tk_act_fn f = tk_act_resolve(act);                            /* resolve ONCE */
        s = now_s();
        for (int i = 0; i < N; i++) v[i] = f(v[i]);
        double t_ptr = now_s() - s;
        g_sink += v[0];

        printf("[act %-7s] switch %.2f ms | fn-ptr %.2f ms | ratio %.2fx\n",
               name, t_switch * 1e3, t_ptr * 1e3, t_switch / t_ptr);
    }

    free(W); free(x); free(b); free(y); free(v);
    return 0;
}
