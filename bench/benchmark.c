/* Micro-benchmarks for the compiled storage dtype.
 *
 *   1. Dense-layer (GEMV) throughput + memory footprint.
 *   2. Activation dispatch: per-element `switch` vs a resolved function pointer
 *      (the optimization discussed in DESIGN.md).
 *
 * Build & run per dtype:  make DTYPE=0 bench  /  make DTYPE=2 bench  ...
 * Timing uses CLOCK_MONOTONIC; a volatile sink stops the optimizer deleting the
 * work. Every timed section runs SAMPLES independent blocks of REPS passes and
 * reports the MEDIAN block time (a single run lies under DVFS -- see
 * docs/PERFORMANCE.md's "medians of 5" convention for the authoritative CI
 * numbers; SAMPLES here matches that). Peak RSS (getrusage's high-water mark)
 * is reported alongside throughput so a memory regression is as visible as a
 * speed one -- it is a whole-process, monotonically-growing figure, so it is
 * printed once per major stage to show where the run's memory actually peaked,
 * not a per-section reset. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#include "ops.h"

#define SAMPLES 5

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static volatile float g_sink = 0.0f;   /* keeps results "used" */

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median_of(double *v, int n) {
    qsort(v, n, sizeof *v, cmp_double);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* Process high-water RSS since start (getrusage(2)'s ru_maxrss is a
 * monotonically non-decreasing peak, never a per-call snapshot). The field's
 * UNIT differs by OS -- bytes on Darwin, kB on Linux -- and reading it as the
 * wrong one silently misreports memory by 1000x, which is exactly the kind of
 * regression this is meant to catch, so the platform branch is load-bearing,
 * not cosmetic. */
static double peak_rss_mb(void) {
    struct rusage u;
    if (getrusage(RUSAGE_SELF, &u) != 0) return -1.0;
#if defined(__APPLE__)
    return (double)u.ru_maxrss / (1024.0 * 1024.0);       /* bytes on Darwin */
#else
    return (double)u.ru_maxrss / 1024.0;                  /* kB on Linux */
#endif
}

int main(int argc, char **argv) {
    const int IN = 2048, OUT = 2048;
    int REPS = (argc > 1) ? atoi(argv[1]) : 200;
    if (REPS <= 0) REPS = 200;
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
    printf("note: single runs lie under DVFS -- reporting medians of %d "
           "interleaved-by-construction samples, %d passes/sample\n", SAMPLES, REPS);

    /* --- weight-matrix memory footprint (computed) vs peak RSS (measured) --- */
    double mb = (double)params * tk_scalar_size() / (1024.0 * 1024.0);
    printf("layer %dx%d = %ld params\n", OUT, IN, params);
    printf("weight memory: %.2f MB   (float32 would be %.2f MB)\n",
           mb, (double)params * 4 / (1024.0 * 1024.0));
    printf("same weights at 1B params: %.2f GB | 7B params: %.2f GB\n",
           (double)tk_scalar_size(), (double)tk_scalar_size() * 7);
    printf("peak RSS after allocating W/x/b/y: %.2f MB\n", peak_rss_mb());

    /* --- 1. GEMV throughput --- */
    tk_linear_forward(W, x, b, y, OUT, IN, TK_ACT_RELU);   /* warm up */
    {
        double samp[SAMPLES];
        for (int s = 0; s < SAMPLES; s++) {
            double t0 = now_s();
            for (int r = 0; r < REPS; r++) {
                tk_linear_forward(W, x, b, y, OUT, IN, TK_ACT_RELU);
                g_sink += y[r % OUT];
            }
            samp[s] = (now_s() - t0) / REPS;
        }
        double dt = median_of(samp, SAMPLES);
        double flops = 2.0 * params;                        /* mul + add per weight */
        printf("\n[GEMV] median %.3f ms/pass, %.2f GFLOP/s\n", dt * 1e3, flops / dt / 1e9);
    }

    /* --- 1a. batch forward (GEMM): W streams once per batch, not per sample --- */
    {
        enum { NS = 64 };
        tk_scalar_t *Xb = malloc((size_t)NS * IN * sizeof(tk_scalar_t));
        float *Yb = malloc((size_t)NS * OUT * sizeof(float));
        if (Xb && Yb) {
            for (long i = 0; i < (long)NS * IN; i++)
                Xb[i] = TK_FROM_FLOAT(((float)(i % 13) - 6.0f) * 0.1f);
            tk_linear_forward_batch(W, Xb, b, Yb, NS, OUT, IN, TK_ACT_RELU); /* warm */
            const int BREPS = REPS / 8 > 0 ? REPS / 8 : 1;
            double samp[SAMPLES];
            for (int s = 0; s < SAMPLES; s++) {
                double t0 = now_s();
                for (int r = 0; r < BREPS; r++) {
                    tk_linear_forward_batch(W, Xb, b, Yb, NS, OUT, IN, TK_ACT_RELU);
                    g_sink += Yb[r % (NS * OUT)];
                }
                samp[s] = (now_s() - t0) / BREPS;
            }
            double bdt = median_of(samp, SAMPLES);
            double bflops = 2.0 * params * NS;
            printf("[GEMM batch=%d] median %.3f ms/sample-equiv, %.2f GFLOP/s\n",
                   NS, bdt / NS * 1e3, bflops / bdt / 1e9);
        }
        free(Xb); free(Yb);
    }

    /* --- 1b. float32-API GEMV (the path every non-C binding calls) --- */
    {
        float *Wf = malloc((size_t)params * sizeof(float));
        float *xf = malloc((size_t)IN * sizeof(float));
        float *bf = malloc((size_t)OUT * sizeof(float));
        if (Wf && xf && bf) {
            for (long i = 0; i < params; i++) Wf[i] = ((float)(i % 17) - 8.0f) * 0.05f;
            for (int i = 0; i < IN; i++)  xf[i] = ((float)(i % 13) - 6.0f) * 0.1f;
            for (int i = 0; i < OUT; i++) bf[i] = 0.01f;

            tk_linear_forward_f32(Wf, xf, bf, y, OUT, IN, TK_ACT_RELU);   /* warm up */
            double samp[SAMPLES];
            for (int s = 0; s < SAMPLES; s++) {
                double t0 = now_s();
                for (int r = 0; r < REPS; r++) {
                    tk_linear_forward_f32(Wf, xf, bf, y, OUT, IN, TK_ACT_RELU);
                    g_sink += y[r % OUT];
                }
                samp[s] = (now_s() - t0) / REPS;
            }
            double fdt = median_of(samp, SAMPLES);
            double flops = 2.0 * params;
            printf("[GEMV f32] median %.3f ms/pass, %.2f GFLOP/s\n",
                   fdt * 1e3, flops / fdt / 1e9);

            /* Prepared path: quantize W once, then call the narrow fast API. */
            tk_scalar_t *Wq = malloc((size_t)params * sizeof(tk_scalar_t));
            tk_scalar_t *xq = malloc((size_t)IN * sizeof(tk_scalar_t));
            tk_scalar_t *bq = malloc((size_t)OUT * sizeof(tk_scalar_t));
            if (Wq && xq && bq) {
                tk_quantize(Wf, Wq, (int)params);
                tk_quantize(bf, bq, OUT);
                tk_quantize(xf, xq, IN);                               /* warm up */
                tk_linear_forward(Wq, xq, bq, y, OUT, IN, TK_ACT_RELU);
                double psamp[SAMPLES];
                for (int s = 0; s < SAMPLES; s++) {
                    double p0 = now_s();
                    for (int r = 0; r < REPS; r++) {
                        tk_quantize(xf, xq, IN);      /* x changes per call; W stays prepared */
                        tk_linear_forward(Wq, xq, bq, y, OUT, IN, TK_ACT_RELU);
                        g_sink += y[r % OUT];
                    }
                    psamp[s] = (now_s() - p0) / REPS;
                }
                double pdt = median_of(psamp, SAMPLES);
                printf("[GEMV f32 prepared] median %.3f ms/pass, %.2f GFLOP/s\n",
                       pdt * 1e3, flops / pdt / 1e9);
            }
            free(Wq); free(xq); free(bq);
        }
        free(Wf); free(xf); free(bf);
    }

    printf("peak RSS after dense-layer sections: %.2f MB\n", peak_rss_mb());

    /* --- 2. activation dispatch: switch (per element) vs resolved pointer --- */
    const int N = 1 << 22;                                 /* 4M elements */
    float *v = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) v[i] = ((float)(i % 200) - 100.0f) * 0.05f;

    for (int trial = 0; trial < 2; trial++) {
        tk_activation_t act = trial ? TK_ACT_SIGMOID : TK_ACT_RELU;
        const char *name = trial ? "sigmoid" : "relu";

        for (int i = 0; i < N; i++) v[i] = tk_act_scalar(v[i], act); /* warm up: both timed loops see hot cache */
        double sw[SAMPLES], pt[SAMPLES];
        for (int s = 0; s < SAMPLES; s++) {
            double t0 = now_s();
            for (int i = 0; i < N; i++) v[i] = tk_act_scalar(v[i], act);   /* per-element switch */
            sw[s] = now_s() - t0;
            g_sink += v[0];

            tk_act_fn f = tk_act_resolve(act);                            /* resolve ONCE */
            t0 = now_s();
            for (int i = 0; i < N; i++) v[i] = f(v[i]);
            pt[s] = now_s() - t0;
            g_sink += v[0];
        }
        double t_switch = median_of(sw, SAMPLES), t_ptr = median_of(pt, SAMPLES);
        printf("[act %-7s] switch %.2f ms | fn-ptr %.2f ms | ratio %.2fx (medians of %d)\n",
               name, t_switch * 1e3, t_ptr * 1e3, t_switch / t_ptr, SAMPLES);
    }

    printf("\npeak RSS for the whole run: %.2f MB\n", peak_rss_mb());

    free(W); free(x); free(b); free(y); free(v);
    return 0;
}
