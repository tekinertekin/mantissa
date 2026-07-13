/* Thread-pool scaling & threshold harness.
 *
 * Modes (argv[1]):
 *   scaling    static-split GEMV throughput at a size sweep — run across
 *              MANTISSA_THREADS values to map how far threading scales
 *   crossover  serial-vs-threaded per size, to locate the TK_MT_MIN_WORK payoff
 *              (build with -DTK_MT_MIN_WORK=1 via `make benchscale-cross`, then
 *              diff a MANTISSA_THREADS=1 run against a multi-thread run)
 *   barrier    fork-join wake+join latency per dispatch (near-empty job)
 *
 * Thread count comes from MANTISSA_THREADS (read once at pool init), so sweep it
 * across processes. All timings are interleaved and reported as medians: the M4
 * clocks its P/E cores on a DVFS curve that drifts within a run, so blocked
 * "all-A-then-all-B" comparisons are meaningless. Build via `make benchscale`. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ops.h"
#include "pool.h"

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static volatile float g_sink = 0.0f;

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, int n) {
    qsort(v, n, sizeof(double), cmp_d);
    return (n & 1) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

/* One timed block: REPS forward passes, returns seconds/pass. */
static double time_gemv(const tk_scalar_t *W, const tk_scalar_t *x,
                        const tk_scalar_t *b, float *y, int out, int in, int reps) {
    double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        tk_linear_forward(W, x, b, y, out, in, TK_ACT_RELU);
        g_sink += y[r % out];
    }
    return (now_s() - t0) / reps;
}

static void alloc_layer(int out, int in, tk_scalar_t **W, tk_scalar_t **x,
                        tk_scalar_t **b, float **y) {
    long params = (long)out * in;
    *W = malloc((size_t)params * sizeof(tk_scalar_t));
    *x = malloc((size_t)in * sizeof(tk_scalar_t));
    *b = malloc((size_t)out * sizeof(tk_scalar_t));
    *y = malloc((size_t)out * sizeof(float));
    for (long i = 0; i < params; i++) (*W)[i] = TK_FROM_FLOAT(((float)(i % 17) - 8.0f) * 0.05f);
    for (int i = 0; i < in; i++)  (*x)[i] = TK_FROM_FLOAT(((float)(i % 13) - 6.0f) * 0.1f);
    for (int i = 0; i < out; i++) (*b)[i] = TK_FROM_FLOAT(0.01f);
}

static double gflops(int out, int in, double spp) { return 2.0 * out * in / spp / 1e9; }

/* Static-split throughput at a size sweep. Run once per MANTISSA_THREADS value
 * to build the scaling table (serial baseline = MANTISSA_THREADS=1). */
static void mode_scaling(int samples, int reps) {
    int sizes[] = {1024, 2048, 4096};
    printf("=== GEMV scaling  (dtype=%s, threads=%d, %dB/param) ===\n",
           tk_dtype_name(), tk_num_threads(), tk_scalar_size());
    printf("  %-6s | %-9s | %s\n", "size", "ms/pass", "GFLOP/s");
    for (unsigned i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        int n = sizes[i];
        tk_scalar_t *W, *x, *b; float *y;
        alloc_layer(n, n, &W, &x, &b, &y);
        double v[64];
        time_gemv(W, x, b, y, n, n, 5);                  /* warm */
        for (int s = 0; s < samples; s++) v[s] = time_gemv(W, x, b, y, n, n, reps);
        double m = median(v, samples);
        printf("  %4dsq | %9.4f | %8.1f\n", n, m*1e3, gflops(n, n, m));
        free(W); free(x); free(b); free(y);
    }
}

/* Serial-vs-threaded crossover. Requires a build with -DTK_MT_MIN_WORK=1 so
 * tk_linear_forward always threads when THREADS>1; compare a THREADS=1 process
 * (serial) against a multi-thread one to find where threading starts to pay. */
static void mode_crossover(int samples, int reps) {
    int sizes[] = {32,48,64,96,128,192,256,384,512,768,1024};
    printf("=== crossover sweep  (dtype=%s, threads=%d) — ms/pass, GFLOP/s ===\n",
           tk_dtype_name(), tk_num_threads());
    printf("  %-6s | %-8s | %-9s | work(out*in)\n", "size", "ms", "GFLOP/s");
    for (unsigned i=0;i<sizeof(sizes)/sizeof(sizes[0]);i++) {
        int n = sizes[i];
        tk_scalar_t *W,*x,*b; float *y;
        alloc_layer(n,n,&W,&x,&b,&y);
        double v[64];
        time_gemv(W,x,b,y,n,n,20);
        for (int s=0;s<samples;s++) v[s]=time_gemv(W,x,b,y,n,n,reps);
        double m = median(v, samples);
        printf("  %4dsq | %8.4f | %8.2f  | %d\n", n, m*1e3, gflops(n,n,m), n*n);
        free(W);free(x);free(b);free(y);
    }
}

/* Barrier wake+join latency: a near-empty job (one trivial row per thread) so
 * the measured time is broadcast + wake + join, not kernel work. */
static void tiny_fn(void *ctx, int b, int e, int w) { (void)ctx;(void)b;(void)e;(void)w; }
static void mode_barrier(int samples) {
    int T = tk_num_threads();
    printf("=== barrier latency  (threads=%d) ===\n", T);
    const int N = 20000;
    double v[64];
    for (int i=0;i<200;i++) tk_parallel_for(T, tiny_fn, NULL);        /* warm */
    for (int s=0;s<samples;s++) {
        double t0=now_s();
        for (int i=0;i<N;i++) tk_parallel_for(T, tiny_fn, NULL);
        v[s] = (now_s()-t0)/N;
    }
    printf("  %.3f us/dispatch\n", median(v, samples)*1e6);
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "scaling";
    int samples = 9, reps = 60;
    if (!strcmp(mode, "scaling"))        mode_scaling(samples, reps);
    else if (!strcmp(mode, "crossover")) mode_crossover(samples, reps);
    else if (!strcmp(mode, "barrier"))   mode_barrier(samples);
    else { fprintf(stderr, "unknown mode %s\n", mode); return 2; }
    return 0;
}
