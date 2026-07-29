/* Conv benchmark: forward + backward throughput at LeNet-5 conv shapes
 * (LeCun et al., 1998, "Gradient-Based Learning Applied to Document
 * Recognition") and one VGG-style 3x3 block (Simonyan & Zisserman, 2014).
 * The conv family is pure float32 -- dtype-independent.
 *
 * The pool reads MANTISSA_THREADS once at creation, so serial vs threaded is
 * two runs of this binary (the `make benchconv` target does both):
 *   MANTISSA_THREADS=1 ./build/bench_conv    # serial
 *   ./build/bench_conv                       # all cores
 *
 * Timing uses CLOCK_MONOTONIC; a volatile sink keeps the work live. Numbers
 * are laptop-noisy -- run a few times. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#include "conv.h"
#include "backprop.h"   /* tk_rng */
#include "pool.h"       /* tk_num_threads */

#define SAMPLES 5

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static volatile float g_sink = 0.0f;

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median_of(double *v, int n) {
    qsort(v, n, sizeof *v, cmp_double);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* Process high-water RSS since start (see bench/benchmark.c for the platform
 * unit rationale: bytes on Darwin's ru_maxrss, kB on Linux's). Conv scratch
 * (im2col / packed panels) is allocated and freed per call, so this is the
 * one number that shows whether that churn ever pushes the process's peak
 * higher than the shape's own X/K/Y/dK/... buffers would need. */
static double peak_rss_mb(void) {
    struct rusage u;
    if (getrusage(RUSAGE_SELF, &u) != 0) return -1.0;
#if defined(__APPLE__)
    return (double)u.ru_maxrss / (1024.0 * 1024.0);
#else
    return (double)u.ru_maxrss / 1024.0;
#endif
}

typedef struct {
    const char *name;
    int n, in_c, in_h, in_w, out_c, kh, kw, stride, pad;
} shape;

static void bench_shape(const shape *sh, int reps) {
    const int oh = tk_conv2d_out_dim(sh->in_h, sh->kh, sh->stride, sh->pad);
    const int ow = tk_conv2d_out_dim(sh->in_w, sh->kw, sh->stride, sh->pad);
    const size_t xsz = (size_t)sh->n * sh->in_c * sh->in_h * sh->in_w;
    const size_t ksz = (size_t)sh->out_c * sh->in_c * sh->kh * sh->kw;
    const size_t ysz = (size_t)sh->n * sh->out_c * oh * ow;
    /* MACs per pass: one GEMM-equivalent forward; backward is two (dK + dX). */
    const double macs = (double)sh->n * sh->out_c * oh * ow
                      * sh->in_c * sh->kh * sh->kw;

    float *X = malloc(xsz * sizeof(float)), *K = malloc(ksz * sizeof(float));
    float *b = malloc((size_t)sh->out_c * sizeof(float));
    float *Z = malloc(ysz * sizeof(float)), *Y = malloc(ysz * sizeof(float));
    float *dY = malloc(ysz * sizeof(float));
    float *dK = malloc(ksz * sizeof(float));
    float *db = malloc((size_t)sh->out_c * sizeof(float));
    float *dX = malloc(xsz * sizeof(float));
    if (!X || !K || !b || !Z || !Y || !dY || !dK || !db || !dX) {
        printf("%s: alloc failed\n", sh->name);
        goto out;
    }
    tk_rng rng = tk_rng_seed(41);
    for (size_t i = 0; i < xsz; i++) X[i] = tk_rng_f01(&rng) - 0.5f;
    for (size_t i = 0; i < ksz; i++) K[i] = (tk_rng_f01(&rng) - 0.5f) * 0.5f;
    for (int i = 0; i < sh->out_c; i++) b[i] = 0.01f;
    for (size_t i = 0; i < ysz; i++) dY[i] = (tk_rng_f01(&rng) - 0.5f) * 0.1f;

    tk_conv2d_forward_f32(X, K, b, Z, Y, sh->n, sh->in_c, sh->in_h, sh->in_w,
                          sh->out_c, sh->kh, sh->kw, sh->stride, sh->pad,
                          TK_ACT_RELU);                            /* warm up */
    double fsamp[SAMPLES];
    for (int s = 0; s < SAMPLES; s++) {
        double t0 = now_s();
        for (int r = 0; r < reps; r++) {
            tk_conv2d_forward_f32(X, K, b, Z, Y, sh->n, sh->in_c, sh->in_h,
                                  sh->in_w, sh->out_c, sh->kh, sh->kw,
                                  sh->stride, sh->pad, TK_ACT_RELU);
            g_sink += Y[(size_t)r % ysz];
        }
        fsamp[s] = (now_s() - t0) / reps;
    }
    const double fdt = median_of(fsamp, SAMPLES);

    tk_conv2d_backward_f32(X, K, Z, dY, dK, db, dX, sh->n, sh->in_c,
                           sh->in_h, sh->in_w, sh->out_c, sh->kh, sh->kw,
                           sh->stride, sh->pad, TK_ACT_RELU);      /* warm up */
    double bsamp[SAMPLES];
    for (int s = 0; s < SAMPLES; s++) {
        double t0 = now_s();
        for (int r = 0; r < reps; r++) {
            tk_conv2d_backward_f32(X, K, Z, dY, dK, db, dX, sh->n, sh->in_c,
                                   sh->in_h, sh->in_w, sh->out_c, sh->kh, sh->kw,
                                   sh->stride, sh->pad, TK_ACT_RELU);
            g_sink += dK[(size_t)r % ksz];
        }
        bsamp[s] = (now_s() - t0) / reps;
    }
    const double bdt = median_of(bsamp, SAMPLES);

    printf("| %-26s | %3d | %8.3f | %7.2f | %8.3f | %7.2f |\n",
           sh->name, sh->n,
           fdt * 1e3, 2.0 * macs / fdt / 1e9,
           bdt * 1e3, 4.0 * macs / bdt / 1e9);
out:
    free(X); free(K); free(b); free(Z); free(Y);
    free(dY); free(dK); free(db); free(dX);
}

int main(int argc, char **argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 50;
    if (reps <= 0) reps = 50;

    /* LeNet-5 conv layers (28x28 input variant) and one VGG 3x3 block. */
    const shape shapes[] = {
        { "LeNet C1 1x28x28->6@5x5",  32, 1, 28, 28,  6, 5, 5, 1, 0 },
        { "LeNet C3 6x14x14->16@5x5", 32, 6, 14, 14, 16, 5, 5, 1, 0 },
        { "VGG 3x32x32->64@3x3 p1",   16, 3, 32, 32, 64, 3, 3, 1, 1 },
        { "VGG 64x32x32->64@3x3 p1",  16, 64, 32, 32, 64, 3, 3, 1, 1 },
    };

    printf("=== mantissa conv benchmark  (float32 family, threads=%d, "
           "%d reps) ===\n", tk_num_threads(), reps);
    printf("note: single runs lie under DVFS -- reporting medians of %d samples, "
           "%d reps/sample\n", SAMPLES, reps);
    printf("| %-26s | bat | fwd ms   | GFLOP/s | bwd ms   | GFLOP/s |\n",
           "shape");
    printf("|----------------------------|-----|----------|---------|----------|---------|\n");
    for (size_t i = 0; i < sizeof shapes / sizeof *shapes; i++)
        bench_shape(&shapes[i], reps);
    printf("\npeak RSS for the whole run: %.2f MB\n", peak_rss_mb());
    return 0;
}
