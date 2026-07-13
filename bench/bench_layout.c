/* Data-layout & cache benchmarks for the compiled storage dtype.
 *
 *   A. SIMD-tail cost: GEMV rate vs in_dim mod 8 (odd widths force a scalar
 *      tail on every row and misalign every following row).
 *   B. Zero-padding the rows (and x) to a multiple of 8 as the layout remedy —
 *      uses the public API unchanged, padding is pure data-layout convention.
 *   C. Base-pointer misalignment of W (element off a 16-byte boundary).
 *   D. Cache residency of layer stacks: per-layer rate as total weight bytes
 *      grow past L1/L2/SLC (thrash between layers vs one resident matrix).
 *
 * All comparisons are interleaved round-robin per sample and reported as the
 * median of SAMPLES runs — M-series DVFS makes back-to-back A/B timing lie.
 * Build & run per dtype:  make DTYPE=2 benchlayout  (env MANTISSA_THREADS=n) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ops.h"
#include "pool.h"

#define SAMPLES 9
#define OUT 2048

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static volatile float g_sink = 0.0f;

static int cmp_d(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}
static double median(double *v, int n) { qsort(v, n, sizeof *v, cmp_d); return v[n / 2]; }

static void fill_w(tk_scalar_t *W, size_t n) {
    for (size_t i = 0; i < n; i++) W[i] = TK_FROM_FLOAT(((float)(i % 17) - 8.0f) * 0.05f);
}
static void fill_x(tk_scalar_t *x, int n) {
    for (int i = 0; i < n; i++) x[i] = TK_FROM_FLOAT(((float)(i % 13) - 6.0f) * 0.1f);
}

/* ---- A + B + C: tail, padding, alignment ---------------------------------- */

typedef struct {
    const char *name;
    const tk_scalar_t *W, *x, *b;
    int out, in;    /* in_dim passed to the kernel (padded variants pass the stride) */
    long useful;    /* FLOPs credited: 2 * out * logical_in */
} variant_t;

static double time_passes(const variant_t *v, float *y, int passes) {
    double t0 = now_s();
    for (int r = 0; r < passes; r++) {
        tk_linear_forward(v->W, v->x, v->b, y, v->out, v->in, TK_ACT_RELU);
        g_sink += y[r % v->out];
    }
    return (now_s() - t0) / passes;
}

static void run_group(const char *title, variant_t *vs, int nv, float *y, int passes,
                      double *out_ms) {
    double t[16][SAMPLES];
    for (int v = 0; v < nv; v++) time_passes(&vs[v], y, 2);   /* warm */
    for (int s = 0; s < SAMPLES; s++)
        for (int v = 0; v < nv; v++) t[v][s] = time_passes(&vs[v], y, passes);
    printf("\n-- %s (median of %d, interleaved, %d passes/sample) --\n",
           title, SAMPLES, passes);
    double base = 0;
    for (int v = 0; v < nv; v++) {
        double ms = median(t[v], SAMPLES) * 1e3;
        double gf = (double)vs[v].useful / (ms * 1e-3) / 1e9;
        if (v == 0) base = ms / (double)vs[v].useful;   /* s per useful FLOP */
        double rel = (ms / (double)vs[v].useful) / base;
        printf("  %-26s %8.4f ms/pass  %7.2f GFLOP/s  %+6.1f%% vs first\n",
               vs[v].name, ms, gf, (rel - 1.0) * 100.0);
        if (out_ms) out_ms[v] = ms;
    }
}

/* ---- D: layer stacks ------------------------------------------------------- */

static void run_stack(int depth, int in, double *ms_per_layer, double *gflops) {
    size_t rowb = (size_t)in * OUT;
    tk_scalar_t **W = malloc((size_t)depth * sizeof *W);
    tk_scalar_t *b = malloc((size_t)OUT * sizeof *b);
    tk_scalar_t *x = malloc((size_t)in * sizeof *x);
    float *y = malloc((size_t)OUT * sizeof *y);
    for (int l = 0; l < depth; l++) { W[l] = malloc(rowb * sizeof(tk_scalar_t)); fill_w(W[l], rowb); }
    fill_x(x, in);
    for (int i = 0; i < OUT; i++) b[i] = TK_FROM_FLOAT(0.01f);

    int passes = 32 / depth > 0 ? 32 / depth : 1;   /* ~equal work per sample */
    double t[SAMPLES];
    for (int s = -1; s < SAMPLES; s++) {            /* s == -1 warms */
        double t0 = now_s();
        for (int r = 0; r < passes; r++) {
            fill_x(x, in);                           /* keep magnitudes bounded */
            for (int l = 0; l < depth; l++) {
                tk_linear_forward(W[l], x, b, y, OUT, in, TK_ACT_RELU);
                tk_quantize(y, x, in);               /* feed next layer, as a model would */
            }
            g_sink += y[0];
        }
        if (s >= 0) t[s] = (now_s() - t0) / passes / depth;
    }
    double m = median(t, SAMPLES);
    *ms_per_layer = m * 1e3;
    *gflops = 2.0 * (double)in * OUT / m / 1e9;
    for (int l = 0; l < depth; l++) free(W[l]);
    free(W); free(b); free(x); free(y);
}

int main(void) {
    printf("=== mantissa layout bench  (dtype=%s, %dB/elem, threads=%d) ===\n",
           tk_dtype_name(), tk_scalar_size(), tk_num_threads());

    /* --- A: tail sweep. out=256 keeps all six matrices (~6 MB total) co-resident
     * in L2, so this isolates the kernel tail from cache effects (an out=2048
     * version measured ~50 MB of interleaved variants: pure DRAM streaming,
     * tail signal buried). --- */
    {
        static const int dims[] = { 2048, 2049, 2050, 2052, 2055, 2056 };
        const int nv = (int)(sizeof dims / sizeof dims[0]);
        const int out = 256;
        variant_t vs[16];
        static char names[16][32];
        tk_scalar_t *b = malloc((size_t)out * sizeof *b);
        float *y = malloc((size_t)out * sizeof *y);
        for (int i = 0; i < out; i++) b[i] = TK_FROM_FLOAT(0.01f);
        for (int v = 0; v < nv; v++) {
            int in = dims[v];
            tk_scalar_t *W = malloc((size_t)in * out * sizeof *W);
            tk_scalar_t *x = malloc((size_t)in * sizeof *x);
            fill_w(W, (size_t)in * out); fill_x(x, in);
            snprintf(names[v], sizeof names[v], "in=%d (mod8=%d)", in, in % 8);
            vs[v] = (variant_t){ names[v], W, x, b, out, in, 2L * out * in };
        }
        run_group("A: tail cost vs in_dim (out=256, L2-resident)", vs, nv, y, 200, NULL);
        for (int v = 0; v < nv; v++) { free((void *)vs[v].W); free((void *)vs[v].x); }
        free(b); free(y);
    }

    /* --- B: zero-padding as remedy (worst tail 2055, and 2050) --- */
    {
        static const int cases[][2] = { { 2050, 2056 }, { 2055, 2056 } };
        const int out = 256;   /* both variants co-resident in L2, as in A */
        for (int c = 0; c < 2; c++) {
            int in = cases[c][0], pad = cases[c][1];
            tk_scalar_t *W  = malloc((size_t)in  * out * sizeof *W);
            tk_scalar_t *Wp = calloc((size_t)pad * out,  sizeof *Wp);
            tk_scalar_t *x  = malloc((size_t)in  * sizeof *x);
            tk_scalar_t *xp = calloc((size_t)pad,        sizeof *xp);
            tk_scalar_t *b  = malloc((size_t)out * sizeof *b);
            float *y = malloc((size_t)out * sizeof *y);
            fill_w(W, (size_t)in * out); fill_x(x, in);
            for (int o = 0; o < out; o++)
                memcpy(Wp + (size_t)o * pad, W + (size_t)o * in, (size_t)in * sizeof *W);
            memcpy(xp, x, (size_t)in * sizeof *x);
            for (int i = 0; i < out; i++) b[i] = TK_FROM_FLOAT(0.01f);

            /* padded terms are exact zeros; only reduction order can differ */
            float *y2 = malloc((size_t)out * sizeof *y2);
            tk_linear_forward(W, x, b, y, out, in, TK_ACT_RELU);
            tk_linear_forward(Wp, xp, b, y2, out, pad, TK_ACT_RELU);
            for (int o = 0; o < out; o++)
                if (fabsf(y[o] - y2[o]) > 1e-3f * (fabsf(y[o]) + 1e-6f)) {
                    printf("PAD MISMATCH o=%d %g vs %g\n", o, y[o], y2[o]); return 1;
                }

            char t[64], n0[32], n1[32];
            snprintf(t, sizeof t, "B: pad %d -> %d (out=256)", in, pad);
            snprintf(n0, sizeof n0, "natural in=%d", in);
            snprintf(n1, sizeof n1, "padded to %d", pad);
            variant_t vs[2] = {
                { n0, W,  x,  b, out, in,  2L * out * in },
                { n1, Wp, xp, b, out, pad, 2L * out * in },   /* useful FLOPs unchanged */
            };
            run_group(t, vs, 2, y, 200, NULL);
            free(W); free(Wp); free(x); free(xp); free(b); free(y); free(y2);
        }
    }

    /* --- C: W base misalignment (one element off) --- */
    {
        int in = 2048;
        tk_scalar_t *Wbuf = malloc(((size_t)in * OUT + 16) * sizeof *Wbuf);
        tk_scalar_t *x = malloc((size_t)in * sizeof *x);
        tk_scalar_t *b = malloc((size_t)OUT * sizeof *b);
        float *y = malloc((size_t)OUT * sizeof *y);
        fill_w(Wbuf, (size_t)in * OUT + 16); fill_x(x, in);
        for (int i = 0; i < OUT; i++) b[i] = TK_FROM_FLOAT(0.01f);
        variant_t vs[2] = {
            { "W aligned",        Wbuf,     x, b, OUT, in, 2L * OUT * in },
            { "W base +1 elem",   Wbuf + 1, x, b, OUT, in, 2L * OUT * in },
        };
        run_group("C: W base alignment (in=2048)", vs, 2, y, 20, NULL);
        free(Wbuf); free(x); free(b); free(y);
    }

    /* --- D: layer-stack residency --- */
    {
        static const int depths[] = { 1, 2, 4, 8, 16 };
        printf("\n-- D: layer stack, %dx%d per layer (median of %d) --\n", OUT, OUT, SAMPLES);
        printf("  %-8s %-12s %-14s %s\n", "depth", "weights MB", "ms/layer", "GFLOP/s");
        for (unsigned i = 0; i < sizeof depths / sizeof depths[0]; i++) {
            double ms, gf;
            run_stack(depths[i], OUT, &ms, &gf);
            double mb = (double)depths[i] * OUT * OUT * tk_scalar_size() / (1024.0 * 1024.0);
            printf("  %-8d %-12.1f %-14.4f %.2f\n", depths[i], mb, ms, gf);
        }
    }
    return 0;
}
