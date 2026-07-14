/* Contract and edge-case tests for the CNN primitives (conv.h) that the
 * gradient-check suites (test_conv.c) do not reach: the tk_conv2d_out_dim
 * argument contract, the 1x1-conv == dense cross-check between two code
 * paths, softmax-xent numerical corners, degenerate dense-batch shapes,
 * whole-image pooling, and fixed-thread reduction determinism. Compiled at
 * float32 (see Makefile `testedge`). Same [OK]/ALL PASSED house style as
 * test_conv.c / test_backprop.c. */
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "conv.h"
#include "backprop.h"   /* tk_rng */
#include "pool.h"       /* tk_num_threads */

static int failures = 0;

static void check_ok(int ok, const char *fmt, ...) {
    va_list ap;
    if (!ok) failures++;
    printf("  [%s] ", ok ? "OK" : "!!");
    va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n");
}

static void fill_rand(float *v, size_t n, tk_rng *rng, float scale) {
    for (size_t i = 0; i < n; i++) v[i] = (tk_rng_f01(rng) - 0.5f) * scale;
}

typedef struct { float max_err, scale; } grad_stat;
static void stat_add(grad_stat *st, float num, float ana) {
    const float e = fabsf(num - ana);
    if (e > st->max_err) st->max_err = e;
    if (fabsf(ana) > st->scale) st->scale = fabsf(ana);
}
static float stat_rel(const grad_stat *st) { return st->max_err / st->scale; }

/* ---- tk_conv2d_out_dim contract ------------------------------------------
 * Documented (conv.h): out = (in + 2*pad - k)/stride + 1, clamped to >= 0;
 * returns 0 on non-positive in/k/stride or negative pad. */
static void test_out_dim_contract(void) {
    struct { int in, k, stride, pad, want; const char *why; } t[] = {
        { 28,  5, 1, 0, 24, "valid" },
        { 28,  5, 1, 2, 28, "same (pad keeps size)" },
        { 32,  3, 2, 1, 16, "stride 2 halves" },
        {  5,  5, 1, 0,  1, "kernel == input -> 1" },
        {  7,  2, 2, 0,  3, "floor drops ragged edge" },
        {  5,  7, 1, 0,  0, "kernel larger than input -> 0" },
        {  3,  5, 1, 1,  1, "pad makes a too-big kernel just fit -> 1" },
        {  3,  7, 1, 1,  0, "pad still not enough -> 0" },
        {  0,  3, 1, 0,  0, "in <= 0 -> 0" },
        {  8,  0, 1, 0,  0, "k <= 0 -> 0" },
        {  8,  3, 0, 0,  0, "stride <= 0 -> 0" },
        {  8,  3, 1,-1,  0, "pad < 0 -> 0" },
    };
    int ok = 1;
    for (size_t i = 0; i < sizeof t / sizeof *t; i++) {
        const int got = tk_conv2d_out_dim(t[i].in, t[i].k, t[i].stride, t[i].pad);
        if (got != t[i].want) {
            ok = 0;
            check_ok(0, "out_dim(%d,%d,%d,%d) = %d, want %d (%s)", t[i].in,
                     t[i].k, t[i].stride, t[i].pad, got, t[i].want, t[i].why);
        }
    }
    check_ok(ok, "tk_conv2d_out_dim contract table (%zu cases)",
             sizeof t / sizeof *t);
}

/* ---- 1x1 conv over a 1x1 image == batched dense layer --------------------
 * A conv with in_h=in_w=kh=kw=1, pad=0, stride=1 has byte-identical buffer
 * layouts to tk_linear_forward_batch_f32 (K (out_c,in_c,1,1) == W
 * (out_dim,in_dim); X (n,in_c,1,1) == X (n,in_dim)). Cross-validates the
 * im2col/GEMM conv path against the dense-head path, forward and backward. */
static void test_dense_equals_1x1_conv(void) {
    enum { N = 3, C = 5, OC = 4 };
    float X[N * C], K[OC * C], b[OC], dY[N * OC];
    float Zc[N * OC], Yc[N * OC], Zd[N * OC], Yd[N * OC];
    float dKc[OC * C], dbc[OC], dXc[N * C];
    float dWd[OC * C], dbd[OC], dXd[N * C];
    tk_rng rng = tk_rng_seed(101);
    fill_rand(X, N * C, &rng, 2.0f);
    fill_rand(K, OC * C, &rng, 1.0f);
    fill_rand(b, OC, &rng, 1.0f);
    fill_rand(dY, N * OC, &rng, 1.0f);

    tk_conv2d_forward_f32(X, K, b, Zc, Yc, N, C, 1, 1, OC, 1, 1, 1, 0,
                          TK_ACT_TANH);
    tk_linear_forward_batch_f32(K, X, b, Zd, Yd, N, OC, C, TK_ACT_TANH);
    grad_stat sf = { 0.0f, 1e-6f };
    for (int i = 0; i < N * OC; i++) stat_add(&sf, Yc[i], Yd[i]);
    check_ok(stat_rel(&sf) < 1e-5f, "1x1 conv forward == dense (rel %.2e)",
             stat_rel(&sf));

    tk_conv2d_backward_f32(X, K, Zc, dY, dKc, dbc, dXc, N, C, 1, 1, OC, 1, 1,
                           1, 0, TK_ACT_TANH);
    tk_linear_backward_batch_f32(K, X, Zd, dY, dWd, dbd, dXd, N, OC, C,
                                 TK_ACT_TANH);
    grad_stat sk = { 0.0f, 1e-6f }, sb = sk, sx = sk;
    for (int i = 0; i < OC * C; i++) stat_add(&sk, dKc[i], dWd[i]);
    for (int i = 0; i < OC; i++)     stat_add(&sb, dbc[i], dbd[i]);
    for (int i = 0; i < N * C; i++)  stat_add(&sx, dXc[i], dXd[i]);
    check_ok(stat_rel(&sk) < 1e-5f && stat_rel(&sb) < 1e-5f
          && stat_rel(&sx) < 1e-5f,
          "1x1 conv backward == dense  dK %.2e  db %.2e  dX %.2e",
          stat_rel(&sk), stat_rel(&sb), stat_rel(&sx));
}

/* ---- softmax-xent numerical corners --------------------------------------
 * uniform logits -> uniform softmax -> loss == log(classes); single class
 * (n=1, classes=1) -> loss 0, gradient 0; huge symmetric logits stay finite. */
static void test_softmax_edges(void) {
    enum { NU = 3, CU = 4 };
    float lu[NU * CU], du[NU * CU];
    int32_t yu[NU] = { 0, 2, 3 };
    for (int i = 0; i < NU * CU; i++) lu[i] = 2.5f;   /* all equal per row */
    const float Lu = tk_softmax_xent_f32(lu, yu, du, NU, CU);
    int grad_ok = 1;
    for (int s = 0; s < NU; s++)
        for (int j = 0; j < CU; j++) {
            const float want = (0.25f - (j == yu[s] ? 1.0f : 0.0f)) / NU;
            if (fabsf(du[s * CU + j] - want) > 1e-6f) grad_ok = 0;
        }
    check_ok(fabsf(Lu - logf((float)CU)) < 1e-5f && grad_ok,
             "uniform logits: loss %.5f == log(%d) %.5f, dlogits exact",
             Lu, CU, logf((float)CU));

    float l1[1] = { 3.7f }, d1[1] = { -9.0f };
    int32_t y1[1] = { 0 };
    const float L1 = tk_softmax_xent_f32(l1, y1, d1, 1, 1);
    check_ok(L1 == 0.0f && d1[0] == 0.0f,
             "single class n=1: loss 0, gradient 0 (got %.3g / %.3g)",
             (double)L1, (double)d1[0]);

    float big[2 * 3] = { 1e4f, -1e4f, 0.0f, -1e4f, 1e4f, 5e3f };
    int32_t yb[2] = { 0, 1 };
    float dbg[6];
    const float Lb = tk_softmax_xent_f32(big, yb, dbg, 2, 3);
    int fin = isfinite(Lb);
    for (int i = 0; i < 6; i++) fin &= isfinite(dbg[i]);
    check_ok(fin && Lb >= 0.0f, "logits +-1e4 stay finite (loss %.4f)", Lb);
}

/* ---- degenerate dense-batch shapes: n=1 and out_dim=1 --------------------
 * The chunk/GEMM boundaries differ at n=1 and out_dim=1; gradient-check both
 * against central differences (compact inline FD). */
static float linb_loss(const float *W, const float *X, const float *b,
                       const float *coef, int ns, int od, int id,
                       tk_activation_t act) {
    float Y[64];
    tk_linear_forward_batch_f32(W, X, b, NULL, Y, ns, od, id, act);
    float L = 0.0f;
    for (int i = 0; i < ns * od; i++) L += coef[i] * Y[i];
    return L;
}

static void check_linb_shape(int ns, int od, int id, const char *name) {
    float W[64], X[64], b[16], Z[64], Y[64], coef[64], dW[64], db[16], dX[64];
    tk_rng rng = tk_rng_seed(202);
    fill_rand(W, (size_t)od * id, &rng, 1.0f);
    fill_rand(X, (size_t)ns * id, &rng, 2.0f);
    fill_rand(b, od, &rng, 1.0f);
    fill_rand(coef, (size_t)ns * od, &rng, 2.0f);

    tk_linear_forward_batch_f32(W, X, b, Z, Y, ns, od, id, TK_ACT_TANH);
    tk_linear_backward_batch_f32(W, X, Z, coef, dW, db, dX, ns, od, id,
                                 TK_ACT_TANH);
    const float h = 1e-3f;
    grad_stat sw = { 0.0f, 1e-6f }, sb = sw, sx = sw;
    for (int k = 0; k < od * id; k++) {
        const float w0 = W[k];
        W[k] = w0 + h; const float Lp = linb_loss(W, X, b, coef, ns, od, id, TK_ACT_TANH);
        W[k] = w0 - h; const float Lm = linb_loss(W, X, b, coef, ns, od, id, TK_ACT_TANH);
        W[k] = w0;
        stat_add(&sw, (Lp - Lm) / (2.0f * h), dW[k]);
    }
    for (int k = 0; k < ns * id; k++) {
        const float x0 = X[k];
        X[k] = x0 + h; const float Lp = linb_loss(W, X, b, coef, ns, od, id, TK_ACT_TANH);
        X[k] = x0 - h; const float Lm = linb_loss(W, X, b, coef, ns, od, id, TK_ACT_TANH);
        X[k] = x0;
        stat_add(&sx, (Lp - Lm) / (2.0f * h), dX[k]);
    }
    check_ok(stat_rel(&sw) < 1e-2f && stat_rel(&sx) < 1e-2f,
             "gradcheck linear_batch %-14s dW %.2e  dX %.2e",
             name, stat_rel(&sw), stat_rel(&sx));
    (void)sb;
}

static void test_linear_batch_degenerate(void) {
    check_linb_shape(1, 4, 5, "n=1");
    check_linb_shape(4, 1, 6, "out_dim=1");
    check_linb_shape(1, 1, 7, "n=1 out_dim=1");
}

/* ---- max pool covering the whole input -----------------------------------
 * pool == in_h == in_w -> a single 1x1 output per plane == that plane's max,
 * argmax the flat index of the max, and backward routes the full incoming
 * gradient to exactly that element. */
static void test_maxpool_whole_input(void) {
    enum { N = 1, C = 2, IH = 3, IW = 3 };
    float X[N * C * IH * IW] = {
        0.1f, 0.2f, 0.3f,  0.4f, 0.9f, 0.5f,  0.6f, 0.7f, 0.8f,   /* max 0.9 @4 */
        8.0f, 1.0f, 2.0f,  3.0f, 4.0f, 5.0f,  6.0f, 7.0f, 0.0f,   /* max 8.0 @0 */
    };
    float Y[N * C], dX[N * C * IH * IW];
    float dY[N * C] = { 2.0f, -3.0f };
    int32_t am[N * C];
    const int oh = tk_conv2d_out_dim(IH, IH, IH, 0);
    const int ow = tk_conv2d_out_dim(IW, IW, IW, 0);
    check_ok(oh == 1 && ow == 1, "whole-input pool -> %dx%d (want 1x1)", oh, ow);

    tk_maxpool2d_f32(X, Y, am, N, C, IH, IW, IH, IW);
    check_ok(Y[0] == 0.9f && am[0] == 4 && Y[1] == 8.0f && am[1] == 0,
             "whole-input pool takes the plane max (%.1f@%d, %.1f@%d)",
             Y[0], am[0], Y[1], am[1]);

    tk_maxpool2d_backward_f32(dY, am, dX, N, C, IH, IW, oh, ow);
    int routed = 1;
    for (int i = 0; i < N * C * IH * IW; i++) {
        const int plane = i / (IH * IW), off = i % (IH * IW);
        const float want = (off == am[plane]) ? dY[plane] : 0.0f;
        if (dX[i] != want) routed = 0;
    }
    check_ok(routed, "whole-input pool backward routes grad to the max only");
}

/* ---- fixed-thread reduction determinism ----------------------------------
 * conv.c documents dK as bit-reproducible at a fixed MANTISSA_THREADS (the
 * per-worker partials are reduced in a worker order fixed by pool width, not
 * by finish order) and the forward as reduction-order-stable. A shape large
 * enough to dispatch the threaded path must therefore give bit-identical
 * results on a repeat call. This runs under both the default pool and
 * MANTISSA_THREADS=1 (Makefile gate), so at default it exercises the
 * multi-worker reduction. */
static void test_thread_determinism(void) {
    enum { N = 8, IC = 8, IH = 12, IW = 12, OC = 16, KH = 3, KW = 3 };
    const int stride = 1, pad = 1;
    const int oh = tk_conv2d_out_dim(IH, KH, stride, pad);
    const int ow = tk_conv2d_out_dim(IW, KW, stride, pad);
    const size_t xsz = (size_t)N * IC * IH * IW;
    const size_t ksz = (size_t)OC * IC * KH * KW;
    const size_t ysz = (size_t)N * OC * oh * ow;

    float *X = malloc(xsz * sizeof(float)), *K = malloc(ksz * sizeof(float));
    float *b = malloc(OC * sizeof(float));
    float *Z = malloc(ysz * sizeof(float)), *Y = malloc(ysz * sizeof(float));
    float *Y2 = malloc(ysz * sizeof(float)), *dY = malloc(ysz * sizeof(float));
    float *dK = malloc(ksz * sizeof(float)), *db = malloc(OC * sizeof(float));
    float *dX = malloc(xsz * sizeof(float));
    float *dK2 = malloc(ksz * sizeof(float)), *db2 = malloc(OC * sizeof(float));
    float *dX2 = malloc(xsz * sizeof(float));
    if (!X || !K || !b || !Z || !Y || !Y2 || !dY || !dK || !db || !dX
        || !dK2 || !db2 || !dX2) {
        check_ok(0, "thread determinism: alloc failed");
        goto out;
    }
    tk_rng rng = tk_rng_seed(303);
    fill_rand(X, xsz, &rng, 2.0f);
    fill_rand(K, ksz, &rng, 0.5f);
    fill_rand(b, OC, &rng, 0.5f);
    fill_rand(dY, ysz, &rng, 1.0f);

    tk_conv2d_forward_f32(X, K, b, Z, Y, N, IC, IH, IW, OC, KH, KW,
                          stride, pad, TK_ACT_TANH);
    tk_conv2d_forward_f32(X, K, b, Z, Y2, N, IC, IH, IW, OC, KH, KW,
                          stride, pad, TK_ACT_TANH);
    check_ok(memcmp(Y, Y2, ysz * sizeof(float)) == 0,
             "conv forward bit-identical across repeat calls (T=%d)",
             tk_num_threads());

    tk_conv2d_backward_f32(X, K, Z, dY, dK, db, dX, N, IC, IH, IW,
                           OC, KH, KW, stride, pad, TK_ACT_TANH);
    tk_conv2d_backward_f32(X, K, Z, dY, dK2, db2, dX2, N, IC, IH, IW,
                           OC, KH, KW, stride, pad, TK_ACT_TANH);
    check_ok(memcmp(dK, dK2, ksz * sizeof(float)) == 0
          && memcmp(db, db2, OC * sizeof(float)) == 0
          && memcmp(dX, dX2, xsz * sizeof(float)) == 0,
          "conv backward dK/db/dX bit-reproducible at fixed threads (T=%d)",
          tk_num_threads());
out:
    free(X); free(K); free(b); free(Z); free(Y); free(Y2); free(dY);
    free(dK); free(db); free(dX); free(dK2); free(db2); free(dX2);
}

int main(void) {
    printf("CNN primitive contract / edge tests (float32):\n");
    test_out_dim_contract();
    test_dense_equals_1x1_conv();
    test_softmax_edges();
    test_linear_batch_degenerate();
    test_maxpool_whole_input();
    test_thread_determinism();
    printf("\n%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
