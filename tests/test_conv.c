/* Gradient check for the CNN primitives (conv.h): analytic dK/db/dX vs a
 * central finite-difference of a scalar loss, plus maxpool backward, the
 * batched dense layer, and softmax-xent. Compiled at float32 (see Makefile
 * `testconv`) because finite differences need the precision. Loss is a fixed
 * random weighted sum L = sum_i coef[i] * Y[i], so dY = coef and the check
 * exercises the full act(conv(...)) composition. Same tolerance style as
 * test_backprop.c: max |numeric - analytic| over the gradient's scale. */
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "conv.h"
#include "backprop.h"   /* tk_rng */

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

/* Track max |num - ana| and the gradient scale (test_backprop.c's metric). */
typedef struct { float max_err, scale; } grad_stat;
static void stat_add(grad_stat *st, float num, float ana) {
    const float e = fabsf(num - ana);
    if (e > st->max_err) st->max_err = e;
    if (fabsf(ana) > st->scale) st->scale = fabsf(ana);
}
static float stat_rel(const grad_stat *st) { return st->max_err / st->scale; }

/* ---- conv2d ---------------------------------------------------------------
 * For each config: forward, analytic backward with dY = coef, then central
 * differences on every element of K, bias and X. */
typedef struct {
    int n, in_c, in_h, in_w, out_c, kh, kw, stride, pad;
    tk_activation_t act;
    const char *name;
} conv_cfg;

static float conv_loss(const float *X, const float *K, const float *b,
                       const float *coef, float *Y, const conv_cfg *cf,
                       size_t ysz) {
    tk_conv2d_forward_f32(X, K, b, NULL, Y, cf->n, cf->in_c, cf->in_h,
                          cf->in_w, cf->out_c, cf->kh, cf->kw,
                          cf->stride, cf->pad, cf->act);
    float L = 0.0f;
    for (size_t i = 0; i < ysz; i++) L += coef[i] * Y[i];
    return L;
}

static void check_conv(const conv_cfg *cf) {
    const int oh = tk_conv2d_out_dim(cf->in_h, cf->kh, cf->stride, cf->pad);
    const int ow = tk_conv2d_out_dim(cf->in_w, cf->kw, cf->stride, cf->pad);
    const size_t xsz = (size_t)cf->n * cf->in_c * cf->in_h * cf->in_w;
    const size_t ksz = (size_t)cf->out_c * cf->in_c * cf->kh * cf->kw;
    const size_t ysz = (size_t)cf->n * cf->out_c * oh * ow;

    float *X = malloc(xsz * sizeof(float)), *K = malloc(ksz * sizeof(float));
    float *b = malloc((size_t)cf->out_c * sizeof(float));
    float *Z = malloc(ysz * sizeof(float)), *Y = malloc(ysz * sizeof(float));
    float *coef = malloc(ysz * sizeof(float));
    float *dK = malloc(ksz * sizeof(float));
    float *db = malloc((size_t)cf->out_c * sizeof(float));
    float *dX = malloc(xsz * sizeof(float));
    if (!X || !K || !b || !Z || !Y || !coef || !dK || !db || !dX) {
        check_ok(0, "conv %s: alloc failed", cf->name);
        goto out;
    }
    tk_rng rng = tk_rng_seed(13);
    fill_rand(X, xsz, &rng, 2.0f);
    fill_rand(K, ksz, &rng, 1.0f);
    fill_rand(b, cf->out_c, &rng, 1.0f);
    fill_rand(coef, ysz, &rng, 2.0f);

    /* relu is non-differentiable at z == 0: nudge the offending channel's
     * bias until every pre-activation clears the kink by >> the FD step. */
    if (cf->act == TK_ACT_RELU) {
        const int npatch = (int)(ysz / ((size_t)cf->n * cf->out_c));
        for (int it = 0; it < 200; it++) {
            tk_conv2d_forward_f32(X, K, b, Z, Y, cf->n, cf->in_c, cf->in_h,
                                  cf->in_w, cf->out_c, cf->kh, cf->kw,
                                  cf->stride, cf->pad, TK_ACT_IDENTITY);
            size_t worst = 0;
            for (size_t i = 1; i < ysz; i++)
                if (fabsf(Z[i]) < fabsf(Z[worst])) worst = i;
            if (fabsf(Z[worst]) > 0.05f) break;
            b[(worst / npatch) % cf->out_c] += 0.11f;
        }
    }

    tk_conv2d_forward_f32(X, K, b, Z, Y, cf->n, cf->in_c, cf->in_h, cf->in_w,
                          cf->out_c, cf->kh, cf->kw, cf->stride, cf->pad,
                          cf->act);
    if (cf->act == TK_ACT_RELU) {
        float zmin = 1e9f;
        for (size_t i = 0; i < ysz; i++)
            if (fabsf(Z[i]) < zmin) zmin = fabsf(Z[i]);
        check_ok(zmin > 0.05f, "conv %s: relu z clear of kink (min |z| %.3f)",
                 cf->name, zmin);
    }
    tk_conv2d_backward_f32(X, K, Z, coef, dK, db, dX, cf->n, cf->in_c,
                           cf->in_h, cf->in_w, cf->out_c, cf->kh, cf->kw,
                           cf->stride, cf->pad, cf->act);

    const float h = 1e-3f;
    grad_stat sk = { 0.0f, 1e-6f }, sb = sk, sx = sk;
    for (size_t k = 0; k < ksz; k++) {
        const float k0 = K[k];
        K[k] = k0 + h; const float Lp = conv_loss(X, K, b, coef, Y, cf, ysz);
        K[k] = k0 - h; const float Lm = conv_loss(X, K, b, coef, Y, cf, ysz);
        K[k] = k0;
        stat_add(&sk, (Lp - Lm) / (2.0f * h), dK[k]);
    }
    for (int k = 0; k < cf->out_c; k++) {
        const float b0 = b[k];
        b[k] = b0 + h; const float Lp = conv_loss(X, K, b, coef, Y, cf, ysz);
        b[k] = b0 - h; const float Lm = conv_loss(X, K, b, coef, Y, cf, ysz);
        b[k] = b0;
        stat_add(&sb, (Lp - Lm) / (2.0f * h), db[k]);
    }
    for (size_t k = 0; k < xsz; k++) {
        const float x0 = X[k];
        X[k] = x0 + h; const float Lp = conv_loss(X, K, b, coef, Y, cf, ysz);
        X[k] = x0 - h; const float Lm = conv_loss(X, K, b, coef, Y, cf, ysz);
        X[k] = x0;
        stat_add(&sx, (Lp - Lm) / (2.0f * h), dX[k]);
    }
    check_ok(stat_rel(&sk) < 1e-2f && stat_rel(&sb) < 1e-2f
          && stat_rel(&sx) < 1e-2f,
          "gradcheck conv %-22s dK %.2e  db %.2e  dX %.2e",
          cf->name, stat_rel(&sk), stat_rel(&sb), stat_rel(&sx));
out:
    free(X); free(K); free(b); free(Z); free(Y); free(coef);
    free(dK); free(db); free(dX);
}

/* Forward against an independent naive direct conv (no im2col). */
static void test_conv_vs_naive(void) {
    enum { N = 2, IC = 3, IH = 6, IW = 5, OC = 4, KH = 3, KW = 2 };
    const int stride = 2, pad = 1;
    const int oh = tk_conv2d_out_dim(IH, KH, stride, pad);
    const int ow = tk_conv2d_out_dim(IW, KW, stride, pad);
    float X[N * IC * IH * IW], K[OC * IC * KH * KW], b[OC];
    float *Y = malloc((size_t)N * OC * oh * ow * sizeof(float));
    tk_rng rng = tk_rng_seed(29);
    fill_rand(X, sizeof X / sizeof *X, &rng, 2.0f);
    fill_rand(K, sizeof K / sizeof *K, &rng, 1.0f);
    fill_rand(b, OC, &rng, 1.0f);
    tk_conv2d_forward_f32(X, K, b, NULL, Y, N, IC, IH, IW, OC, KH, KW,
                          stride, pad, TK_ACT_IDENTITY);
    float max_err = 0.0f;
    for (int s = 0; s < N; s++)
        for (int oc = 0; oc < OC; oc++)
            for (int oy = 0; oy < oh; oy++)
                for (int ox = 0; ox < ow; ox++) {
                    float z = b[oc];
                    for (int c = 0; c < IC; c++)
                        for (int ky = 0; ky < KH; ky++)
                            for (int kx = 0; kx < KW; kx++) {
                                const int iy = oy * stride - pad + ky;
                                const int ix = ox * stride - pad + kx;
                                if (iy < 0 || iy >= IH || ix < 0 || ix >= IW) continue;
                                z += K[((oc * IC + c) * KH + ky) * KW + kx]
                                   * X[((s * IC + c) * IH + iy) * IW + ix];
                            }
                    const float got = Y[((s * OC + oc) * oh + oy) * ow + ox];
                    if (fabsf(got - z) > max_err) max_err = fabsf(got - z);
                }
    check_ok(max_err < 1e-4f, "conv forward == naive direct (max err %.2e)",
             max_err);
    free(Y);
}

/* ---- maxpool: floor semantics, argmax scatter, FD through the pool ------- */
static void test_maxpool(void) {
    enum { N = 2, C = 3, IH = 5, IW = 7 };            /* ragged both edges */
    const int pool = 2, stride = 2;
    const int oh = tk_conv2d_out_dim(IH, pool, stride, 0);
    const int ow = tk_conv2d_out_dim(IW, pool, stride, 0);
    check_ok(oh == 2 && ow == 3, "maxpool floor semantics: 5x7/2 -> %dx%d "
             "(want 2x3, ragged edge dropped)", oh, ow);

    float X[N * C * IH * IW], Y[N * C * 2 * 3], coef[N * C * 2 * 3];
    float dX[N * C * IH * IW];
    int32_t am[N * C * 2 * 3];
    tk_rng rng = tk_rng_seed(31);
    fill_rand(X, sizeof X / sizeof *X, &rng, 2.0f);
    fill_rand(coef, sizeof coef / sizeof *coef, &rng, 2.0f);
    tk_maxpool2d_f32(X, Y, am, N, C, IH, IW, pool, stride);

    int ok = 1;                                        /* Y == X[argmax] */
    for (int pl = 0; pl < N * C; pl++)
        for (int i = 0; i < oh * ow; i++)
            ok &= Y[pl * oh * ow + i] == X[pl * IH * IW + am[pl * oh * ow + i]];
    check_ok(ok, "maxpool argmax points at the winning element");

    tk_maxpool2d_backward_f32(coef, am, dX, N, C, IH, IW, oh, ow);
    const float h = 1e-4f;                             /* small: don't flip winners */
    grad_stat st = { 0.0f, 1e-6f };
    for (size_t k = 0; k < sizeof X / sizeof *X; k++) {
        const float x0 = X[k];
        float Lp = 0.0f, Lm = 0.0f, Yt[N * C * 2 * 3];
        int32_t amt[N * C * 2 * 3];
        X[k] = x0 + h;
        tk_maxpool2d_f32(X, Yt, amt, N, C, IH, IW, pool, stride);
        for (size_t i = 0; i < sizeof Yt / sizeof *Yt; i++) Lp += coef[i] * Yt[i];
        X[k] = x0 - h;
        tk_maxpool2d_f32(X, Yt, amt, N, C, IH, IW, pool, stride);
        for (size_t i = 0; i < sizeof Yt / sizeof *Yt; i++) Lm += coef[i] * Yt[i];
        X[k] = x0;
        stat_add(&st, (Lp - Lm) / (2.0f * h), dX[k]);
    }
    check_ok(stat_rel(&st) < 1e-2f, "gradcheck maxpool dX %.2e", stat_rel(&st));

    /* overlapping windows (stride < pool) accumulate in the scatter */
    enum { OH3 = 2, OW3 = 2 };                         /* 3x3, pool 2, stride 1 */
    float X3[9] = { 0, 1, 2, 3, 9, 5, 6, 7, 8 };       /* center wins all 4 */
    float Y3[OH3 * OW3], dY3[OH3 * OW3] = { 1, 1, 1, 1 }, dX3[9];
    int32_t am3[OH3 * OW3];
    tk_maxpool2d_f32(X3, Y3, am3, 1, 1, 3, 3, 2, 1);
    tk_maxpool2d_backward_f32(dY3, am3, dX3, 1, 1, 3, 3, OH3, OW3);
    check_ok(dX3[4] == 4.0f && dX3[0] == 0.0f,
             "maxpool overlap accumulates (center got %.0f, want 4)", dX3[4]);
}

/* ---- batched dense head --------------------------------------------------- */
enum { LB_NS = 3, LB_OD = 4, LB_ID = 5 };

static float linb_loss(const float *W, const float *X, const float *b,
                       const float *coef, tk_activation_t act) {
    float Yt[LB_NS * LB_OD];
    tk_linear_forward_batch_f32(W, X, b, NULL, Yt, LB_NS, LB_OD, LB_ID, act);
    float L = 0.0f;
    for (int i = 0; i < LB_NS * LB_OD; i++) L += coef[i] * Yt[i];
    return L;
}

static void check_linear_batch(tk_activation_t act, const char *name) {
    enum { NS = LB_NS, OD = LB_OD, ID = LB_ID };
    float W[OD * ID], X[NS * ID], b[OD], Z[NS * OD], Y[NS * OD];
    float coef[NS * OD], dW[OD * ID], db[OD], dX[NS * ID];
    tk_rng rng = tk_rng_seed(17);
    fill_rand(W, OD * ID, &rng, 1.0f);
    fill_rand(X, NS * ID, &rng, 2.0f);
    fill_rand(b, OD, &rng, 1.0f);
    fill_rand(coef, NS * OD, &rng, 2.0f);

    if (act == TK_ACT_RELU) {       /* same kink-clearing nudge as check_conv */
        for (int it = 0; it < 200; it++) {
            tk_linear_forward_batch_f32(W, X, b, Z, Y, NS, OD, ID,
                                        TK_ACT_IDENTITY);
            int worst = 0;
            for (int i = 1; i < NS * OD; i++)
                if (fabsf(Z[i]) < fabsf(Z[worst])) worst = i;
            if (fabsf(Z[worst]) > 0.05f) break;
            b[worst % OD] += 0.11f;
        }
    }

    tk_linear_forward_batch_f32(W, X, b, Z, Y, NS, OD, ID, act);
    if (act == TK_ACT_RELU) {
        float zmin = 1e9f;
        for (int i = 0; i < NS * OD; i++)
            if (fabsf(Z[i]) < zmin) zmin = fabsf(Z[i]);
        check_ok(zmin > 0.05f, "linear_batch relu z clear of kink (%.3f)", zmin);
    }
    tk_linear_backward_batch_f32(W, X, Z, coef, dW, db, dX, NS, OD, ID, act);

    const float h = 1e-3f;
    grad_stat sw = { 0.0f, 1e-6f }, sb = sw, sx = sw;
    for (int k = 0; k < OD * ID; k++) {
        const float w0 = W[k];
        W[k] = w0 + h; const float Lp = linb_loss(W, X, b, coef, act);
        W[k] = w0 - h; const float Lm = linb_loss(W, X, b, coef, act);
        W[k] = w0;
        stat_add(&sw, (Lp - Lm) / (2.0f * h), dW[k]);
    }
    for (int k = 0; k < OD; k++) {
        const float b0 = b[k];
        b[k] = b0 + h; const float Lp = linb_loss(W, X, b, coef, act);
        b[k] = b0 - h; const float Lm = linb_loss(W, X, b, coef, act);
        b[k] = b0;
        stat_add(&sb, (Lp - Lm) / (2.0f * h), db[k]);
    }
    for (int k = 0; k < NS * ID; k++) {
        const float x0 = X[k];
        X[k] = x0 + h; const float Lp = linb_loss(W, X, b, coef, act);
        X[k] = x0 - h; const float Lm = linb_loss(W, X, b, coef, act);
        X[k] = x0;
        stat_add(&sx, (Lp - Lm) / (2.0f * h), dX[k]);
    }
    check_ok(stat_rel(&sw) < 1e-2f && stat_rel(&sb) < 1e-2f
          && stat_rel(&sx) < 1e-2f,
          "gradcheck linear_batch %-8s dW %.2e  db %.2e  dX %.2e",
          name, stat_rel(&sw), stat_rel(&sb), stat_rel(&sx));
}

/* ---- softmax + cross-entropy: dlogits vs FD of the returned loss --------- */
static void test_softmax_xent(void) {
    enum { NS = 4, CL = 5 };
    float logits[NS * CL], dl[NS * CL], tmp[NS * CL];
    int32_t labels[NS] = { 0, 3, 4, 2 };
    tk_rng rng = tk_rng_seed(23);
    fill_rand(logits, NS * CL, &rng, 6.0f);           /* wide range: stability */

    const float L = tk_softmax_xent_f32(logits, labels, dl, NS, CL);
    check_ok(isfinite(L) && L > 0.0f, "softmax_xent loss finite (%.4f)", L);

    float rowsum_err = 0.0f;      /* softmax rows sum to 1 => dlogits rows to 0 */
    for (int s = 0; s < NS; s++) {
        float rs = 0.0f;
        for (int j = 0; j < CL; j++) rs += dl[s * CL + j];
        if (fabsf(rs) > rowsum_err) rowsum_err = fabsf(rs);
    }
    check_ok(rowsum_err < 1e-6f, "softmax_xent dlogits rows sum to 0 (%.2e)",
             rowsum_err);

    const float h = 1e-3f;
    grad_stat st = { 0.0f, 1e-6f };
    for (int k = 0; k < NS * CL; k++) {
        const float l0 = logits[k];
        logits[k] = l0 + h;
        const float Lp = tk_softmax_xent_f32(logits, labels, tmp, NS, CL);
        logits[k] = l0 - h;
        const float Lm = tk_softmax_xent_f32(logits, labels, tmp, NS, CL);
        logits[k] = l0;
        stat_add(&st, (Lp - Lm) / (2.0f * h), dl[k]);
    }
    check_ok(stat_rel(&st) < 1e-2f, "gradcheck softmax_xent dlogits %.2e",
             stat_rel(&st));

    /* stability: huge logits must not overflow to inf/nan */
    float big[2 * 2] = { 1000.0f, -1000.0f, 500.0f, 499.0f };
    int32_t lb[2] = { 0, 1 };
    float dbig[4];
    const float Lb = tk_softmax_xent_f32(big, lb, dbig, 2, 2);
    check_ok(isfinite(Lb) && isfinite(dbig[0]) && isfinite(dbig[3]),
             "softmax_xent stable at logit 1000 (loss %.4f)", Lb);
}

static void test_sgd_update(void) {
    float W[3] = { 1.0f, -2.0f, 0.5f }, dW[3] = { 0.5f, -1.0f, 0.0f };
    tk_sgd_update_f32(W, dW, 3, 0.1f);
    check_ok(fabsf(W[0] - 0.95f) < 1e-7f && fabsf(W[1] + 1.9f) < 1e-7f
          && W[2] == 0.5f, "sgd_update_f32 matches closed form");
}

/* SGD sanity: one conv layer + softmax head must reduce the loss. */
static void test_conv_training_reduces_loss(void) {
    enum { N = 4, IC = 1, IH = 6, IW = 6, OC = 3, KH = 3, KW = 3, CL = 3 };
    const int oh = 4, ow = 4, FD = OC * oh * ow;
    float X[N * IC * IH * IW], K[OC * IC * KH * KW], kb[OC];
    float Z[N * FD], Y[N * FD];
    float W[CL * FD], wb[CL], Zl[N * CL], logits[N * CL], dlog[N * CL];
    float dK[OC * IC * KH * KW], dkb[OC], dW[CL * FD], dwb[CL], dY[N * FD];
    int32_t labels[N] = { 0, 1, 2, 1 };
    tk_rng rng = tk_rng_seed(5);
    fill_rand(X, N * IC * IH * IW, &rng, 2.0f);
    fill_rand(K, OC * IC * KH * KW, &rng, 0.5f);
    fill_rand(W, CL * FD, &rng, 0.2f);
    memset(kb, 0, sizeof kb); memset(wb, 0, sizeof wb);

    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 30; step++) {
        tk_conv2d_forward_f32(X, K, kb, Z, Y, N, IC, IH, IW, OC, KH, KW,
                              1, 0, TK_ACT_TANH);
        tk_linear_forward_batch_f32(W, Y, wb, Zl, logits, N, CL, FD,
                                    TK_ACT_IDENTITY);
        const float L = tk_softmax_xent_f32(logits, labels, dlog, N, CL);
        if (step == 0) first = L;
        last = L;
        tk_linear_backward_batch_f32(W, Y, Zl, dlog, dW, dwb, dY, N, CL, FD,
                                     TK_ACT_IDENTITY);
        tk_conv2d_backward_f32(X, K, Z, dY, dK, dkb, NULL, N, IC, IH, IW,
                               OC, KH, KW, 1, 0, TK_ACT_TANH);
        tk_sgd_update_f32(W, dW, CL * FD, 0.1f);
        tk_sgd_update_f32(wb, dwb, CL, 0.1f);
        tk_sgd_update_f32(K, dK, OC * IC * KH * KW, 0.1f);
        tk_sgd_update_f32(kb, dkb, OC, 0.1f);
    }
    check_ok(last < first, "conv+softmax SGD reduced loss %.4f -> %.4f",
             first, last);
}

int main(void) {
    printf("Conv/pool/dense-batch gradient check (float32, central differences):\n");

    /* stride 1/2, pad 0/1/2, kh != kw, non-square inputs, three activations.
     * relu inputs get a bias nudge so no z sits on the z==0 kink (checked). */
    const conv_cfg cfgs[] = {
        { 2, 2, 5, 6, 3, 3, 2, 1, 0, TK_ACT_TANH,     "s1 p0 3x2 tanh" },
        { 1, 1, 7, 5, 2, 3, 3, 2, 1, TK_ACT_IDENTITY, "s2 p1 3x3 ident" },
        { 2, 2, 6, 4, 2, 2, 3, 2, 2, TK_ACT_TANH,     "s2 p2 2x3 tanh" },
        { 2, 3, 4, 5, 4, 3, 3, 1, 1, TK_ACT_RELU,     "s1 p1 3x3 relu" },
        { 1, 2, 5, 5, 3, 1, 1, 1, 0, TK_ACT_TANH,     "s1 p0 1x1 tanh" },
    };
    for (size_t i = 0; i < sizeof cfgs / sizeof *cfgs; i++) check_conv(&cfgs[i]);
    test_conv_vs_naive();

    test_maxpool();

    check_linear_batch(TK_ACT_IDENTITY, "identity");
    check_linear_batch(TK_ACT_TANH,     "tanh");
    check_linear_batch(TK_ACT_RELU,     "relu");

    test_softmax_xent();
    test_sgd_update();
    test_conv_training_reduces_loss();

    printf("\n%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
