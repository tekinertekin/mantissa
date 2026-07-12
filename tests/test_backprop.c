/* Gradient check for tk_linear_backward: compare the analytic dW/dx against a
 * central finite-difference of the loss. Compiled at float32 (see Makefile
 * `testbp`) because finite differences need the precision. If the analytic
 * backward matches numerical gradients, back-propagation is correct. */
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include "ops.h"
#include "loss.h"
#include "backprop.h"

enum { OUT = 3, IN = 4 };
static int failures = 0;

static void check_ok(int ok, const char *fmt, ...) {
    va_list ap;
    if (!ok) failures++;
    printf("  [%s] ", ok ? "OK" : "!!");
    va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n");
}

static float forward_loss(const tk_scalar_t *W, const tk_scalar_t *x,
                          const tk_scalar_t *b, const float *t,
                          tk_activation_t act) {
    float z[OUT], y[OUT], dy[OUT];
    tk_linear_forward(W, x, b, z, OUT, IN, TK_ACT_IDENTITY);
    for (int i = 0; i < OUT; i++) y[i] = z[i];
    tk_activate(y, OUT, act);
    return tk_loss(y, t, dy, OUT, TK_LOSS_MSE);
}

static void check_layer(tk_activation_t act, const char *name) {
    tk_scalar_t W[OUT * IN], x[IN], b[OUT];
    float t[OUT] = { 0.2f, -0.5f, 0.8f };
    tk_rng rng = tk_rng_seed(7);
    for (int i = 0; i < OUT * IN; i++) W[i] = TK_FROM_FLOAT(tk_rng_f01(&rng) - 0.5f);
    for (int i = 0; i < IN;       i++) x[i] = TK_FROM_FLOAT(tk_rng_f01(&rng) - 0.5f);
    for (int i = 0; i < OUT;      i++) b[i] = TK_FROM_FLOAT(tk_rng_f01(&rng) - 0.5f);

    /* analytic gradients */
    float z[OUT], y[OUT], dy[OUT], dW[OUT * IN], db[OUT], dx[IN];
    tk_linear_forward(W, x, b, z, OUT, IN, TK_ACT_IDENTITY);
    for (int i = 0; i < OUT; i++) y[i] = z[i];
    tk_activate(y, OUT, act);
    tk_loss(y, t, dy, OUT, TK_LOSS_MSE);
    tk_linear_backward(W, x, z, dy, dW, db, dx, OUT, IN, act);

    /* Robust metric: largest |numeric - analytic| over all components, divided
     * by the gradient's scale. Normalizing per-component would let a near-zero
     * entry's finite-difference noise dominate a false failure. */
    const float h = 1e-3f;
    float max_err = 0.0f, grad_scale = 1e-6f;

    for (int k = 0; k < OUT * IN; k++) {
        float w0 = TK_TO_FLOAT(W[k]);
        W[k] = TK_FROM_FLOAT(w0 + h); float Lp = forward_loss(W, x, b, t, act);
        W[k] = TK_FROM_FLOAT(w0 - h); float Lm = forward_loss(W, x, b, t, act);
        W[k] = TK_FROM_FLOAT(w0);
        float num = (Lp - Lm) / (2.0f * h);
        if (fabsf(num - dW[k]) > max_err)   max_err = fabsf(num - dW[k]);
        if (fabsf(dW[k])       > grad_scale) grad_scale = fabsf(dW[k]);
    }
    for (int k = 0; k < IN; k++) {
        float x0 = TK_TO_FLOAT(x[k]);
        x[k] = TK_FROM_FLOAT(x0 + h); float Lp = forward_loss(W, x, b, t, act);
        x[k] = TK_FROM_FLOAT(x0 - h); float Lm = forward_loss(W, x, b, t, act);
        x[k] = TK_FROM_FLOAT(x0);
        float num = (Lp - Lm) / (2.0f * h);
        if (fabsf(num - dx[k]) > max_err)   max_err = fabsf(num - dx[k]);
        if (fabsf(dx[k])       > grad_scale) grad_scale = fabsf(dx[k]);
    }
    float max_rel = max_err / grad_scale;

    check_ok(max_rel < 1e-2f, "gradcheck %-8s max rel error %.2e", name, max_rel);
}

static void test_sgd_reduces_loss(void) {
    /* SGD must not increase the loss over a few steps. */
    tk_scalar_t W[OUT * IN], x[IN], b[OUT];
    float t[OUT] = { 0.2f, -0.5f, 0.8f };
    tk_rng rng = tk_rng_seed(3);
    for (int i = 0; i < OUT * IN; i++) W[i] = TK_FROM_FLOAT(tk_rng_f01(&rng) - 0.5f);
    for (int i = 0; i < IN;       i++) x[i] = TK_FROM_FLOAT(tk_rng_f01(&rng) - 0.5f);
    for (int i = 0; i < OUT;      i++) b[i] = TK_FROM_FLOAT(0.0f);
    tk_optim opt = tk_optim_default(0.1f);
    float L0 = forward_loss(W, x, b, t, TK_ACT_TANH), L1 = L0;
    for (int step = 0; step < 50; step++) {
        float z[OUT], y[OUT], dy[OUT], dW[OUT * IN], db[OUT];
        tk_linear_forward(W, x, b, z, OUT, IN, TK_ACT_IDENTITY);
        for (int i = 0; i < OUT; i++) y[i] = z[i];
        tk_activate(y, OUT, TK_ACT_TANH);
        L1 = tk_loss(y, t, dy, OUT, TK_LOSS_MSE);
        tk_linear_backward(W, x, z, dy, dW, db, NULL, OUT, IN, TK_ACT_TANH);
        tk_sgd_step(W, dW, OUT * IN, &opt, &rng);
        tk_sgd_step(b, db, OUT,      &opt, &rng);
    }
    check_ok(L1 < L0, "SGD reduced loss %.4f -> %.4f", L0, L1);
}

static void test_empty_batch(void) {
    /* Empty batch/layer must yield 0, not 0/0 = NaN (guards in tk_loss and
     * tk_train_step_f32). */
    float y1[1] = { 0.5f }, t1[1] = { 0.5f }, g1[1];
    float L = tk_loss(y1, t1, g1, 0, TK_LOSS_MSE);
    check_ok((L == 0.0f) && !isnan(L), "empty-batch loss == 0 (got %g)", (double)L);

    float Wf[4] = { 0 }, xf[4] = { 0 }, tf[1] = { 0 };
    float Ls = tk_train_step_f32(Wf, NULL, xf, tf, 0, 4, TK_ACT_IDENTITY, 0.1f);
    check_ok((Ls == 0.0f) && !isnan(Ls), "empty-layer train_step == 0 (got %g)", (double)Ls);
}

static void test_bce(void) {
    /* BCE: value against the closed form, gradient against central differences,
     * and the eps-clamped extremes staying finite. */
    enum { N = 3 };
    float y[N]  = { 0.3f, 0.7f, 0.9f };
    float t[N]  = { 0.0f, 1.0f, 1.0f };
    float dy[N];
    float L    = tk_loss(y, t, dy, N, TK_LOSS_BCE);
    float want = -(logf(1.0f - 0.3f) + logf(0.7f) + logf(0.9f)) / 3.0f;
    check_ok(fabsf(L - want) < 1e-5f, "BCE loss %.6f (want %.6f)", L, want);

    const float hb = 1e-3f;
    float max_rel = 0.0f, gtmp[N];
    for (int k = 0; k < N; k++) {
        float y0 = y[k];
        y[k] = y0 + hb; float Lp = tk_loss(y, t, gtmp, N, TK_LOSS_BCE);
        y[k] = y0 - hb; float Lm = tk_loss(y, t, gtmp, N, TK_LOSS_BCE);
        y[k] = y0;
        float num = (Lp - Lm) / (2.0f * hb);
        float rel = fabsf(num - dy[k]) / (fabsf(dy[k]) + 1e-6f);
        if (rel > max_rel) max_rel = rel;
    }
    check_ok(max_rel < 1e-2f, "BCE gradcheck max rel error %.2e", max_rel);

    float ye[2] = { 0.0f, 1.0f }, te[2] = { 1.0f, 0.0f }, ge[2];
    float Le = tk_loss(ye, te, ge, 2, TK_LOSS_BCE);
    check_ok(isfinite(Le) && isfinite(ge[0]) && isfinite(ge[1]),
             "BCE extremes stay finite (loss %.3f)", Le);
}

static void test_dropout(void) {
    /* Dropout: rate 0 = identity, rate 1 = all-zero without NaN, backward
     * mirrors the forward mask, and a seed fully determines the mask. */
    enum { N = 64 };
    float y[N], y2[N], dy[N]; uint8_t m[N], m2[N];
    tk_rng r = tk_rng_seed(11);
    for (int i = 0; i < N; i++) y[i] = 1.0f + 0.01f * (float)i;
    tk_dropout_forward(y, m, N, 0.0f, &r);
    int ok = 1;
    for (int i = 0; i < N; i++) ok &= (m[i] == 1) && (y[i] == 1.0f + 0.01f * (float)i);
    check_ok(ok, "dropout rate=0 is the identity");

    for (int i = 0; i < N; i++) y[i] = 1.0f;
    tk_dropout_forward(y, m, N, 1.0f, &r);
    ok = 1;
    for (int i = 0; i < N; i++) ok &= (y[i] == 0.0f) && !isnan(y[i]);
    check_ok(ok, "dropout rate=1 zeroes without NaN");

    int kept = 0;
    r = tk_rng_seed(42);
    for (int i = 0; i < N; i++) { y[i] = 1.0f; dy[i] = 1.0f; }
    tk_dropout_forward(y, m, N, 0.5f, &r);
    tk_dropout_backward(dy, m, N, 0.5f);
    ok = 1;
    for (int i = 0; i < N; i++) {
        ok &= (y[i] == dy[i]);                          /* fwd/bwd scale identically */
        ok &= m[i] ? (y[i] == 2.0f) : (y[i] == 0.0f);   /* inverted scaling, 1/(1-rate) */
        kept += m[i];
    }
    ok &= kept > 0 && kept < N;
    r = tk_rng_seed(42);
    for (int i = 0; i < N; i++) y2[i] = 1.0f;
    tk_dropout_forward(y2, m2, N, 0.5f, &r);
    for (int i = 0; i < N; i++) ok &= (m[i] == m2[i]);  /* seeded determinism */
    check_ok(ok, "dropout mask/backward consistency (kept %d/%d)", kept, N);
}

static void test_train_epoch_matches_step(void) {
    /* tk_train_epoch_f32 must be bit-identical to per-sample tk_train_step_f32
     * calls: same weights, same bias, matching mean loss. */
    enum { NS = 8, OD = 2, ID = 5 };
    float Wa[OD * ID], Wb[OD * ID], ba[OD], bb[OD], X[NS * ID], T[NS * OD];
    for (int i = 0; i < OD * ID; i++) Wa[i] = Wb[i] = 0.05f * (float)(i % 7 - 3);
    for (int i = 0; i < OD; i++)      ba[i] = bb[i] = 0.01f * (float)i;
    for (int i = 0; i < NS * ID; i++) X[i] = 0.1f * (float)(i % 11 - 5);
    for (int i = 0; i < NS * OD; i++) T[i] = 0.2f * (float)(i % 3);

    float mean = tk_train_epoch_f32(Wa, ba, X, T, NS, OD, ID, TK_ACT_TANH, 0.05f);
    float sum = 0.0f;
    for (int s = 0; s < NS; s++)
        sum += tk_train_step_f32(Wb, bb, X + s * ID, T + s * OD,
                                 OD, ID, TK_ACT_TANH, 0.05f);
    check_ok(memcmp(Wa, Wb, sizeof Wa) == 0 && memcmp(ba, bb, sizeof ba) == 0
          && fabsf(mean - sum / NS) < 1e-6f,
          "train_epoch == per-sample train_step (mean loss %.4f)", mean);
}

static void test_l1l2(void) {
    /* L1/L2 regularization path of tk_sgd_step (float32 build: write-back is
     * exact, so the update can be checked against the closed form). */
    tk_scalar_t w[2] = { TK_FROM_FLOAT(1.0f), TK_FROM_FLOAT(-1.0f) };
    float g[2] = { 0.1f, 0.0f };
    tk_optim opt = { 0.5f, 0.001f, 0.01f, 0 };
    tk_sgd_step(w, g, 2, &opt, NULL);
    /* w0: g' = 0.1 + 0.01*1  + 0.001*(+1) = 0.111  -> 1 - 0.5*0.111  =  0.9445
     * w1: g' = 0   + 0.01*-1 + 0.001*(-1) = -0.011 -> -1 + 0.5*0.011 = -0.9945 */
    check_ok(fabsf(TK_TO_FLOAT(w[0]) - 0.9445f) < 1e-6f &&
             fabsf(TK_TO_FLOAT(w[1]) + 0.9945f) < 1e-6f,
             "SGD L1+L2 update matches the closed form");
}

int main(void) {
    printf("Backprop gradient check (float32, central differences):\n");
    check_layer(TK_ACT_TANH,    "tanh");
    check_layer(TK_ACT_SIGMOID, "sigmoid");
    check_layer(TK_ACT_RELU,    "relu");
    check_layer(TK_ACT_GELU,    "gelu");

    test_sgd_reduces_loss();
    test_empty_batch();
    test_bce();
    test_dropout();
    test_train_epoch_matches_step();
    test_l1l2();

    printf("\n%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
