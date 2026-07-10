/* Gradient check for tk_linear_backward: compare the analytic dW/dx against a
 * central finite-difference of the loss. Compiled at float32 (see Makefile
 * `testbp`) because finite differences need the precision. If the analytic
 * backward matches numerical gradients, back-propagation is correct. */
#include <stdio.h>
#include <math.h>
#include "ops.h"
#include "loss.h"
#include "backprop.h"

enum { OUT = 3, IN = 4 };
static int failures = 0;

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

    int ok = max_rel < 1e-2f;
    if (!ok) failures++;
    printf("  [%s] gradcheck %-8s max rel error %.2e\n", ok ? "OK" : "!!", name, max_rel);
}

int main(void) {
    printf("Backprop gradient check (float32, central differences):\n");
    check_layer(TK_ACT_TANH,    "tanh");
    check_layer(TK_ACT_SIGMOID, "sigmoid");
    check_layer(TK_ACT_RELU,    "relu");
    check_layer(TK_ACT_GELU,    "gelu");

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
    int ok = L1 < L0;
    if (!ok) failures++;
    printf("  [%s] SGD reduced loss %.4f -> %.4f\n", ok ? "OK" : "!!", L0, L1);

    printf("\n%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
