/* A 3-layer MLP forward pass where every layer is configured differently:
 *
 *   layer 1:  4 -> 6,  with bias,    tanh
 *   layer 2:  6 -> 5,  NO bias,      relu
 *   layer 3:  5 -> 2,  with bias,    sigmoid
 *
 * This shows the core API supports heterogeneous ("mixed") architectures: bias
 * is a per-call NULL-able pointer and the activation is a per-call argument, so
 * each layer picks its own. The same pattern stacks into any depth and is what
 * an MLP / Transformer feed-forward block is made of.
 *
 * Weights here are arbitrary constants (no training). Build: `make mlp`. */
#include <stdio.h>
#include "ops.h"

/* Fill n weights with a small deterministic pattern, quantized to storage. */
static void fill(tk_scalar_t *w, int n, float base) {
    for (int i = 0; i < n; i++) w[i] = TK_FROM_FLOAT(base + 0.1f * (float)(i % 7) - 0.3f);
}

int main(void) {
    tk_scalar_t W1[6 * 4], b1[6], W2[5 * 6], W3[2 * 5], b3[2];
    float x[4] = { 0.5f, -1.0f, 2.0f, 0.25f };
    float h1[6], h2[5], y[2];

    fill(W1, 6 * 4, 0.2f);  fill(b1, 6, 0.0f);
    fill(W2, 5 * 6, -0.1f);
    fill(W3, 2 * 5, 0.15f);  fill(b3, 2, 0.05f);

    tk_scalar_t xq[4];
    for (int i = 0; i < 4; i++) xq[i] = TK_FROM_FLOAT(x[i]);

    printf("MLP forward  (dtype=%s)\n", tk_dtype_name());

    tk_scalar_t h1q[6], h2q[5];
    tk_linear_forward(W1, xq,  b1,   h1, 6, 4, TK_ACT_TANH);      /* bias + tanh    */
    for (int i = 0; i < 6; i++) h1q[i] = TK_FROM_FLOAT(h1[i]);
    tk_linear_forward(W2, h1q, NULL, h2, 5, 6, TK_ACT_RELU);      /* no bias + relu */
    for (int i = 0; i < 5; i++) h2q[i] = TK_FROM_FLOAT(h2[i]);
    tk_linear_forward(W3, h2q, b3,   y,  2, 5, TK_ACT_SIGMOID);   /* bias + sigmoid */

    printf("  output: [% .4f, % .4f]\n", y[0], y[1]);
    return 0;
}
