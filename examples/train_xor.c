/* Train a 2-4-1 MLP on XOR with back-propagation + SGD.
 *
 * XOR is the textbook problem a single perceptron *cannot* solve (it is not
 * linearly separable) but a hidden layer can -- so a falling loss here is
 * end-to-end proof that the backward pass and weight update are correct.
 *
 *   layer1: 2 -> 4, tanh
 *   layer2: 4 -> 1, sigmoid,  MSE loss
 *
 * Stochastic rounding is enabled so the tiny weight updates survive narrow
 * storage (e.g. bfloat16); on float32 it is a no-op. Build: `make train`. */
#include <stdio.h>
#include "ops.h"
#include "loss.h"
#include "backprop.h"

enum { IN = 2, H = 4, OUT = 1, EPOCHS = 4000 };

static const float X[4][IN] = { {0,0}, {0,1}, {1,0}, {1,1} };
static const float T[4]     = {  0,     1,     1,     0    };   /* XOR */

int main(void) {
    tk_rng rng = tk_rng_seed(12345);

    /* Parameters, stored in the configured type. Init spans [-1,1]: wide enough
     * to break the symmetric saddle where XOR otherwise stalls at output 0.5. */
    tk_scalar_t W1[H * IN], b1[H], W2[OUT * H], b2[OUT];
    for (int i = 0; i < H * IN;  i++) W1[i] = TK_FROM_FLOAT((tk_rng_f01(&rng) - 0.5f) * 2.0f);
    for (int i = 0; i < H;       i++) b1[i] = TK_FROM_FLOAT(0.0f);
    for (int i = 0; i < OUT * H; i++) W2[i] = TK_FROM_FLOAT((tk_rng_f01(&rng) - 0.5f) * 2.0f);
    for (int i = 0; i < OUT;     i++) b2[i] = TK_FROM_FLOAT(0.0f);

    tk_optim opt = tk_optim_default(0.5f);
    opt.stochastic = 1;                       /* needed for narrow storage */

    float z1[H], h[H], z2[OUT], y[OUT], dy[OUT], dh[H];
    float dW1[H * IN], db1[H], dW2[OUT * H], db2[OUT];

    printf("Training XOR  (dtype=%s, stochastic_rounding=%d)\n", tk_dtype_name(), opt.stochastic);

    for (int epoch = 0; epoch <= EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        for (int s = 0; s < 4; s++) {
            tk_scalar_t xq[IN], hq[H];
            tk_quantize(X[s], xq, IN);

            /* forward (IDENTITY gives the pre-activation z; then activate) */
            tk_linear_forward(W1, xq, b1, z1, H, IN, TK_ACT_IDENTITY);
            for (int i = 0; i < H; i++) { h[i] = z1[i]; }
            tk_activate(h, H, TK_ACT_TANH);
            tk_quantize(h, hq, H);
            tk_linear_forward(W2, hq, b2, z2, OUT, H, TK_ACT_IDENTITY);
            for (int i = 0; i < OUT; i++) { y[i] = z2[i]; }
            tk_activate(y, OUT, TK_ACT_SIGMOID);

            epoch_loss += tk_loss(y, &T[s], dy, OUT, TK_LOSS_MSE);

            /* backward */
            tk_linear_backward(W2, hq, z2, dy, dW2, db2, dh, OUT, H, TK_ACT_SIGMOID);
            tk_linear_backward(W1, xq, z1, dh, dW1, db1, NULL, H, IN, TK_ACT_TANH);

            /* update */
            tk_sgd_step(W2, dW2, OUT * H, &opt, &rng);
            tk_sgd_step(b2, db2, OUT,     &opt, &rng);
            tk_sgd_step(W1, dW1, H * IN,  &opt, &rng);
            tk_sgd_step(b1, db1, H,       &opt, &rng);
        }
        if (epoch % 1000 == 0) printf("  epoch %4d  loss %.5f\n", epoch, epoch_loss / 4.0f);
    }

    printf("predictions:\n");
    for (int s = 0; s < 4; s++) {
        tk_scalar_t xq[IN], hq[H];
        tk_quantize(X[s], xq, IN);
        tk_linear_forward(W1, xq, b1, h, H, IN, TK_ACT_TANH);
        tk_quantize(h, hq, H);
        tk_linear_forward(W2, hq, b2, y, OUT, H, TK_ACT_SIGMOID);
        printf("  (%.0f,%.0f) -> %.3f  (target %.0f)\n", X[s][0], X[s][1], y[0], T[s]);
    }
    return 0;
}
