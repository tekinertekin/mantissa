/* Build a single perceptron from mantissa primitives.
 *
 *   y = activation(W . x + bias)
 *
 * This models a 2-input OR gate. No training here (forward pass only); weights
 * are set by hand. Build & run: `make example`. */
#include <stdio.h>
#include "ops.h"

int main(void) {
    enum { IN = 2, OUT = 1 };

    /* Weights/bias are stored in the configured type (bfloat16 by default). */
    tk_scalar_t W[OUT * IN] = { TK_FROM_FLOAT(1.0f), TK_FROM_FLOAT(1.0f) };
#if TK_USE_BIAS
    tk_scalar_t bias[OUT]   = { TK_FROM_FLOAT(-0.5f) };
    const tk_scalar_t *bp   = bias;
#else
    const tk_scalar_t *bp   = NULL;
#endif

    const float inputs[4][IN] = { {0,0}, {0,1}, {1,0}, {1,1} };

    printf("OR perceptron  (dtype=%s)\n", tk_dtype_name());
    for (int i = 0; i < 4; i++) {
        tk_scalar_t x[IN] = { TK_FROM_FLOAT(inputs[i][0]), TK_FROM_FLOAT(inputs[i][1]) };
        float y;
        tk_linear_forward(W, x, bp, &y, OUT, IN, TK_ACT_STEP);
        printf("  (%.0f, %.0f) -> %.0f\n", inputs[i][0], inputs[i][1], y);
    }
    return 0;
}
