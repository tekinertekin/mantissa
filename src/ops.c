#include "ops.h"

float tk_dot(const tk_scalar_t *restrict a, const tk_scalar_t *restrict b, int n) {
    /* Four independent accumulators break the loop-carried dependency on the FP
     * adder, letting the CPU pipeline/vectorize the FMAs instead of stalling on
     * each add's latency. The compiler folds mul+add into FMA under
     * -ffp-contract=fast. */
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += TK_TO_FLOAT(a[i + 0]) * TK_TO_FLOAT(b[i + 0]);
        s1 += TK_TO_FLOAT(a[i + 1]) * TK_TO_FLOAT(b[i + 1]);
        s2 += TK_TO_FLOAT(a[i + 2]) * TK_TO_FLOAT(b[i + 2]);
        s3 += TK_TO_FLOAT(a[i + 3]) * TK_TO_FLOAT(b[i + 3]);
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) s += TK_TO_FLOAT(a[i]) * TK_TO_FLOAT(b[i]);
    return s;
}

void tk_linear_forward(const tk_scalar_t *restrict W,
                       const tk_scalar_t *restrict x,
                       const tk_scalar_t *restrict bias,
                       float *restrict y,
                       int out_dim, int in_dim,
                       tk_activation_t act) {
    for (int o = 0; o < out_dim; o++) {
        float z = tk_dot(W + (size_t)o * in_dim, x, in_dim);
        if (bias) z += TK_TO_FLOAT(bias[o]);
        y[o] = tk_act_scalar(z, act);
    }
}
