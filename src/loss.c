#include "loss.h"
#include <math.h>

float tk_loss(const float *y, const float *target, float *grad, int n, tk_loss_t which) {
    if (n <= 0) return 0.0f;                 /* empty batch: 0, not 0/0 = NaN */
    const float inv = 1.0f / (float)n;
    float sum = 0.0f;

    if (which == TK_LOSS_BCE) {
        const float eps = 1e-7f;
        for (int i = 0; i < n; i++) {
            float p = y[i] < eps ? eps : (y[i] > 1.0f - eps ? 1.0f - eps : y[i]);
            sum    += -(target[i] * logf(p) + (1.0f - target[i]) * logf(1.0f - p));
            grad[i] = (p - target[i]) / (p * (1.0f - p)) * inv;   /* dL/dy */
        }
        return sum * inv;
    }

    /* MSE: L = mean((y - t)^2),  dL/dy = 2 (y - t) / n */
    for (int i = 0; i < n; i++) {
        float d  = y[i] - target[i];
        sum     += d * d;
        grad[i]  = 2.0f * d * inv;
    }
    return sum * inv;
}
