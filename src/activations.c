#include "activations.h"

void tk_activate(float *y, int n, tk_activation_t a) {
    for (int i = 0; i < n; i++) y[i] = tk_act_scalar(y[i], a);
}
