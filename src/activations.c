#include "activations.h"
#include <math.h>

float tk_act_identity(float z) { return z; }
float tk_act_step(float z)     { return 1.0f - (float)(tk__act_f2u(z) >> 31); }
float tk_act_sign(float z)     { return copysignf(1.0f, z); }
float tk_act_relu(float z)     { return fmaxf(z, 0.0f); }
float tk_act_sigmoid(float z)  { return tk__sigmoid(z); }
float tk_act_tanh(float z)     { return tk__tanh(z); }
float tk_act_gelu(float z) {
    return tk__gelu(z);
}

tk_act_fn tk_act_resolve(tk_activation_t a) {
    /* Table indexed by the enum -> single load, no branch chain. */
    static const tk_act_fn table[TK_ACT_COUNT] = {
        tk_act_identity, tk_act_step, tk_act_sign,
        tk_act_relu, tk_act_sigmoid, tk_act_tanh, tk_act_gelu,
    };
    return (a >= 0 && a < TK_ACT_COUNT) ? table[a] : tk_act_identity;
}

void tk_activate(float *y, int n, tk_activation_t a) {
    /* Per-element inline switch, NOT a resolved function pointer: the compiler
     * vectorizes this loop, whereas an indirect call per element cannot be
     * vectorized. `make bench` shows the switch ~3x faster for cheap
     * activations. tk_act_resolve() remains for genuinely pluggable dispatch. */
    for (int i = 0; i < n; i++) y[i] = tk_act_scalar(y[i], a);
}
