#ifndef MANTISSA_OPS_H
#define MANTISSA_OPS_H

#include "dtypes.h"
#include "activations.h"
#include "tk_export.h"

/* Core linear-algebra primitives. A dense layer (matrix-vector product + bias
 * + activation) is the shared building block of every model this library
 * targets: an MLP stacks it, an RNN/GRU/LSTM applies it per timestep, a
 * Transformer uses it for the Q/K/V and feed-forward projections. Building it
 * once, well, is the whole point of mantissa. */

/* Dot product of two stored vectors. Inputs are narrow (tk_scalar_t); the
 * accumulator is float32. Narrow-store / wide-accumulate is the standard
 * mixed-precision recipe (Micikevicius et al., 2017) and keeps error bounded
 * over the millions of terms in a large layer. */
TK_API float tk_dot(const tk_scalar_t *restrict a,
                    const tk_scalar_t *restrict b, int n);

/* y = activation(W x + bias), the dense-layer forward pass.
 *   W    : out_dim x in_dim, row-major
 *   x    : in_dim
 *   bias : out_dim, or NULL to disable (runtime per-layer bias control)
 *   y    : out_dim, float32 output
 * W and x are stored narrow; y is float32 so it can feed the next layer or be
 * requantized by the caller. */
TK_API void tk_linear_forward(const tk_scalar_t *restrict W,
                              const tk_scalar_t *restrict x,
                              const tk_scalar_t *restrict bias,
                              float *restrict y,
                              int out_dim, int in_dim,
                              tk_activation_t act);

/* Narrow n floats into the configured storage type (dst[i] = FROM_FLOAT(src[i])).
 * A caller doing repeated inference narrows W once with this, then calls the fast
 * tk_linear_forward -- avoiding tk_linear_forward_f32's re-quantization of every
 * weight on each call. dst is tk_scalar_t*; size it with tk_scalar_size(). */
TK_API void tk_quantize(const float *restrict src, tk_scalar_t *restrict dst, int n);

/* Same forward pass, but all buffers are plain float32. Each value is quantized
 * through the configured storage type before the multiply, so the caller sees
 * that type's numerical behavior while passing ordinary floats. This is the
 * entry point the Python binding uses, so the library works from Python
 * whatever dtype it was compiled for. */
TK_API void tk_linear_forward_f32(const float *restrict W,
                                  const float *restrict x,
                                  const float *restrict bias,
                                  float *restrict y,
                                  int out_dim, int in_dim,
                                  tk_activation_t act);

#endif /* MANTISSA_OPS_H */
