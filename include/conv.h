#ifndef MANTISSA_CONV_H
#define MANTISSA_CONV_H

#include <stdint.h>
#include "activations.h"
#include "tk_export.h"

/* CNN primitives: 2-D convolution, max pooling, the batched float32 dense
 * head, fused softmax + cross-entropy, and a plain float32 SGD update. The
 * training path here is pure float32 end to end (the `_f32` family, like
 * tk_train_epoch_f32) -- no storage-dtype quantization; narrow-storage conv
 * is a roadmap item. Layout is NCHW, row-major, batch outermost; all buffers
 * are caller-allocated. */

/* Output spatial size: (in + 2*pad - k) / stride + 1, clamped to >= 0.
 * Returns 0 on non-positive in/k/stride or negative pad. */
TK_API int tk_conv2d_out_dim(int in, int k, int stride, int pad);

/* Convolution forward, batched.
 *   X    : n x in_c x in_h x in_w
 *   K    : out_c x in_c x kh x kw
 *   bias : out_c, or NULL
 *   Z    : n x out_c x out_h x out_w pre-activation, or NULL if act ==
 *          TK_ACT_IDENTITY and only Y is wanted (backward needs Z)
 *   Y    : n x out_c x out_h x out_w, Y = act(Z)
 * im2col + GEMM (Chellapilla et al., 2006), one sample's patch matrix at a
 * time; accumulates in float32. */
TK_API void tk_conv2d_forward_f32(const float *X, const float *K, const float *bias,
                                  float *Z, float *Y,
                                  int n, int in_c, int in_h, int in_w,
                                  int out_c, int kh, int kw,
                                  int stride, int pad, int act);

/* Convolution backward, batched. Inputs as forward plus the saved
 * pre-activation Z and the incoming gradient dY (n x out_c x oh x ow);
 * dz = dy * act'(z), the tk_linear_backward convention. Outputs (written,
 * not accumulated into):
 *   dK : out_c x in_c x kh x kw, summed over the batch
 *   db : out_c, summed, or NULL
 *   dX : n x in_c x in_h x in_w, or NULL for the first layer */
TK_API void tk_conv2d_backward_f32(const float *X, const float *K, const float *Z,
                                   const float *dY,
                                   float *dK, float *db, float *dX,
                                   int n, int in_c, int in_h, int in_w,
                                   int out_c, int kh, int kw,
                                   int stride, int pad, int act);

#endif /* MANTISSA_CONV_H */
