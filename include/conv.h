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

/* Max pooling forward, no padding, FLOOR semantics: out = (in - pool)/stride
 * + 1, so a ragged right/bottom edge narrower than the window is dropped.
 *   Y      : n x c x oh x ow
 *   argmax : same shape, int32: flat index (h*in_w + w) of the winner within
 *            its channel plane -- exactly what the backward scatter needs. */
TK_API void tk_maxpool2d_f32(const float *X, float *Y, int32_t *argmax,
                             int n, int c, int in_h, int in_w,
                             int pool, int stride);

/* Max pooling backward: zeroes dX (callee-side), then scatters dY through
 * argmax. Overlapping windows (stride < pool) accumulate. */
TK_API void tk_maxpool2d_backward_f32(const float *dY, const int32_t *argmax,
                                      float *dX,
                                      int n, int c, int in_h, int in_w,
                                      int out_h, int out_w);

/* Dense layer forward, batched float32 (the CNN head): Y = act(X @ W^T + bias).
 *   X : n x in_dim,  W : out_dim x in_dim (row-major),  Y/Z : n x out_dim.
 * Z (pre-activation) may be NULL. */
TK_API void tk_linear_forward_batch_f32(const float *W, const float *X,
                                        const float *bias, float *Z, float *Y,
                                        int n, int out_dim, int in_dim, int act);

/* Dense layer backward, batched float32:
 *   dW : out_dim x in_dim, summed over the batch
 *   db : out_dim, summed, or NULL
 *   dX : n x in_dim, or NULL */
TK_API void tk_linear_backward_batch_f32(const float *W, const float *X,
                                         const float *Z, const float *dY,
                                         float *dW, float *db, float *dX,
                                         int n, int out_dim, int in_dim, int act);

/* Softmax + cross-entropy, fused and numerically stable (row-max
 * subtraction): logits n x classes, labels n int32 class ids. Writes
 * dlogits = (softmax - onehot)/n and returns the mean loss. */
TK_API float tk_softmax_xent_f32(const float *logits, const int32_t *labels,
                                 float *dlogits, int n, int classes);

/* Plain SGD on a float32 buffer: W -= lr * dW, n elements. (tk_sgd_step is
 * the narrow-storage update; the CNN family trains pure float32.) */
TK_API void tk_sgd_update_f32(float *W, const float *dW, int n, float lr);

#endif /* MANTISSA_CONV_H */
