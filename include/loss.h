#ifndef MANTISSA_LOSS_H
#define MANTISSA_LOSS_H

#include "tk_export.h"

typedef enum {
    TK_LOSS_MSE = 0,   /* mean squared error */
    TK_LOSS_BCE = 1    /* binary cross-entropy (expects sigmoid outputs) */
} tk_loss_t;

/* Returns the scalar loss over the length-n vectors and writes the seed
 * gradient dL/dy into `grad` (length n) -- the start of back-propagation. */
TK_API float tk_loss(const float *y, const float *target,
                     float *grad, int n, tk_loss_t which);

#endif /* MANTISSA_LOSS_H */
