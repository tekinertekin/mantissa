/* Back-propagation benchmarks for the compiled storage dtype:
 *   1. dense backward (tk_linear_backward) throughput,
 *   2. SGD weight-update throughput,
 *   3. stochastic rounding vs round-to-nearest on a real training run.
 *
 * Run per dtype:  make DTYPE=2 benchbp  (SR effect is only visible on narrow
 * types; on float32 both round modes are exact). */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ops.h"
#include "loss.h"
#include "backprop.h"

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static volatile float g_sink = 0.0f;

/* Train the 2-4-1 XOR net for `epochs`, return final mean loss. */
static float train_xor(int stochastic, int epochs) {
    enum { IN = 2, H = 4, OUT = 1 };
    static const float X[4][2] = {{0,0},{0,1},{1,0},{1,1}}, T[4] = {0,1,1,0};
    tk_rng rng = tk_rng_seed(1);
    tk_scalar_t W1[H*IN], b1[H], W2[OUT*H], b2[OUT];
    for (int i = 0; i < H*IN; i++)  W1[i] = TK_FROM_FLOAT((tk_rng_f01(&rng)-0.5f)*2.0f);
    for (int i = 0; i < H; i++)     b1[i] = TK_FROM_FLOAT(0.0f);
    for (int i = 0; i < OUT*H; i++) W2[i] = TK_FROM_FLOAT((tk_rng_f01(&rng)-0.5f)*2.0f);
    b2[0] = TK_FROM_FLOAT(0.0f);
    tk_optim opt = tk_optim_default(0.5f);
    opt.stochastic = stochastic;
    float z1[H], h[H], z2[OUT], y[OUT], dy[OUT], dh[H];
    float dW1[H*IN], db1[H], dW2[OUT*H], db2[OUT], loss = 0;
    for (int e = 0; e < epochs; e++) {
        loss = 0;
        for (int s = 0; s < 4; s++) {
            tk_scalar_t xq[IN], hq[H];
            for (int i = 0; i < IN; i++) xq[i] = TK_FROM_FLOAT(X[s][i]);
            tk_linear_forward(W1, xq, b1, z1, H, IN, TK_ACT_IDENTITY);
            for (int i = 0; i < H; i++) h[i] = z1[i];
            tk_activate(h, H, TK_ACT_TANH);
            for (int i = 0; i < H; i++) hq[i] = TK_FROM_FLOAT(h[i]);
            tk_linear_forward(W2, hq, b2, z2, OUT, H, TK_ACT_IDENTITY);
            y[0] = z2[0];
            tk_activate(y, OUT, TK_ACT_SIGMOID);
            loss += tk_loss(y, &T[s], dy, OUT, TK_LOSS_MSE);
            tk_linear_backward(W2, hq, z2, dy, dW2, db2, dh, OUT, H, TK_ACT_SIGMOID);
            tk_linear_backward(W1, xq, z1, dh, dW1, db1, NULL, H, IN, TK_ACT_TANH);
            tk_sgd_step(W2, dW2, OUT*H, &opt, &rng);
            tk_sgd_step(b2, db2, OUT, &opt, &rng);
            tk_sgd_step(W1, dW1, H*IN, &opt, &rng);
            tk_sgd_step(b1, db1, H, &opt, &rng);
        }
    }
    return loss/4.0f;
}

int main(int argc, char **argv) {
    const int IN = 2048, OUT = 2048;
    int REPS = (argc > 1) ? atoi(argv[1]) : 200;
    if (REPS <= 0) REPS = 200;
    const long params = (long)IN * OUT;

    tk_scalar_t *W = malloc((size_t)params*sizeof(tk_scalar_t));
    tk_scalar_t *x = malloc((size_t)IN*sizeof(tk_scalar_t));
    float *z = malloc(OUT*sizeof(float)), *dy = malloc(OUT*sizeof(float));
    float *dW = malloc((size_t)params*sizeof(float)), *db = malloc(OUT*sizeof(float));
    float *dx = malloc(IN*sizeof(float));
    tk_rng rng = tk_rng_seed(9);
    for (long i = 0; i < params; i++) W[i] = TK_FROM_FLOAT(((float)(i%17)-8.0f)*0.05f);
    for (int i = 0; i < IN; i++) x[i] = TK_FROM_FLOAT(((float)(i%13)-6.0f)*0.1f);
    for (int i = 0; i < OUT; i++) {
        z[i] = 0.1f;
        dy[i] = 0.01f;
    }

    printf("=== mantissa backprop benchmark  (dtype=%s, %d bytes/param) ===\n",
           tk_dtype_name(), tk_scalar_size());
    printf("note: single runs lie under DVFS -- compare medians of interleaved runs\n");
    printf("layer %dx%d = %ld params\n", OUT, IN, params);

    /* --- 1. dense backward throughput --- */
    tk_linear_backward(W, x, z, dy, dW, db, dx, OUT, IN, TK_ACT_RELU);   /* warm up */
    double t0 = now_s();
    for (int r = 0; r < REPS; r++) {
        tk_linear_backward(W, x, z, dy, dW, db, dx, OUT, IN, TK_ACT_RELU);
        g_sink += dW[r%params];
    }
    double dt = now_s()-t0;
    double flops = 3.0*params*REPS;                              /* dW + dx (mul+add) */
    printf("[backward] %.3f ms/pass, %.2f GFLOP/s\n", dt/REPS*1e3, flops/dt/1e9);

    /* --- 1a. float32 training step (forward+backward+update), the binding's path --- */
    {
        float *Wf = malloc((size_t)params*sizeof(float)), *bf = malloc(OUT*sizeof(float));
        float *xf = malloc(IN*sizeof(float)), *tf = malloc(OUT*sizeof(float));
        if (Wf && bf && xf && tf) {
            for (long i = 0; i < params; i++) Wf[i] = ((float)(i%17)-8.0f)*0.05f;
            for (int i = 0; i < OUT; i++) {
                bf[i] = 0.01f;
                tf[i] = 0.02f;
            }
            for (int i = 0; i < IN; i++) xf[i] = ((float)(i%13)-6.0f)*0.1f;
            tk_train_step_f32(Wf, bf, xf, tf, OUT, IN, TK_ACT_RELU, 0.001f);  /* warm up */
            double s0 = now_s();
            for (int r = 0; r < REPS; r++) {
                g_sink += tk_train_step_f32(Wf, bf, xf, tf, OUT, IN, TK_ACT_RELU, 0.001f);
            }
            double sdt = now_s()-s0;
            double sflops = 4.0*params*REPS;   /* forward mul+add, backward mul+update */
            printf("[train_step_f32] %.3f ms/pass, %.2f GFLOP/s\n", sdt/REPS*1e3, sflops/sdt/1e9);
        }
        free(Wf);
        free(bf);
        free(xf);
        free(tf);
    }

    /* --- 2. SGD weight-update throughput --- */
    tk_optim opt = tk_optim_default(0.01f);
    tk_sgd_step(W, dW, (int)params, &opt, &rng);   /* warm up */
    t0 = now_s();
    for (int r = 0; r < REPS; r++) {
        tk_sgd_step(W, dW, (int)params, &opt, &rng);
        g_sink += TK_TO_FLOAT(W[r%params]);
    }
    dt = now_s()-t0;
    printf("[sgd_step] %.3f ms/pass  (%.0f M weights/s)\n", dt/REPS*1e3, params*REPS/dt/1e6);

    /* --- 3. stochastic rounding vs round-to-nearest: same run, final XOR loss --- */
    float rn = train_xor(0, 4000);
    float sr = train_xor(1, 4000);
    printf("\n[stochastic rounding] XOR final loss after 4000 epochs:\n");
    printf("  round-to-nearest : %.5f%s\n", rn, rn > 0.05f ? "  (stalled)" : "");
    printf("  stochastic       : %.5f%s\n", sr, sr < 0.05f ? "  (converged)" : "");

    free(W);
    free(x);
    free(z);
    free(dy);
    free(dW);
    free(db);
    free(dx);
    return 0;
}
