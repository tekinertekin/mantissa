"""End-to-end MNIST demo on the mantissa C core — one command, no ML frameworks.

Trains a linear classifier (784->10) in float32 through the core, then runs
inference with the weights quantized into the build's storage format (whatever
`make DTYPE=N` produced). Reports accuracy, timing, and the memory footprint,
and writes a grid of test digits with the predictions.

    make lib                 # or: make DTYPE=4 lib   (fp8 — 4x smaller weights)
    python examples/mnist_demo.py

Needs only numpy + matplotlib (the core itself has no dependencies).
"""
import os
import time
import urllib.request

import numpy as np

# Load the core through its Python binding (source tree or installed wheel).
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "python"))
from mantissa import Mantissa, IDENTITY

DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CACHE = os.path.join(_HERE, ".data", "mnist.npz")
NTR, NTE, IN, OUT, EPOCHS, LR = 20000, 10000, 784, 10, 120, 0.03
BYTES = {"float32": 4, "fp16": 2, "bfloat16": 2, "tekin32": 4,
         "tekin8": 1, "fp8_e5m2": 1, "fp4_e2m1": 0.5}


def load_mnist():
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        print("downloading MNIST (~11 MB, once)...")
        urllib.request.urlretrieve(DATA_URL, CACHE)
    with np.load(CACHE) as d:
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def main():
    tk = Mantissa()
    fmt = tk.dtype
    bpp = BYTES.get(fmt, 4)
    print(f"\nmantissa build storage format: {fmt}  ({bpp} bytes/param)\n")

    xtr, ytr, xte, yte = load_mnist()
    xtr = np.ascontiguousarray(xtr[:NTR].reshape(NTR, -1) / 255.0, dtype=np.float32)
    xte = np.ascontiguousarray(xte[:NTE].reshape(-1, IN) / 255.0, dtype=np.float32)
    ytr, yte = ytr[:NTR], yte[:NTE]
    mu = xtr.mean(0)
    xtr = np.ascontiguousarray(xtr - mu, dtype=np.float32)
    xte = np.ascontiguousarray(xte - mu, dtype=np.float32)

    # --- train (float32) ---
    W = np.zeros(OUT * IN, dtype=np.float32)
    bias = np.zeros(OUT, dtype=np.float32)
    T = np.zeros((NTR, OUT), dtype=np.float32); T[np.arange(NTR), ytr] = 1.0
    trn = tk.trainer(W, xtr.reshape(-1), np.ascontiguousarray(T.reshape(-1)),
                     NTR, OUT, IN, bias)
    t0 = time.time()
    for _ in range(EPOCHS):
        loss = trn.train_epoch(IDENTITY, LR)
    train_s = time.time() - t0
    print(f"trained {NTR} images x {EPOCHS} epochs in {train_s:.1f}s  (final mse {loss:.4f})")

    # --- inference at the build's storage precision (weights quantized) ---
    prep = tk.prepare(W, OUT, IN, bias)
    preds = np.empty(NTE, dtype=np.int64)
    t0 = time.time()
    for i in range(NTE):
        y = prep.forward(np.ascontiguousarray(xte[i]), IDENTITY)
        preds[i] = int(np.argmax(y))
    infer_s = time.time() - t0
    acc = (preds == yte).mean()

    model_kb = OUT * IN * bpp / 1024.0
    print(f"\n  test accuracy      {acc * 100:.2f}%  on {NTE} images")
    print(f"  inference          {NTE / infer_s:,.0f} img/s  ({infer_s * 1e6 / NTE:.1f} us/img)")
    print(f"  model footprint    {model_kb:.1f} KB  ({OUT * IN} params x {bpp} B)")
    print(f"                     {4 / bpp:.0f}x smaller than float32\n"
          if bpp < 4 else "")

    # --- visual proof: a grid of predictions ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 8, figsize=(8, 4.4))
        for ax, i in zip(axes.ravel(), range(32)):
            ax.imshow((xte[i] + mu).reshape(28, 28), cmap="gray")
            ok = preds[i] == yte[i]
            ax.set_title(str(preds[i]), color=("#2b6cb0" if ok else "#e53e3e"),
                         fontsize=10, pad=1)
            ax.axis("off")
        fig.suptitle(f"mantissa @ {fmt} — {acc*100:.1f}% on MNIST "
                     f"({model_kb:.1f} KB, {NTE/infer_s:,.0f} img/s)", fontsize=11)
        fig.tight_layout()
        out = os.path.join(_HERE, "mnist_demo.png")
        fig.savefig(out, dpi=130)
        print(f"  wrote {out}  (blue = correct, red = wrong)\n")
    except ImportError:
        print("  (install matplotlib for the prediction grid)\n")


if __name__ == "__main__":
    main()
