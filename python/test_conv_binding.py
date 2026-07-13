"""Smoke test for the CNN-primitive bindings: conv2d forward cross-checked
against a naive numpy im2col reference, conv2d backward against numpy analytic
gradients, maxpool round-trip, batched dense, softmax-xent and sgd_update on
random data. Exact float32 equality is NOT expected across different summation
orders (C reduces 8-wide NEON chains, numpy pairwise) — np.allclose with
rtol=1e-4 is the bar. Run from python/ after `make dist`:

    /path/to/python python/test_conv_binding.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mantissa import Mantissa, IDENTITY, RELU, TANH  # noqa: E402

failures = 0


def check(ok, msg):
    global failures
    if not ok:
        failures += 1
    print(f"  [{'OK' if ok else '!!'}] {msg}")


def im2col_ref(X, kh, kw, stride, pad):
    """(n, c, h, w) -> (n, oh*ow, c*kh*kw), zero-padded patches."""
    n, c, h, w = X.shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((n, oh * ow, c * kh * kw), dtype=X.dtype)
    for oy in range(oh):
        for ox in range(ow):
            patch = Xp[:, :, oy * stride:oy * stride + kh,
                       ox * stride:ox * stride + kw]
            cols[:, oy * ow + ox, :] = patch.reshape(n, -1)
    return cols, oh, ow


def act_ref(z, act):
    return {IDENTITY: lambda v: v, RELU: lambda v: np.maximum(v, 0),
            TANH: np.tanh}[act](z)


def act_grad_ref(z, act):
    return {IDENTITY: lambda v: np.ones_like(v),
            RELU: lambda v: (v > 0).astype(v.dtype),
            TANH: lambda v: 1 - np.tanh(v) ** 2}[act](z)


def test_conv(tk, n, in_c, in_h, in_w, out_c, kh, kw, stride, pad, act, name):
    rng = np.random.default_rng(7)
    f32 = np.float32
    X = rng.standard_normal(n * in_c * in_h * in_w).astype(f32)
    K = rng.standard_normal(out_c * in_c * kh * kw).astype(f32) * f32(0.5)
    b = rng.standard_normal(out_c).astype(f32)
    oh = tk.conv2d_out_dim(in_h, kh, stride, pad)
    ow = tk.conv2d_out_dim(in_w, kw, stride, pad)
    ysz = n * out_c * oh * ow
    Z = np.empty(ysz, dtype=f32)
    Y = np.empty(ysz, dtype=f32)
    tk.conv2d_forward(X, K, b, Z, Y, n, in_c, in_h, in_w, out_c, kh, kw,
                      stride, pad, act)

    # numpy reference: im2col patches @ K^T, channel-major output
    cols, oh_r, ow_r = im2col_ref(X.reshape(n, in_c, in_h, in_w),
                                  kh, kw, stride, pad)
    assert (oh, ow) == (oh_r, ow_r)
    Zr = np.einsum("npk,ok->nop", cols, K.reshape(out_c, -1)) \
        + b[None, :, None]
    check(np.allclose(Z, Zr.ravel(), rtol=1e-4, atol=1e-5),
          f"conv2d_forward Z == numpy im2col reference ({name})")
    check(np.allclose(Y, act_ref(Zr, act).ravel(), rtol=1e-4, atol=1e-5),
          f"conv2d_forward Y == act(Z) reference ({name})")

    # backward vs numpy analytic gradients
    dY = rng.standard_normal(ysz).astype(f32)
    dK = np.empty_like(K)
    db = np.empty_like(b)
    dX = np.empty_like(X)
    tk.conv2d_backward(X, K, Z, dY, dK, db, dX, n, in_c, in_h, in_w,
                       out_c, kh, kw, stride, pad, act)
    dz = dY.reshape(n, out_c, oh * ow) * act_grad_ref(Zr, act).astype(f32)
    dK_r = np.einsum("nop,npk->ok", dz, cols).ravel()
    db_r = dz.sum(axis=(0, 2))
    dcols = np.einsum("nop,ok->npk", dz, K.reshape(out_c, -1))
    # col2im: scatter patch gradients back (loop mirrors im2col_ref)
    dX_r = np.zeros((n, in_c, in_h + 2 * pad, in_w + 2 * pad), dtype=f32)
    for oy in range(oh):
        for ox in range(ow):
            dX_r[:, :, oy * stride:oy * stride + kh,
                 ox * stride:ox * stride + kw] += \
                dcols[:, oy * ow + ox, :].reshape(n, in_c, kh, kw)
    dX_r = dX_r[:, :, pad:pad + in_h, pad:pad + in_w].ravel()
    check(np.allclose(dK, dK_r, rtol=1e-4, atol=1e-4),
          f"conv2d_backward dK == numpy reference ({name})")
    check(np.allclose(db, db_r, rtol=1e-4, atol=1e-4),
          f"conv2d_backward db == numpy reference ({name})")
    check(np.allclose(dX, dX_r, rtol=1e-4, atol=1e-4),
          f"conv2d_backward dX == numpy reference ({name})")


def test_maxpool(tk):
    rng = np.random.default_rng(11)
    n, c, ih, iw, pool, stride = 2, 3, 5, 7, 2, 2   # ragged both edges
    oh, ow = tk.conv2d_out_dim(ih, pool, stride, 0), tk.conv2d_out_dim(iw, pool, stride, 0)
    check((oh, ow) == (2, 3), f"maxpool floor semantics 5x7/2 -> {oh}x{ow}")
    X = rng.standard_normal(n * c * ih * iw).astype(np.float32)
    Y = np.empty(n * c * oh * ow, dtype=np.float32)
    am = np.empty(n * c * oh * ow, dtype=np.int32)
    tk.maxpool2d(X, Y, am, n, c, ih, iw, pool, stride)
    Xr = X.reshape(n, c, ih, iw)
    Yr = Xr[:, :, :oh * stride, :ow * stride] \
        .reshape(n, c, oh, stride, ow, stride).max(axis=(3, 5))
    check(np.array_equal(Y, Yr.ravel()), "maxpool2d Y == numpy reference")

    dY = rng.standard_normal(Y.size).astype(np.float32)
    dX = np.empty_like(X)
    tk.maxpool2d_backward(dY, am, dX, n, c, ih, iw, oh, ow)
    check(np.isclose(dX.sum(), dY.sum(), rtol=1e-5)
          and np.count_nonzero(dX) == Y.size,
          "maxpool2d_backward scatters every gradient exactly once")


def test_dense_softmax_sgd(tk):
    rng = np.random.default_rng(13)
    f32 = np.float32
    n, od, idim = 8, 10, 32
    W = rng.standard_normal(od * idim).astype(f32) * f32(0.3)
    X = rng.standard_normal(n * idim).astype(f32)
    b = rng.standard_normal(od).astype(f32)
    Z = np.empty(n * od, dtype=f32)
    Y = np.empty(n * od, dtype=f32)
    tk.linear_forward_batch(W, X, b, Z, Y, n, od, idim, TANH)
    Zr = X.reshape(n, idim) @ W.reshape(od, idim).T + b
    check(np.allclose(Z, Zr.ravel(), rtol=1e-4, atol=1e-5)
          and np.allclose(Y, np.tanh(Zr).ravel(), rtol=1e-4, atol=1e-5),
          "linear_forward_batch == numpy reference")

    dY = rng.standard_normal(n * od).astype(f32)
    dW = np.empty_like(W)
    db = np.empty_like(b)
    dX = np.empty_like(X)
    tk.linear_backward_batch(W, X, Z, dY, dW, db, dX, n, od, idim, TANH)
    dz = dY.reshape(n, od) * (1 - np.tanh(Zr) ** 2).astype(f32)
    check(np.allclose(dW, (dz.T @ X.reshape(n, idim)).ravel(), rtol=1e-4, atol=1e-4)
          and np.allclose(db, dz.sum(axis=0), rtol=1e-4, atol=1e-4)
          and np.allclose(dX, (dz @ W.reshape(od, idim)).ravel(), rtol=1e-4, atol=1e-4),
          "linear_backward_batch == numpy reference")

    logits = rng.standard_normal(n * od).astype(f32) * f32(3.0)
    labels = rng.integers(0, od, n).astype(np.int32)
    dl = np.empty(n * od, dtype=f32)
    loss = tk.softmax_xent(logits, labels, dl, n, od)
    lr = logits.reshape(n, od).astype(np.float64)
    p = np.exp(lr - lr.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    loss_r = -np.log(p[np.arange(n), labels]).mean()
    onehot = np.zeros((n, od))
    onehot[np.arange(n), labels] = 1.0
    check(np.isclose(loss, loss_r, rtol=1e-5),
          f"softmax_xent loss {loss:.5f} == numpy {loss_r:.5f}")
    check(np.allclose(dl, ((p - onehot) / n).ravel(), rtol=1e-4, atol=1e-6),
          "softmax_xent dlogits == (softmax - onehot)/n")

    W0 = W.copy()
    tk.sgd_update(W, dW, W.size, 0.01)
    check(np.allclose(W, W0 - np.float32(0.01) * dW),
          "sgd_update W -= lr*dW")


def main():
    tk = Mantissa()
    print(f"CNN binding smoke test (lib dtype={tk.dtype}; conv family is f32):")
    check(hasattr(tk, "conv2d_forward"), "feature detection: conv2d_forward")
    test_conv(tk, n=2, in_c=3, in_h=8, in_w=6, out_c=5, kh=3, kw=2,
              stride=1, pad=1, act=TANH, name="s1 p1 3x2 tanh")
    test_conv(tk, n=3, in_c=2, in_h=9, in_w=7, out_c=4, kh=3, kw=3,
              stride=2, pad=2, act=RELU, name="s2 p2 3x3 relu")
    test_conv(tk, n=1, in_c=1, in_h=28, in_w=28, out_c=6, kh=5, kw=5,
              stride=1, pad=0, act=IDENTITY, name="LeNet C1")
    test_maxpool(tk)
    test_dense_softmax_sgd(tk)
    print("FAILED" if failures else "ALL PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
