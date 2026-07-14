"""Binding-contract tests for the Python-side plumbing in python/mantissa:
argument marshalling (_as_c_float writeback paths, zero-copy detection),
pre-bound pointers (Trainer, _bind_c_float error contracts), the Session
identity memo, Prepared-vs-linear_forward consistency, and the documented
edge cases of the CNN methods. Complements test_conv_binding.py, which
cross-checks the C kernels against numpy references; this file checks the
BINDING — that every input path (numpy / array('f') / list / wrong dtype)
reaches the same C call or fails with the promised exception, verbatim.

Everything is tiny, seeded and deterministic. Run from the repo root after
`make dist`:

    /path/to/python python/test_binding.py
"""
import sys
from array import array
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


def expect(exc_type, msg, fn, label):
    """fn() must raise exactly exc_type with exactly `msg` (the error strings
    are part of the binding's contract — callers grep for them)."""
    try:
        fn()
    except exc_type as e:
        check(str(e) == msg, f"{label}: {exc_type.__name__}: {e}")
    except Exception as e:  # noqa: BLE001 — wrong exception type is a failure
        check(False, f"{label}: wrong exception {type(e).__name__}: {e}")
    else:
        check(False, f"{label}: no exception raised")


def bf16(a):
    """Round-to-nearest-even bfloat16 quantization of float32 (numpy mirror
    of tk_q for TK_DTYPE_BFLOAT16), for an exact-value linear_forward
    reference when the library is built at the default dtype."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    round_bit = ((u >> 16) & 1) + 0x7FFF
    return ((u + round_bit) & 0xFFFF0000).view(np.float32)


# -- _as_c_float: every input path reaches the same C call -----------------------

def test_marshalling(tk):
    rng = np.random.default_rng(2)
    od, idim = 3, 4
    Wn = rng.standard_normal(od * idim).astype(np.float32)
    bn = rng.standard_normal(od).astype(np.float32)
    x = rng.standard_normal(idim).astype(np.float32)

    y_np = np.array(tk.linear_forward(Wn, x, bn, od, idim, TANH),
                    dtype=np.float32)
    y_list = np.array(tk.linear_forward(Wn.tolist(), x.tolist(), bn.tolist(),
                                        od, idim, TANH), dtype=np.float32)
    check(np.array_equal(y_np, y_list),
          "linear_forward: list inputs == numpy inputs (same C call)")

    if tk.dtype == "bfloat16":
        y_id = np.array(tk.linear_forward(Wn, x, bn, od, idim, IDENTITY),
                        dtype=np.float32)
        ref = bf16(Wn).reshape(od, idim) @ bf16(x) + bf16(bn)
        check(np.allclose(y_id, ref, rtol=1e-6, atol=1e-7),
              "linear_forward == numpy through explicit bf16 quantization")
    else:
        print(f"  [--] bf16 reference check skipped (lib dtype={tk.dtype})")

    out_list = [0.0] * od
    r = tk.linear_forward(Wn, x, bn, od, idim, TANH, out=out_list)
    check(r is out_list and np.array_equal(np.float32(out_list), y_np),
          "out= plain list: written back and the same object returned")

    out_arr = array("f", [0.0] * od)
    tk.linear_forward(array("f", Wn.tolist()), x, bn, od, idim, TANH,
                      out=out_arr)
    check(np.array_equal(np.frombuffer(out_arr, np.float32), y_np),
          "array('f') W and out=: zero-copy path, same values")

    expect(ValueError, "W: expected 12 float32 values, got 7",
           lambda: tk.linear_forward(np.zeros(7, np.float32), x, bn,
                                     od, idim, IDENTITY),
           "wrong-length W")


def test_train_step_writeback(tk):
    rng = np.random.default_rng(3)
    od, idim = 3, 4
    W0 = rng.standard_normal(od * idim).astype(np.float32)
    x = rng.standard_normal(idim).astype(np.float32)
    t = rng.standard_normal(od).astype(np.float32)

    Wn, bn = W0.copy(), np.zeros(od, dtype=np.float32)
    loss_np = tk.train_step(Wn, x, t, od, idim, TANH, 0.1, bias=bn)

    Wl, bl = W0.tolist(), [0.0] * od          # plain lists: copy in, copy back
    loss_list = tk.train_step(Wl, x, t, od, idim, TANH, 0.1, bias=bl)
    check(loss_list == loss_np
          and np.array_equal(np.float32(Wl), Wn)
          and np.array_equal(np.float32(bl), bn),
          "train_step: plain-list W/bias mutated via writeback, same loss")

    W64 = W0.astype(np.float64)               # documented silent copy fallback
    loss_64 = tk.train_step(W64, x, t, od, idim, TANH, 0.1,
                            bias=np.zeros(od, dtype=np.float32))
    check(loss_64 == loss_np and np.array_equal(W64.astype(np.float32), Wn),
          "train_step: float64 W takes the copy path, still written back")

    Wa = array("f", W0.tolist())              # array('f'): zero-copy mutation
    loss_a = tk.train_step(Wa, x, t, od, idim, TANH, 0.1,
                           bias=array("f", [0.0] * od))
    check(loss_a == loss_np and np.array_equal(np.frombuffer(Wa, np.float32), Wn),
          "train_step: array('f') W mutated in place, same trajectory")


# -- _bind_c_float: Trainer's zero-copy-only contract -----------------------------

def test_bind_errors(tk):
    ns, od, idim = 4, 2, 4
    X = np.zeros(ns * idim, dtype=np.float32)
    T = np.zeros(ns * od, dtype=np.float32)

    expect(TypeError, "W: need a C-contiguous float32 array for a pre-bound "
           "pointer (got dtype=float64)",
           lambda: tk.trainer(np.zeros(od * idim, np.float64), X, T,
                              ns, od, idim),
           "trainer float64 W")
    expect(TypeError, "W: need a C-contiguous float32 numpy array or "
           "array('f') for a pre-bound pointer",
           lambda: tk.trainer([0.0] * (od * idim), X, T, ns, od, idim),
           "trainer list W")
    # A non-contiguous float32 slice is rejected too, but the message blames
    # the dtype ("got dtype=float32") — misleading wording, pinned as-is.
    big = np.zeros(2 * od * idim, dtype=np.float32)
    expect(TypeError, "W: need a C-contiguous float32 array for a pre-bound "
           "pointer (got dtype=float32)",
           lambda: tk.trainer(big[::2], X, T, ns, od, idim),
           "trainer non-contiguous W (note: message says dtype)")
    expect(ValueError, "W: expected 8 float32 values, got 7",
           lambda: tk.trainer(np.zeros(7, np.float32), X, T, ns, od, idim),
           "trainer wrong-size W")
    expect(ValueError, "targets: expected 8 float32 values, got 4",
           lambda: tk.trainer(np.zeros(od * idim, np.float32), X,
                              np.zeros(4, np.float32), ns, od, idim),
           "trainer wrong-size targets")


def test_trainer_bit_identity(tk):
    """Trainer docstring promise: same C entry points, bit-identical weight
    trajectories — only the per-call pointer work is gone."""
    rng = np.random.default_rng(5)
    ns, od, idim = 8, 3, 4
    W0 = rng.standard_normal(od * idim).astype(np.float32)
    b0 = rng.standard_normal(od).astype(np.float32)
    X = rng.standard_normal(ns * idim).astype(np.float32)
    T = np.sign(rng.standard_normal(ns * od)).astype(np.float32)
    perm = rng.permutation(ns).astype(np.int32)

    Wt, bt = W0.copy(), b0.copy()
    Wu, bu = W0.copy(), b0.copy()
    tr = tk.trainer(Wt, X, T, ns, od, idim, bias=bt)

    rt = tr.train_epoch(TANH, 0.05, order=perm, mistakes=True)
    ru = tk.train_epoch(Wu, X, T, ns, od, idim, TANH, 0.05, bias=bu,
                        order=perm, mistakes=True)
    check(isinstance(rt, tuple) and isinstance(rt[1], int) and rt == ru,
          f"train_epoch(mistakes=True) (loss, mistakes) tuple identical {rt}")
    check(np.array_equal(Wt, Wu) and np.array_equal(bt, bu),
          "train_epoch via Trainer: bit-identical W and bias")

    lt = tr.train_epoch(TANH, 0.05)           # plain path returns a bare float
    lu = tk.train_epoch(Wu, X, T, ns, od, idim, TANH, 0.05, bias=bu)
    check(isinstance(lt, float) and lt == lu and np.array_equal(Wt, Wu),
          "train_epoch without order/mistakes: float loss, still identical")

    mt = tr.perceptron_epoch(0.1, order=perm)
    mu = tk.perceptron_epoch(Wu, X, T, ns, od, idim, 0.1, bias=bu, order=perm)
    check(mt == mu and np.array_equal(Wt, Wu) and np.array_equal(bt, bu),
          f"perceptron_epoch via Trainer: same mistakes ({mt}), identical W")


def test_trainer_margins(tk):
    rng = np.random.default_rng(6)
    ns, idim = 8, 4
    w = rng.standard_normal(idim).astype(np.float32)
    X = rng.standard_normal(ns * idim).astype(np.float32)
    t = np.sign(rng.standard_normal(ns)).astype(np.float32)

    tr = tk.trainer(w, X, t, ns, 1, idim)
    out = np.empty(ns, dtype=np.float32)
    r = tr.margins(out)
    # margins() is the same GEMV with X as the weight matrix and w as the
    # input — literally the same C call, so bit-identical, storage dtype
    # included.
    ref = np.array(tk.linear_forward(X, w, None, ns, idim, IDENTITY),
                   dtype=np.float32)
    check(r is out and np.array_equal(out, ref),
          "margins(): bit-identical to the swapped linear_forward GEMV")

    W3 = np.zeros(3 * idim, dtype=np.float32)
    tr3 = tk.trainer(W3, X, np.zeros(ns * 3, np.float32), ns, 3, idim)
    expect(ValueError, "margins() needs out_dim == 1, got 3",
           lambda: tr3.margins(out), "margins with out_dim=3")


# -- order=: zero-copy int32 vs boxed int64 ---------------------------------------

def test_order_paths(tk):
    rng = np.random.default_rng(7)
    ns, od, idim = 8, 3, 5
    W0 = rng.standard_normal(od * idim).astype(np.float32)
    X = rng.standard_normal(ns * idim).astype(np.float32)
    T = rng.standard_normal(ns * od).astype(np.float32)
    perm64 = rng.permutation(ns)              # int64: boxed into a ctypes buf

    Wa, Wb = W0.copy(), W0.copy()
    la = tk.train_epoch(Wa, X, T, ns, od, idim, TANH, 0.05,
                        order=perm64.astype(np.int32))
    lb = tk.train_epoch(Wb, X, T, ns, od, idim, TANH, 0.05, order=perm64)
    check(la == lb and np.array_equal(Wa, Wb),
          "order=: int32 zero-copy == int64 boxed, bit-identical")

    expect(ValueError, "order: expected 8 int32 indices, got 5",
           lambda: tk.train_epoch(Wa, X, T, ns, od, idim, TANH, 0.05,
                                  order=np.arange(5, dtype=np.int32)),
           "wrong-length order")


# -- Session: identity memo, rebinding, strict zero-copy errors --------------------

def test_session(tk):
    s = tk.session()

    W = np.arange(4, dtype=np.float32)
    dW = np.ones(4, dtype=np.float32)
    s.sgd_update(W, dW, 4, 0.5)
    s.sgd_update(W, dW, 4, 0.5)               # second call: memo hit
    check(np.array_equal(W, np.arange(4, dtype=np.float32) - 1.0),
          "Session sgd_update: two calls on the same buffers, correct result")
    check(s._fp(W, 4, "W") is s._fp(W, 4, "W"),
          "identity memo: same array object -> the cached pointer is reused")

    # A REPLACED array object must rebind — results land in the new storage.
    W2 = np.arange(4, dtype=np.float32)
    s.sgd_update(W2, dW, 4, 0.5)
    check(np.array_equal(W2, np.arange(4, dtype=np.float32) - 0.5),
          "replaced array object: fresh bind, mutation lands in the new array")

    expect(TypeError, "W: need a C-contiguous float32 array for a pre-bound "
           "pointer (got dtype=float64)",
           lambda: s.sgd_update(np.zeros(4, np.float64), dW, 4, 0.1),
           "Session float64")
    big = np.zeros(8, dtype=np.float32)
    expect(TypeError, "W: need a C-contiguous float32 array for a pre-bound "
           "pointer (got dtype=float32)",
           lambda: s.sgd_update(big[::2], dW, 4, 0.1),
           "Session non-contiguous")
    expect(TypeError, "W: need a C-contiguous float32 numpy array or "
           "array('f') for a pre-bound pointer",
           lambda: s.sgd_update([0.0] * 4, dW, 4, 0.1),
           "Session list (no silent copy)")
    expect(ValueError, "W: expected 4 float32 values, got 5",
           lambda: s.sgd_update(np.zeros(5, np.float32), dW, 4, 0.1),
           "Session wrong size")

    # int32 path (labels/argmax): same strictness.
    lg = np.zeros(8, dtype=np.float32)
    dl = np.zeros(8, dtype=np.float32)
    expect(TypeError, "labels: Session needs a C-contiguous int32 numpy array",
           lambda: s.softmax_xent(lg, np.zeros(2, np.int64), dl, 2, 4),
           "Session int64 labels")
    expect(ValueError, "labels: expected 2 int32 values, got 3",
           lambda: s.softmax_xent(lg, np.zeros(3, np.int32), dl, 2, 4),
           "Session wrong-size labels")


def test_session_gc(tk):
    """Weakref-memo contract: the Session must not keep dead arrays alive nor
    grow unboundedly across fresh-array-per-call patterns. Regression guard
    for two ways this was broken before: the original memo held strong
    references (every fresh inference slice retained for the model's
    lifetime), and a first weakref version stored `arr.ctypes.data_as(...)`
    pointers, whose `._arr` backref (numpy's use-after-free guard) re-pinned
    every array through the weakref anyway. The memo must hold neither."""
    import gc
    import weakref

    s = tk.session()
    dW = np.ones(4, dtype=np.float32)
    W = np.arange(4, dtype=np.float32)
    s.sgd_update(W, dW, 4, 0.5)
    wr = weakref.ref(W)
    del W
    gc.collect()
    check(wr() is None,
          "Session does not keep a deleted float32 array alive")

    lg = np.zeros(2 * 2, dtype=np.float32)
    dl = np.zeros(2 * 2, dtype=np.float32)
    lb = np.zeros(2, dtype=np.int32)
    s.softmax_xent(lg, lb, dl, 2, 2)
    wi = weakref.ref(lb)
    del lb
    gc.collect()
    check(wi() is None,
          "Session does not keep a deleted int32 array alive")

    sweep = getattr(type(s), "_SWEEP_AT", 4096)
    s2 = tk.session()                         # fresh session: exact counts
    for _ in range(sweep + 64):               # fresh array per call, all dead
        s2.sgd_update(np.zeros(4, dtype=np.float32), dW, 4, 0.1)
    gc.collect()
    check(len(s2._fmemo) <= sweep,
          f"memo stays bounded by the sweep ({len(s2._fmemo)} entries after "
          f"{sweep + 64} fresh arrays, purge threshold {sweep})")

    # array('f') buffers are accepted by Session (bindable) but are not
    # weakrefable — the documented contract is: correct, just unmemoized.
    Wa = array("f", [0.0, 1.0, 2.0, 3.0])
    s.sgd_update(Wa, dW, 4, 0.5)
    check(np.array_equal(np.frombuffer(Wa, np.float32),
                         np.arange(4, dtype=np.float32) - 0.5),
          "Session accepts array('f') (unmemoized), mutation lands")


def test_session_matches_unbound(tk):
    rng = np.random.default_rng(9)
    s = tk.session()
    n, ic, ih, iw, oc, k = 2, 2, 5, 5, 3, 3
    X = rng.standard_normal(n * ic * ih * iw).astype(np.float32)
    K = rng.standard_normal(oc * ic * k * k).astype(np.float32)
    b = rng.standard_normal(oc).astype(np.float32)
    oh = tk.conv2d_out_dim(ih, k, 1, 1)
    ysz = n * oc * oh * oh
    Zs, Ys = np.empty(ysz, np.float32), np.empty(ysz, np.float32)
    Zu, Yu = np.empty(ysz, np.float32), np.empty(ysz, np.float32)
    s.conv2d_forward(X, K, b, Zs, Ys, n, ic, ih, iw, oc, k, k, 1, 1, RELU)
    tk.conv2d_forward(X, K, b, Zu, Yu, n, ic, ih, iw, oc, k, k, 1, 1, RELU)
    check(np.array_equal(Ys, Yu) and np.array_equal(Zs, Zu),
          "Session conv2d_forward: bit-identical to the unbound method")

    logits = rng.standard_normal(4 * 3).astype(np.float32)
    labels = np.array([0, 2, 1, 2], dtype=np.int32)
    ds, du = np.empty(12, np.float32), np.empty(12, np.float32)
    ls = s.softmax_xent(logits, labels, ds, 4, 3)
    lu = tk.softmax_xent(logits, labels, du, 4, 3)
    check(ls == lu and np.array_equal(ds, du),
          "Session softmax_xent: identical loss and dlogits")


# -- Prepared: resident narrow weights --------------------------------------------

def test_prepared(tk):
    rng = np.random.default_rng(10)
    od, idim = 6, 16
    W = rng.standard_normal(od * idim).astype(np.float32) * np.float32(0.3)
    b = rng.standard_normal(od).astype(np.float32)
    x = rng.standard_normal(idim).astype(np.float32)

    p = tk.prepare(W, od, idim, bias=b)
    # Docstring promise: differs from linear_forward by at most a ULP or two
    # (same quantization, different reduction order) — hold it to that.
    for act, name in ((IDENTITY, "identity"), (TANH, "tanh"), (RELU, "relu")):
        yp = np.array(p.forward(x, act), dtype=np.float32)
        yl = np.array(tk.linear_forward(W, x, b, od, idim, act),
                      dtype=np.float32)
        check(np.allclose(yp, yl, rtol=1e-6, atol=1e-7),
              f"Prepared.forward({name}) == linear_forward within ULP envelope")

    y_relu = np.array(p.forward(x, RELU), dtype=np.float32)
    check(y_relu.min() >= 0.0, "Prepared relu output is non-negative")

    out = np.empty(od, dtype=np.float32)
    r = p.forward(x, TANH, out=out)
    check(r is out and np.array_equal(out, np.float32(p.forward(x, TANH))),
          "Prepared out=: same object returned, same values as the list path")

    p0 = tk.prepare(W, od, idim)              # bias=None
    y0 = np.array(p0.forward(x, IDENTITY), dtype=np.float32)
    yl0 = np.array(tk.linear_forward(W, x, None, od, idim, IDENTITY),
                   dtype=np.float32)
    check(np.allclose(y0, yl0, rtol=1e-6, atol=1e-7),
          "Prepared without bias == linear_forward without bias")


# -- CNN-method edge contracts ------------------------------------------------------

def test_cnn_edges(tk):
    rng = np.random.default_rng(11)

    check(tk.conv2d_out_dim(2, 5, 1, 0) == 0
          and tk.conv2d_out_dim(0, 1, 1, 0) == 0
          and tk.conv2d_out_dim(28, 5, 1, 2) == 28,
          "conv2d_out_dim: degenerate arguments clamp to 0, normal case exact")

    # Z=None with a non-identity activation: the docstring says "(identity
    # only)" but the C side null-checks Z everywhere, so Y is still exact.
    # Pinned: only BACKWARD needs Z.
    n, ic, ih, iw, oc, k = 1, 2, 5, 5, 3, 3
    X = rng.standard_normal(n * ic * ih * iw).astype(np.float32)
    K = rng.standard_normal(oc * ic * k * k).astype(np.float32)
    oh = tk.conv2d_out_dim(ih, k, 1, 1)
    ysz = n * oc * oh * oh
    Z = np.empty(ysz, np.float32)
    Y1, Y2 = np.empty(ysz, np.float32), np.empty(ysz, np.float32)
    tk.conv2d_forward(X, K, None, Z, Y1, n, ic, ih, iw, oc, k, k, 1, 1, RELU)
    tk.conv2d_forward(X, K, None, None, Y2, n, ic, ih, iw, oc, k, k, 1, 1, RELU)
    check(np.array_equal(Y1, Y2),
          "conv2d_forward Z=None with relu: Y identical to the Z-passed call")

    # SUSPICIOUS (pinned, flagged, not fixed): maxpool2d's argmax is an
    # OUTPUT, but _as_c_int32 has no writeback — any argmax that is not a
    # C-contiguous int32 numpy array is boxed and the winners written to the
    # discarded temp. int64 argmax comes back UNTOUCHED, with no error.
    Xp = rng.standard_normal(1 * 1 * 4 * 4).astype(np.float32)
    Yp = np.empty(4, np.float32)
    am64 = np.full(4, -7, dtype=np.int64)
    tk.maxpool2d(Xp, Yp, am64, 1, 1, 4, 4, 2, 2)
    check(np.all(am64 == -7),
          "maxpool2d int64 argmax: silently discarded (current behavior, flagged)")
    am32 = np.full(4, -7, dtype=np.int32)
    Yq = np.empty(4, np.float32)
    tk.maxpool2d(Xp, Yq, am32, 1, 1, 4, 4, 2, 2)
    check(np.all(am32 != -7) and np.array_equal(Yp, Yq),
          "maxpool2d int32 argmax: written (Y identical either way)")
    # ... and the int32 argmax actually drives backward:
    dY = np.ones(4, dtype=np.float32)
    dX = np.empty(16, dtype=np.float32)
    tk.maxpool2d_backward(dY, am32, dX, 1, 1, 4, 4, 2, 2)
    check(dX.sum() == 4.0 and np.count_nonzero(dX) == 4,
          "maxpool2d_backward scatters through the int32 argmax")

    tk.sgd_update(np.empty(0, np.float32), np.empty(0, np.float32), 0, 1.0)
    check(True, "sgd_update n=0 on empty buffers: no-op, no crash")
    expect(ValueError, "W: expected 0 float32 values, got 2",
           lambda: tk.sgd_update(np.zeros(2, np.float32),
                                 np.zeros(2, np.float32), 0, 1.0),
           "sgd_update n=0 with non-empty W (n must match the buffer)")


def main():
    tk = Mantissa()
    print(f"binding contract test (lib dtype={tk.dtype}):")
    test_marshalling(tk)
    test_train_step_writeback(tk)
    test_bind_errors(tk)
    test_trainer_bit_identity(tk)
    test_trainer_margins(tk)
    test_order_paths(tk)
    test_session(tk)
    test_session_gc(tk)
    test_session_matches_unbound(tk)
    test_prepared(tk)
    test_cnn_edges(tk)
    print("FAILED" if failures else "ALL PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
