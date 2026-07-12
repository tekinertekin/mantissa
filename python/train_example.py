"""Back-propagation from Python: train a single linear neuron via the C core.

The neuron learns to map x = [1, 2, 3] to the target 14 (i.e. weights ~[1,2,3]),
using mantissa's float32 training step over the ctypes binding.

    make dist
    python3 python/train_example.py
"""
from array import array

from mantissa import Mantissa, IDENTITY

def main() -> None:
    tk = Mantissa()
    print(f"backend dtype: {tk.dtype}")

    # array('f') (or a float32 numpy array) takes the zero-copy path: C mutates
    # W/bias in place, no per-element boxing. Plain lists work too, just slower.
    W      = array("f", [0.0, 0.0, 0.0])
    bias   = array("f", [0.0])
    x      = array("f", [1.0, 2.0, 3.0])
    target = array("f", [14.0])

    print("training a linear neuron:")
    for step in range(201):
        loss = tk.train_step(W, x, target, out_dim=1, in_dim=3,
                             act=IDENTITY, lr=0.01, bias=bias)
        if step % 50 == 0:
            print(f"  step {step:3d}  loss {loss:.6f}")

    pred = tk.linear_forward(W, x, bias, out_dim=1, in_dim=3, act=IDENTITY)
    print(f"final prediction {pred[0]:.3f} (target 14)   weights "
          f"[{W[0]:.2f}, {W[1]:.2f}, {W[2]:.2f}]")


if __name__ == "__main__":
    main()
