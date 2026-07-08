"""A single perceptron in Python, computed by the C core via ctypes.

Run:
    make lib            # builds ../build/libmantissa.<ext>
    python3 python/perceptron_example.py
"""
from mantissa import Mantissa, STEP

def main() -> None:
    tk = Mantissa()
    print(f"backend dtype: {tk.dtype}")

    # OR gate: y = step(x0*1 + x1*1 - 0.5)
    W = [1.0, 1.0]          # 1 x 2, row-major
    bias = [-0.5]

    print("OR perceptron:")
    for x in ([0, 0], [0, 1], [1, 0], [1, 1]):
        y = tk.linear_forward(W, x, bias, out_dim=1, in_dim=2, act=STEP)
        print(f"  {tuple(x)} -> {int(y[0])}")


if __name__ == "__main__":
    main()
