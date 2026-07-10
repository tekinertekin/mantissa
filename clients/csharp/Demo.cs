// Use the downloaded mantissa shared library from C# via P/Invoke.
//
//   dotnet run                         # place the library alongside (see README)
//
// Shows a forward pass (OR-gate perceptron) and a simple back-prop training loop.
using System;
using System.Runtime.InteropServices;

class Demo
{
    // Resolves to libmantissa.so (Linux), libmantissa.dylib (macOS), mantissa.dll
    // (Windows) — rename the downloaded release asset to this base name.
    const string LIB = "mantissa";

    [DllImport(LIB)] static extern IntPtr tk_dtype_name();

    [DllImport(LIB)]
    static extern void tk_linear_forward_f32(
        [In] float[] W, [In] float[] x, [In] float[] bias, [Out] float[] y,
        int outDim, int inDim, int act);

    [DllImport(LIB)]
    static extern float tk_train_step_f32(
        [In, Out] float[] W, [In, Out] float[] bias, [In] float[] x, [In] float[] target,
        int outDim, int inDim, int act, float lr);

    static void Main()
    {
        Console.WriteLine("backend dtype: " + Marshal.PtrToStringAnsi(tk_dtype_name()));

        // Forward: 2-input OR gate (activation ids: IDENTITY=0, STEP=1)
        float[] W = { 1, 1 }, bias = { -0.5f };
        float[][] inputs = { new float[] {0,0}, new float[] {0,1}, new float[] {1,0}, new float[] {1,1} };
        Console.WriteLine("OR perceptron:");
        foreach (var xin in inputs)
        {
            float[] y = new float[1];
            tk_linear_forward_f32(W, xin, bias, y, 1, 2, 1);
            Console.WriteLine($"  ({xin[0]},{xin[1]}) -> {y[0]}");
        }

        // Back-prop: learn a linear neuron mapping [1,2,3] -> 14
        float[] Wt = { 0, 0, 0 }, bt = { 0 };
        float[] xx = { 1, 2, 3 }, target = { 14 };
        Console.WriteLine("training a linear neuron:");
        for (int s = 0; s <= 200; s++)
        {
            float loss = tk_train_step_f32(Wt, bt, xx, target, 1, 3, 0, 0.01f);
            if (s % 50 == 0) Console.WriteLine($"  step {s}  loss {loss:F6}");
        }
        float[] yy = new float[1];
        tk_linear_forward_f32(Wt, xx, bt, yy, 1, 3, 0);
        Console.WriteLine($"final prediction {yy[0]:F3} (target 14)");
    }
}
