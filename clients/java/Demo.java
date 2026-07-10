// Use the downloaded mantissa shared library from Java via JNA.
//
//   javac -cp jna.jar Demo.java
//   java  -cp .:jna.jar -Djna.library.path=../../dist Demo
//
// Shows a forward pass (OR-gate perceptron) and a simple back-prop training loop.
import com.sun.jna.Library;
import com.sun.jna.Native;

public class Demo {
    // JNA maps "mantissa" -> libmantissa.so / libmantissa.dylib / mantissa.dll.
    // Point -Djna.library.path at the folder holding the downloaded library.
    public interface Mantissa extends Library {
        String tk_dtype_name();
        void tk_linear_forward_f32(float[] W, float[] x, float[] bias, float[] y,
                                   int outDim, int inDim, int act);
        float tk_train_step_f32(float[] W, float[] bias, float[] x, float[] target,
                                int outDim, int inDim, int act, float lr);
    }

    public static void main(String[] args) {
        Mantissa m = Native.load("mantissa", Mantissa.class);
        System.out.println("backend dtype: " + m.tk_dtype_name());

        // Forward: 2-input OR gate (activation ids: IDENTITY=0, STEP=1)
        float[] W = {1, 1}, bias = {-0.5f};
        float[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        System.out.println("OR perceptron:");
        for (float[] xin : inputs) {
            float[] y = new float[1];
            m.tk_linear_forward_f32(W, xin, bias, y, 1, 2, 1);
            System.out.printf("  (%.0f,%.0f) -> %.0f%n", xin[0], xin[1], y[0]);
        }

        // Back-prop: learn a linear neuron mapping [1,2,3] -> 14
        float[] Wt = {0, 0, 0}, bt = {0};
        float[] x = {1, 2, 3}, target = {14};
        System.out.println("training a linear neuron:");
        for (int s = 0; s <= 200; s++) {
            float loss = m.tk_train_step_f32(Wt, bt, x, target, 1, 3, 0, 0.01f);
            if (s % 50 == 0) System.out.printf("  step %3d  loss %.6f%n", s, loss);
        }
        float[] y = new float[1];
        m.tk_linear_forward_f32(Wt, x, bt, y, 1, 3, 0);
        System.out.printf("final prediction %.3f (target 14)%n", y[0]);
    }
}
