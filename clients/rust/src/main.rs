// Use the downloaded mantissa shared library from Rust via `libloading`.
//
//   cargo run -- ../../dist/libmantissa.dylib      # or your downloaded file
//
// Shows a forward pass (OR-gate perceptron) and a simple back-prop training loop.
use libloading::{Library, Symbol};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

fn default_lib() -> &'static str {
    if cfg!(target_os = "macos") { "libmantissa-macos-arm64.dylib" }
    else if cfg!(target_os = "windows") { "libmantissa-windows-x86_64.dll" }
    else { "libmantissa-linux-x86_64.so" }
}

// Activation ids (activations.h): IDENTITY = 0, STEP = 1.
type DtypeName = unsafe extern "C" fn() -> *const c_char;
type Forward = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, c_int, c_int, c_int);
type Train = unsafe extern "C" fn(*mut f32, *mut f32, *const f32, *const f32, c_int, c_int, c_int, f32) -> f32;

fn main() {
    let path = std::env::args().nth(1)
        .or_else(|| std::env::var("MANTISSA_LIB").ok())
        .unwrap_or_else(|| default_lib().to_string());

    unsafe {
        let lib = Library::new(&path).expect("failed to load library");
        let dtype_name: Symbol<DtypeName> = lib.get(b"tk_dtype_name").unwrap();
        let forward: Symbol<Forward> = lib.get(b"tk_linear_forward_f32").unwrap();
        let train: Symbol<Train> = lib.get(b"tk_train_step_f32").unwrap();

        println!("backend dtype: {}", CStr::from_ptr(dtype_name()).to_str().unwrap());

        // Forward: 2-input OR gate
        let w = [1.0f32, 1.0];
        let bias = [-0.5f32];
        println!("OR perceptron:");
        for xin in [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] {
            let mut y = [0.0f32];
            forward(w.as_ptr(), xin.as_ptr(), bias.as_ptr(), y.as_mut_ptr(), 1, 2, 1);
            println!("  ({},{}) -> {}", xin[0], xin[1], y[0]);
        }

        // Back-prop: learn a linear neuron mapping [1,2,3] -> 14
        let mut wt = [0.0f32; 3];
        let mut bt = [0.0f32];
        let x = [1.0f32, 2.0, 3.0];
        let target = [14.0f32];
        println!("training a linear neuron:");
        for s in 0..=200 {
            let loss = train(wt.as_mut_ptr(), bt.as_mut_ptr(), x.as_ptr(), target.as_ptr(), 1, 3, 0, 0.01);
            if s % 50 == 0 {
                println!("  step {s}  loss {loss:.6}");
            }
        }
        let mut y = [0.0f32];
        forward(wt.as_ptr(), x.as_ptr(), bt.as_ptr(), y.as_mut_ptr(), 1, 3, 0);
        println!("final prediction {:.3} (target 14)", y[0]);
    }
}
