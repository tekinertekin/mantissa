# mantissa
#
# Storage type (see include/config.h; default is bfloat16):
#   DTYPE=0 float32 | 1 fp16 | 2 bfloat16 | 3 tekin32 | 4 tekin8 | 5 fp8_e5m2 | 6 fp4_e2m1
#
#   make test             build + run tests            (default dtype = bfloat16)
#   make DTYPE=4 bench     build + run benchmark with tekin8
#   make example / mlp     C perceptron / mixed-MLP demos
#   make dist              shared library for the Python binding
#   make clean

CC      := cc
DTYPE   ?= 2
# -O3 + fast FP contraction lets tk_dot fold multiply-add into FMA and vectorize.
# Add `march=native` locally for full SIMD width; kept off by default for portable builds.
CFLAGS  := -O3 -funroll-loops -ffp-contract=fast -Wall -Wextra -std=c11 \
           -Iinclude -DTK_DTYPE=$(DTYPE) -fvisibility=hidden -pthread
LDFLAGS := -lm -pthread

SRC     := src/dtypes.c src/activations.c src/ops.c src/loss.c src/backprop.c src/pool.c
BUILD   := build

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBEXT := dylib
else
    LIBEXT := so
endif

.PHONY: test testbp lib dist example mlp train bench benchbp clean

test: $(SRC) tests/test_dtypes.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/test_dtypes $^ $(LDFLAGS)
	@./$(BUILD)/test_dtypes

lib: $(SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DTK_BUILD_DLL -fPIC -shared -o $(BUILD)/libmantissa.$(LIBEXT) $^ $(LDFLAGS)
	@echo "built $(BUILD)/libmantissa.$(LIBEXT)  (dtype=$(DTYPE))"

# Prebuilt, committed shared library so the Python binding works without a
# toolchain. The same command emits libmantissa.dll on Windows and .so on Linux.
dist: $(SRC)
	@mkdir -p dist
	$(CC) $(CFLAGS) -DTK_BUILD_DLL -fPIC -shared -o dist/libmantissa.$(LIBEXT) $^ $(LDFLAGS)
	@echo "built dist/libmantissa.$(LIBEXT)  (dtype=$(DTYPE))"

example: $(SRC) examples/perceptron_example.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/perceptron_example $^ $(LDFLAGS)
	@./$(BUILD)/perceptron_example

mlp: $(SRC) examples/mlp_example.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/mlp_example $^ $(LDFLAGS)
	@./$(BUILD)/mlp_example

train: $(SRC) examples/train_xor.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/train_xor $^ $(LDFLAGS)
	@./$(BUILD)/train_xor

# Gradient check always runs at float32 (finite differences need precision).
testbp: $(SRC) tests/test_backprop.c
	@mkdir -p $(BUILD)
	$(CC) -O3 -funroll-loops -ffp-contract=fast -Wall -Wextra -std=c11 \
	      -Iinclude -DTK_DTYPE=0 -o $(BUILD)/test_backprop $^ $(LDFLAGS)
	@./$(BUILD)/test_backprop

bench: $(SRC) bench/benchmark.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/benchmark $^ $(LDFLAGS)
	@./$(BUILD)/benchmark

benchbp: $(SRC) bench/bench_backprop.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/bench_backprop $^ $(LDFLAGS)
	@./$(BUILD)/bench_backprop

# Data-layout / cache harness (SIMD tail, padding, alignment, residency).
benchlayout: $(SRC) bench/bench_layout.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $(BUILD)/bench_layout $^ $(LDFLAGS)
	@./$(BUILD)/bench_layout

# Thread-pool scaling harness. benchscale builds the standard binary;
# benchscale-cross forces TK_MT_MIN_WORK=1 so threading always engages, to map
# the serial-vs-threaded crossover. Drive thread count via MANTISSA_THREADS.
benchscale: $(SRC) bench/bench_scaling.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $(BUILD)/bench_scaling $^ $(LDFLAGS)

benchscale-cross: $(SRC) bench/bench_scaling.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -Isrc -DTK_MT_MIN_WORK=1 -o $(BUILD)/bench_scaling_cross $^ $(LDFLAGS)

clean:
	rm -rf $(BUILD)
