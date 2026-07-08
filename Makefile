# mantissa
#
# Storage type (see include/config.h):
#   DTYPE=0 float32 | 1 fp16 | 2 bfloat16 | 3 tekin32 | 4 tekin8
#
#   make DTYPE=2 test     build + run tests with bfloat16
#   make lib              build the shared library for the Python binding
#   make example          build + run the C perceptron example
#   make clean

CC      := cc
DTYPE   ?= 0
# -O3 + fast FP contraction lets tk_dot fold multiply-add into FMA and vectorize.
# Add `march=native` locally for full SIMD width; kept off by default for portable builds.
CFLAGS  := -O3 -funroll-loops -ffp-contract=fast -Wall -Wextra -std=c11 \
           -Iinclude -DTK_DTYPE=$(DTYPE) -fvisibility=hidden
LDFLAGS := -lm

SRC     := src/dtypes.c src/activations.c src/ops.c
BUILD   := build

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBEXT := dylib
else
    LIBEXT := so
endif

.PHONY: test lib example clean

test: $(SRC) tests/test_dtypes.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/test_dtypes $^ $(LDFLAGS)
	@./$(BUILD)/test_dtypes

lib: $(SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -DTK_BUILD_DLL -fPIC -shared -o $(BUILD)/libmantissa.$(LIBEXT) $^ $(LDFLAGS)
	@echo "built $(BUILD)/libmantissa.$(LIBEXT)  (dtype=$(DTYPE))"

example: $(SRC) examples/perceptron_example.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/perceptron_example $^ $(LDFLAGS)
	@./$(BUILD)/perceptron_example

clean:
	rm -rf $(BUILD)
