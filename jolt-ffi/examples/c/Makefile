# Makefile for Jolt FFI C Example

# Paths
JOLT_ROOT = ../../..
TARGET_DIR = $(JOLT_ROOT)/target
DEBUG_DIR = $(TARGET_DIR)/debug
HEADER = $(DEBUG_DIR)/jolt-ffi.h

# Compiler settings
CC = clang
CFLAGS = -Wall -Wextra -I$(DEBUG_DIR)
LDFLAGS = -L$(DEBUG_DIR)
LIBS = -ljolt_ffi -lm -liconv -framework CoreFoundation -framework Security -framework SystemConfiguration

# Output
TARGET = jolt_example

.PHONY: all clean run

all: $(TARGET)

# Build the Rust FFI library first
$(DEBUG_DIR)/libjolt_ffi.a:
	@echo "Building jolt-ffi library..."
	cd $(JOLT_ROOT) && cargo build -p jolt-ffi --lib

# Build the C example
$(TARGET): example.c $(DEBUG_DIR)/libjolt_ffi.a
	@echo "Compiling C example..."
	$(CC) $(CFLAGS) example.c -o $(TARGET) $(LDFLAGS) $(LIBS)
	@echo "✓ Built $(TARGET) successfully"

clean:
	rm -f $(TARGET)
	rm -f *.o

help:
	@echo "Jolt FFI C Example Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build the example (default)"
	@echo "  clean   - Remove built files"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make"
	@echo "  ./$(TARGET) <elf_file> <proof_output> <preprocessing_file>"
