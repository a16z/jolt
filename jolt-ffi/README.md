# Jolt FFI

C Foreign Function Interface (FFI) bindings for the Jolt zkVM prover.

Integrate Jolt zero-knowledge proofs into any language that can call C functions (C, C++, Swift, iOS, Python with ctypes, etc.).

## Quick Start

One script does everything:

```bash
cd jolt-ffi/examples/c
./prepare_and_run.sh fibonacci-guest
```

This will:
1. ✅ Build the Fibonacci guest program as a RISC-V ELF
2. ✅ Generate preprocessing data
3. ✅ Build the C FFI library
4. ✅ Compile the C example
5. ✅ Optionally run the example and generate a proof

## Available Guest Programs

Try any of these:
- `fibonacci-guest` - Fibonacci calculator
- `sha2-guest` - SHA2 hashing
- `sha3-guest` - SHA3 hashing
- `merkle-tree-guest` - Merkle tree operations
- `collatz-guest` - Collatz conjecture
- `btreemap-guest` - BTreeMap operations

## Building

Build just the FFI library:

```bash
cargo build --release -p jolt-ffi
```

This generates:
- Static library: `target/release/libjolt_ffi.a`
- C header: `target/release/jolt-ffi.h`

## C API Overview

### Simple Example

```c
#include "jolt-ffi.h"

int main() {
    // 1. Load preprocessing
    JoltProverPreprocessingHandle* prep =
        jolt_prover_preprocessing_load("fibonacci-preprocessing.bin");

    // 2. Read ELF file
    uint8_t* elf = read_file("fibonacci.elf", &elf_size);

    // 3. Create prover from ELF
    JoltCpuProverHandle* prover = jolt_cpu_prover_gen_from_elf(
        prep,
        elf, elf_size,
        inputs, inputs_len,
        NULL, 0,  // untrusted advice (optional)
        NULL, 0   // trusted advice (optional)
    );

    // 4. Generate proof
    jolt_cpu_prover_prove(prover, "proof.bin");

    // 5. Cleanup
    jolt_prover_preprocessing_free(prep);

    return 0;
}
```

### API Functions

**Preprocessing:**
```c
// Load preprocessing from file
JoltProverPreprocessingHandle* jolt_prover_preprocessing_load(const char* path);

// Save preprocessing to file
int jolt_prover_preprocessing_save(JoltProverPreprocessingHandle* prep, const char* path);

// Free preprocessing
void jolt_prover_preprocessing_free(JoltProverPreprocessingHandle* prep);
```

**Prover:**
```c
// Create prover from ELF file
JoltCpuProverHandle* jolt_cpu_prover_gen_from_elf(
    JoltProverPreprocessingHandle* preprocessing,
    const uint8_t* elf_contents, size_t elf_len,
    const uint8_t* inputs, size_t inputs_len,
    const uint8_t* untrusted_advice, size_t untrusted_advice_len,
    const uint8_t* trusted_advice, size_t trusted_advice_len
);

// Generate proof and save to file (consumes prover)
int jolt_cpu_prover_prove(JoltCpuProverHandle* prover, const char* output_path);
```

**Error Handling:**
```c
// Get last error message (returns NULL if no error)
const char* jolt_last_error();
```

- Functions returning pointers return `NULL` on error
- Functions returning `int` return `0` on success, `-1` on error
- Always check return values and call `jolt_last_error()` for details

## Manual Workflow (Advanced)

If you want to prepare guest programs separately:

### Step 1: Prepare Guest Program

```bash
cargo run -p jolt-ffi --bin prepare-guest -- \
    --guest fibonacci-guest \
    --elf-output fibonacci.elf \
    --preprocessing-output fibonacci-preprocessing.bin
```

Options:
- `--guest <NAME>` - Guest program name (required)
- `--elf-output <PATH>` - ELF output path (default: guest.elf)
- `--preprocessing-output <PATH>` - Preprocessing output path (default: preprocessing.bin)
- `--max-trace-length <SIZE>` - Max trace length, power of 2 (default: 65536)

### Step 2: Build C Example

```bash
cd jolt-ffi/examples/c
make
```

### Step 3: Run

```bash
./jolt_example <elf_file> <proof_output> <preprocessing_file>
```

## Linking to Your C/C++ Project

Add to your build:

**macOS:**
```makefile
CFLAGS = -I/path/to/jolt/target/release
LDFLAGS = -L/path/to/jolt/target/release
LIBS = -ljolt_ffi -lm -liconv \
       -framework CoreFoundation \
       -framework Security \
       -framework SystemConfiguration
```

**Linux:**
```makefile
CFLAGS = -I/path/to/jolt/target/release
LDFLAGS = -L/path/to/jolt/target/release
LIBS = -ljolt_ffi -lm -ldl -lpthread
```

## Memory Management

- **Preprocessing**: Call `jolt_prover_preprocessing_free()` when done. Can be reused across provers.
- **Prover**: Automatically freed by `jolt_cpu_prover_prove()`. Only call `jolt_cpu_prover_free()` if you create but don't prove.
- **Input Buffers**: Caller owns all input byte arrays. Safe to free after function returns.
- **Error Messages**: Thread-local, no need to free.

## Architecture Details

- **Field**: BN254 Fr (ark-bn254)
- **Commitment Scheme**: Dory
- **Transcript**: Blake2b
- **ISA**: RV64IMAC

## Files in This Directory

- `src/lib.rs` - Main FFI implementation
- `src/bin/prepare_guest.rs` - Utility to prepare guest programs
- `examples/c/` - Complete C example with Makefile
- `examples/c/prepare_and_run.sh` - One-command demo script

## Platform Support

- ✅ **macOS** (Apple Silicon & Intel) - Fully tested
- ✅ **Linux** - Should work (may need additional system libs)
- ❓ **Windows** - Not yet tested

## Troubleshooting

**"library 'jolt_ffi' not found"**
- Run `cargo build -p jolt-ffi --lib` first
- Check that `libjolt_ffi.a` exists in `target/debug/` or `target/release/`

**"Failed to build guest"**
- Ensure guest name is correct (use `./prepare_and_run.sh` without args to see list)
- Verify RISC-V toolchain: `rustup target add riscv64imac-unknown-none-elf`

**Preprocessing takes too long**
- First time is slow (compiling dependencies)
- Preprocessing is a one-time cost per guest program
- Reduce `--max-trace-length` if your program has short traces

## Examples

See [examples/c/](examples/c/) for:
- Complete C program (`example.c`)
- Working Makefile
- Helper script that automates everything

## License

Same as parent Jolt project.
