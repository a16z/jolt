# Panic Detection Example

Demonstrates panic detection in Jolt zkVM guests. When a guest program panics, Jolt captures this through a panic bit mechanism, allowing the host to detect and handle panics while still generating valid proofs.

## How It Works

- **no_std guests**: Panic handler in `zeroos-runtime-nostd` calls `__platform_abort()` which sets the panic bit
- **std guests**: `std::panic!` → `abort()` → `SIGABRT` signal → ZeroOS signal handler → `__platform_abort()`

The panic location (file:line) is printed to stdout during execution. To see full stack backtraces, set `JOLT_BACKTRACE=1` (see below).

## Running

From the workspace root:

```bash
# Run no_std demo
cargo run --release -p backtrace-demo --no-default-features --features nostd

# Run std demo
cargo run --release -p backtrace-demo --no-default-features --features std
```

## Expected Output

```
nostd panicked: true
nostd proof valid: true
```

The `panicked: true` indicates Jolt successfully detected the guest panic through `JoltCommitments::has_panic()`.

## Full Stack Backtraces

To see complete call stack backtraces when a panic occurs, run with the `JOLT_BACKTRACE` environment variable:

```bash
# Basic backtrace (call stack with symbolized function names)
JOLT_BACKTRACE=1 cargo run --release -p backtrace-demo --no-default-features --features std

# Extended backtrace (includes emulator state info)
JOLT_BACKTRACE=full cargo run --release -p backtrace-demo --no-default-features --features std
```

Example backtrace output:

```
Guest Program panicked in "foundation::kfn::trap::ksyscall" at ".../zeroos-foundation/src/kfn/trap.rs:17:22"
stack backtrace:
   0: 0x800053de - foundation::kfn::trap::ksyscall
                   trap_handler at /workspaces/jolt/jolt-sdk/src/runtime/trap.rs:62:23
  ...
  14: 0x800079cc - panic_abort::__rust_start_panic::abort
  15: 0x800079be - __rustc::__rust_start_panic
  ...
  31: 0x80019a06 - std::io::Write::write_fmt
```

**Note:** `JOLT_BACKTRACE` uses the tracer's software-based call stack tracking (via JAL/JALR instructions), not DWARF-based unwinding. This works for both `std` and `nostd` guests.
