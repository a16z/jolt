# jolt-host

Host-side guest program compilation, decoding, and tracing for Jolt.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate provides the `Program` builder for compiling, decoding, and tracing guest RISC-V programs on the host side. It takes a guest crate name, invokes `cargo build` with the appropriate RISC-V target and linker configuration, then decodes the resulting ELF into RISC-V instructions and traces execution to produce `Cycle` vectors for the proving pipeline.

The crate is independent of the proving system — it bridges the guest build toolchain and the RISC-V emulator (`tracer`) without any cryptographic dependencies.

## Public API

### Core Types

- **`Program`** -- Host-side builder for guest RISC-V programs. Configure via `set_*` methods (heap size, stack size, I/O sizes, std support, build profile), then call `build()`, `decode()`, `trace()`, or `trace_analyze()`.
- **`CycleRow`** -- Trait abstracting a single execution cycle's flag/operand extraction. Provides `circuit_flags()`, `instruction_flags()`, `lookup_table()`, and `operands()` from a traced `Cycle`.
- **`ProgramSummary`** -- Post-trace analysis: instruction frequency breakdown and trace length.

### Re-exports

These types are re-exported from `common` and `tracer` for convenience:

- **`Cycle`** / **`Instruction`** -- Execution trace types from the RISC-V emulator.
- **`JoltDevice`** / **`MemoryConfig`** -- Guest I/O device and memory layout configuration.
- **`Memory`** -- Emulator memory state.
- **`LazyTraceIterator`** -- Streaming trace iterator for large programs.

### Functions

- **`decode(elf)`** -- Decode a raw ELF byte slice into `(instructions, memory_init, entry_point)`.

### Constants

- **`DEFAULT_TARGET_DIR`** -- Default path for guest build artifacts (`/tmp/jolt-guest-targets`).

## Dependency Position

`jolt-host` depends on `common` (memory config), `tracer` (RISC-V emulation), and `jolt-instructions` (flag/table dispatch). It is used by `jolt-zkvm`.

## Feature Flags

This crate has no feature flags.

## License

MIT OR Apache-2.0
