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
cargo run --release -p backtrace --no-default-features --features nostd

# Run std demo
cargo run --release -p backtrace --no-default-features --features std
```

## Expected Output

```
nostd panicked: true
nostd proof valid: true
```

The `panicked: true` indicates Jolt successfully detected the guest panic through `JoltCommitments::has_panic()`.

## Backtrace Output Reference

The `JOLT_BACKTRACE` environment variable controls how much information is shown when a guest panics. Below are the outputs for every combination of guest mode and backtrace level.

### no_std guest

#### No `JOLT_BACKTRACE` (default)

```bash
cargo run --release -p backtrace --no-default-features --features nostd
```

```
PANIC: panicked at examples/backtrace/guest-nostd/src/lib.rs:33:5: backtrace demo (no-std)
ERROR tracer: Guest program terminated due to panic after 3696 cycles.
  <no backtrace available>
note: run `trace_and_analyze` with `JOLT_BACKTRACE=1` environment variable to enable backtraces
```

Only the panic location and cycle count are shown. No call stack.

#### `JOLT_BACKTRACE=1`

```bash
JOLT_BACKTRACE=1 cargo run --release -p backtrace --no-default-features --features nostd
```

```
PANIC: panicked at examples/backtrace/guest-nostd/src/lib.rs:33:5: backtrace demo (no-std)
ERROR tracer: Guest program terminated due to panic after 3696 cycles.
Guest Program panicked in "core::fmt::write" at ".../library/core/src/fmt/mod.rs:1471:9"
stack backtrace:
   0: 0x80000b38 - core::fmt::write
                               at .../library/core/src/fmt/mod.rs:1471:9
   ...
  15: 0x800017f4 - __rustc::rust_begin_unwind
  16: 0x800003c2 - core::panicking::panic_fmt
                               at .../library/core/src/panicking.rs:75:14
  17: 0x800001ea - backtrace_guest_nostd::level_two
  18: 0x80000192 - backtrace_guest_nostd::level_one
  19: 0x80000208 - main
  20: 0x80000d12 - __default_main_entry
  21: 0x80000d3c - __platform_bootstrap
  22: 0x80000034 - __bootstrap
  23: 0x8000002c - __bootstrap
  24: 0x80000018 - _start
note: run with `JOLT_BACKTRACE=full` environment variable to display extended emulator state info
```

Symbolized function names and source locations are displayed. The full call chain from `_start` through `level_one` → `level_two` → `panic_fmt` is visible.

#### `JOLT_BACKTRACE=full`

```bash
JOLT_BACKTRACE=full cargo run --release -p backtrace --no-default-features --features nostd
```

```
PANIC: panicked at examples/backtrace/guest-nostd/src/lib.rs:33:5: backtrace demo (no-std)
ERROR tracer: Guest program terminated due to panic after 3696 cycles.
Guest Program panicked in "core::fmt::write" at ".../library/core/src/fmt/mod.rs:1471:9"
stack backtrace:
   ...
  16: 0x800003c2 - core::panicking::panic_fmt
                               at .../library/core/src/panicking.rs:75:14
                   registers: ra=0x800013be, sp=0x80102fc0, gp=0x80003700, ...
                   cycle: 357
  17: 0x800001ea - backtrace_guest_nostd::level_two
                   registers: ra=0x800001e6, sp=0x80102fd0, gp=0x80003700, ...
                   cycle: 351
  18: 0x80000192 - backtrace_guest_nostd::level_one
                   registers: ra=0x8000018e, sp=0x80103030, gp=0x80003700, ...
                   cycle: 326
  19: 0x80000208 - main
                   registers: ra=0x80000204, sp=0x80103070, gp=0x80003700, ...
                   cycle: 313
  20: 0x80000d12 - __default_main_entry
                   registers: ra=0x7ffffd0e, sp=0x80103090, gp=0x80003700, ...
                   cycle: 296
  ...
  24: 0x80000018 - _start
                   registers: ra=0x80001014, sp=0x801030a0, gp=0x80003700
                   cycle: 6
```

Same as `JOLT_BACKTRACE=1`, plus non-zero register values and the cycle count at each frame.

---

### std guest

#### No `JOLT_BACKTRACE` (default)

```bash
cargo run --release -p backtrace --no-default-features --features std
```

```
thread '<unnamed>' panicked at examples/backtrace/guest-std/src/lib.rs:22:5:
backtrace demo (std)
stack backtrace:
   0:         0x800112b2 - <unknown>
   1:         0x8000180e - <unknown>
   ...
  11:         0x80034e48 - <unknown>
ERROR tracer: Guest program terminated due to panic after 463691 cycles.
  <no backtrace available>
note: run `trace_and_analyze` with `JOLT_BACKTRACE=1` environment variable to enable backtraces
```

The guest's own std backtrace fires but shows `<unknown>` because symbols are stripped in release builds. The tracer has no symbolized backtrace either.

#### `JOLT_BACKTRACE=1`

```bash
JOLT_BACKTRACE=1 cargo run --release -p backtrace --no-default-features --features std
```

```
thread '<unnamed>' panicked at examples/backtrace/guest-std/src/lib.rs:22:5:
backtrace demo (std)
stack backtrace:
   0:         0x800112b2 - <unknown>
   ...
ERROR tracer: Guest program terminated due to panic after 463691 cycles.
stack backtrace:
   0: 0x800067b2 - trap_handler
   1: 0x800001f6 - _default_trap_handler
   2: 0x800067b2 - trap_handler
   3: 0x800001f6 - _default_trap_handler
   4: 0x80037eb0 - raise
                               at .../obj_musl/../src_musl/src/signal/raise.c:9:2
   5: 0x80035280 - abort
                               at .../obj_musl/../src_musl/src/exit/abort.c:11:2
   6: 0x8000914e - panic_abort::__rust_start_panic::abort
   7: 0x8000913c - __rustc::__rust_start_panic
   8: 0x8001076e - __rustc::rust_panic
   9: 0x8000fe8c - std::panicking::rust_panic_with_hook
  10: 0x8000fe22 - std::panicking::rust_panic_with_hook
  11: 0x8000fdde - std::panicking::rust_panic_with_hook
  12: 0x800067b2 - trap_handler
  ...
note: run with `JOLT_BACKTRACE=full` environment variable to display extended emulator state info
```

The tracer's backtrace now shows symbolized function names. The guest's own `<unknown>` backtrace still appears first (from std's built-in backtrace), followed by the tracer's symbolized backtrace.

#### `JOLT_BACKTRACE=full`

```bash
JOLT_BACKTRACE=full cargo run --release -p backtrace --no-default-features --features std
```

```
...
ERROR tracer: Guest program terminated due to panic after 463691 cycles.
stack backtrace:
   0: 0x800067b2 - trap_handler
                   registers: ra=0x800001fa, sp=0x80157e00, gp=0x80056b60, ...
                   cycle: 463653
   1: 0x800001f6 - _default_trap_handler
                   registers: ra=0x800061f2, sp=0x80157e20, gp=0x80056b60, ...
                   cycle: 463628
   ...
   4: 0x80037eb0 - raise
                               at .../obj_musl/../src_musl/src/signal/raise.c:9:2
                   registers: ra=0x80035284, sp=0x80157f40, gp=0x80056b60, ...
                   cycle: 463382
   5: 0x80035280 - abort
                               at .../obj_musl/../src_musl/src/exit/abort.c:11:2
                   registers: ra=0x80009150, sp=0x80157ff0, gp=0x80056b60, ...
                   cycle: 463374
   6: 0x8000914e - panic_abort::__rust_start_panic::abort
                   registers: ra=0x8000913e, sp=0x80158030, gp=0x80056b60, ...
                   cycle: 463370
   ...
```

Same as `JOLT_BACKTRACE=1`, plus non-zero register values and cycle counts per frame.

---

### Summary

| Guest | `JOLT_BACKTRACE` | Panic location | Call stack | Symbols | Registers & cycles |
|---|---|---|---|---|---|
| nostd | *(unset)* | Yes | No | No | No |
| nostd | `1` | Yes | Yes | Yes | No |
| nostd | `full` | Yes | Yes | Yes | Yes |
| std | *(unset)* | Yes | Guest-side only (`<unknown>`) | No | No |
| std | `1` | Yes | Yes (guest `<unknown>` + tracer symbolized) | Yes | No |
| std | `full` | Yes | Yes (guest `<unknown>` + tracer symbolized) | Yes | Yes |

**Note:** `JOLT_BACKTRACE` uses the tracer's software-based call stack tracking (via JAL/JALR instructions), not DWARF-based unwinding. This works for both `std` and `nostd` guests.

---

### How `JOLT_BACKTRACE` and `backtrace = "dwarf"` interact

The guest always runs inside the tracer (emulator). There are **two independent** backtrace mechanisms at play:

#### 1. Tracer-side backtrace (controlled by `JOLT_BACKTRACE`)

The tracer **always** records a software call stack by watching every JAL/JALR instruction as it emulates the guest — this happens unconditionally, regardless of any `backtrace` setting. The call stack is stored in host memory, not inside the guest.

When the guest panics, what `JOLT_BACKTRACE` controls is **symbol resolution**:

- **Unset**: The tracer has the call stack but doesn't have access to symbols (stripped ELF) or the ELF path. It prints `<no backtrace available>`.
- **`=1`**: The `jolt build` step skips `-Cstrip=symbols`, preserving symbol/debug info in the ELF. The tracer then uses `addr2line` on the **host** to resolve call-site addresses → function names + file:line. Zero extra guest cycles.
- **`=full`**: Same as `=1`, plus prints register snapshots and cycle counts per frame.

#### 2. Guest-side backtrace (controlled by `backtrace = "dwarf"`)

This only affects **std guests**. When `backtrace = "dwarf"` is set in `#[jolt::provable]`, the build adds `-Cforce-frame-pointers=yes`, which:

- Generates frame pointer setup (`fp`/`s0`) in every function prologue
- Enables std's built-in panic handler to do **DWARF unwinding inside the guest** as emulated RISC-V instructions
- The guest writes its backtrace via `write()` syscalls, which the emulator intercepts and prints to stdout

This is the `<unknown>` output you see for std guests — it's the **guest code itself** printing a backtrace, but symbols are stripped in release builds so it can only show raw addresses. This guest-side unwinding is expensive (~446k extra cycles vs ~17k with `"off"`).

For **no_std guests**, there is no built-in panic handler that does unwinding, so `backtrace = "dwarf"` has essentially no effect.

#### Comparison

Both mechanisms run "in the emulator", but at different layers:

| | Tracer-side (`JOLT_BACKTRACE`) | Guest-side (`backtrace = "dwarf"`) |
|---|---|---|
| **What runs** | Host code (addr2line on the host CPU) | Emulated RISC-V code (DWARF unwinder in guest) |
| **When** | After guest terminates | During guest execution (part of panic handler) |
| **Modifies generated code?** | No (only preserves symbols) | Yes (adds frame pointer prologues) |
| **Extra guest cycles** | 0 | ~446k for std, negligible for no_std |
| **Affects no_std?** | Yes (full backtrace) | No (no built-in unwinder) |
| **Affects std?** | Yes (full backtrace) | Yes (enables guest-side `<unknown>` backtrace) |

#### What `backtrace = "off"` changes

With `backtrace = "off"`, the `-Cforce-frame-pointers=yes` flag is **not** set:

| | `backtrace = "dwarf"` | `backtrace = "off"` |
|---|---|---|
| **no_std** | ~3696 cycles | ~3604 cycles (slightly fewer) |
| **no_std + JOLT_BACKTRACE=1** | Full tracer backtrace | Full tracer backtrace (identical) |
| **std** | ~463k cycles, guest prints `<unknown>` frames | ~17k cycles, **no** guest backtrace at all |
| **std + JOLT_BACKTRACE=1** | Tracer backtrace + guest `<unknown>` | Tracer backtrace only (cleaner, faster) |

**Key takeaway:** For most debugging, `JOLT_BACKTRACE=1` alone is sufficient — it works regardless of the `backtrace` attribute and adds zero guest cycles. The `backtrace = "dwarf"` attribute is only needed when you want frame pointers in the generated code (e.g., for ZeroOS-level DWARF unwinding or external tooling), and it comes with a significant cycle cost for std guests.
