# Jolt Memory Layout

This document describes the memory layout used by the Jolt zkVM emulator and prover.

## Overview

Jolt divides the address space into two main regions:
1. **IO Region** - Below `RAM_START_ADDRESS` (0x80000000), contains advice, inputs, outputs, and control bits
2. **RAM Region** - At and above `RAM_START_ADDRESS`, contains program code, stack, and heap

## Memory Layout Diagram

```
                              Address Space
    ┌─────────────────────────────────────────────────────┐
    │                                                     │  Higher addresses
    │                                                     │
    │  ┌───────────────────────────────────────────────┐  │
    │  │                                               │  │  memory_end
    │  │                  HEAP                         │  │
    │  │             (grows upward ↑)                  │  │
    │  │                                               │  │
    │  ├───────────────────────────────────────────────┤  │  stack_start
    │  │                                               │  │
    │  │                  STACK                        │  │
    │  │            (grows downward ↓)                 │  │
    │  │                                               │  │
    │  ├───────────────────────────────────────────────┤  │  stack_end
    │  │              STACK CANARY                     │  │
    │  ├───────────────────────────────────────────────┤  │
    │  │                                               │  │
    │  │                PROGRAM                        │  │
    │  │          (.text, .data, .bss)                 │  │
    │  │                                               │  │
    │  └───────────────────────────────────────────────┘  │  RAM_START_ADDRESS = 0x80000000
    │                                                     │
    │  ═══════════════════════════════════════════════    │  (gap between IO and RAM)
    │                                                     │
    │  ┌───────────────────────────────────────────────┐  │  io_end
    │  │            TERMINATION BIT                    │  │
    │  ├───────────────────────────────────────────────┤  │
    │  │              PANIC BIT                        │  │
    │  ├───────────────────────────────────────────────┤  │  output_end
    │  │               OUTPUTS                         │  │
    │  ├───────────────────────────────────────────────┤  │  output_start = input_end
    │  │                INPUTS                         │  │
    │  ├───────────────────────────────────────────────┤  │  input_start = untrusted_advice_end
    │  │          UNTRUSTED ADVICE                     │  │
    │  ├───────────────────────────────────────────────┤  │  untrusted_advice_start = trusted_advice_end
    │  │           TRUSTED ADVICE                      │  │
    │  └───────────────────────────────────────────────┘  │  trusted_advice_start (~0x7FFF8000)
    │                                                     │  Lower addresses
    └─────────────────────────────────────────────────────┘
```

## Address Calculations

### IO Region (below RAM_START_ADDRESS)

The IO region is laid out from `RAM_START_ADDRESS` going downward:

```
io_end              = RAM_START_ADDRESS
termination         = io_end - 8
panic               = termination - 8
output_end          = panic
output_start        = output_end - max_output_size
input_end           = output_start
input_start         = input_end - max_input_size
untrusted_advice_end   = input_start
untrusted_advice_start = untrusted_advice_end - max_untrusted_advice_size
trusted_advice_end     = untrusted_advice_start
trusted_advice_start   = trusted_advice_end - max_trusted_advice_size
```

### RAM Region (at/above RAM_START_ADDRESS)

```
stack_end   = RAM_START_ADDRESS + program_size
stack_start = stack_end + stack_size
memory_end  = stack_start + memory_size  (original calculation)
```

## Key Data Structures

### MemoryConfig

Configuration provided by the user/macro:

| Field | Description |
|-------|-------------|
| `max_input_size` | Maximum size of public inputs |
| `max_output_size` | Maximum size of outputs |
| `max_trusted_advice_size` | Maximum size of trusted (private) advice |
| `max_untrusted_advice_size` | Maximum size of untrusted advice |
| `stack_size` | Size of the stack region |
| `memory_size` | Size of the heap region |
| `program_size` | Size of program sections (computed from ELF) |

### MemoryLayout

Computed layout with absolute addresses:

| Field | Description |
|-------|-------------|
| `program_size` | Size of program in bytes |
| `stack_size` | Stack size in bytes |
| `stack_end` | Bottom of stack (lowest address) |
| `memory_size` | Heap size in bytes |
| `memory_end` | Top of accessible RAM (highest valid address + 1) |
| `trusted_advice_start/end` | Trusted advice region bounds |
| `untrusted_advice_start/end` | Untrusted advice region bounds |
| `input_start/end` | Input region bounds |
| `output_start/end` | Output region bounds |
| `panic` | Address of panic bit |
| `termination` | Address of termination bit |
| `io_end` | End of IO region |

## ZeroOS/cargo-jolt Linker Layout

The ZeroOS `jolt-platform` uses a different memory layout (defined in `linker.ld.template`) where:
- Stack is placed at the **top** of RAM
- Heap is placed **below** the stack (with a guard gap)
- Program is at the bottom

### Diagram

```
    ┌───────────────────────────────────────────────┐  RAM_END (e.g., 0x88000000 for 128MB)
    │                  STACK                        │  __stack_top (aligned to 16 bytes)
    │            (grows downward ↓)                 │
    ├───────────────────────────────────────────────┤  __stack_bottom = __stack_top - __stack_size
    │           GUARD GAP (4KB)                     │
    ├───────────────────────────────────────────────┤  __heap_end (page-aligned)
    │                                               │
    │                  HEAP                         │
    │             (grows upward ↑)                  │
    │                                               │
    ├───────────────────────────────────────────────┤  __heap_start (page-aligned)
    │                                               │
    │              (unused space)                   │
    │                                               │
    ├───────────────────────────────────────────────┤  __bss_end
    │                 .bss                          │
    ├───────────────────────────────────────────────┤  __bss_start
    │            .tdata / .tbss (TLS)               │
    ├───────────────────────────────────────────────┤
    │                 .data                         │
    ├───────────────────────────────────────────────┤
    │         .init_array / .fini_array             │
    ├───────────────────────────────────────────────┤
    │        .eh_frame (if backtrace enabled)       │
    ├───────────────────────────────────────────────┤
    │                .rodata                        │
    ├───────────────────────────────────────────────┤
    │                 .text                         │
    │            (.text.boot first)                 │
    └───────────────────────────────────────────────┘  RAM_START (0x80000000)
```

### Address Calculations (from linker.ld.template)

```
__stack_top     = ALIGN((ORIGIN(RAM) + LENGTH(RAM)) - 15, 16)   // Top of RAM, 16-byte aligned
__stack_bottom  = __stack_top - __stack_size

__stack_guard_size = 4096   // 1 page guard gap

__heap_end      = ALIGN((__stack_bottom - __stack_guard_size) - 4095, 4096)  // Page-aligned
__heap_start    = ALIGN((__heap_end - __heap_size) - 4095, 4096)             // Page-aligned
```

### Safety Assertions

The linker script includes safety checks:
```
ASSERT(__heap_start >= __bss_end, "heap overlaps .bss/.data")
ASSERT(__heap_end <= __stack_bottom, "heap overlaps stack")
ASSERT(__heap_end + __stack_guard_size <= __stack_bottom, "heap/stack guard gap violated")
```

### Example with 128MB RAM, 64KB Stack, 1MB Heap

```
RAM_START       = 0x80000000
RAM_END         = 0x88000000  (128MB)

__stack_top     = 0x88000000  (top of RAM)
__stack_bottom  = 0x87FF0000  (64KB below top)

Guard gap       = 0x87FF0000 - 0x87FEF000 = 4KB

__heap_end      = 0x87FEF000  (below guard gap, page-aligned)
__heap_start    = 0x87EEF000  (1MB below __heap_end)
```

### Supporting This Layout in Jolt

To support linker layouts that place stack/heap near the **top of RAM**, the guest binary should export a symbol describing the end of RAM.

Jolt’s emulator will use `__ram_end` (or `__memory_end`) if present to:

1. Size the emulator’s backing memory.
2. Widen the RAM bounds check (`memory_end`).

In a linker script, this typically looks like:

```ld
__ram_end = ORIGIN(RAM) + LENGTH(RAM);
```

### Comparison: Original Jolt vs ZeroOS Layout

| Aspect | Original Jolt | ZeroOS/cargo-jolt |
|--------|---------------|-------------------|
| Stack location | After program | Top of RAM |
| Heap location | After stack | Below stack |
| Guard gap | Stack canary only | 4KB page gap |
| memory_end | program + stack + heap | widened to `__ram_end` when present |
| Flexibility | Fixed layout | Configurable via linker |

## Proving Considerations

The ZK proving cost is determined by **actual memory operations**, not by the configured RAM window:

1. `ram_K` (the memory table size) is computed from the maximum address actually accessed during execution
2. Only memory addresses that appear in the trace contribute to proving cost
3. Setting a large `__ram_end` increases emulator memory allocation but does NOT increase proving cost if the program only accesses a small region

## Constants

From `common/src/constants.rs`:

| Constant | Value | Description |
|----------|-------|-------------|
| `RAM_START_ADDRESS` | 0x80000000 | Start of RAM region |
| `EMULATOR_MEMORY_CAPACITY` | 128 MB | Default linker `LENGTH` used by the host template |
| `DEFAULT_MEMORY_SIZE` | 64 MB | Default heap size |
| `DEFAULT_STACK_SIZE` | 8 MB | Default stack size |
| `STACK_CANARY_SIZE` | 128 bytes | Stack overflow detection |
| `DEFAULT_MAX_INPUT_SIZE` | 4 KB | Default max input size |
| `DEFAULT_MAX_OUTPUT_SIZE` | 4 KB | Default max output size |
