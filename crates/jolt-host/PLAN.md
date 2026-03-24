# jolt-host: CycleRow Trait & Tracer Encapsulation Plan

## Goal

Make jolt-zkvm's witness layer generic over a `CycleRow` trait defined in jolt-host,
so that jolt-zkvm never imports `tracer` directly. The tracer becomes an implementation
detail behind jolt-host's API.

## Current State

- jolt-host re-exports concrete tracer types: `Cycle`, `Instruction`, `Memory`, `LazyTraceIterator`
- jolt-zkvm depends on `tracer` directly and uses `Cycle`, `Instruction`, `RAMAccess`,
  `NormalizedInstruction` in 6 source files across `witness/`, `tables.rs`, `evaluators/`
- jolt-witness already has `TraceSource` trait and `CycleData` (tracer-agnostic), but
  jolt-zkvm doesn't use them at the jolt-host boundary
- `flags.rs` in jolt-zkvm contains the full ISA dispatch table (~310 lines) mapping every
  `Instruction` variant to boolean flag arrays

## Architecture

```
tracer ──▶ jolt-host (defines CycleRow, implements for Cycle) ──▶ jolt-zkvm (generic over CycleRow)
                                                                       │
                                                              jolt-witness (TraceSource<Row=CycleData>)
```

jolt-host is the **sole tracer boundary**. Everything downstream sees only `CycleRow`.

---

## Phase 1: Define `CycleRow` trait in jolt-host

### New file: `src/cycle_row.rs`

```rust
use jolt_instructions::flags::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

/// Abstract interface for one execution cycle of a RISC-V trace.
///
/// This is the boundary between the tracer (which produces concrete `Cycle` values)
/// and the proving system (which consumes per-cycle data to build witnesses).
/// All ISA-specific logic (instruction dispatch, flag computation, operand routing)
/// is pushed into the `CycleRow` implementation, so the prover sees only scalars
/// and boolean arrays.
///
/// jolt-zkvm's witness layer is generic over `CycleRow`. The concrete implementation
/// for `tracer::Cycle` lives in jolt-host.
pub trait CycleRow: Copy {
    // ── Identity ──

    /// A no-op (padding) cycle.
    ///
    /// Used by `generate_witnesses` to pad traces to the next power of two.
    /// Must satisfy `Self::noop().is_noop() == true`.
    fn noop() -> Self;

    /// True if this cycle is a no-op (padding).
    fn is_noop(&self) -> bool;

    // ── Program counter & sequencing ──

    /// The unexpanded (pre-virtual-expansion) program counter.
    fn unexpanded_pc(&self) -> u64;

    /// Remaining steps in a virtual instruction sequence, or `None` if
    /// this is a real (non-virtual) instruction.
    fn virtual_sequence_remaining(&self) -> Option<u16>;

    /// True if this is the first instruction in a virtual sequence.
    fn is_first_in_sequence(&self) -> bool;

    /// True if this is a virtual (expanded) instruction.
    fn is_virtual(&self) -> bool;

    // ── Register operations ──

    /// RS1 register read: `(register_index, value)`, or `None` if unused.
    fn rs1_read(&self) -> Option<(u8, u64)>;

    /// RS2 register read: `(register_index, value)`, or `None` if unused.
    fn rs2_read(&self) -> Option<(u8, u64)>;

    /// RD register write: `(register_index, pre_value, post_value)`, or `None`.
    fn rd_write(&self) -> Option<(u8, u64, u64)>;

    /// The static `rd` operand from the instruction encoding (for write-address
    /// polynomial), independent of whether a write actually occurs.
    fn rd_operand(&self) -> Option<u8>;

    // ── RAM operations ──

    /// RAM access address, or `None` if this cycle has no RAM access.
    /// Returns the address regardless of read/write direction.
    fn ram_access_address(&self) -> Option<u64>;

    /// RAM read value. For reads: the loaded value. For writes: the pre-write value.
    /// `None` if no RAM access.
    fn ram_read_value(&self) -> Option<u64>;

    /// RAM write value. For writes: the post-write value. For reads: same as read value.
    /// `None` if no RAM access.
    fn ram_write_value(&self) -> Option<u64>;

    // ── Instruction metadata ──

    /// The immediate operand, sign-extended.
    fn imm(&self) -> i128;

    /// R1CS circuit flags (opflags from the Jolt paper).
    /// Indexed by `CircuitFlags` variants.
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS];

    /// Non-R1CS instruction flags for witness generation and operand routing.
    /// Indexed by `InstructionFlags` variants.
    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS];

    // ── Lookup computation (ISA-specific, pushed to tracer boundary) ──

    /// Combined lookup index for RA polynomial construction.
    /// Encodes the instruction identity + operands into a single index.
    ///
    /// Encoding depends on circuit flags:
    /// - **AddOperands**: `left_input + right_input`
    /// - **SubtractOperands**: `left_input + (2^64 - right_input)` (two's complement)
    /// - **MultiplyOperands**: `left_input * right_input`
    /// - **Advice / NoOp**: `0`
    /// - **Default (interleaved)**: `interleave_bits(left_input, right_input)`
    fn lookup_index(&self) -> u128;
}
```

### Trait bounds

`CycleRow: Copy` — required because `generate_witnesses` currently copies `Cycle` values
during padding (`Cycle` is `Copy`). This is a tight bound that matches the existing usage.
If a future tracer produces non-Copy rows, this can be relaxed to `Clone`.

### Why `noop()` is on the trait

`generate_witnesses` pads the trace to the next power of two with NoOp cycles. With a
concrete `Cycle`, this is `Cycle::NoOp`. With a generic `CycleRow`, the trait must
provide a way to construct padding cycles. A `noop()` associated function is the minimal
addition that preserves the current architecture.

### Why `lookup_operands`, `lookup_output`, `lookup_table_id` are NOT on the trait

These values are computable from the other CycleRow methods (flags + registers + PC + imm)
and the R1CS witness builder already computes them inline. Adding them would create
redundant paths. If a future consumer needs them as pre-computed values, they can be
added as provided methods with default implementations.

The `lookup_index` IS on the trait because it requires `interleave_bits` from
`jolt-instructions`, which is ISA-specific logic that belongs at the tracer boundary.

### Dependencies

jolt-host gains a dependency on `jolt-instructions` (for `NUM_CIRCUIT_FLAGS`,
`NUM_INSTRUCTION_FLAGS` constants). This is a Level 1 crate with no heavy deps.

### Re-exports from `lib.rs`

```rust
mod cycle_row;
pub use cycle_row::CycleRow;

// Concrete type for tests and host-side code
pub use tracer::instruction::Cycle;

// Flag types needed by downstream CycleRow consumers
pub use jolt_instructions::flags::{
    CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
```

---

## Phase 2: Implement `CycleRow for tracer::Cycle` in jolt-host

### New file: `src/cycle_row_impl.rs`

This file implements `CycleRow` for `tracer::instruction::Cycle`. It absorbs:

1. **`flags.rs` from jolt-zkvm** — the full ISA dispatch table mapping `Instruction`
   variants to `circuit_flags()` and `instruction_flags()` arrays. This is ~310 lines
   of match arms that belong at the tracer boundary, not inside the prover.

2. **`compute_lookup_index` from `cycle_data.rs`** — the ISA-specific lookup index
   computation using circuit flags and `interleave_bits`. This is the only function
   from cycle_data.rs that moves.

3. **RAM access decomposition** — extracting address/read/write from `RAMAccess` enum
   variants into the trait's `Option<u64>` returns.

4. **Sequencing fields** — normalizing the instruction and extracting PC, virtual
   sequence info, `is_first_in_sequence`, `is_virtual`, `is_compressed`.

5. **`noop()` constructor** — returns `Cycle::NoOp`.

The implementation calls `self.instruction()` and `self.instruction().normalize()`
internally — these are tracer-internal methods that never leak through the trait.

### What moves out of jolt-zkvm

| jolt-zkvm file | What moves to jolt-host |
|---|---|
| `witness/flags.rs` | Entire file → `cycle_row_impl.rs` (ISA dispatch) |
| `witness/cycle_data.rs` | `compute_lookup_index` → `CycleRow::lookup_index()` impl |

### What stays in jolt-zkvm

- `instruction_inputs` — stays in `cycle_data.rs`, signature changes to
  `fn instruction_inputs(cycle: &impl CycleRow, ...) -> (u64, i128)`.
  This is pure flag dispatch (not ISA dispatch) — it reads `instruction_flags()`,
  `rs1_read()`, `rs2_read()` through the trait.
- `cycle_to_cycle_data` — converts `&impl CycleRow` → `CycleData` (protocol-level)
- `trace_to_cycle_data` — iterates trace, calls above
- All sumcheck witness builders — now generic over `CycleRow`
- `cycle_to_witness` in r1cs_inputs.rs — computes lookup operands and output
  from CycleRow methods + flags (no ISA dispatch needed)

---

## Phase 3: Update jolt-zkvm to be generic over `CycleRow`

### File-by-file changes

| File | Current signature | New signature |
|---|---|---|
| `witness/generate.rs` | `generate_witnesses(trace: &[Cycle])` | `generate_witnesses<R: CycleRow>(trace: &[R])` |
| `witness/cycle_data.rs` | `cycle_to_cycle_data(cycle: &Cycle, ...)` | `cycle_to_cycle_data(cycle: &impl CycleRow, ...)` |
| `witness/cycle_data.rs` | `instruction_inputs(cycle: &Cycle, ...)` | `instruction_inputs(cycle: &impl CycleRow, ...)` |
| `witness/cycle_data.rs` | `compute_lookup_index(cycle: &Cycle)` | **Deleted** — moved to `CycleRow::lookup_index()` |
| `witness/r1cs_inputs.rs` | `cycle_to_witness(cycle: &Cycle, next: Option<&Cycle>, ...)` | `cycle_to_witness(cycle: &impl CycleRow, next: Option<&impl CycleRow>, ...)` |
| `witness/bytecode.rs` | `BytecodePreprocessing::new(trace: &[Cycle])` | `BytecodePreprocessing::new(trace: &[impl CycleRow])` |
| `witness/bytecode.rs` | `BytecodePreprocessing::get_pc(&self, cycle: &Cycle)` | `get_pc(&self, cycle: &impl CycleRow)` |
| `witness/flags.rs` | **Deleted** — moved to jolt-host | N/A |
| `witness/mod.rs` | `pub mod flags;` | Line removed, doc comment updated |
| `tables.rs` | `from_witness(..., trace: &[Cycle], ...)` | `from_witness(..., trace: &[impl CycleRow], ...)` |
| `evaluators/sparse_rw.rs` | `ram_entries_from_trace(trace: &[Cycle], ...)` | `ram_entries_from_trace(trace: &[impl CycleRow], ...)` |

**Note:** `prover.rs` (`GraphProverInput`) does NOT have a `trace` field — it operates
on `PolynomialTables<F>`, which is fully field-element based. The trace is consumed
entirely at the witness generation boundary.

### Bytecode preprocessing

`BytecodePreprocessing::new` currently does `cycle.instruction().normalize()` to get
PC and virtual sequence info. With `CycleRow`, it uses:
- `cycle.unexpanded_pc()`
- `cycle.virtual_sequence_remaining()`
- `cycle.is_noop()`

`BytecodePreprocessing::get_pc` currently takes `&Cycle` and calls `instruction().normalize()`.
With `CycleRow`, it uses the same trait methods.

### R1CS witness generation

`cycle_to_witness` currently calls 6+ methods on `Cycle` plus `flags::circuit_flags()`
and `flags::instruction_flags()`. With `CycleRow`, every field is a direct trait
method call — no `normalize()`, no `RAMAccess` pattern matching, no `Instruction`
variant dispatch. The function body simplifies.

The lookup operands (`V_LEFT_LOOKUP_OPERAND`, `V_RIGHT_LOOKUP_OPERAND`) and lookup
output (`V_LOOKUP_OUTPUT`) are computed inline from CycleRow methods:
- `circuit_flags()` for add/sub/mul/advice mode selection
- `rs1_read()`, `rs2_read()` for register values
- `rd_write()` for rd_write_value (advice output)
- `unexpanded_pc()` and `imm()` for branch target computation

No additional trait methods are needed — the existing CycleRow surface fully covers
the R1CS computation.

### Instruction inputs

`instruction_inputs` stays in `cycle_data.rs` as a standalone function. Its logic is
pure flag dispatch:

```rust
pub(crate) fn instruction_inputs(
    cycle: &impl CycleRow,
    iflags: &[bool; NUM_INSTRUCTION_FLAGS],
    unexpanded_pc: u64,
    imm: i128,
) -> (u64, i128) { ... }
```

Called from:
1. `cycle_to_witness` (r1cs_inputs.rs) — for `V_LEFT_INSTRUCTION_INPUT` / `V_RIGHT_INSTRUCTION_INPUT`
2. `CycleRow::lookup_index()` impl (cycle_row_impl.rs in jolt-host) — for index computation

Since the function's signature becomes `&impl CycleRow`, it can be called from both
jolt-host and jolt-zkvm. **However**, this creates a circular dependency
(jolt-host → jolt-zkvm → jolt-host). Resolution: duplicate the logic.

The `instruction_inputs` computation is 10 lines of flag dispatch. The jolt-host
impl calls the same logic inline (it already has all the data from `self`). The
jolt-zkvm copy takes `&impl CycleRow`. No shared code needed.

### Padding

`generate_witnesses` pads the trace to the next power of two. Currently:
```rust
let padded_trace: Vec<Cycle> = trace.iter().copied()
    .chain(std::iter::repeat_n(Cycle::NoOp, padded_len - trace.len()))
    .collect();
```

With CycleRow:
```rust
let padded_trace: Vec<R> = trace.iter().copied()
    .chain(std::iter::repeat_n(R::noop(), padded_len - trace.len()))
    .collect();
```

The `CycleRow: Copy` bound makes `.copied()` work. The `noop()` associated function
provides padding cycles.

### `compute_ram_k`

This function in `generate.rs` currently pattern-matches on `RAMAccess`:
```rust
fn compute_ram_k(trace: &[Cycle]) -> usize {
    let max_addr = trace.iter().filter_map(|c| match c.ram_access() {
        RAMAccess::Read(r) => Some(r.address),
        RAMAccess::Write(w) => Some(w.address),
        RAMAccess::NoOp => None,
    }).max().unwrap_or(0);
    ...
}
```

With CycleRow:
```rust
fn compute_ram_k(trace: &[impl CycleRow]) -> usize {
    let max_addr = trace.iter()
        .filter_map(|c| c.ram_access_address())
        .max()
        .unwrap_or(0);
    ...
}
```

### `ram_entries_from_trace` in sparse_rw.rs

Currently pattern-matches on `RAMAccess::Read/Write/NoOp`. With CycleRow:

```rust
pub fn ram_entries_from_trace<F: Field>(
    trace: &[impl CycleRow],
    padded_len: usize,
) -> Vec<RwEntry<F>> {
    let mut entries = Vec::new();
    for (j, cycle) in trace.iter().enumerate() {
        if let Some(addr) = cycle.ram_access_address() {
            let read_val = cycle.ram_read_value().unwrap();
            let write_val = cycle.ram_write_value().unwrap();
            entries.push(RwEntry {
                bind_pos: j,
                free_pos: addr as usize,
                ra: F::one(),
                val: F::from_u64(read_val),
                prev_val: F::from_u64(read_val),
                next_val: F::from_u64(write_val),
            });
        }
    }
    entries
}
```

### `extract_flag_poly` and `extract_register_addresses` in tables.rs

Currently call `cycle.instruction()` → `flags::instruction_flags()`. With CycleRow:
- `extract_flag_poly` calls `cycle.instruction_flags()[flag_idx]` directly
- `extract_register_addresses` calls `cycle.rs1_read()`, `cycle.rs2_read()`,
  `cycle.rd_write()` — already trait methods

### Next-cycle lookahead

R1CS needs `next_unexpanded_pc`, `next_is_virtual`, `next_is_first_in_sequence`,
`next_is_noop`. These are NOT methods on `CycleRow` — they're computed during
iteration with lookahead:

```rust
for i in 0..trace.len() {
    let next = trace.get(i + 1);
    let witness = cycle_to_witness(&trace[i], next, ...);
}
```

Inside `cycle_to_witness`, the `next: Option<&impl CycleRow>` provides:
- `next.circuit_flags()` → `VirtualInstruction`, `IsFirstInSequence`
- `next.instruction_flags()` → `IsNoop`
- `next.unexpanded_pc()`

---

## Phase 4: Dependency cleanup

### jolt-zkvm Cargo.toml

```diff
 [dependencies]
+jolt-host = { workspace = true }
 jolt-instructions = { workspace = true }
 jolt-witness = { workspace = true }
 # ... other jolt-* deps ...
-tracer = { workspace = true }

 [dev-dependencies]
-jolt-host = { workspace = true }
+# jolt-host is now a regular dependency
```

### What jolt-zkvm imports from jolt-host

```rust
use jolt_host::CycleRow;  // trait — used in all generic signatures
use jolt_host::Cycle;      // concrete type — used in tests only
```

### What jolt-zkvm still imports from jolt-instructions

```rust
use jolt_instructions::flags::{CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};
```

These are needed to index into the flag arrays returned by `CycleRow::circuit_flags()`
and `CycleRow::instruction_flags()`. jolt-instructions is a lightweight crate (flag
enums + lookup tables) with no tracer dep.

---

## Phase 5: Verify

1. `cargo clippy -p jolt-host --all-targets --message-format=short -q -- -D warnings`
2. `cargo nextest run -p jolt-host --cargo-quiet`
3. `cargo clippy -p jolt-zkvm --all-targets --message-format=short -q -- -D warnings`
4. `cargo nextest run -p jolt-zkvm --cargo-quiet`
5. E2E: `cargo nextest run -p jolt-zkvm graph_driven_muldiv --cargo-quiet`

The e2e test in `e2e_graph.rs` does NOT need to change — it uses
`jolt_host::Program::trace()` which returns `Vec<Cycle>`, and `Cycle: CycleRow` so
`&[Cycle]` satisfies `&[impl CycleRow]`.

---

## Sumcheck Coverage

Every sumcheck stage's witness builder becomes generic over `CycleRow`:

| Stage | Sumcheck | CycleRow methods used |
|---|---|---|
| S1 | Spartan outer | All R1CS variables derive from CycleRow via `cycle_to_witness` |
| S2 | Register RW checking | `rs1_read`, `rs2_read`, `rd_write` |
| S2 | RAM RW checking | `ram_access_address`, `ram_read_value`, `ram_write_value` |
| S2 | RAM Hamming booleanity | `ram_access_address` (is_some check) |
| S2 | RAM RAF evaluation | `ram_access_address` |
| S3 | Spartan shift | `unexpanded_pc`, `is_virtual`, `is_first_in_sequence`, `is_noop` |
| S4 | Registers val evaluation | `rd_operand` |
| S5 | Bytecode read-RAF | `unexpanded_pc`, `virtual_sequence_remaining` |
| S5 | Instruction read-RAF | `lookup_index`, `circuit_flags`, `instruction_flags` |
| S5 | Instruction RA virtual | `lookup_index` |
| S6 | RAM RA virtual | `ram_access_address` |
| S6 | Booleanity | `lookup_index`, `ram_access_address` (via RaIndices) |
| S7 | Inc claim reduction | `rd_write` (RdInc), `ram_read_value`/`ram_write_value` (RamInc) |
| S7 | Registers claim reduction | `rs1_read`, `rs2_read`, `rd_write` |
| S7 | RAM RA claim reduction | `ram_access_address` |

No stage requires access to `tracer::Cycle`, `Instruction`, `RAMAccess`, or
`NormalizedInstruction` directly. All ISA knowledge is encapsulated in the
`CycleRow` implementation.

---

## Decisions

1. **`lookup_table_id` representation**: NOT on the trait. Currently unused by any
   consumer in jolt-zkvm — the prover reads from `PolynomialTables`, not from CycleRow.
   Can be added as a provided method later if needed.

2. **`lookup_operands` / `lookup_output`**: NOT on the trait. These are computed inline
   by the R1CS witness builder from circuit_flags + register values. Adding them would
   create redundant computation paths with no current consumer.

3. **Where `CycleData` conversion lives**: Stays in `jolt-zkvm/witness/cycle_data.rs`.
   CycleData is a protocol-level type (jolt-witness), not a tracer-level type.
   Moving it to jolt-host would create a jolt-host → jolt-witness dependency that
   doesn't belong.

4. **`BytecodePreprocessing` ownership**: Stays in jolt-zkvm. It's prover infrastructure,
   not tracer abstraction.

5. **`instruction_inputs` duplication**: The 10-line flag dispatch logic is duplicated:
   once in jolt-host's `CycleRow::lookup_index()` impl (inlined), once in jolt-zkvm's
   `cycle_data.rs` (standalone function taking `&impl CycleRow`). This avoids a circular
   dependency between jolt-host and jolt-zkvm.

6. **`CycleRow: Copy` bound**: Required by `generate_witnesses` padding. Matches `Cycle`
   which is `Copy`. Can be relaxed to `Clone` if a future tracer needs heap-allocated rows.
