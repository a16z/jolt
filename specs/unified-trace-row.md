# Design: one trace row from execution through proving

Status: implemented on `refactor/unify-trace-rows`, based on upstream
[`47130f3dc`](https://github.com/a16z/jolt/commit/47130f3dc9a51a7ac2754a98ff0aa31981a6b810)
(2026-10-03), after [#1818](https://github.com/a16z/jolt/pull/1818) and
[#1734](https://github.com/a16z/jolt/pull/1734). This change addresses
[#1839](https://github.com/a16z/jolt/issues/1839). Implementation and validation
results, including measurement limits, are recorded below.

## Implemented design

`jolt_riscv::JoltTraceRow` is the single stored row throughout execution,
analysis, replay, and proving. It is 64 bytes and `Copy`, including under
`field-inline`. It retains full encoded instruction operands, exact
virtual-sequence metadata, and captured integer-register presence, while
preserving the integer-only proof register accessors. Field-inline payloads
live in a sparse event vector beside the core rows in `TraceData`.

Every emitted row already contains its bytecode PC. Execution receives a
`JoltProgram` with expanded bytecode and builds the existing PC mapper before
producing rows. There is no unbound-row state or later PC assignment pass.
Backends return the producer's allocation as `TraceOutput::trace`, an
`Arc<TraceData>` that analysis and witness construction retain; another live
owner does not cause a row copy.

The separate `jolt_program::TraceRow` implementation and the alternate compact
execution/witness entrypoints have been removed. Logical register/RAM input
and view types now live in `jolt-riscv`; program mapping and field payloads
remain in `jolt-program`.

## Historical baseline at `47130f3dc`

This section describes the upstream revision before this implementation.
Its pinned source links are historical evidence, not links to the new APIs.

| Finding at the baseline | Design consequence |
| --- | --- |
| Without `field-inline`, SDK proving and profiling called `trace_compact` and then `from_compact`. | These paths already retained one compact row allocation. Their acceptance target is unchanged proof behavior without a material performance regression. |
| Generic `TraceBackend::try_new` allocated compact rows. Under `field-inline`, it reused `source.shared_rows()` for `raw_trace_rows` when available. | Ordinary `OwnedTrace` field-inline proving retained two row representations. Unification removes that duplication; it does not remove a third independent row allocation. |
| SDK/host preparation populated `JoltProgram` with expanded bytecode, and `BytecodePCMapper::try_new` accepted it without the leading no-op. | PCs can be bound during production using the existing numbering rule. |
| `JoltTraceRow` had three reserved bytes; its metadata word used 23 bits normally and 31 under `field-inline`. | The sequence count and presence masks fit in the reserved bytes, but the masks do not fit in the field-enabled metadata word. |
| `JoltTraceRow` stored `instruction.integer_operands()`; `TraceRow` retained encoded operands and independent captures. | Store encoded IDs once and cache integer-projection presence, keeping field IDs out of integer proof columns. |
| Field instruction roles and validation came from ordinary bytecode, including memory accumulation and advice-limb instructions. | Reuse the operand projections and shape validator instead of restoring the removed field-bytecode metadata table. |
| The constructors accepted different malformed inputs; recorded register IDs could disagree with instruction operands. | The unified row needs an explicit checked contract, beyond a type alias. |
| The legacy prover and its byte-diff acceptance lane had been removed. | Use current prover/verifier fixtures and a before/after baseline for acceptance. |

Historical sources: [SDK proof path](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/jolt-sdk/src/host_utils.rs#L313),
[profile trace path](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-prover/src/profile.rs#L1029),
[witness construction](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-witness/src/backend/trace/mod.rs#L179),
[compact layout](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-riscv/src/trace_row.rs#L203),
[execution constructor](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-program/src/execution/trace/row.rs#L262).

## Row contract and layout

The private value slots retain their existing aliasing. The checked
constructor packs them for the three final row classes, and the logical
accessors read them back by the cached `Load`/`Store` circuit flags:

| Row class | Slot 0 | Slot 1 | Slot 2 | Slot 3 |
| --- | --- | --- | --- | --- |
| Non-memory | rs1 value | rs2 value | rd pre-value | rd post-value |
| Load | rs1 value | RAM address | rd pre-value | rd post-value / loaded value |
| Store | rs1 value | rs2 value / RAM post-value | RAM pre-value | RAM address |

The physical layout is:

| Field | Bytes | Offset |
| --- | ---: | ---: |
| Four value slots | 32 | 0 |
| Guest instruction address | 8 | 32 |
| Immediate magnitude | 8 | 40 |
| Bytecode PC | 4 | 48 |
| Packed metadata | 4 | 52 |
| Stable instruction tag | 2 | 56 |
| Virtual-sequence remaining | 2 | 58 |
| Three full encoded instruction operand IDs | 3 | 60 |
| Row-control presence masks | 1 | 63 |
| **Total, alignment 8** | **64** | |

The metadata word retains its feature-dependent layout: 16 circuit-flag bits
in base builds or 24 under `field-inline`, followed by six instruction flags
and the immediate sign. Nine bits remain unused in base builds and one in
field-enabled builds. Byte 63 has the same definition in every feature mode:

| Row-control bits | Meaning |
| --- | --- |
| 0–2 | Captured integer rs1, rs2, and rd presence |
| 3 | Virtual-sequence count present |
| 4–6 | Integer rs1, rs2, and rd operand presence |
| 7 | Reserved, zero |

Construction caches integer-operand presence from
`instruction.integer_operands()`. That projection removes encoded operands
without moving IDs between slots, so proof register accessors gate each
stored ID with its cached integer bit. This avoids repeating instruction-role
dispatch in hot loops. The capture mask is independent: a declared integer
operand and an observed register value are different facts. Coverage of field
operand shapes pins this slot-preserving property; a future projection that
remaps slots must revisit the representation.

An operand byte uses `0xff` for absence and accepts IDs through 254. This is a
storage bound, not validation against the architectural register count.
`virtual_sequence_remaining()` exposes `Option<u16>` and preserves `None`
versus `Some(0)` without a count sentinel. The PC mapper separately enforces
its sequence-length limit. `instruction()` reconstructs all accepted
instruction metadata, including encoded field operands, immediate, address,
and sequence metadata. The no-op boolean restriction below is necessary for
that exact reconstruction.

A compile-time assertion pins the 64-byte size. Tests cover alignment and
`Copy`; compile-time width/non-overlap assertions pin metadata and control
masks. Serialization uses logical values and does not depend on these offsets
or bit assignments. No unused bytes remain.

### Checked construction

The logical `RegisterRead`, `RegisterWrite`, `RegisterState`, `RamRead`,
`RamWrite`, and `RamAccess` types are exported by `jolt-riscv`. They are
constructor inputs, execution views, and wire values, not another stored row
representation. The construction boundary is:

```rust
JoltTraceRow::new(
    instruction: JoltInstructionRow,
    registers: RegisterState,
    ram_access: RamAccess,
    bytecode_pc: u32,
) -> Result<JoltTraceRow, TraceRowError>
```

The constructor validates logical observations and is the single
slot-packing implementation. Interpreter and x86 adapters extract
observations and resolve the PC. `from_components` and the `CapturedState`
view it consumed have been removed. Producer failures carry context through
`TraceError`.

Release checks enforce:

- Loads have `RamAccess::Read`, no captured integer rs2, and a loaded value
  equal to the integer rd post-value.
- Stores have `RamAccess::Write`, no captured integer rd, and a RAM post-value
  equal to the integer rs2 value.
- Non-memory rows have `RamAccess::NoOp`, even when a malformed RAM operation
  would have all-zero values.
- Each present captured register ID matches the corresponding
  `instruction.integer_operands()` entry. Captures in field-only slots are
  rejected even when the numeric IDs match.
- Immediate magnitude fits `u64`, and encoded IDs fit the storage bound.
  PC zero is reserved for no-ops; all no-ops use it and all non-noops use a
  nonzero PC.
- No-ops have no captured register or RAM effects. They reject
  `is_first_in_sequence` or `is_compressed` with `TraceRowError::NoOpMetadata`,
  because no-op proof flags do not retain those booleans.

Absent captures remain legal and distinct from observed zero. Their scalar
proof values are zero; memory-alias checks use that same zero when a capture
is absent. The constructor does not require every declared operand to have a
capture. Accepted no-op addresses, operands, immediates, and sequence counts
are preserved, so not every accepted no-op is canonical padding.
`JoltTraceRow::default()` produces the canonical padding row with its
existing proof columns.

Both execution producers derive captured IDs from integer operands. The
[baseline mismatched-ID test](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-program/src/execution/trace/row.rs#L702)
deliberately recorded register 200 for operand 2. Rejecting that discrepancy
is an intentional constructor change. Fixtures now use the
checked constructor with valid observations instead of value-only shortcuts.

Capture and operand presence differ for field memory operations.
`FIELD_LOAD_ACCUMULATE_FROM_MEMORY` encodes an integer base in rs1, an integer
scratch destination in rd, and a field accumulator in rs2. Its RAM read equals
integer rd's post-value. It has no integer rs2 capture or proof index despite
having an encoded rs2. The field accumulator values reside in the sparse
payload, so the three existing value-slot classes suffice.

Canonical `integer_operands()` and `field_operands()` still own field-role
projection, including implicit accumulator reads and destinations encoded in
rs2. `validate_field_inline_instruction` owns program-level instruction shape
checks, including the x0 restrictions for memory accumulation and advice-limb
writes. Field witness installation retains payload shape, bridge, bytecode,
profile, and register-continuity checks. Row-local construction does not
replace these checks or proof constraints.

Execution accessors (`instruction`, `virtual_sequence_remaining`, `rs1_read`,
`rs2_read`, `rd_write`, `registers`, and `ram_access`) are implemented on the
unified row. The row does not implement `JoltCycle`; witness lookup queries
adapt it through the proof accessors in `jolt-witness`. A load's proof
`ram_write_value()` equals its loaded value, while its execution view reports
a RAM read. Proof `rs1_index`, `rs2_index`, and `rd_index` retain
integer-operand semantics; execution register views use capture presence.
`instruction_kind()` retains its existing optional return type.

Raw emulator `Cycle` and instruction-specific `RegisterSnapshot` types remain
unchanged. They precede final instruction expansion and rd=x0 rewriting and
are not interchangeable with stored final rows. The implementation is in
[`jolt-riscv/src/trace_row.rs`](../crates/jolt-riscv/src/trace_row.rs).

## PCs are bound during production

`BytecodePCMapper::get_instruction_pc` owns instruction-aware lookup,
including no-op-to-zero. `BytecodePreprocessing::get_pc` delegates to it.
Mapper construction assigns the same PCs to expanded bytecode with or without
a leading canonical no-op; this remains the sole numbering rule.

Generic execution builds a mapper from `program.expanded_bytecode` once per
execution. Replay workers share it through `Arc<WorkerSeed>` instead of
rebuilding it per chunk. The x86 `CompiledProgram` also owns the mapper and
uses it during observation reassembly, without independently computing
`row_index + 1`. A separate preprocessing map is acceptable: its size follows
program length, not trace length. No mapper is cached inside the publicly
mutable `JoltProgram`.

Missing mappings, oversized PCs, and source-only instructions produce typed
errors. An ELF-only `JoltProgram` is no longer sufficient to produce final
rows; final-row tests prepare expanded bytecode. Raw `Cycle` tracing/debugging
APIs remain available. SDK and host execution already prepare expanded
bytecode.

The constructor's `u32` PC type and no-op check enforce local representation.
Program mapping establishes PC identity: producers resolve each PC through
`BytecodePCMapper::get_instruction_pc` before calling the public
`JoltTraceRow::new`, which does not check program membership. No-ops use slot
zero independently of source address, satisfy the local no-effects/boolean
contract, and need not equal canonical padding.

Sources: [PC mapper](../crates/jolt-program/src/preprocess/bytecode.rs),
[interpreter and replay](../tracer/src/execution_backend.rs).

## One owner of rows and field payloads

The retained storage in `jolt-program` has this shape; `TraceData` fields are
private:

```text
TraceOutput {
    trace: Arc<TraceData>,
    device, final_memory, advice_tape,
}
TraceData {
    rows: Vec<JoltTraceRow>,
    proof_len: usize,
    field_events: Vec<FieldEvent>, // field-inline only
}
FieldEvent { cycle: usize, data: FieldInlineTraceData }
```

Each event stores its `Copy` payload inline. `Arc<TraceData>` is the only
sharing handle, so payloads carry no reference count or separate allocation.
A transient `TraceEvent` contains a core row and, in field-enabled builds, an
optional payload. `TraceData::push` appends them together. `from_parts`
transfers parallel producer buffers and rejects unsorted, duplicate, or
out-of-range field events. Payload semantics remain the responsibility of
field witness validation. There is no per-cycle payload slot in retained
rows, and absent payloads consume no event entries. Products and inverse
products are derived from decoded field register values rather than stored
as extra payload fields.

`ExecutionBackend::trace` returns a non-generic `TraceOutput`; backends have no
trace-source associated type and there is no row cursor or trace-source trait.
`TraceOutput::new` takes the freshly produced `TraceData` and becomes its first
owner. `ChunkedExecutionBackend::replay_chunk` returns each replayed chunk as
its own `TraceData`. There is no `into_rows` method that might clone a shared
row vector.

`TraceBackend::try_new` accepts the non-generic `JoltVmWitnessInputs`, rejects
a `proof_len` beyond the cycle domain, and retains the input `TraceOutput`
with the producer's `Arc`. `TraceBackend` has no trace-source type parameter or
`PhantomData`. `RandomAccessRows` and field witnesses borrow or share the
aggregate. `from_compact`, `compact_trace_row`, and `raw_trace_rows` have been
removed. Chunked replay stays on `ChunkedExecutionBackend`; witness
construction accepts only a complete `TraceOutput`.

Sequential consumers merge rows and sparse events: serialization and full
field validation use sequential event cursors. Sparse Spartan and
field-register witness scans iterate events directly. Isolated random access
uses `TraceData::field_inline` and a binary search.
`FieldInlineWitnessOracle::fill_rd_increments(start, values)` gives dense
chunk consumers an efficient path: the trace-backed implementation
locates the first event once and merges the rest sequentially. Akita uses
this method; it does not binary-search the sparse table once per cycle.

Interpreter parallel collection writes one final core vector. Field-enabled
collection fills that vector by indexed chunks and combines sparse event
batches in order, without a full intermediate `(row, optional_payload)`
vector. The combined event vector is allocated once from the summed batch
lengths, so combining copies each payload once instead of regrowing; the
per-batch vectors still grow by push, and they coexist with the combined
vector while batches are moved in. The interpreter's raw `Vec<Cycle>` and
x86 observation buffer remain.

### Execution length and proof length

Analysis and replay retain all produced rows. `TraceData::proof_len()` and
`proof_rows()` exclude only canonical trailing padding, without truncating or
copying the row allocation. Interior no-ops, noncanonical no-ops, and rows
with field payloads are retained. `push` tracks the bound while emitting;
`TraceData::new` finds it by scanning backward over the canonical trailing
suffix of an already collected vector. `from_parts` uses that same suffix
scan and also retains the final field event's cycle. There is no full trace
conversion or normalization copy.

`ProverConfig::derive` now consumes `&[JoltTraceRow]`; callers supply
`proof_rows()`. Its existing minimum domain and final-no-op sizing law are
unchanged. Its last-row Jump precondition (`ProverError::TraceDoesNotEndInJump`)
also depends on that prefix: `rows()` can end in canonical padding and would
reject an honest trace. Witness and field construction validate the same retained
`proof_len` against the cycle domain, and random-access proof consumers use
that same prefix. This shared bound prevents padding from changing proof
shape near a power-of-two boundary while keeping full analysis content.

Sources: [aggregate](../crates/jolt-program/src/execution/trace/data.rs),
[execution output](../crates/jolt-program/src/execution/trace.rs),
[witness handoff](../crates/jolt-witness/src/backend/trace/mod.rs),
[field consumers](../crates/jolt-witness/src/field_inline/mod.rs),
[configuration](../crates/jolt-prover/src/config.rs).

## Serialization and public API

Rows and `TraceData` implement `Serialize` only. Core row serialization uses a
logical wire shim containing instruction, captured registers, RAM access, and
bytecode PC; packed storage is not a serialized API. The aggregate wire is a
sequence of logical `TraceEvent` values, joining each payload with its row, so
`field-inline` builds emit an optional payload on every event. The physical
sparse event vector is not serialized. Serialization borrows rows/payloads and
emits them incrementally.

`ProgramSummary.trace` is `Arc<TraceData>` and retains the full trace, including
padding and field payloads. Common access is `summary.trace.len()`,
`summary.trace.rows().iter()`, or `summary.trace.field_inline(cycle)` in
field-enabled builds. `Program::trace_analyze` moves the execution output's
`Arc<TraceData>` into the summary.

`ProgramSummary` is a debugging and analysis output, and it derives
`Serialize` only. `write_to_file` streams its bincode encoding through a
buffered writer. Nothing reads a summary back, so the file carries no format
header. A future summary reader must define its own format contract alongside
its first caller.

This is a source and analysis-wire breaking change: imports, constructors,
summary iteration, and direct `row.field_inline` access need migration.
`OwnedTrace`, `TraceSource` (with its SDK re-exports `jolt::OwnedTrace` and
`jolt::TraceSource`), `ExecutionBackend::Trace`, and the generic parameters on
`TraceOutput` and `JoltVmWitnessInputs` are removed. Callers read
`TraceOutput::trace` (`Arc<TraceData>`) directly, and generated `trace_*`
functions return `jolt::TraceOutput`. `CapturedState` with its
`NonMemoryState`, `LoadState`, and `StoreState` payloads is removed; slot
values are read through the row accessors. `JoltTraceRow::no_op()` is folded
into `Default`. `JoltInstruction` converts from `JoltInstructionRow` through an
infallible `From`, replacing a `TryFrom` whose match over the same kind enum
could not fail. Neither `ProgramSummary` nor the trace row (formerly
`jolt_program::TraceRow`) implements `Deserialize`. The deployed
proof and preprocessing formats are unchanged.
`Program::trace_to_file` still writes raw `Cycle` records and is outside this
wire migration. See [summary implementation](../crates/jolt-host/src/analyze.rs).

The accompanying telemetry change advances the span taxonomy to version 4.
`ProverConfig::derive_compact` is removed; every feature mode emits
`ProverConfig::derive`. This configuration span belongs to the SDK/profile
prelude, outside the root proving span. Telemetry consumers must account for
the removed label. Proof and preprocessing serialization are unaffected.

## Structural memory effect and scope

The following estimates compare with `47130f3dc`; they are element-storage
accounting, not measured RSS or speedups. Capacity, other live buffers, and
peak timing affect process memory.

For a generic non-field trace of N physical rows, the baseline conversion
held approximately `64N + 64N` row bytes during handoff. The shared allocation
now retains `64N`. One full vector at N = 2^26 is 4 GiB of elements. The
baseline non-field SDK/profile path already used one compact vector, so it
does not gain that reduction again.

Under `field-inline`, the baseline execution row occupied 72 bytes and
witness construction retained a separate 64-byte core row, totaling `136N`
bytes before payloads and capacity. `OwnedTrace::shared_rows` shared the
execution vector; the live input handle was not a third allocation. The new
owner retains `64N + 192M` element bytes for M field events on a 64-bit
host. A `FieldEvent` is its 8-byte cycle plus the 184-byte
`FieldInlineTraceData`: two 34-byte read options, a 66-byte write option, a
1-byte op option, and a 48-byte bridge option, padded to 8-byte alignment.
Measurements must include actual event density and vector capacities rather
than quote a universal savings percentage.

The change preserves cycle ordering, advice tapes, final memory, and device
outputs. It does not eliminate the interpreter's raw `Vec<Cycle>` or x86's
observation buffer and does not claim fully streaming execution. Proof
equations, sumcheck constraints, lookup routing, instruction tags, field
encodings, and commitment layouts are unchanged. Bytecode storage and
emulator execution have not been redesigned.

## Implementation map

| Area | Implementation |
| --- | --- |
| Canonical row and logical captures | `crates/jolt-riscv/src/trace_row.rs` |
| PC numbering and contextual lookup | `crates/jolt-program/src/preprocess/bytecode.rs` |
| Shared aggregate and sparse events | `crates/jolt-program/src/execution/trace/data.rs` |
| Execution output and chunked replay interface | `crates/jolt-program/src/execution/{trace,backend}.rs` |
| Interpreter, replay, and x86 production | `tracer/src/execution_backend.rs`, `crates/jolt-tracer-x86/src/` |
| Witness ownership and sparse field consumers | `crates/jolt-witness/src/backend/trace/`, `crates/jolt-witness/src/field_inline/` |
| Configuration, SDK, profiling, and kernels | Callers consume the same row and aggregate; compact-only entrypoints are removed |
| Full analysis and summary files | `crates/jolt-host/src/{analyze,program}.rs` |

## Validation and acceptance

The completed runs below cover the unified-row implementation through
`4e9f1cdd8`, based on upstream `47130f3dc`. They precede the accompanying
telemetry taxonomy version 4 change. Counts are per run, not a total of unique
tests across feature configurations.

| Scope | Configuration | Result |
| --- | --- | --- |
| `jolt-riscv` core | Base | 67 passed |
| `jolt-riscv` core | Field-inline | 79 passed |
| Affected-crate suites | Base | 654 passed, 8 skipped |
| Affected-crate suites | Field-inline | 635 passed, 3 skipped |
| Full `jolt-prover` suite | `prover-fixtures` | 23 passed |
| Full `jolt-prover` suite | `prover-fixtures,zk` | 29 passed |
| Full `jolt-prover` suite | `prover-fixtures,akita` | 30 passed |
| Full `jolt-prover` suite | `prover-fixtures,field-inline` | 33 passed |
| Full `jolt-prover` suite | `prover-fixtures,field-inline,zk` | 23 passed |
| Full `jolt-prover` suite | `prover-fixtures,field-inline,akita` | 30 passed |
| `tracer` | `test-utils,field-inline,fp128-field-inline` | 208 passed |

The skips are feature- or ignore-related, not missing guest-toolchain
substitutes for execution. Guest end-to-end tests executed in the completed
acceptance runs. Workspace clippy passed with both `host` and `host,zk`;
selected field-inline clippy checks also passed. No-default-feature checks
passed for `jolt-riscv` and `jolt-program` in plain, serialization, and
field-inline configurations.

Deterministic clear Dory fixtures were regenerated in separate worktrees at
`47130f3dc` and `4e9f1cdd8`, using the same guest inputs and four Rayon threads.
The proof, preprocessing, and public-input sections are byte-identical:

| Fixture | Proof bytes | Proof SHA-256 |
| --- | ---: | --- |
| `standard-muldiv-small` (optimized) | 64,800 | `c9b841c1dce5cb5562aa91c7e2eb42f784634c75bb4b05e763d5a0bba87262e9` |
| `standard-field-inline-eqpoly-modular-v4` (reference) | 70,186 | `c07071ec31c31bb05a6ca982c0edc376f04944b22b8833b92e3af97d3a199d12` |

Both fresh proofs verified before their respective tamper-rejection checks.
At `b233f6f5e`, the telemetry version 4 profiling smoke test passed (one test,
no skips), and profiling-enabled clippy passed. All five full verifier
configurations also passed:

| `jolt-verifier` features | Passed | Skipped |
| --- | ---: | ---: |
| `prover-fixtures` | 162 | 14 |
| `prover-fixtures,zk` | 137 | 25 |
| `prover-fixtures,akita` | 102 | 9 |
| `prover-fixtures,field-inline` | 156 | 7 |
| `prover-fixtures,field-inline,akita` | 105 | 6 |

The final implementation at `b7a9d1eb9` adds a targeted hot-path adjustment:
`proof_rows()` is inlineable across crates, and each random-access window
reuses one proof slice. Disassembly confirms that the two new out-of-line
getter calls per window are eliminated in both feature modes. This preserves
the checked prefix bound and avoids adding call overhead to row extraction.
After this adjustment, all 38 ordinary and 53 field-inline witness tests passed,
as did both workspace clippy configurations, formatting, and style checks.

Performance/memory measurements are recorded below. They distinguish retained
row storage from process peak memory and row production from proving time.

Permanent tests cover independent layout and semantic properties: 64-byte
`Copy` rows, metadata/control bounds, exact accepted instruction reconstruction,
field operand projection, capture absence versus observed zero, memory
aliasing, malformed identity/RAM rejection, and no-op PC/boolean rules. Tests
use independent properties or live production paths rather than a copy of
removed row conversion code.

Existing interpreter/x86 differential and chunk-composition tests check rows,
PCs, sparse payload association, and execution outputs. Guest-dependent
checks must actually execute; a missing toolchain that causes skips is not a
passing backend acceptance run. Field coverage includes arithmetic,
`LoadAccumulateFromRegister`, `LoadAccumulateFromMemory`, and `AdviceLimb`,
including missing/extraneous payloads and register continuity failures.
Memory accumulation must retain encoded field rs2 while exposing no integer
rs2 and binding its RAM read to integer rd's value.

Ownership tests check that witness construction retains the producer's
allocation when another `Arc` owner exists. Shared proof bounds,
padding/lookahead, and lengths around powers of two have distinct failure
signals.

The acceptance matrix covers affected-crate suites (`jolt-riscv`,
`jolt-program`, `jolt-host`, `tracer`, `jolt-tracer-x86`, `jolt-witness`, and
`jolt-kernels`) and the following commands. The tables above record the
completed runs; test runs always use nextest.

```bash
cargo nextest run -p jolt-verifier standard_muldiv --features prover-fixtures --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures,zk --cargo-quiet
cargo nextest run -p jolt-prover --features akita,prover-fixtures --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures,field-inline --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures,field-inline,zk --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures,field-inline,akita --cargo-quiet
cargo nextest run -p jolt-verifier --features prover-fixtures --test-threads 1 --cargo-quiet
cargo nextest run -p jolt-verifier --features prover-fixtures,zk --test-threads 1 --cargo-quiet
cargo nextest run -p jolt-verifier --features prover-fixtures,akita --test-threads 1 --cargo-quiet
cargo nextest run -p jolt-verifier --features prover-fixtures,field-inline --test-threads 1 --cargo-quiet
cargo nextest run -p jolt-verifier --features prover-fixtures,field-inline,akita --test-threads 1 --cargo-quiet
cargo nextest run -p jolt-witness --features test-utils,parallel,field-inline --cargo-quiet
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo clippy -p jolt-witness --features test-utils,parallel,field-inline -q --all-targets -- -D warnings
cargo fmt --all --check
```

Also run affected no-default-feature/serialization builds and CI field-inline
checks, including witness and kernels and the tracer's
`field-inline,fp128-field-inline` configuration. Keep Akita and ZK in separate
builds. The current `e2e_matrix` covers ordinary guests and active `field_ops`
/ inactive `muldiv` field-profile execution across clear, ZK, and Akita modes;
retain specialized field reference/optimized and tamper coverage.

Compare deterministic clear Dory proof bytes with the pre-change revision
using identical programs, inputs, configuration, and preprocessing. Use
verification and tamper rejection for randomized ZK proofs. Preserve Akita
and committed-program/advice fixture coverage without restoring deleted
legacy-prover machinery.

### Measured behavior

Measurements compare baseline `47130f3dc` with implementation `b7a9d1eb9`
on the same Linux VM (16 vCPUs, AMD EPYC 9554P, 62 GiB RAM, Rust 1.95.0),
using four Rayon workers and the optimized `ci` profile (`opt-level=3`, no
LTO). `TRACER_PARALLEL` was unset. Separate target directories and copied
binaries prevented Cargo artifacts from crossing revisions.

A temporary diagnostic invoked the production trace/config/witness
constructors in fresh processes, with identical pinned guest ELFs and inputs.
Five alternating samples per variant excluded guest compilation and
preprocessing from their timers. The diagnostic was removed afterward;
no measurement-only API or benchmark remains in the workspace.

| Handoff path | Actual cycles / field events | Baseline witness construction, median ms | Unified, median ms |
| --- | ---: | ---: | ---: |
| Ordinary compact Fibonacci | 197,605 / 0 | 0.0010 | 0.0010 |
| Ordinary generic Fibonacci | 197,605 / 0 | 24.4148 | 0.0010 |
| Ordinary generic SHA-256 chain | 136,766 / 0 | 18.4076 | 0.0007 |
| Fibonacci with field support, no field activity | 197,605 / 0 | 22.4824 | 1.1953 |
| Active field operations | 1,761 / 68 | 0.1921 | 0.0272 |

The generic conversion cost disappears. The old ordinary compact path already
had constant-time handoff. Field-enabled handoff still validates field state;
it shares the row allocation and sparse events.

For inactive-field Fibonacci, retained row-element storage decreases from
26,874,280 to 12,646,720 bytes, and median RSS after witness construction
falls from 31.047 to 17.027 MiB. For active field operations, the row and
event element lower bound decreases from 239,496 to 125,760 bytes (`136N`
versus `64N + 192M`); this tiny workload's process RSS remains about 5.2 MiB.
These element totals exclude unused capacity and other allocations; the
baseline total also excludes its 68 separately allocated payloads, which the
new events store inline. The new aggregate keeps capacities private, so they
were not inferred from lengths. Process peak RSS is essentially unchanged
in these trace diagnostics: approximately 55 MiB for ordinary Fibonacci,
88 MiB with field support, and 37 MiB for the small active field workload.
The emulator still determines that peak.

Cold trace timing varied substantially. The final ordinary compact/unified
Fibonacci medians were 47.219/54.114 ms, with ranges 45.967–54.282 and
37.858–59.967 ms; SHA medians were 43.973/51.544 ms, with ranges
37.864–50.385 and 32.448–52.711 ms. Earlier paired batches were flat for
Fibonacci and faster for SHA. The initial active-field slowdown reversed in
a dedicated ten-pair repetition. These samples do not establish a stable
trace-throughput change. The reported handoff and storage improvements do
not depend on claiming one.

The existing optimized Dory profiler separately measures proving time,
excluding tracing and witness construction. Three alternating final pairs
used Fibonacci and SHA-256 chaining at scale 18, plus the fixed-input field
workload at scale 16. The field guest has 1,009 actual cycles in this profiler;
its inputs differ from the larger-limb handoff diagnostic, and `--scale` does
not sweep its payload density.

| Optimized Dory workload | Baseline median [min–max], seconds | Unified median [min–max], seconds | Median change |
| --- | ---: | ---: | ---: |
| Fibonacci, scale 18 | 13.265 [13.104–13.705] | 13.411 [13.017–13.552] | +1.10% |
| SHA-256 chain, scale 18 | 12.470 [12.367–12.705] | 12.616 [12.575–12.668] | +1.17% |
| Field operations, scale 16 | 1.003 [0.999–1.061] | 1.010 [0.952–1.027] | +0.78% |

All three final median changes are below 2%; no reproducible regression above
that target was observed in this measured matrix.

These are bounded `ci`-profile measurements, not a large-scale or fat-LTO
release performance guarantee. The end-to-end target is no reproducible
prover-time regression above 2%; the sample ranges and configuration must
accompany any claim about that target. Reproduce the proving measurements
in separate worktrees with:

```bash
RAYON_NUM_THREADS=4 cargo run --profile ci -p jolt-prover --features profiling -- profile --name fibonacci --scale 18 --backend optimized --format chrome
RAYON_NUM_THREADS=4 cargo run --profile ci -p jolt-prover --features profiling -- profile --name sha2-chain --scale 18 --backend optimized --format chrome
RAYON_NUM_THREADS=4 cargo run --profile ci -p jolt-prover --features profiling,field-inline -- profile --name field-ops --scale 16 --backend optimized --format chrome
```

## Alternatives considered

| Alternative | Assessment |
| --- | --- |
| Late-assignable PC on every row | Adds an unnecessary state and proof-access checks when the map is available during execution. |
| Per-cycle sequence-count and recorded-ID side tables | Adds memory traffic; the count fits inline and valid captured IDs match operands. Only field payloads need separate storage. |
| A 72-byte, non-`Copy` row under field-inline | Puts a pointer on every ordinary cycle and loses the compact-row contract. Sparse storage also removes the second retained representation. |
| Resolve PC in witness accessors | Repeats mapping work in hot consumers despite room to retain the result. |
| Move the canonical row into `jolt-program` | Couples low-level proof/lookup consumers to execution; `jolt-riscv` already owns the needed vocabulary. |
| Share only the packing helper | Leaves competing row APIs, conversions, and allocation ownership. |

The main risks remain changed acceptance of malformed observations, field
versus integer capture semantics, padding-induced proof-shape changes,
analysis-wire compatibility, and accidental copying during handoff. The
construction, ownership, serialization, and acceptance contracts above make
those boundaries explicit.
