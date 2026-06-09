# Spec: Proof Trace Row Layout

| Field       | Value |
|-------------|-------|
| Author(s)   | Quang Dao |
| Created     | 2026-05-14 |
| Status      | in progress (foundation implemented) |
| PR          | follow-up to [#1522](https://github.com/a16z/jolt/pull/1522) |

> **Implementation status (foundation slice).** The accessor API, builder,
> parity tests, and one default layout have landed; consumer cutover and the
> performance gate are deferred follow-ups.
>
> The type lives in the new `crates/` (not the deprecating `jolt-core`):
>
> **Landed (Execution slices 1–3):**
> - `crates/jolt-riscv/src/trace_row.rs`: `JoltTraceRow` with the **Option C
>   balanced-packed 64-byte layout** (checked `size_of` assertion), private
>   physical slots with memory-row aliasing, native `CircuitFlagSet`/
>   `InstructionFlagSet`, and logical accessors for all columns. Checked newtype
>   `BytecodeIndex(u64)` over `u32` storage; `u64` guest addresses;
>   construction-time range checks for immediates and register ids; loud
>   `TraceRowError` (a `thiserror` enum) on memory-row contract violations.
>   `from_components` is the producer contract and depends only on `jolt-riscv`
>   (no `jolt-core`, no `tracer`, no lookup tables).
> - `crates/jolt-lookup-tables/src/traits.rs`: `InstructionLookupTable for
>   JoltTraceRow`, delegating through the row's cached `instruction_kind()` (this
>   crate owns `LookupTableKind`, so the accessor cannot live in `jolt-riscv`).
> - `tracer/src/trace_row.rs`: `cycle_to_trace_row` / `build_trace_rows` — the
>   `Cycle` → `JoltTraceRow` conversion (extracts logical values, rejects
>   source-only cycles, computes the bytecode PC via `jolt-program`); errors via
>   the `thiserror` enum `CycleConversionError`. The dependency now points
>   `tracer -> jolt-riscv`.
> - Tests: jolt-riscv per-class aliasing (non-memory / LD / SD), no-op/`Default`
>   parity, size/overflow/rejection; and the layout guard — a `host`-gated **real
>   fibonacci-trace** parity test in `jolt-core` comparing every accessor (built
>   via the tracer conversion) against `R1CSCycleInputs::from_trace`.
>
> **Deferred (Execution slices 4–11):** the `jolt-eval`
> `trace_row_accessor_parity` invariant; consumer cutover (R1CS inputs, Spartan
> outer, RAM/register/instruction-lookup/bytecode phases);
> `JoltTraceCycle::try_new` hot-path removal; `CompactBytecodeRow` co-design; and
> the ≤2% end-to-end prover-time benchmark gate. The layout choice is documented
> as the conservative default; it must be confirmed against that gate before the
> hot-path cutover relies on it.

## Summary

PR #1522 separates decoded source instructions from final Jolt instruction
rows, and introduces `JoltTraceCycle<'a>` as an interim proof-facing adapter
over tracer-owned `Cycle` values. That fixes the ownership boundary for static
proof metadata, but it still leaves prover hot paths repeatedly adapting a raw
tracer cycle and pulling values from a large enum-shaped representation.

This follow-up should design and implement a materialized proof trace row:
`JoltTraceRow`. It is built once after tracing and final bytecode expansion, then
used by witness generation, Spartan, RAM/register read-write checking, and
instruction lookup phases. The key design choice is to separate logical proof
columns from physical storage. Proof code still asks for `Rs1Value`,
`RamReadValue`, `RamWriteValue`, flags, bytecode PC, and lookup metadata, but the
row layout is free to pack values that are mutually exclusive or derivable for
final Jolt rows.

The trace-row choice is coupled to bytecode storage. Today
`BytecodePreprocessing` stores bytecode as padded `Vec<JoltInstructionRow>`,
where each row is 48 bytes in memory, plus a `BytecodePCMapper` implemented as
`Vec<Vec<(u16, usize)>>`. A lookup-heavy trace row only makes sense if bytecode
metadata lookups are compact and cache-friendly. This spec therefore treats
proof trace rows and preprocessed bytecode rows as one layout design problem.
The current design space ranges from a lookup-heavy 40-48 byte trace row to a
more expanded 80 byte row similar to the companion C++ tracer.

## Intent

### Goal

Introduce a proof-facing `JoltTraceRow` representation that replaces direct
prover use of `tracer::instruction::Cycle` while preserving the existing logical
proof columns and prover/verifier semantics.

The follow-up should introduce or clarify these boundaries:

- `JoltTraceRow`: a compact, copyable row used by prover-side witness and
  sumcheck code after source tracing has been converted to final Jolt rows.
- `JoltTraceRows` or an equivalent builder/sink: constructs rows once from
  final trace events, final bytecode preprocessing, and selected profile
  metadata. The first implementation may adapt the existing `Vec<Cycle>`
  boundary, but the API should not require materializing a full `Vec<Cycle>` as
  the permanent producer contract.
- logical accessors: methods such as `rs1_value()`, `rs2_value()`,
  `rd_pre_value()`, `rd_write_value()`, `ram_address()`, `ram_read_value()`,
  `ram_write_value()`, `bytecode_pc()`, `unexpanded_pc()`, `circuit_flags()`,
  `instruction_flags()`, and `lookup_table()`.
- physical storage slots: private fields that may alias logical columns when
  the selected final row contract proves the values are equal or mutually
  exclusive.
- optional phase-local materialization: temporary SoA columns or dense vectors
  created for phases that repeatedly scan one derived value and benefit from
  avoiding recomputation.
- compact preprocessed bytecode rows: an optional replacement for the current
  48-byte `JoltInstructionRow` bytecode table when trace rows choose to recover
  static metadata through `bytecode_pc`.

The implementation should make proof code depend on logical accessors, not raw
layout fields. That keeps the row representation benchmarkable: 40B, 48B, 64B,
and 80B layouts can be compared without changing proof semantics.

The implementation deliverable should be one production representation, not a
family of permanent layout variants. The code PR should land the accessor API,
the builder, parity tests, and one selected default layout. Candidate layouts
may be compared in local benchmark branches or benchmark-only scaffolding, but
they should not remain as public feature flags or alternate production paths
unless a later optimization PR makes that explicit.

### Invariants

- `JoltTraceRow` is proof-facing. It is produced after tracing/decode has been
  finalized and after each executed row has a final `JoltInstructionRow`.
- Source-only rows must not enter `JoltTraceRow` construction. Construction
  must fail loudly if a raw cycle cannot be associated with a final Jolt row.
- Prover/verifier semantics do not change. For every cycle index `t`, the
  logical values exposed by `JoltTraceRow` must match the values currently
  derived from `Cycle`/`JoltTraceCycle`:
  - `PC`, `UnexpandedPC`, `NextPC`, `NextUnexpandedPC`;
  - `Rs1Value`, `Rs2Value`, `RdWriteValue`, register read/write addresses;
  - `RamAddress`, `RamReadValue`, `RamWriteValue`, RAM read/write address;
  - instruction inputs, lookup operands, lookup output, flags, and table
    routing.
- The logical proof columns remain separate even if the physical layout aliases
  their storage. `RamWriteValue` and `Rs2Value` are still different logical
  columns, even if a store row returns both from the same physical slot.
- The current final Jolt memory-row contract is:

  ```text
  LD:
    RamAddress     = effective address
    RamReadValue   = RdWriteValue
    RamWriteValue  = RdWriteValue

  SD:
    RamAddress     = effective address
    RamReadValue   = old memory value
    RamWriteValue  = Rs2Value
  ```

  This contract follows from the existing R1CS constraints and from PR #1522's
  source/final split: narrow loads/stores, atomics, and store-conditionals are
  source-level operations that lower to final `LD`, compute rows, and final
  `SD` rows before proving.
- If a future final instruction violates the current memory-row contract, one
  of these must happen before enabling it:
  - lower it into canonical `LD`/compute/`SD` rows;
  - introduce an explicit new final row class and extend `JoltTraceRow`
    accessors/tests for that class;
  - choose a less packed layout that stores the newly independent values.
- `JoltTraceRow` must not become a new source of instruction identity. Final
  row identity remains `JoltInstructionRow` plus its canonical operation name
  and compact final tag.
- Dense bytecode/profile indexes are local indexes. They may be stored in a row
  for speed, but they are not persistent instruction identity. Public proof code
  should see a logical bytecode index accessor, not a public `u32` field.
- `bytecode_pc` may be stored physically as `u32` only if construction checks
  `bytecode_preprocessing.code_size <= u32::MAX as usize`. This is a practical
  compact-storage choice for current prover memory budgets, not a semantic fact
  of the protocol.
- If Jolt ever supports bytecode tables beyond `u32::MAX` entries, we should not
  need to rewrite every proving loop. The row's physical fields should be
  private, so the implementation can widen the local index in one storage module,
  introduce a vector-level compact/wide trace storage enum, or materialize a
  phase-local wide index column. The initial implementation may reject oversized
  bytecode explicitly, but that rejection must be documented as a storage-format
  limitation rather than an instruction-catalog or proof-system limitation.
- `unexpanded_pc` and guest addresses must not be stored as `usize` in a fixed
  trace or bytecode layout. Use `u64` for absolute RV64 addresses, or use a
  checked narrower offset/delta with an explicit base and range check. `usize`
  is acceptable in transient host APIs, but it should not define row size or
  serialized/preprocessed layout.
- Any narrowed cached index, offset, immediate, register id, tag, or packed
  field must have a construction-time range check. For example, packed register
  ids require an explicit maximum-register bound, and a compact immediate field
  requires proof that all final-row immediates fit the chosen encoding.
- The row layout must be asserted in tests with `size_of::<JoltTraceRow>()`.
  Any future size increase should be intentional and reviewed.
- If bytecode storage is compacted, `size_of::<CompactBytecodeRow>()` or an
  equivalent storage-budget assertion must be added as well.
- The selected layout must have a zero/default no-op row whose logical accessors
  return the same values as current `Cycle::NoOp` paths.
- The verifier should not depend on `JoltTraceRow`. The verifier consumes
  bytecode preprocessing and proof data, not prover trace storage.
- If phase-local SoA columns are added, they must be derived from
  `JoltTraceRow` accessors and must not reintroduce independent semantics.

### Non-Goals

- Do not implement this inside PR #1522 unless the current PR is explicitly
  expanded. This is a follow-up design and implementation.
- Do not change RISC-V, expansion, RAM, register, lookup, or R1CS semantics.
- Do not change proof serialization or verifier public inputs.
- Do not expose `JoltTraceRow` as a stable persisted trace format in this
  follow-up. It is an internal prover representation unless a separate spec
  says otherwise.
- Do not keep `JoltTraceCycle<'a>` as a long-term compatibility layer after the
  full cutover. It may remain temporarily during migration, but the end state
  should have proof hot paths consume `JoltTraceRow`.
- Do not add local caching hacks around repeated `JoltTraceCycle::try_new`.
  The fix is to build the proof-facing trace representation once.
- Do not make `jolt-riscv` own prover memory layout policy. Static row identity
  and flags may be queried through final-row metadata, but trace storage layout
  belongs to the prover/tracing boundary.
- Do not silently change persisted verifier preprocessing or proof
  serialization. If compact bytecode rows change serialized preprocessing, the
  implementation PR must make that format change explicit and update
  serialization tests.

## Evaluation

### Acceptance Criteria

- [ ] A new proof-facing row type exists, tentatively named `JoltTraceRow`.
- [ ] The row type has private physical fields and public logical accessors.
- [ ] The selected default layout has a checked size budget. The implementation
      PR ships exactly one production layout; 40B, 48B, 64B, and 80B candidates
      may inform the choice but should not all land as maintained variants.
- [ ] A builder constructs `Vec<JoltTraceRow>` once from final trace data and
      bytecode preprocessing.
- [ ] The builder rejects source-only cycles that do not have a final
      `JoltInstructionRow`.
- [ ] Unit tests compare every logical accessor against the current
      `Cycle`/`JoltTraceCycle` derivation on representative rows.
- [ ] A `jolt-eval` parity invariant captures the reference-vs-optimized
      property that `JoltTraceRow` accessors match the current
      `Cycle`/`JoltTraceCycle` derivation. The first implementation should at
      least generate deterministic tests; fuzz/red-team targets may be added
      once useful trace-row input generation exists.
- [ ] Focused memory-row tests cover final `LD`, final `SD`, expanded `LB/LH/LW`
      families, expanded `SB/SH/SW` families, LR/SC, and AMO source expansions.
- [ ] `R1CSCycleInputs::from_trace` or its replacement consumes
      `JoltTraceRow`, not raw `Cycle`.
- [ ] Spartan outer sumcheck hot loops consume `JoltTraceRow` or
      phase-local columns derived from it.
- [ ] RAM read-write, RAM RA, RAM hamming, register read-write, bytecode RA,
      and instruction lookup phases no longer repeatedly adapt raw cycles for
      final-row metadata.
- [ ] `JoltTraceCycle::try_new` call sites in proof hot paths are removed or
      reduced to construction-time checks.
- [ ] Standard and ZK prover tests continue to pass.
- [ ] If the chosen trace-row layout relies on bytecode metadata lookups, a
      compact bytecode-row design is implemented or explicitly deferred with
      benchmark evidence showing the current 48-byte `JoltInstructionRow` table
      is not a material bottleneck under the performance gate below.
- [ ] Any `u32` `bytecode_pc` or compact bytecode index has a checked
      construction path and a test for overflow/rejection.
- [ ] Proving code does not depend directly on the physical width of
      `bytecode_pc`. It should use an accessor/logical index type, so widening
      the physical storage later is localized to trace construction and storage.
- [ ] Any fixed-layout guest address field is `u64` or a checked offset/delta,
      not raw `usize`.

### Testing Strategy

Run the normal checks for any implementation PR:

```bash
cargo fmt -q
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

Add focused tests:

- `JoltTraceRow` accessor parity tests against raw `Cycle` for all final Jolt
  instruction kinds supported by the selected profile.
- a `jolt-eval` invariant, tentatively `trace_row_accessor_parity`, that checks
  each logical accessor against the existing reference derivation through
  `Cycle`/`JoltTraceCycle`. This is the stable guard for future layout changes:
  alternate physical storage is allowed only when the logical accessor outputs
  remain unchanged.
- property-style tests over random final rows where the old `Cycle` path can
  generate meaningful register/RAM state.
- expansion-driven tests for source memory instructions proving that source
  narrow/atomic/store-conditional rows lower into final rows that satisfy the
  memory-row contract.
- no-op/padding tests proving default rows match current no-op semantics.
- size/layout tests for the chosen row representation.
- regression tests around bytecode PC mapping, since several layouts may store
  only a local `bytecode_pc` and derive static metadata from bytecode tables.
- bytecode storage parity tests proving any compact bytecode row reconstructs
  the same final `JoltInstructionRow` logical metadata as the current table.
- overflow tests for narrowed indexes and offsets, especially `bytecode_pc:
  u32` and any compact unexpanded-PC encoding.

### Performance

The implementation should benchmark trace construction memory and prover hot
paths before committing to a default layout. The decision should be based on at
least these measurements:

- peak RSS and row-vector bytes for large traces;
- time to build `Vec<JoltTraceRow>` from raw trace data;
- Spartan outer sumcheck time;
- instruction lookup read-RAF time;
- RAM read-write, RAM RA, RAM hamming, and RAM val-check time;
- register read-write and register val-check time;
- bytecode read-RAF time;
- cache behavior when bytecode is small, medium, and intentionally large enough
  that repeated bytecode metadata lookups may miss cache.

Performance is a gate for the implementation PR, not just an informational
appendix. The selected layout should satisfy:

- no proof, verifier, or serialization correctness changes;
- no more than a 2% end-to-end prover-time regression on
  `prover_time_fibonacci_100` and `prover_time_sha2_chain_100`, measured against
  the same base revision on the same machine with the same feature set;
- material trace-row storage smaller than today's 96-byte `Cycle`, preferably
  no larger than 64 bytes unless the expanded layout has clear phase-level
  evidence, with a checked `size_of::<JoltTraceRow>()` assertion;
- phase-level measurements for Spartan outer, RAM/register phases,
  instruction lookup read-RAF, and bytecode read-RAF recorded as diagnostic
  evidence for the layout choice.

If benchmark variance makes a 2% wall-clock decision unreliable, the code PR
should report the raw measurements and profile evidence and choose the smaller
or simpler layout unless there is a clear phase-level win.

Compact bytecode rows may be deferred only with named benchmark or profile
evidence. Acceptable evidence includes one of:

- lookup-heavy and ordinary guest benchmarks, including
  `prover_time_fibonacci_100` and `prover_time_sha2_chain_100`, showing the
  trace-row layout is neutral within the 2% end-to-end gate while keeping current
  `Vec<JoltInstructionRow>` bytecode storage;
- profiler data showing bytecode metadata lookup/read-RAF is less than 5% of
  end-to-end prover time for the selected layout and is not the source of a
  measured phase regression;
- a direct comparison showing a compact bytecode prototype does not materially
  improve the selected layout on the named benches.

Expected layout targets:

| Layout | Approx. row size | Idea | Expected risk |
|--------|------------------|------|---------------|
| Minimal indexed | 40B | store dynamic value slots plus `bytecode_pc` and a 32-bit packed metadata word | more bytecode/profile table traffic |
| Minimal plus PC | 48B | minimal row plus cached `unexpanded_pc` | still lookup-heavy, but avoids a common PC lookup |
| Balanced packed | 56-64B | packed dynamic slots plus hot metadata such as `imm`, flags, table id, maybe register ids | best likely default if bytecode lookups are costly |
| Address-cached | 64-72B | balanced or minimal row with explicit RAM address if deriving address is expensive | spends bytes on a value used heavily by RAM phases |
| Expanded row | about 80B | store most values directly, similar to companion C++ tracer layout | fewer lookups but higher memory bandwidth |
| Current Rust `Cycle` | 96B today | enum-shaped tracer representation | too large and semantically tied to tracer internals |

No one layout is obviously right without measurements. The benchmark should
include both ordinary guest programs with locality and adversarial bytecode
access patterns where bytecode-table lookup costs are more visible.

Current bytecode storage baselines to measure:

| Item | Current in-memory size | Notes |
|------|------------------------|-------|
| `JoltInstructionKind` | 2B | final tag enum |
| `NormalizedOperands` | 32B | dominated by `imm: i128` and 16-byte alignment |
| `JoltInstructionRow` | 48B | current bytecode table entry |
| `Vec<JoltInstructionRow>` | 48B per padded entry | bytecode is prepended with no-op and padded to power of two |
| `(u16, usize)` | 16B | current PC-map inner entry |
| `Vec<(u16, usize)>` | 24B header plus entries | each outer PC-map slot has a Vec header |
| `BytecodePreprocessing` | 64B header plus heap allocations | `code_size`, bytecode table, PC mapper, entry address |

These sizes are not a criticism of the current representation; it is simple and
typed. They matter because a trace row that stores only `bytecode_pc` shifts
more pressure onto this table.

## Design

### Architecture

Current proof-facing path after PR #1522:

```text
tracer::Cycle
  |
  | repeated in many phases
  v
JoltTraceCycle::try_new(&cycle)
  |
  +-- dynamic values from Cycle
  +-- final static metadata from JoltInstructionRow
  v
R1CS / RAM / registers / instruction lookups
```

Target path:

```text
final trace events
  |
  | one checked construction pass
  v
Vec<JoltTraceRow>
  |
  +-- logical accessors expose proof columns
  +-- physical slots stay private and benchmarkable
  v
R1CS / RAM / registers / instruction lookups
```

For a scoped first implementation, "final trace events" can be the current
materialized `Vec<tracer::Cycle>`. That keeps the cutover small and gives parity
tests a direct reference path. The longer-term target is a trace-row sink or
builder that the tracer can populate directly once final bytecode metadata is
available, avoiding simultaneous materialization of both `Vec<Cycle>` and
`Vec<JoltTraceRow>`.

The important API shape is:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct JoltTraceRow {
    values: TraceValueSlots,
    meta: TraceRowMeta,
}

impl JoltTraceRow {
    #[inline(always)]
    pub fn rs1_value(&self) -> u64;

    #[inline(always)]
    pub fn rs2_value(&self) -> u64;

    #[inline(always)]
    pub fn rd_pre_value(&self) -> u64;

    #[inline(always)]
    pub fn rd_write_value(&self) -> u64;

    #[inline(always)]
    pub fn ram_address(&self) -> u64;

    #[inline(always)]
    pub fn ram_read_value(&self) -> u64;

    #[inline(always)]
    pub fn ram_write_value(&self) -> u64;
}
```

Callers should use the accessors. They should not match on private storage
slots or reinterpret field names as logical proof columns.

### Physical Aliasing For Memory Rows

The central packing opportunity is that final memory rows have mutually
exclusive or equal register/RAM values.

One possible physical slot layout:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct TraceValueSlots {
    rs1_value: u64,
    rs2_value_or_load_addr: u64,
    rd_pre_or_store_pre: u64,
    rd_post_or_store_addr: u64,
}
```

Logical interpretation:

```text
Non-memory row:
  rs1_value             -> Rs1Value
  rs2_value_or_load_addr-> Rs2Value
  rd_pre_or_store_pre   -> Rd pre-value
  rd_post_or_store_addr -> RdWriteValue
  RAM logical columns   -> 0

LD row:
  rs1_value             -> Rs1Value
  rs2_value_or_load_addr-> RamAddress
  rd_pre_or_store_pre   -> Rd pre-value
  rd_post_or_store_addr -> RdWriteValue, RamReadValue, RamWriteValue
  Rs2Value              -> 0

SD row:
  rs1_value             -> Rs1Value
  rs2_value_or_load_addr-> Rs2Value, RamWriteValue
  rd_pre_or_store_pre   -> RamReadValue
  rd_post_or_store_addr -> RamAddress
  Rd pre/Rd write       -> 0
```

This is Igor's suggestion in its safest form: merge physical storage for memory
operations, but do not merge the logical proof columns.

Accessors can make the contract explicit:

```rust
impl JoltTraceRow {
    #[inline(always)]
    pub fn ram_address(&self) -> u64 {
        if self.is_load() {
            self.values.rs2_value_or_load_addr
        } else if self.is_store() {
            self.values.rd_post_or_store_addr
        } else {
            0
        }
    }

    #[inline(always)]
    pub fn ram_read_value(&self) -> u64 {
        if self.is_load() {
            self.values.rd_post_or_store_addr
        } else if self.is_store() {
            self.values.rd_pre_or_store_pre
        } else {
            0
        }
    }

    #[inline(always)]
    pub fn ram_write_value(&self) -> u64 {
        if self.is_load() {
            self.values.rd_post_or_store_addr
        } else if self.is_store() {
            self.values.rs2_value_or_load_addr
        } else {
            0
        }
    }
}
```

The accessors can branch on cached circuit flags, an explicit compact row class,
or final `JoltInstructionKind`. The exact choice should be benchmarked.

### Layout Options

#### Option A: Minimal Indexed Row

```rust
#[repr(C)]
pub struct JoltTraceRow {
    values: TraceValueSlots, // 32 bytes
    bytecode_pc: u32,        // local bytecode/preprocessing index
    packed_meta: u32,        // e.g. circuit flags, instruction flags, table id
}
```

Approximate size: 40 bytes.

The 32-bit word after `bytecode_pc` should not remain dead padding. One useful
packing is:

```text
bits 0..15    circuit_flags
bits 16..23   instruction_flags
bits 24..31   lookup_table_id_plus_one, where 0 means no lookup table
```

Other packings are possible, but this word should either be removed by a
different field ordering or spent on measured hot metadata.

Pros:

- smallest row vector;
- better memory bandwidth for phases that stream over trace rows;
- bytecode/profile metadata stays canonical in preprocessing tables.
- caches flags/table routing without increasing the 40B size.

Cons:

- most static metadata needs a bytecode table lookup;
- repeated random-ish bytecode lookup patterns may hurt cache behavior;
- immediate, register ids, and unexpanded PC are not local;
- harder to reason about performance without profiling bytecode locality.

This is attractive if trace memory bandwidth dominates and bytecode metadata
has excellent locality.

#### Option B: Minimal Plus Unexpanded PC

```rust
#[repr(C)]
pub struct JoltTraceRow {
    values: TraceValueSlots, // 32 bytes
    bytecode_pc: u32,
    packed_meta: u32,
    unexpanded_pc: u64,
}
```

Approximate size: 48 bytes.

Pros:

- still compact;
- avoids repeatedly recovering the source PC for shift/PC constraints;
- keeps most static metadata in bytecode tables.

Cons:

- still lookup-heavy for immediates, flags, lookup table routing, and register
  ids;
- if PC is not the dominant static lookup, the extra 8 bytes may not be the
  best cache trade.

This is the smallest layout that likely feels ergonomic for current proof code.

#### Option C: Balanced Packed Row

```rust
#[repr(C)]
pub struct JoltTraceRow {
    values: TraceValueSlots, // 32 bytes
    unexpanded_pc: u64,
    imm_abs: u64,
    bytecode_pc: u32,
    register_pack: u32,
    jolt_tag: u16,
    circuit_flags: u16,
    instruction_flags: u8,
    lookup_table_id: u8,
    imm_is_negative: u8,
    row_class: u8,
}
```

Approximate size: 64 bytes, depending on exact ordering and alignment.

Pros:

- caches the metadata most likely to appear in row-wise hot loops;
- still much smaller than today's 96B `Cycle`;
- avoids many bytecode table lookups while preserving packed RAM/register
  values;
- row class can make memory accessors branch cheaply.

Cons:

- more bytes than the lookup-heavy layouts;
- duplicates metadata that is already available in bytecode/profile tables;
- requires careful consistency tests so cached metadata cannot drift from
  preprocessing.

This is the likely default if benchmarks show that repeated bytecode lookups
are costly. Otherwise, the implementation should prefer the smallest layout
that satisfies the accessor contract and performance gate.

#### Option D: Address-Cached Row

This is a variant of Option B or C that stores `ram_address` as an explicit
`u64` instead of aliasing it into unused load/store register slots or deriving
it from `rs1 + imm`.

Pros:

- RAM phases use addresses heavily;
- no branch or immediate arithmetic to recover the address;
- simpler accessor implementation.

Cons:

- spends 8 bytes per row, including non-memory rows;
- may duplicate information already in the packed slots;
- phase-local materialization might be a better place to cache addresses.

This option should be selected only if RAM-address recovery shows up in
profiles.

#### Option E: Expanded Row

An expanded row stores most logical values directly:

```rust
#[repr(C)]
pub struct ExpandedJoltTraceRow {
    rs1_value: u64,
    rs2_value: u64,
    rd_pre_value: u64,
    rd_post_value: u64,
    ram_address: u64,
    ram_read_value: u64,
    ram_write_value: u64,
    unexpanded_pc: u64,
    imm: i64,
    packed_metadata: u64,
}
```

Approximate size: 80 bytes.

Pros:

- simplest mapping from row fields to proof columns;
- fewer bytecode/profile lookups;
- close to the companion C++ tracer's 80B row shape;
- easier to debug.

Cons:

- stores redundant values for loads and stores;
- higher memory bandwidth than packed layouts;
- less pressure to keep static metadata canonical in preprocessing tables.

This may still win if bytecode lookups dominate and trace memory bandwidth is
not the bottleneck.

#### Option F: SoA Or Phase-Local Materialization

Instead of choosing one AoS row layout for every phase, keep a compact canonical
row and materialize columns for phases that repeatedly scan one derived value:

```text
Vec<JoltTraceRow>
  |
  +-- RAM phase builds Vec<ram_address_chunk>
  +-- register phase builds Vec<rd_inc>
  +-- instruction lookup phase builds Vec<lookup_index>
```

Pros:

- compact canonical trace plus phase-specific locality;
- avoids bloating every row for a value only one phase needs;
- can align with future GPU/SIMD-friendly SoA paths.

Cons:

- extra construction time and temporary memory;
- more implementation complexity;
- risks rebuilding columns in several places unless ownership is disciplined.

This should be considered an optimization layer on top of `JoltTraceRow`, not a
replacement for a clean logical trace-row API.

### Metadata Lookup Policy

The core open question is how much static metadata to cache per row. A row can
store only `bytecode_pc` and recover everything else:

```text
row.bytecode_pc
  -> bytecode[row.bytecode_pc]
  -> JoltInstructionRow
  -> flags / imm / operands / lookup table / tag
```

Or it can cache hot metadata:

```text
row.circuit_flags
row.instruction_flags
row.lookup_table_id
row.imm_abs + sign
row.register_pack
row.unexpanded_pc
```

The lookup-heavy version is closer to MLIR bytecode's idea that names/tables are
the real metadata and bytecode holds compact indexes. The packed version is
closer to a CPU/GPU execution trace optimized for repeated streaming. Jolt has
both pressures: bytecode metadata is canonical and mostly static, but prover
loops repeatedly scan the trace and may not tolerate cache-missing bytecode
lookups.

The decision should be empirical. The implementation should make metadata
caching a layout decision behind accessor methods, not a semantic dependency at
call sites.

### Bytecode Storage Co-Design

Current bytecode preprocessing stores final rows directly:

```rust
pub struct BytecodePreprocessing {
    pub code_size: usize,
    pub bytecode: Vec<JoltInstructionRow>,
    pub pc_map: BytecodePCMapper,
    pub entry_address: u64,
}

pub struct JoltInstructionRow {
    pub instruction_kind: JoltInstructionKind,
    pub address: usize,
    pub operands: NormalizedOperands,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}

pub struct NormalizedOperands {
    pub rs1: Option<u8>,
    pub rs2: Option<u8>,
    pub rd: Option<u8>,
    pub imm: i128,
}
```

This is a good source-level and construction-level representation, but it is
not obviously the right storage representation for repeated prover bytecode
lookups. On a 64-bit host today, `JoltInstructionRow` is 48 bytes because
`NormalizedOperands` is 32 bytes and aligned to 16 bytes by `imm: i128`.

If `JoltTraceRow` stores only a compact `bytecode_pc`, then bytecode rows should
also have a compact proof-facing representation. A possible row shape is:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CompactBytecodeRow {
    /// Stable final instruction tag, not profile-local dense index.
    tag: u16,
    /// Sequence metadata for expanded virtual rows.
    virtual_sequence_remaining: u16,
    /// `rd`, `rs1`, `rs2` packed with a sentinel for None.
    packed_registers: u16,
    /// Circuit flags, instruction flags, table id, compression/first-row bits.
    packed_flags_and_table: u32,
    /// Absolute RV64 PC, or a checked offset in a different layout variant.
    unexpanded_pc: u64,
    /// Magnitude or compact representation of the immediate.
    imm_abs_or_inline_key: u64,
    /// Sign/extension metadata and spare checked bits.
    imm_and_row_class: u32,
}
```

That sketch is roughly 32 bytes depending on exact ordering and padding. It is
not a final proposal; it illustrates the target direction:

- keep `JoltInstructionRow` as the typed construction/API row;
- derive `CompactBytecodeRow` during preprocessing for prover hot paths;
- make compact bytecode reconstruct the same logical metadata as
  `JoltInstructionRow`;
- store absolute RV64 addresses as `u64`, or store checked offsets with a base.

The current `BytecodePCMapper` should also be revisited. It is implemented as
`Vec<Vec<(u16, usize)>>`, so it pays a 24-byte outer `Vec` header for every
address index, plus 16 bytes for each `(virtual_sequence_remaining, pc)` pair.
That is simple, but it is not compact. Follow-up options include:

- build the map only during preprocessing/trace-row construction, then drop it
  from proof-facing preprocessing if no later phase needs it;
- store a sorted `Vec<(address_or_index, virtual_sequence_remaining, pc)>` and
  binary search during construction;
- store a dense compact vector with fixed-size entries if address density makes
  that worthwhile;
- store first PC plus a compact sequence range for common expanded sequences.

The right choice depends on when `get_pc(address, sequence)` remains necessary.
If trace rows store `bytecode_pc`, most prover phases should no longer need to
recover PC from `(address, virtual_sequence_remaining)`.

### Integer Width Policy

There are two different PC notions in this design:

```text
bytecode_pc    = dense local index into expanded bytecode/preprocessing
unexpanded_pc  = original RV64 guest/source instruction address
```

They intentionally answer different questions. `bytecode_pc` identifies one
final row in the expanded bytecode table:

```text
expanded bytecode:
  0: NoOp
  1: ADDI at source pc 0x8000_0000
  2: LD   from source LW at pc 0x8000_0004
  3: SLL  from source LW at pc 0x8000_0004
  4: SRAI from source LW at pc 0x8000_0004
```

Here final row `3` has `bytecode_pc = 3`.

`unexpanded_pc` identifies the source instruction address before expansion.
Multiple final rows emitted for one source instruction share the same
`unexpanded_pc`:

```text
source LW at 0x8000_0004
  -> final LD    unexpanded_pc = 0x8000_0004
  -> final SLL   unexpanded_pc = 0x8000_0004
  -> final SRAI  unexpanded_pc = 0x8000_0004
```

So `bytecode_pc` is a local table position, while `unexpanded_pc` is a guest
architectural address. They should not be given the same type merely because
both contain "pc" in the name.

`bytecode_pc` is not persisted instruction identity and does not need to be
pointer-sized. It should still have a logical type that is wider than the
initial compact storage, so public prover code does not bake in the `u32`
choice:

```rust
#[repr(transparent)]
pub struct BytecodeIndex(u64);
```

The compact row can store a checked `u32` and expose the logical index through
an accessor:

```rust
#[repr(C)]
pub struct JoltTraceRowCompact {
    values: TraceValueSlots,
    bytecode_pc: u32,
    packed_meta: u32,
}

impl JoltTraceRowCompact {
    #[inline]
    pub fn bytecode_index(&self) -> BytecodeIndex {
        BytecodeIndex(u64::from(self.bytecode_pc))
    }
}
```

Using compact `u32` storage is reasonable if the builder enforces:

```rust
let pc = bytecode_preprocessing.get_pc(&row).ok_or(...)?;
let pc = u32::try_from(pc).map_err(|_| TraceRowError::BytecodePcTooWide { pc })?;
```

This implies a maximum of `u32::MAX + 1` bytecode entries for that trace-row
layout. In practice, a 4-billion-row bytecode table is far beyond current prover
memory budgets, but the code should reject it explicitly rather than relying on
that practical limit.

If this assumption stops being true, widening should be a storage decision, not
a semantic redesign. Two viable upgrade paths are:

```rust
#[repr(C)]
pub struct JoltTraceRowWide {
    values: TraceValueSlots,
    bytecode_pc: u64,
    packed_meta: u32,
    // padding or additional measured hot metadata
}
```

or a vector-level enum that chooses the layout once for the whole trace:

```rust
pub enum JoltTraceRows {
    Compact32(Vec<JoltTraceRowCompact>),
    Wide64(Vec<JoltTraceRowWide>),
}
```

The enum should be at the trace-vector level, not inside each row, so ordinary
hot loops do not pay a per-row tag. This keeps the initial compact layout agile:
the first implementation can reject oversized bytecode, while the public API and
proof phases are shaped so a future wide layout is a localized extension.

`unexpanded_pc` is a guest RV64 address, not a local table index. The semantic
representation is:

```rust
unexpanded_pc: u64
```

The compact representation is a checked delta from a `u64` base:

```rust
pc_base: u64,          // stored once in preprocessing
unexpanded_pc_delta: u32 // checked per row
```

That compact form is valid only when every source address used by the selected
program satisfies:

```rust
let delta = unexpanded_pc.checked_sub(pc_base).ok_or(...)?;
let delta = u32::try_from(delta).map_err(|_| TraceRowError::PcDeltaTooWide { delta })?;
```

An absolute `u32 unexpanded_pc` is not the preferred contract. It would quietly
impose a 32-bit guest address limit. A `u32` delta says something more precise:
this particular program's code range fits in 32 bits relative to `pc_base`.

The fixed row layout should not use `usize` for guest addresses. `usize` changes
with the host architecture and makes memory layout depend on the prover machine
instead of the guest ISA contract. Existing APIs can continue using `usize`
where the surrounding code already does, but `JoltTraceRow` and compact
bytecode storage should choose explicit widths.

### Construction Point

`JoltTraceRow` should be produced after:

1. guest execution produces final trace events, initially exposed as raw
   `Cycle` data;
2. source-only rows have been expanded to final Jolt rows;
3. bytecode preprocessing can map each final row to a local `bytecode_pc`;
4. final-row metadata is available from `JoltInstructionRow`.

Conceptually:

```text
ELF/source bytes
  -> SourceInstruction
  -> expansion
  -> JoltInstructionRow bytecode
  -> tracer execution
  -> final trace events
  -> JoltTraceRow construction or direct trace-row sink
  -> prover witness/sumcheck phases
```

The construction pass is where invariants should be checked once. Hot loops
should not repeatedly rediscover that a cycle is final-row backed.

The implementation should avoid making `Vec<Cycle>` a new architectural
dependency. It is acceptable as an initial adapter source because current tracer
and prover APIs already expose it, but the row builder should be shaped so a
future tracer path can emit `JoltTraceRow` directly without changing proof-phase
call sites.

## Alternatives Considered

### Keep `JoltTraceCycle<'a>` Forever

This is the smallest diff, but it preserves repeated adaptation in hot loops and
keeps prover code tied to tracer's enum-shaped `Cycle`. It is a good interim
boundary for PR #1522, not the clean end state.

### Store The Raw `Cycle` And Add Caches Beside It

This avoids a full cutover but creates two sources of truth: raw tracer data and
proof metadata caches. It also keeps the 96B `Cycle` allocation and adds more
memory on top. This is not the right long-term shape.

### Fully Minimal Row Only

A 40B row is appealing, but it may push too much work into bytecode table
lookups. It should be benchmarked together with compact bytecode storage, not
assumed in isolation.

### Fully Expanded Row Only

An 80B row is easy to understand and may perform well, but it ignores the
strong equalities already present in final memory rows. It should be a benchmark
candidate and maybe a debug layout, but not the only design.

### Derive RAM Address From `rs1 + imm`

For load/store rows, `RamAddress == Rs1Value + Imm` is already an R1CS
constraint. Deriving address can save one physical slot, but it adds an
accessor branch and signed/wrapping immediate arithmetic. Since RAM phases use
addresses heavily, this should be benchmarked against storing or aliasing the
address.

### Leave Bytecode As `Vec<JoltInstructionRow>`

This keeps preprocessing simple and typed, but the current row is 48 bytes and
contains construction-oriented fields such as `usize` address and `i128`
immediate. If trace rows rely on bytecode lookups for hot metadata, this table
may become part of the critical path. Keeping it unchanged is acceptable only if
benchmarks show the lookup-heavy trace layouts remain competitive.

### Use `usize` For Local And Guest Addresses

This matches several current APIs, but it is the wrong fixed-layout contract.
`usize` is host-width dependent. A proof-facing trace row should use `u32` or
`u64` depending on whether the field is a checked local index or a guest RV64
address.

### Treat RAM And Registers As One Logical Event

This is too aggressive. A load is both a memory read and a register write; a
store is both a memory write and register reads. The proof protocols still need
separate logical columns. Only the physical storage can be packed.

## Documentation

This is an internal prover representation, so no Jolt book update is required
for the first implementation PR. The code should document:

- why `JoltTraceRow` is proof-facing and not a source trace format;
- the final memory-row contract used by packed accessors;
- which fields are physical storage slots versus logical proof columns;
- the selected row size and why it was chosen.

If a later PR exposes trace serialization or profiling output based on
`JoltTraceRow`, that PR should update user-facing documentation.

## Execution

Suggested implementation slices:

1. Add `JoltTraceRow` with logical accessors and a conservative layout. Add
   parity tests against raw `Cycle` derivations.
2. Add explicit checked newtypes or construction helpers for narrowed fields
   such as `BytecodePc(u32)` and any compact PC offsets.
3. Add a construction pass that builds `Vec<JoltTraceRow>` once in the prover
   setup path from the current `Vec<Cycle>` source, while keeping the builder
   API compatible with a future direct tracer sink.
4. Choose and land one production layout. Benchmark-only alternatives can exist
   during development, but the merged implementation should expose a single
   default representation behind the accessor API.
5. If benchmarks favor lookup-heavy trace rows, add `CompactBytecodeRow` or an
   equivalent proof-facing bytecode table before relying on repeated
   `bytecode_pc` lookups in hot loops.
6. Cut `R1CSCycleInputs` and `ProductCycleInputs` over to `JoltTraceRow`.
7. Cut Spartan outer loops over to `JoltTraceRow` or derived phase-local
   columns.
8. Cut RAM/register/instruction-lookup/bytecode phases over to
   `JoltTraceRow` accessors or derived phase-local columns.
9. Add the `jolt-eval` accessor-parity invariant and keep size assertions.
10. Remove proof hot-path `JoltTraceCycle::try_new` usage once each phase is cut
    over.
11. Document the selected layout, the benchmark evidence, and any explicitly
    deferred bytecode compaction work.

Implementation should prefer `#[inline(always)]` for tiny accessors that sit in
sumcheck loops, but only after confirming they are on hot paths. The row type
should remain `Copy`, and construction should avoid heap allocation per row.

## References

- [#1522: Split source/Jolt instruction catalogs](https://github.com/a16z/jolt/pull/1522)
- [`specs/source-jolt-instruction-split.md`](./source-jolt-instruction-split.md)
- [`specs/inline-expansion-grammar.md`](./inline-expansion-grammar.md)
- Current Rust tracer `Cycle` size assertion:
  `tracer/src/instruction/mod.rs::rv64imac_cycle_size`
- Current R1CS memory/register equality constraints:
  `jolt-core/src/zkvm/r1cs/constraints.rs`
- Companion C++ tracer prior art in Quang's local companion checkout, not a
  repo-local `a16z/jolt` path:
  `/Users/quang.dao/Documents/SNARKs/jolt-cpp/src/tracer/trace_owner.hpp`
