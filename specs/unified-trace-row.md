# Design: one trace row from execution through proving

Status: proposed. Inspected upstream `main` at
[`47130f3dc`](https://github.com/a16z/jolt/commit/47130f3dc9a51a7ac2754a98ff0aa31981a6b810),
on 2026-10-03. Both [#1818](https://github.com/a16z/jolt/pull/1818) and
[#1734](https://github.com/a16z/jolt/pull/1734) have landed.
This proposal addresses [#1839](https://github.com/a16z/jolt/issues/1839).
The working branch used to write this document predates #1818;
implementation should start from the inspected
revision or newer. Source links below are pinned to that revision.

## Recommendation

Use `jolt_riscv::JoltTraceRow` as the single stored row throughout execution,
analysis, replay, and proving. Keep it 64 bytes and `Copy`, including under
`field-inline`. Retain full encoded instruction operands for reconstruction,
while preserving the current integer-only proof register accessors. Extend it
with exact virtual-sequence metadata and captured integer-register presence.
Store field-inline payloads separately in the same trace owner, indexed only
at cycles that have a payload.

Every emitted row should already contain its bytecode PC. There is no need for
an unbound-row state: execution receives a `JoltProgram` whose bytecode is
already expanded, and the existing PC mapper can index that bytecode before
execution. This is simpler than the late-assignment proposal in the issue and
keeps the current total, constant-time proof PC accessor.

Complete the change by transferring the producer's row allocation into the
witness backend. Unifying the Rust type without fixing that handoff would leave
the allocation problem in place. Keep the existing name `JoltTraceRow`, migrate
imports, and delete the separate `jolt_program::TraceRow` implementation rather
than maintaining two public names for the same concept.

## What has changed since the issue

| Finding at the inspected revision | Consequence |
| --- | --- |
| Without `field-inline`, SDK proving and profiling call `trace_compact` and then `from_compact`. Field-inline uses execution rows and `try_new`. | Non-field paths already share one compact row allocation. Do not promise another 4 GiB reduction at 2^26 cycles on them. |
| `TraceBackend::try_new` still allocates compact rows. Under `field-inline`, it now reuses `source.shared_rows()` for `raw_trace_rows`; only sources without a shared view require a raw-row copy. | Ordinary `OwnedTrace` field-inline proving retains two row representations, not three independent row allocations. Unification removes the compact/raw duplication. |
| SDK/host preparation populates `JoltProgram` with final expanded bytecode. `BytecodePCMapper::try_new` accepts it without the prepended no-op. | Bind PCs at production; neither a second trace walk nor optional PCs are necessary. Explicitly migrate the tested ELF-only generic-trace behavior described below. |
| `JoltTraceRow` has three reserved bytes. Its `u32` metadata uses 23 bits normally and 31 under `field-inline`. | Put the sequence count in two reserved bytes and presence masks in the third; the new masks cannot all fit in the existing metadata word. |
| `JoltTraceRow` now stores `instruction.integer_operands()`; `TraceRow` retains encoded operands and independent captures. | Store encoded IDs once and cache their integer projection's presence, keeping field-register IDs out of proof register accessors. |
| Field-inline instruction roles and validation now come from ordinary bytecode, including memory accumulation and advice-limb instructions. | Reuse `integer_operands`, `field_operands`, and the existing shape validator; do not restore the removed field-bytecode metadata table. |
| The two constructors accept different malformed inputs. Recorded register IDs can currently disagree with instruction operands. | Define a checked final-row contract explicitly; this is more than a mechanical type alias. |
| The legacy prover and its byte-diff acceptance lane are gone. | Validate against current prover/verifier fixtures and a before/after baseline, not removed commands. |

Evidence: [SDK proof path](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/jolt-sdk/src/host_utils.rs#L313),
[profile trace path](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-prover/src/profile.rs#L1029),
[witness construction](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-witness/src/backend/trace/mod.rs#L179),
[program definition](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-program/src/execution/trace.rs#L15),
[PC mapper](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-program/src/preprocess/bytecode.rs#L119).

## Row contract and layout

Keep the existing private, aliased value slots and logical proof accessors.
`CapturedState` remains the owner of packing the three final row classes:

| Row class | Slot 0 | Slot 1 | Slot 2 | Slot 3 |
| --- | --- | --- | --- | --- |
| Non-memory | rs1 value | rs2 value | rd pre-value | rd post-value |
| Load | rs1 value | RAM address | rd pre-value | rd post-value / loaded value |
| Store | rs1 value | rs2 value / RAM post-value | RAM pre-value | RAM address |

The proposed physical layout is:

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

Keep the current metadata word's feature-dependent layout: 16 circuit-flag
bits in base builds, 24 under `field-inline`, followed by six instruction flags
and the immediate sign. That leaves nine spare bits in base builds but only
one under `field-inline`. The original draft's proposal to add four presence
bits to that word does not fit the current field-inline build.

Use byte 63 instead, with the same definition in every feature mode:

| Row-control bits | Meaning |
| --- | --- |
| 0–2 | Captured integer rs1, rs2, and rd presence |
| 3 | Virtual-sequence count present |
| 4–6 | Integer rs1, rs2, and rd operand presence |
| 7 | Reserved, zero |

Cache integer-operand presence once from `instruction.integer_operands()` at
construction. The current projection only removes encoded operands; it does
not move IDs between slots. Thus each proof register accessor can gate the
corresponding stored encoded ID with its cached integer-presence bit, preserving
today's integer-only result without repeating instruction-shape dispatch in
hot loops. The capture mask is separate: a declared integer operand and an
observed register value are different facts. Compute these caches from the
canonical helper, not a second opcode-role table. Pin the slot-preserving
projection property in coverage of every field-inline operand shape; a future
projection that remaps slots requires revisiting this representation.

Represent `virtual_sequence_remaining` as `Option<u16>` through its accessor;
do not use a
count value as an absence sentinel. In particular, preserve `None` versus
`Some(0)`. Full encoded IDs, immediate, sequence metadata, and the existing
first-in-sequence and compressed flags support `JoltInstructionRow`
reconstruction, including its field operands. The PC mapper still enforces its
own virtual-sequence length limit.

Retain the compile-time size assertion and add feature-enabled coverage for
size, alignment, and `Copy`. No serialized representation depends on these
offsets or metadata bits. Define metadata masks in one place, with compile-time
width and non-overlap checks for both the metadata word and row-control byte.
There are no unused bytes left; future additions must fit the documented bit
budget or deliberately revise the layout.

### One checked construction boundary

Move the small `RegisterRead`, `RegisterWrite`, `RegisterState`, `RamRead`,
`RamWrite`, and `RamAccess` value types into `jolt-riscv`. They remain useful as
logical constructor inputs, execution accessor results, and wire values; they
are not another stored trace representation. `jolt-program` continues to own
program mapping and field-inline payloads, avoiding a dependency back from
`jolt-riscv` to `jolt-program`.

Use one checked row constructor, conceptually:

```rust
JoltTraceRow::new(instruction, registers, ram_access, bytecode_pc)
    -> Result<JoltTraceRow, TraceRowError>
```

It validates logical inputs, constructs `CapturedState`, and invokes its single
packing implementation. Interpreter and x86 adapters only extract observations
and resolve the PC; they do not implement their own slot collapse. Consolidate
the two row-error definitions, with producer context added by `TraceError`.
Remove or privatize `from_components` so callers cannot bypass the new
construction contract through the old value-only API.

Enforce these conditions in release builds:

- Loads have a RAM read, no captured integer rs2, and equal loaded and integer
  rd post-values.
- Stores have a RAM write, no captured integer rd, and equal RAM post- and
  integer rs2 values.
- Non-memory rows have `RamAccess::NoOp`, including when malformed RAM data
  happens to consist entirely of zeros.
- Every present captured integer-register ID matches the corresponding
  `instruction.integer_operands()` entry. Reject captures in field-only
  operand slots even when their numeric IDs happen to match. Preserve absent
  captures explicitly; do not infer them solely from operand presence.
- Immediates and operand IDs satisfy the existing storage bounds. PC zero is
  reserved for no-op rows; non-noops cannot use it, and no-ops must use it.
  No-ops have no captured register or RAM effects. The canonical padding row
  retains its existing proof columns.

Both real producers already obtain captured register IDs from their
instruction's integer operands. The current packing test deliberately stores
register 200 for operand 2, but no production need for retaining that
discrepancy was found. Rejecting
it is an intentional public constructor/deserializer change, not a claim that
all previously accepted inputs are preserved. Likewise, stop using
`from_instruction` to construct incomplete load/store observations; such test
fixtures must supply valid captures.

Capture presence matters independently of IDs. For example,
`FIELD_LOAD_ACCUMULATE_FROM_MEMORY` encodes an integer base in rs1, an integer
scratch destination in rd, and a field accumulator in rs2. It is a load-class
row: the RAM read equals integer rd's post-value, and it has no captured integer
rs2 despite the encoded rs2 being present. Its field accumulator values live
in the sparse payload. The three existing value-slot classes still suffice.

Preserve canonical `integer_operands()` and `field_operands()` projections,
including implicit accumulator reads and field destinations encoded in rs2.
Reuse `validate_field_inline_instruction` at program/bytecode boundaries;
among other checks it rejects x0 destinations for memory accumulation and
advice-limb writes. Preserve payload shape, bridge, and register-continuity
checks in the field witness. These checks use ordinary bytecode now; the old
separate field-inline bytecode metadata table has been removed. Row-local
construction validates representation invariants and does not replace these
program checks or the proof constraints.

Keep the tracer's instruction-specific `RegisterSnapshot` types and raw
`Cycle` representation. They capture execution before this final-row boundary;
source-only cycles and low-level loads to x0 are not automatically valid final
rows. Consume completed observations after the existing instruction expansion
and rd=x0 rewriting. This unification does not merge raw emulator snapshots
with the stored final row.

Expose the existing execution views (`instruction`, register reads/writes,
`ram_access`, and `JoltCycle`) on the unified row. Preserve the distinction
between a load's proof `ram_write_value()`—equal to its loaded value—and its
execution view, which reports a RAM read and no RAM write. Keep proof register
index accessors' current **integer operand** semantics through the cached
integer mask. `instruction()` returns all encoded operands; execution register
views use the separate capture mask. No field-register ID may leak into the
integer proof columns during this cutover.

Evidence: [existing compact layout](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-riscv/src/trace_row.rs#L203),
[execution constructor](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-program/src/execution/trace/row.rs#L262),
[mismatched-ID test](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-program/src/execution/trace/row.rs#L702),
[interpreter captures](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/tracer/src/instruction/mod.rs#L551).
The field-specific rules are owned by the [operand projections](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-riscv/src/row.rs#L85)
and [instruction validator](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-program/src/field_inline.rs#L44).

## Bind PCs during the existing production pass

Give `BytecodePCMapper` the instruction-aware lookup currently implemented by
`BytecodePreprocessing::get_pc`, including the no-op-to-zero rule. Make
preprocessing delegate to it. Its existing constructor already assigns the
same PCs to unpadded expanded bytecode and to bytecode with the leading no-op;
retain that implementation as the sole numbering rule.

Generic execution builds the mapper once from `program.expanded_bytecode`,
using the same implementation as preprocessing. Accept a separate map
allocation when preprocessing also exists: it is proportional to the program,
not trace length, and avoids adding a second execution entrypoint just to pass
a map. Share this per-execution map with replay workers. Do not cache it inside
the currently mutable `JoltProgram`: its public bytecode vector could invalidate
that cache.

The interpreter uses this context in its existing `Cycle` conversion pass.
Chunked execution retains an `Arc` to the mapping context in `WorkerSeed`, so
each replay can bind rows without reconstructing the map. The x86 backend does
the same during observation reassembly. Although x86 observations contain an
expanded row index, do not encode `row_index + 1` independently.

Missing mappings, oversized PCs, and source-only instructions return typed
errors. An externally constructed ELF-only `JoltProgram`, or execution at an
address missing from its expanded bytecode, no longer produces a final unified
row. This intentionally narrows existing public behavior: the current
`tracer_backend_traces_a_guest_elf_into_jolt_rows` test successfully traces an
ELF-only program. Migrate that final-row test to `build_jolt_program`, and retain
ELF-only execution coverage through the raw `Cycle` tracing/debugging APIs.
Production SDK/host paths already prepare expanded bytecode. Preparing it
implicitly inside the generic tracer would be an alternative API decision,
not a reason to introduce optional PCs. See the [existing ELF-only regression](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/tracer/src/execution_backend.rs#L607).

The row constructor can enforce PC width and the no-op rule; only the program
mapper can establish the correct PC for a particular bytecode. Treat row-local
deserialization as structural validation, not proof that the row belongs to an
arbitrary supplied program. Validate imported traces against that program as
part of import, before admitting them to witness construction. For non-noops,
check both the mapped PC and full reconstructed instruction against the
bytecode entry: address/sequence lookup alone does not validate kind, operands,
or flags. No-ops use reserved slot zero independent of source address and must
satisfy the no-effects contract above; canonical padding has the default
instruction metadata as well.

## One owner of rows and field payloads

Evolve the existing `OwnedTrace` storage into a shared trace aggregate in
`jolt-program`, conceptually:

```text
OwnedTrace { data: Arc<TraceData>, next: usize }
TraceData {
    rows: Vec<JoltTraceRow>,
    field_events: Vec<FieldEvent>, // field-inline only
}
FieldEvent { cycle: usize, data: Arc<FieldInlineTraceData> }
```

Keep fields private. A builder appends a core row and its optional payload
together. Field events are sorted, unique, and in range. Payload shape and
presence must agree with the corresponding instruction; keep the existing
field witness validation for operation, operands, bridges, and register-state
continuity. Reuse those rules rather than duplicating them in the storage
builder. Preserve the current full-instruction comparison with bytecode and
profile checks when installing the field witness (`with_field_inline` today).
An ordinary trace has no payload entries or per-cycle payload slots. Keep the
current payload shape: products and inverse-products are derived from decoded
register values, not additional stored fields.

Sequential field consumers merge rows and events in order. Random field
access uses a lookup in the sorted events; avoid a dense index column or a
hash map unless measurements justify it. Parallel column scans locate the
event range once per chunk and then merge sequentially; do not turn every row
visit into a binary search. In particular, Akita's existing dense increment
walks call `rd_increment_at` for every cycle: adapt those walks to the chunk
merge/cursor instead of hiding a sparse-table search inside that accessor.
The field witness already emits sparse Spartan and register rows; feed these
consumers from sparse events without materializing a dense field row array.
Parallel construction writes rows into one final allocation and combines
sparse event batches in cycle order;
do not collect a full intermediate vector of `(row, optional_payload)`.

Move the shared aggregate, not its elements, into witness construction:

- Evolve the existing `TraceSource::shared_rows` / `OwnedTrace` sharing hook
  to expose the aggregate and add a consuming handoff returning the same
  `Arc<TraceData>`. Reject a partially consumed cursor. Do not call the existing
  `into_rows`, which clones when the `Arc` is shared.
- Make `TraceBackend` accept retained trace data directly and remove its
  unused `T: TraceSource` type parameter and `PhantomData`. Its current
  constructor already refuses iterator-only sources; preserve that boundary.
- Let `RandomAccessRows`, field witnesses, and kernel consumers share that
  aggregate. Delete `raw_trace_rows` and `compact_trace_row`.
- Keep the streaming/replay trait separate from retained witness storage.
  Its per-item result must carry a core row and optional field payload
  together. This transient event is not another stored row format. Its slice
  view must expose the associated aggregate, with the existing rule that a
  partially consumed source cannot claim to expose the full remaining trace.

This ownership change is essential even without `field-inline`: replacing
`TraceRow` with `JoltTraceRow` and continuing to collect a new vector in
`try_new` would save no row memory.

Retain existing cycle ordering, chunk boundaries, advice tapes, final memory,
and device outputs. Padding stays a witness-domain concern. Before deriving
proof configuration, preserve the current compact path's canonical trailing
padding normalization; never discard interior no-ops or trim rows with field
payloads. Analysis/replay keep their reported execution lengths. A retained
proof view pairs the shared aggregate with a checked `physical_len` excluding
canonical trailing padding. `TraceBackend`, `RandomAccessRows`, and field
witnesses use that same view and validate its bound against the proof domain,
rather than consulting the allocation's full length independently. Compute
the bound during row production or the existing configuration pass; do not add
a normalization walk or copy. Test the exact power-of-two boundary:
`ProverConfig` includes a final no-op in its sizing, so changing which rows
count can change the proof.
Consolidate `derive`/`derive_compact` on the unified row without changing their
sizing law.

Evidence: [existing shared trace wrapper](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-program/src/execution/trace.rs#L218),
[field witness](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-witness/src/field_inline/mod.rs#L209),
[proof configuration](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-prover/src/config.rs#L61).

## Serialization and public API

Serialize logical values through a checked wire shim. The core row wire contains
the instruction, captured registers, RAM access, and bytecode PC. Deserialization
calls the same constructor. Never derive serde over private packed fields, and
never invent PC zero for a non-noop read from an old wire format.

The trace aggregate's wire joins field payloads with their logical rows; its
physical sparse event table and cursor are not the wire contract. Serialize
rows incrementally through borrowed views rather than assembling another full
wire-row vector. Deserialize directly into the aggregate builder.

Keep `ProgramSummary`'s full trace functionality in this change. Change its
trace field to `Arc<TraceData>` and provide `len`, iteration, and
read-only row access, so the SDK's common `summary.trace.len()` use remains
simple. Do not silently reduce it to a count or histogram. That would be a
separate product/API decision. Field payload access now belongs to the trace,
not to a `JoltTraceRow` member.

This is a source and analysis-wire breaking change: imports, constructors,
trace iteration, summary trace type, and direct `row.field_inline` access need
migration. Make `ProgramSummary`'s public serde boundary a checked envelope with
fixed magic and a schema version; `write_to_file` uses that same representation.
This check must also run for direct serde callers, not just a file helper. Put
bytecode before trace records in the new wire so the importer can validate PCs
and instructions while constructing the one row allocation. Document the new
schema and reject old unversioned files; do not add a speculative legacy reader.
If compatibility with a deployed reader is required before implementation, a bytecode-aware
import adapter can bind old logical rows, but it must not introduce a second
resident row vector or weaken validation.

The deployed proof and preprocessing formats remain unchanged. The separate
`Program::trace_to_file` path writes raw `Cycle` records and is outside this
wire migration.

Evidence: [current logical row wire](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-program/src/execution/trace/row.rs#L197),
[public summary and bincode output](https://github.com/a16z/jolt/blob/47130f3dc9a51a7ac2754a98ff0aa31981a6b810/crates/jolt-host/src/analyze.rs#L9).

## Expected effect and scope

For a generic non-field trace with N physical rows, the current conversion
temporarily holds approximately `64N + 64N` bytes of row elements; afterward,
the source can be dropped. The proposed handoff holds `64N` throughout. Removing
one full vector is 4 GiB of element storage at N = 2^26, but process peak RSS
depends on allocation capacity, other live buffers, and when the peak occurs.
This is a structural estimate, not a measured speedup or RSS result.

With `field-inline`, the current execution row is 72 bytes, and witness
construction keeps a 64-byte core copy plus the 72-byte execution rows.
For `OwnedTrace`, `shared_rows()` now retains the original execution allocation;
the live input handle does not represent a third row allocation. Only the
fallback for non-sharing sources copies raw rows. The ordinary retained row
baseline is therefore `136N`, excluding payloads and capacities. The proposal
retains `64N` plus sparse payload entries and the payloads themselves. On a
64-bit host an entry containing
`usize` and `Arc` costs approximately 16 bytes; include event density and vector
capacities in measurements rather than quoting a universal percentage saving.

The non-field SDK/profile path already has one compact vector. Its acceptance
target is unchanged proof behavior and no material performance regression.
Unification also leaves the interpreter's temporary `Vec<Cycle>` and x86's
observation buffer in place. Removing those requires a separate change to
instruction emission and parallel tracing; this proposal does not claim true
streaming or elimination of all trace-sized allocations.

Do not change proof equations, sumcheck constraints, lookup routing, instruction
tags, field encodings, or commitment layouts. Preserve analysis content and
behavior. Do not redesign bytecode storage or emulator execution as part of
this work.

## Implementation sequence

1. **Pin the baseline and constructor contract.** Capture current fixture and
   benchmark results on landed main. Extend the `jolt-riscv` row and logical
   input/view types, with strict construction and semantic serialization.
   Move instruction-aware PC lookup to the mapper. Preserve the existing
   value-slot packing owner.
2. **Unify producers and ownership.** Add the retained aggregate and sparse
   payload builder, migrate interpreter, x86, and replay to emit the same row,
   and share the mapper at checkpoint creation. Preserve bounded replay and
   indexed parallel collection. Cut witness and field consumers over to
   ownership transfer, including `RandomAccessRows`.
3. **Collapse public paths.** Migrate SDK, profiling, analysis, configuration,
   evaluation tools, and fixtures. Delete `trace_compact` as a separate public
   output format, `from_compact`, the old row module, duplicated conversion
   logic, and phantom witness generics. Retain one production execution API
   backed by one conversion implementation.
4. **Finish acceptance and documentation.** Version analysis output, document
   source migration, update `proof-trace-row-layout.md` to reflect the new
   construction point, and remove stale row-size claims in trace benchmarks.
   Remove temporary parity probes before handoff.

These can be reviewable commits in one coordinated PR. Do not ship an
intermediate public API that unifies the type but still copies the full trace,
or leave field-inline on the old storage indefinitely.

## Validation and acceptance

Permanent tests should cover independent layout/semantic properties: 64-byte
`Copy` rows in both feature modes; independent metadata/control-mask bounds;
exact instruction reconstruction including field operands; cached integer
projections matching the canonical helper for every operand shape; register
capture absence versus an observed zero; all three memory classes; malformed
identity/RAM rejection;
no-op PC zero; missing mappings; and checked semantic wire roundtrips with a
versioned archive fixture. Extend existing tests rather than preserving a copy
of deleted production logic as an oracle.

Use existing interpreter/x86 differential tests and chunk-composition tests to
check identical logical rows, PCs, sidecar association, and execution outputs.
Guest-dependent tests must actually execute; a missing guest toolchain that
causes skips is not a passing backend acceptance run. Exercise field arithmetic,
`LoadAccumulateFromRegister`, `LoadAccumulateFromMemory`, and `AdviceLimb`,
including missing/extraneous payload and register continuity failures. In
particular, the memory-accumulation row must preserve its encoded field rs2
while exposing no integer rs2 and binding its RAM read to integer rd's value.

Add one ownership regression test that proves witness construction retains the
producer's allocation, including when another owner exists; a shared `Arc`
must not trigger cloning. Cover partially consumed sources, padding and
lookahead, and trace lengths around powers of two. Imported traces must reject
PC/program mismatches.

Run affected-crate suites (`jolt-riscv`, `jolt-program`, `tracer`,
`jolt-tracer-x86`, `jolt-witness`, and `jolt-kernels`) and the current acceptance
commands, always using nextest:

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
cargo nextest run -p jolt-verifier --features prover-fixtures,field-inline --test-threads 1 --cargo-quiet
cargo nextest run -p jolt-verifier --features prover-fixtures,field-inline,akita --test-threads 1 --cargo-quiet
cargo nextest run -p jolt-witness --features test-utils,parallel,field-inline --cargo-quiet
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo clippy -p jolt-witness --features test-utils,parallel,field-inline -q --all-targets -- -D warnings
cargo fmt --all --check
```

Also run the affected no-default-feature/serialization builds and the current
CI field-inline checks, which now explicitly include witness and kernels, plus
the tracer's `field-inline,fp128-field-inline` configuration. Keep Akita and ZK
in separate builds. The current `e2e_matrix` covers ordinary guests and active
`field_ops` / inactive `muldiv` field-profile execution across clear, ZK, and
Akita modes; retain specialized field reference/optimized and tamper tests.

Compare deterministic clear Dory proof bytes with the pre-change revision using
identical programs, inputs, configuration, and preprocessing. Use verification
and tamper rejection for randomized ZK proofs. Preserve current Akita and
committed-program/advice fixture coverage. Do not resurrect the legacy prover
or its deleted byte-diff machinery.

Measure three paths separately: existing compact SDK/profile proving, generic
execution-to-witness handoff, and field-inline handoff. Record row and sidecar
capacities, subprocess peak RSS and physical footprint where available,
tracing time, handoff time, and prover time.
Use the existing fibonacci and SHA tracing benchmarks, plus optimized fibonacci
and a memory-heavy prover workload on the same machine and thread count.
Repeat an apparent regression; target no reproducible end-to-end prover-time
regression above 2%. Existing profiling excludes tracing, so its proving timer
alone cannot establish faster row production or handoff.

Use the now-landed field profiling workload as well:

```bash
cargo run --release -p jolt-prover --features profiling,field-inline -- profile --name field-ops --backend optimized --format chrome
cargo run --release -p jolt-prover --features profiling,field-inline,akita -- profile --name field-ops --backend optimized --format chrome
```

`field-ops` has fixed-size input; changing `--scale` does not generate a sweep
of field-payload densities. Report its actual row/event counts and include a
base workload under the field-enabled build to measure sparse-event overhead.

## Alternatives considered

| Alternative | Assessment |
| --- | --- |
| Late-assignable PC on every row | Fits physically, but introduces an unnecessary state and proof-access checks when the program map is already available. Reserve contextual rebinding for legacy import, if needed. |
| Per-cycle side tables for sequence count and recorded register IDs | Unnecessary memory traffic: the count fits inline, and real captured IDs already match operands. Only field payloads need side storage. |
| Keep a 72-byte, non-`Copy` row under field-inline | Smaller API migration, but puts a pointer on every ordinary cycle in feature-enabled builds and loses the existing compact-row contract. The sparse aggregate also eliminates the second retained row representation. |
| Remove cached PC and resolve it in witness accessors | Adds repeated mapping work to hot consumers; there is room to retain it. |
| Move the canonical row into `jolt-program` | Couples low-level proof/lookup consumers to program execution. `jolt-riscv` already owns the required instruction and flag vocabulary. |
| Share only the packing helper | Removes one duplicated rule but leaves competing row APIs, conversions, and allocation ownership. It does not complete this issue. |

The main implementation risks are changed acceptance of malformed execution
rows, field versus integer capture semantics, padding-induced proof-shape
changes, analysis-wire compatibility, and accidental copying at the `Arc`
handoff. Each has an explicit contract and acceptance check above. This document
is a source-based design review; no implementation or new benchmark results are
claimed.
