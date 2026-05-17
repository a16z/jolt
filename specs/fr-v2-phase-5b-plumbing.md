# Phase 5b plumbing — feed tracer FR events into Stage45SparseTraceWitness

This is a working note for the next session, not a long-lived spec.
Delete after Phase 5b plumbing lands.

## State at HEAD

- `5f9cbe882 feat(phase-5b): FieldRegReplay materializers for FR Twist witness polys`
- `10a9306c4 feat(phase-5b): wire FieldRegReplay into Stage45SparseTraceWitness`

What works now:
- `jolt_witness::field_reg::FieldRegReplay { num_cycles, bytecode, events }`
  + `FrCycleBytecode { frs1, frs2, frd, reads_frs1, reads_frs2 }`
  + 5 materializer methods (`field_reg_val` / `frs1_ra` / `frs2_ra` /
  `frd_wa` / `frd_inc`).
- `Stage45SparseTraceWitness::with_field_reg_replay(replay)` builder.
- Stage 4/5 prover-input borrows already point at the per-poly buffers
  (`witness.field_reg_val`, etc.). Inert path stays all-zero.

What's dormant: jolt-host's `prove_program` discards the tracer's
FieldRegEvent stream (jolt-core/src/host/program.rs:325 captures it
as `_field_reg_events` and drops it) and never constructs a
`FieldRegReplay`. For FR-active programs (poseidon2-sdk, etc.) the
Stage 4/5 FR sumchecks see all-zero buffers and fail with non-trivial
input claims.

## Remaining plumbing (3 steps)

### 1. Plumb FR events out of `Program::trace`

`jolt-core/src/host/program.rs:308` currently returns `(LazyTraceIterator,
Vec<Cycle>, Memory, JoltDevice)`. Drop `_field_reg_events` from the
discard list and return it as a 5th tuple element:

```rust
pub fn trace(...) -> (
    LazyTraceIterator,
    Vec<Cycle>,
    Memory,
    JoltDevice,
    Vec<tracer::emulator::cpu::FieldRegEvent>,
) { ... }
```

Update all call sites (grep `program.trace(`) — the main one is
`crates/jolt-host/src/lib.rs:354`.

### 2. Extract per-cycle FrCycleBytecode from the trace

Walk `trace: &[TraceRow]` in `crates/jolt-host/src/lib.rs::assemble_and_prove`
and decode FR metadata per cycle. Each `TraceRow.instruction` is a
`NormalizedInstruction`; match on opcode/funct7 to classify:

- `FieldOp` (opcode 0x0B, funct7=0x40): `reads_frs1=true`,
  `reads_frs2=true` for FMUL/FADD/FSUB/FAssertEq; `reads_frs1=true`,
  `reads_frs2=false` for FINV. `frs1=rs1 & 0xF`, `frs2=rs2 & 0xF`,
  `frd=rd & 0xF`.
- `FieldMov` / `FieldSll*` (bridge ops reading integer registers):
  `reads_frs1=false`, `reads_frs2=false`. `frd=rd & 0xF`.
- Non-FR cycles: `FrCycleBytecode::default()` (all zero / all false).

Implement as a `crates/jolt-witness/src/field_reg.rs` helper:
```rust
pub fn fr_cycle_bytecode_from_trace(
    trace: &[NormalizedInstruction],
) -> Vec<FrCycleBytecode>;
```

### 3. Build FieldRegReplay + attach to witness

In `assemble_and_prove` (jolt-host), right after `extract_trace_rows`
and before `stage4_5_sparse_trace_witness_from_accesses` (which
currently lives in `jolt-prover/src/prover.rs:707`):

```rust
let fr_bytecode = fr_cycle_bytecode_from_trace(&trace_instructions);
let fr_events = field_reg_events
    .into_iter()
    .map(|ev| jolt_witness::field_reg::FieldRegEvent {
        cycle: ev.cycle_index as u64,
        frs1: 0,           // unused by materializers
        frs2: 0,           // unused by materializers
        frd: ev.slot,
        rs1_pre: jolt_witness::field_reg::FrLimbs::ZERO,
        rs2_pre: jolt_witness::field_reg::FrLimbs::ZERO,
        rd_post: jolt_witness::field_reg::FrLimbs(ev.new),
        rd_written: true,  // tracer only emits writes
    })
    .collect();
let replay = jolt_witness::field_reg::FieldRegReplay {
    num_cycles: trace_length,
    bytecode: fr_bytecode,
    events: fr_events,
};
```

Then in `prove_jolt_with_stage_inputs` (jolt-prover/src/prover.rs:706),
the `stage45_witness` build needs to take an optional `&FieldRegReplay`
parameter and chain `.with_field_reg_replay(replay)` onto the result.
Plumb through `JoltProverInputs` (or `BoltProverInputs`).

### 4. Gates

```
source ./.bolt-dev-env
cargo nextest run -p jolt-witness --cargo-quiet
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p bolt --test commitment_ir --cargo-quiet --no-fail-fast
cargo clippy -p jolt-witness -p jolt-r1cs -p jolt-kernels -p bolt \
    --message-format=short -q --all-targets -- -D warnings
```

muldiv has no FR events → replay.events.is_empty() → materializers
return zero buffers → no behavior change. Phase 5c (poseidon2-sdk
example) is the first FR-active validation.

## After Phase 5b plumbing lands → Phase 5c

Cherry-pick `examples/bn254-fr-poseidon2-sdk/` from source commit
`11fd62596`. Mirror Phase 5a's pattern:
- Adapt `#[jolt::provable]` → `#[jolt::provable(backend = "modular", ...)]`
- Clamp `max_trace_length` to ≤ 2^18 (fixture ceiling)
- Rewrite host main.rs for modular-sdk's `compile_*` / `prove_*` /
  `verify_*` pattern

Validation: prove + verify succeeds, `valid: true`, prove time is
significantly lower than the arkworks baseline (5a) — the FR
coprocessor is the whole point.

## After 5c → Phase 5d (audit fixes C1-C11)

Per `specs/fr-v2-port-plan.md` lines 117-122. Carry over from source
branch.
