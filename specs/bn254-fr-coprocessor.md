# Spec: BN254 Fr Native-Field Coprocessor

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @sagar-a16z                    |
| Created     | 2026-05-13                     |
| Status      | implemented                    |
| PR          |                                |

## Summary

Cryptographic guests that operate over BN254's scalar field (Fr) currently
emulate each Fr operation as hundreds of RV64 u64 instructions: a single Fr
multiplication expands to a 256-bit Montgomery multiplication in software, and
a Poseidon2 t=3 permutation balloons to ~250k RV cycles. This is wasteful when
the proving system already speaks Fr natively. The BN254 Fr coprocessor adds
dedicated RISC-V opcodes (`FieldOp`, `FieldMov`, `FieldSLL{64,128,192}`,
`FieldAssertEq`) routed through a 16-slot native-field register file, with a
Twist-style sumcheck integration that mirrors the existing RV register
checking. The same Poseidon2 permutation drops to ~36k cycles — a ~7×
compression of the trace, which is the only reason FR-active proofs are
practical on the modular stack. The coprocessor is **modular-stack only**;
jolt-core monolithic does not (and will not) support FR instructions.

## Intent

### Goal

Provide a native BN254 Fr coprocessor in the modular Jolt zkVM: 9 new
instructions, 16 Fr register slots, Twist read/write checking across Stages 3,
4, and 5, with end-to-end host driving via `jolt_host::prove_program` and an
inline SDK (`jolt-inlines-bn254-fr`) that the guest can call.

### Invariants

The implementation must preserve these properties. Each was learned from a
concrete regression and is enforced by tests or measurement.

1. **No K×T dense materialization for one-hot polynomials.** Any Twist
   polynomial (Rs1Ra, Rs2Ra, RdWa, Val) whose access pattern is one-hot per
   cycle is stored as a sparse access list keyed by FR-active cycles, never as
   a `Vec<F>` of length `K · T`. Materialization to a K-sized dense vector
   happens only at `current_trace_len == 1`, after all `log_T` cycle-variable
   sumcheck rounds. Reference: `SparseRegistersState` in
   `crates/jolt-kernels/src/stage4.rs`. Verified by
   `bolt_sha2_chain_2_16_core_vs_bolt_perf_oracle` — modular `peak_rss_mb`
   must not regress versus upstream `refactor/crates` by more than the linear
   T-overhead from the 13 added FR R1CS rows.

2. **Empty FR events → zero FR-witness allocation.** Guests that emit no
   `FieldRegEvent`s (sha2-chain, muldiv, all non-FR guests) must allocate zero
   bytes for FR Twist polynomials. The sparse representation from invariant 1
   already enforces this — the sparse access list is empty.

3. **Stage state borrows witness slices; never clones.** Sumcheck state
   constructors take `&[F]` references into the witness, not `Vec<F>` clones.

4. **No address-broadcast replication.** Never replicate a T-length factor K
   times to produce a `K · T` Vec. The sparse representation references
   shared T-length factors via the per-entry `col: u8` index.

5. **FR API surface lives in Bolt emit templates, not regen'd code.** Any
   change to the FR prover/verifier API surface goes in
   `crates/bolt/src/protocols/jolt/artifacts.rs`. The `jolt-prover` and
   `jolt-verifier` crates are regenerated from scratch by
   `JOLT_UPDATE_GOLDENS=1`; hand-edits there are wiped on regen.

6. **`jolt_host::prove_program` is a thin host wrapper.** Stage orchestration
   lives in `jolt-prover` (the regen'd crate) via
   `prove_jolt_with_witness_inputs`. The host crate's job is trace + class-A
   data setup + a single top-level prove call.

7. **Stage 6 padding semantics match `Cycle::NoOp`.** Trace-extension helpers
   that index past `trace.len()` must produce values identical to running
   over a NoOp cycle. Concretely:
   `stage5_lookup_trace`'s missing-cycle branch sets
   `is_interleaved_operands = true` (matching `CircuitFlagSet::default()`),
   not `false`. Caught by Stage 6 bytecode_read_raf round-0 input claim check.

### Non-Goals

- **jolt-core monolithic FR support.** `jolt-core::zkvm::instruction` panics
  on `FieldMov` and other FR opcodes; this is intentional. FR is a
  modular-stack feature.
- **BlindFold ZK for FR Twist.** ZK mode for the FR sumcheck chain is tracked
  separately (task #56 in the implementation tracker) and is out of scope here.
- **The `#[jolt::provable_modular]` proc macro.** The host-side
  `prove_program` entry point exists; the SDK macro that wraps it for
  ergonomic guest binding is a separate spec.
- **Multi-shape goldens.** `default_prover_programs()` is pinned to one
  `JoltProtocolParams::fixture()` shape at a time. Runtime-selected per-guest
  shapes is a separate spec.
- **Cross-stack equivalence with jolt-core.** Bolt's Stage 1 R1CS adds 13 FR
  rows that jolt-core doesn't have, so cross-stack assertions in
  `jolt-equivalence` are structurally divergent by design. Self-parity within
  the modular stack is the correctness gate.

## Evaluation

### Acceptance Criteria

- [x] **muldiv smoke through modular path produces a verified proof.**
  `cargo nextest run -p jolt-host muldiv_modular_prove_smoke --release
  --run-ignored only --cargo-quiet --no-capture` succeeds and the returned
  `JoltProof` has `evaluation.is_some()`.
- [x] **bn254-fr-poseidon2-sdk smoke through modular path produces a verified
  proof.** `cargo nextest run -p jolt-host fr_poseidon2_modular_prove_smoke
  --release --run-ignored only ...` succeeds. The FR coprocessor is exercised
  (~16k `FieldRegEvent`s out of ~36k cycles).
- [x] **Modular bolt prove on sha2-chain log_t=16 matches upstream.**
  `cargo nextest run -p jolt-equivalence
  bolt_sha2_chain_2_16_core_vs_bolt_perf_oracle --release --run-ignored only ...`
  reports `prove_ms` within ±10% of upstream and `peak_rss_mb` within ~50 MB
  of upstream (the residual is the linear-in-T overhead from the 13 added FR
  R1CS rows).
- [x] **All 56 `jolt-kernels` unit tests pass.**
- [x] **jolt-core monolithic correctness preserved.**
  `cargo nextest run -p jolt-core muldiv --features host` succeeds — FR work
  has not regressed the monolithic prove path.
- [ ] **FR-active perf comparison: modular FR coprocessor vs modular ark-bn254
  software Fr.** Both routes prove the same Poseidon2 permutation; the
  coprocessor route should run on a ~7× smaller trace.

### Testing Strategy

**Existing tests that must keep passing** (both `--features host` and the
modular path):
- `cargo nextest run -p jolt-core muldiv` (monolithic baseline; unchanged by
  FR work).
- `cargo nextest run -p jolt-kernels` (all unit tests, including the
  sparse-register path FR mirrors).
- `cargo nextest run -p jolt-equivalence
  bolt_sha2_chain_2_16_core_vs_bolt_perf_oracle` (modular path on a non-FR
  workload, gates against memory or prove-time regressions).
- `cargo nextest run -p jolt-equivalence
  bolt_stage3_batched_real_muldiv_self_parity_at_log_t_9` (full Bolt prove +
  modular verifier on muldiv at the small fixture shape).

**New tests added for FR coverage**:
- `crates/jolt-host/tests/prove_program_smoke.rs::muldiv_modular_prove_smoke`
  — FR-rows-present, FR-events-empty end-to-end.
- `crates/jolt-host/tests/prove_program_smoke.rs::fr_poseidon2_modular_prove_smoke`
  — FR-rows-present, FR-events-active end-to-end.
- `crates/jolt-host/tests/fr_perf.rs` — cycle count + prove time + peak RSS
  per scenario, for the four-way comparison documented in Performance.

**Cross-stack note**: `jolt-equivalence::bolt_oracle` cross-stack assertions
(`assert_core_accepts_bolt_stage*`) are skipped on this branch because the
Stage 1 R1CS surface differs by 13 FR rows. Self-parity within the modular
stack (above) is the live correctness gate.

### Performance

Workload: `sha2-chain` at log_t=16 through `assert_bolt_full_real_trace_self_parity`,
on the same machine, best of three runs.

| State | prove_ms (bolt) | peak_rss_mb (bolt) | vs upstream RSS |
|---|---:|---:|---:|
| Upstream `refactor/crates` baseline | 1191 | 1034 | — |
| FR pre-port (dense K×T tables) | 1381 | 1436 | +39% |
| FR with Stage 4 sparse | 1146 | 1167 | +13% |
| FR with Stage 4 + Stage 3 sparse | 1146 | 1114 | +8% |
| **FR with Stage 4 + 3 + 5 sparse (current)** | **1122** | **1075** | **+4%** |

Current state at the same workload: prove time is 6% faster than upstream
(1122 vs 1191 ms), peak RSS is 4% higher (1075 vs 1034 MB). The residual ~41
MB is linear in T and entirely accounted for by the 13 added FR R1CS rows.

For the FR-active workload (Poseidon2 BN256 t=3 permutation, inputs
`(1,2,3)`):

| Path | Raw cycles | Prove time | Peak RSS |
|---|---:|---:|---:|
| jolt-core monolithic + inline FR coprocessor | — | — | ❌ panics (no FR support) |
| jolt-core monolithic + ark-bn254 software Fr | 252,978 | 2.34 s | 851 MB |
| Modular `prove_program` + inline FR coprocessor | **35,890** | 2.30 s | 1,896 MB |
| Modular `prove_program` + ark-bn254 software Fr | 252,978 | 5.01 s | 3,688 MB |

The FR coprocessor's value is the ~7× cycle reduction on the same algebraic
output. Modular memory at log_t=18 is sub-linear in cycles versus the same
workload at log_t=16, consistent with the per-cycle architectural overhead.

### Acceptance Criteria — Memory Invariant Gate

The bolt_perf comparison at sha2-chain log_t=16 is the structural gate for
Invariant 1. Modular `peak_rss_mb` must satisfy
`peak_rss_mb ≤ upstream_peak_rss_mb + C · T` for a small constant `C` derived
from the FR R1CS row count. Today the constant is approximately
`C ≈ 41 MB / 65 536 ≈ 656 B/cycle`. Anything larger indicates a regression of
Invariant 1.

## Design

### Architecture

The coprocessor spans tracer, R1CS, witness, kernels, host wrapper, and the
inline SDK. The new modules are durable (not regenerated by Bolt). FR
plumbing on the regen'd `jolt-prover`/`jolt-verifier` lives in Bolt's emit
templates.

**Tracer** (`tracer/src/instruction/`):
- 6 new `Instruction` variants for FR opcodes (`FieldOp`,
  `FieldAssertEq`, `FieldMov`, `FieldSLL{64,128,192}`). `FieldOp` is a single
  variant shared by FMUL/FADD/FSUB/FINV, discriminated at runtime by
  `funct3`.
- `FieldRegEvent` stream emitted per FR cycle alongside the regular
  `Cycle` stream. Carries `(cycle_index, slot, old, new)` as `[u64; 4]`
  limbs.
- 16-slot `FieldReg` CPU state for emulation correctness.

**R1CS** (`crates/jolt-r1cs/src/constraints/rv64.rs`):
- 13 new R1CS rows (rows 19–31) encoding the FR coprocessor constraint
  surface. `NUM_CONSTRAINTS_PER_CYCLE` increases from 22 to 35, padded to 64.
- 3 new R1CS witness columns: `V_FIELD_RS1`, `V_FIELD_RS2`, `V_FIELD_RD`,
  populated by `extract_trace` from the FR event replay.

**Witness** (`crates/jolt-witness/src/field_reg.rs`):
- `FieldRegisterColumns<F>` (Stage 3) and `FieldRegistersWitness<F>` (Stage
  4/5). Both expose a `sparse_accesses` access list rather than dense K×T
  Vecs; only `field_rd_inc` and `field_rd_wa_at_register_address` are dense
  (length T) because the regular RV equivalents are too.
- `field_registers_witness` (FR-active builder) and
  `zero_field_registers_witness` (FR-less builder, allocates only the
  empty-access-list footprint).

**Kernels** (`crates/jolt-kernels/src/stage{3,4,5}.rs`):
- **Stage 3 `FieldRegClaimReduction`**: `SparseFieldRegistersState<F>`
  mirroring the regular RV ClaimReduction state. T-dimension sumcheck only;
  no register-address phase.
- **Stage 4 `FieldRegistersReadWriteChecking`**: `SparseFieldRegistersState<F>`
  + `Stage4FieldRegisterAccess<F>` per-cycle access struct. Sumcheck has both
  T (cycle) and log_K=4 (FR register address) phases; the dense K-sized
  vector is materialized at `current_trace_len == 1` exactly like the regular
  registers path.
- **Stage 5 `FieldRegValEvaluation`**: `field_rd_wa_at_register_address`
  projection helper, sparse over FR accesses.
- All three share a generic `SparseValBacking<F>` trait so the regular RV
  register sparse state (V = u64) and the FR sparse state (V = F) reuse the
  same Gruen quadratic poly builder, sumcheck round body, and bind logic.

**Bolt emit templates** (`crates/bolt/src/protocols/jolt/`):
- `artifacts.rs` emits the FR API surface: `FrProverWitness<'a, F>` struct,
  `field_registers: Option<FrProverWitness>` field on
  `JoltProverWitnessInputs`, zero-fallback materialization in
  `prove_jolt_with_witness_inputs`, and FR threading through
  `stage{3,4,5}_prover_inputs`. Regen-safe.
- `phases/stage{3,4,5}.rs` and `emit/rust/stage4.rs` emit the
  `stage4_field_registers_rw` Bolt sumcheck instance + point-order
  normalization on the verifier side.

**Host wrapper** (`crates/jolt-host/src/lib.rs`):
- `prove_program(program, inputs, untrusted_advice, trusted_advice)` drives
  the full modular prove path on a real guest ELF: two-pass trace (compute
  advice → real run), class-A data construction, single
  `prove_jolt_with_witness_inputs` call, evaluation proof. Currently larger
  than the target thin-wrapper shape; the per-stage orchestration is open
  follow-up.

**Inline SDK** (`jolt-inlines/bn254-fr/`):
- `Fr::from_limbs / to_limbs / add / sub / mul / inv / mov` guest-side
  callable surface that dispatches to the new opcodes. A `compute_advice`
  feature builds a Pass-1 ELF that pre-computes ark-bn254 results into the
  advice tape, consumed by Pass-2 which emits the real FR opcodes.

### Alternatives Considered

- **Dense K×T tables for FR Twist polynomials.** This was the initial
  implementation; measured at +400 MB peak RSS vs upstream on sha2-chain
  log_t=16. Rejected after the architectural audit traced the regression to
  Stage 4's `DenseStage4State` building six K·T-length Vecs (val, rs1_ra,
  rs2_ra, rd_wa, eq_cycle replicated 16×, rd_inc replicated 16×). The
  regular RV Registers path never does this — it stores ≤3 sparse entries
  per cycle and materializes the K-sized dense vector only when the cycle
  dimension has collapsed. Sparse is the correct mirror.

- **`Option<&FieldRegistersWitness>` plumbing throughout.** Would have saved
  ~140 MB on FR-less workloads by skipping the zero allocation, but does not
  help FR-active workloads (where the witness is non-trivial) and requires
  touching every FR consumer with a Some/None branch. The sparse-state
  refactor subsumes this win — empty access list ≈ zero allocation — and
  also helps FR-active workloads. Strictly dominated.

- **FR support in jolt-core monolithic.** Would have provided a pre-FR vs
  post-FR perf comparison in the same prover. Rejected because jolt-core is
  on track to be replaced by the modular stack; adding FR there would
  duplicate work. The arkworks software-Fr variant
  (`examples/bn254-fr-poseidon2-arkworks`) gives the comparable monolithic
  baseline without requiring jolt-core changes.

- **Reusing `OneHotPolynomial` from `crates/jolt-poly` directly for FR Twist
  polynomials.** Rejected: `OneHotPolynomial` is designed for the
  commitment-side one-hot indicator polynomials over the instruction lookup
  domain; the Twist sumcheck state has additional working data per access
  (the read/write values and the per-row claim) that doesn't fit the
  `OneHotPolynomial` shape. The `SparseRegisterEntry` / `SparseValBacking`
  pattern is the better fit.

## Documentation

No Jolt book changes required. The FR coprocessor is an internal proving
system feature; user-facing surface is the inline SDK
(`jolt-inlines-bn254-fr/README.md` documents the guest-side API
independently). The architectural invariants documented in this spec are
internal contracts.

## Execution

Implementation order has been:
1. Tracer + R1CS + jolt-riscv (durable crates outside Bolt's regen zone).
2. Inline SDK + guest-side bindings.
3. Stage 3/4/5 kernels with dense FR Twist polynomial materialization (initial
   working state).
4. Bolt emit templates for FR API surface (port step 14g).
5. Host wrapper (`jolt-host::prove_program`).
6. Stage 6 padding-semantics fix in `stage5_lookup_trace`.
7. **Sparse port of Stage 4, then Stage 3, then Stage 5** — replacing each
   dense K×T materialization with the sparse-access-list pattern from the
   regular RV registers path. This is the path that brought modular peak RSS
   back within ~50 MB of upstream.

Open follow-ups (separate specs):
- `#[jolt::provable_modular]` proc macro that wraps `prove_program`.
- Multi-shape goldens so muldiv, sha2-chain, and Poseidon2 can each run at
  their natural fixture shape without manual `JOLT_UPDATE_GOLDENS=1` regens.
- `prove_program` collapse to thin wrapper around
  `prove_jolt_with_witness_inputs`.
- BlindFold ZK for FR Twist.

## References

- HorizenLabs Poseidon2 BN256 reference parameters (transcribed into
  `examples/bn254-fr-poseidon2-sdk/guest/src/lib.rs::RC3`):
  `plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs`.
- Twist/Shout lookup argument: Jolt paper, Section 6 (register checking shape
  this work mirrors).
- Upstream `refactor/crates` baseline commit: `313cf54b8` ("Collapse Bolt
  cleanup modules"). Bolt prove perf at sha2-chain log_t=16 on this commit is
  the parity target.
