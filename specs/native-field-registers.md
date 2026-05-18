# Spec: BN254 Fr Native-Field Coprocessor

| Field       | Value          |
|-------------|----------------|
| Author(s)   | Sagar Dhawan   |
| Status      | shipped        |

## Summary

A native-field coprocessor that adds a 16-slot BN254 Fr register file and 9 new RISC-V instructions to the modular Jolt zkVM, with three sumcheck instances (Stage 3 / 4 / 5) wired into the existing Twist register-checking pipeline and a Stage 6 bytecode-RAF anchor binding FR access to bytecode. Guest programs that use the `jolt_inlines_bn254_fr::Fr` SDK type emit one `FieldOp` cycle per (mul, add, sub, inv) operation whose constraint witness is bound natively to the Fr field. Software-emulated Fr arithmetic (arkworks on RV) is replaced end-to-end for FMUL/FADD/FSUB/FINV; FAssertEq closes the prover advice loop, and FieldMov/FieldSLL64/128/192 form the integer→FR bridge ABI for loading inputs and reconstructing outputs.

## Intent

### Goal

Prove BN254 Fr arithmetic at native cost: one FR register-file slot per Fr value, one `FieldOp` cycle per binary op, and one constraint relation tying R1CS witnesses to a 16-slot FR Twist register file. The architectural boundaries:

- **Tracer**: `tracer::emulator::cpu::FieldRegEvent`, `Cpu.field_regs: [[u64; 4]; 16]`, `Cpu.field_reg_events: Vec<FieldRegEvent>` (`tracer/src/emulator/cpu.rs`), and the per-instruction files under `tracer/src/instruction/field_*.rs`.
- **ISA**: 9 instruction kinds — `FieldOp` (funct7=0x40, opcode=0x0B; funct3 selects FMUL/FADD/FSUB/FINV), `FieldAssertEq`, `FieldMov`, and `FieldSLL{64,128,192}` (funct7=0x41). FR-access classification (which slots are read/written per kind) lives in `JoltInstructionKind::fr_access_flags()` (`crates/jolt-riscv/src/kind.rs`) as the single source of truth shared by host and kernel sites.
- **R1CS**: 13 new eq-constraint rows (rows 19–31 in `crates/jolt-r1cs/src/constraints/rv64.rs`) over slots `V_FIELD_RS1_VALUE=45`, `V_FIELD_RS2_VALUE=46`, `V_FIELD_RD_WRITE_VALUE=47`. `NUM_R1CS_INPUTS=47`, `NUM_VARS_PER_CYCLE=50`, `NUM_EQ_CONSTRAINTS=32`.
- **Witness types**: `jolt_witness::field_reg::{FrLimbs, FieldRegEvent, FrCycleBytecode, FieldRegReplay, FIELD_REG_COUNT (=16), LOG_K_FR (=4), FIELD_REG_ADDR_MASK (=0xF), limbs_to_field, sub_limbs}` in `crates/jolt-witness/src/field_reg.rs`. Sparse-first: only `materialize_frd_inc<F>` (T-length) is retained, and it is cached once per proof in `Stage45SparseTraceWitness.fr_frd_inc` so both Stage 4 and Stage 5 share the materialization.
- **Sumcheck kernels** in `crates/jolt-kernels/`:
  - Stage 3 `FieldRegClaimReduction` (`stage3.rs`) — γ-batched eq sumcheck reducing rd/rs1/rs2 R1CS values to FR write/read claims.
  - Stage 4 `FieldRegRW` (`stage4.rs`, `SparseFieldRegState`) — sparse Hamming-shaped Twist read-write check over `K_FR = 16 × T`. Round-poly accumulation and bind shell out to Rayon `par_chunk_by` once the entry count crosses `DENSE_BIND_PAR_THRESHOLD` (mirrors the integer Twist's parallelization).
  - Stage 5 `FieldRegValEvaluation` (`stage5.rs`) — degree-3 `frd_inc · frd_wa_at_address · lt` evaluation.
  - Stage 6 bytecode-RAF (`stage6.rs::bytecode_entry_stage_values`, `BYTECODE_READ_RAF_STAGE_COUNT=6`) — adds an FR stage group `γ_fr·writes_frd·register_eq(frd) + γ_fr²·reads_frs1·register_eq(frs1) + γ_fr³·reads_frs2·register_eq(frs2)` that binds FR access against bytecode. Mirrored on the verifier side in `crates/jolt-verifier/src/stages/common.rs` (`STAGE67_BYTECODE_STAGE_COUNT=6`).
- **SDK**: `jolt_inlines_bn254_fr::Fr` (`jolt-inlines/bn254-fr/src/sdk.rs`) with three compile-time dispatch paths (host arkworks, `compute_advice` pass populating the advice tape, RV pass-2 emitting the FieldOp instruction sequence). `#[jolt::provable(backend = "modular")]` (`jolt-sdk/macros/src/lib.rs`) auto-builds the two-pass advice-tape ELF.

### Invariants

- **Replay/trace length parity**: `FieldRegReplay.num_cycles == trace_len` and `replay.bytecode.len() == trace_len` (asserted in `SparseFieldRegState::new`).
- **Cycle bound**: every `FieldRegEvent.cycle < trace_len`.
- **Monotone event order**: events are sorted strictly increasing by cycle; duplicates and out-of-order events are rejected at `SparseFieldRegState::new`.
- **Address mask**: FR slot indices are masked `& FIELD_REG_ADDR_MASK` at producer and consumer sites (host `fr_bytecode_from_trace`, `populate_r1cs_fr_slots`, `populate_fr_cycle_fields`, kernel Stage 4 / Stage 5). `FIELD_REG_COUNT` must be a power of two; `SparseFieldRegState::new` and `frd_wa_at_field_reg_address` both enforce this.
- **Initial FR state is zero, algebraically enforced**: Stage 5 `FieldRegValEvaluation` proves `FieldRegVal(addr, t) = Σ_j FrdInc(j) · FrdWa(addr, j) · lt(t, j)`. At `t = 0` the `lt(0, j)` MLE is identically 0, so the RHS is empty and `FieldRegVal(addr, 0) = 0` is forced — no separate init-eval opening required. This is the same mechanism integer registers use. RAM is the outlier (its `RamValInit` opening is bound to a precomputed public ELF image because RAM init values aren't all zero).
- **FINV(0) is unsatisfiable**: the tracer panics in `field_op.rs`; the SDK guards via `Fr::inverse() -> Option<Fr>`. Either fails closed before any proof is attempted.
- **Bytecode/event consistency**: every event-bearing cycle has at least one FR access flag set on the corresponding bytecode row. `SparseFieldRegState::new` rejects stray events that land on bc-all-false cycles (defense in depth against tracer / preprocessing classifier drift).
- **Bytecode-anchored FR access (drop-event soundness)**: the Stage 6 bytecode-RAF sumcheck adds an FR stage group whose verifier-recomputable claim depends on the *bytecode-derived* (frd, frs1, frs2, reads_*, writes_frd) — not on event presence. A malicious prover that drops an FR event on a bc-active cycle leaves the Stage 4 FrdWa polynomial short of that cycle's contribution, so the Stage 6 input-claim equality fails. This is the load-bearing check; FR write-slot indicators must be sourced from `bc.frd` (committed) rather than `event.frd` (uncommitted prover input) in both the FrdInc materializer and the Stage 4 sparse-entry construction.
- **Bit ordering**: Stage 4 binds `r_cycle` LowToHigh inside `SparseFieldRegState`; `final_frs2_read_eval` reverses both the cycle and address halves of the bound point so `EqPolynomial::evals` receives MSB-first input matching `sparse_register_read_eval`.
- **Read claim decomposition**: `frs1_ra` is recovered from the combined eval as `(combined_read_ra − γ² · frs2_ra) · γ⁻¹`; the equality `combined_read_ra == γ · frs1_ra + γ² · frs2_ra` is asserted under `debug_assertions`.
- **R1CS witness coupling**: `populate_r1cs_fr_slots` stamps `V_FIELD_RS1/RS2/RD_WRITE_VALUE` from the same `FieldRegReplay` that feeds Stage 3/4/5. Stage 3 input claim consumes those slots via `Stage3Cycle.{field_rs1, field_rs2, field_rd, is_field_op}` populated by `populate_fr_cycle_fields`. Stage 4's `expected_field_reg_rw` is the verifier mirror of the prover's claim.
- **No new `OracleGeneration` variant**: FR oracles (`FieldRegInc` dense, `FieldRegRa_d` one-hot) reuse the existing `DenseTrace`/`OneHotChunk` shapes and are gated behind `params.field_reg_d > 0`. Today `field_reg_d = 0` and the FR Twist witness flows in-process via `Stage4FieldRegWitness` / `Stage5FieldRegValWitness` directly from `FieldRegReplay`.

### Non-Goals

- **ZK mode for the FR Twist sumchecks.** BlindFold integration (`input_claim_constraint` / `output_claim_constraint` synchronization, Pedersen-committed round polynomials) is not implemented for the three FR sumcheck instances. The modular stack's BlindFold port itself is also incomplete; this work is contingent on that landing.
- **Multi-cycle FINV / FAssertEq.** Both are single-cycle today. FINV(0) is unsatisfiable and fails closed (tracer panic + SDK `Option<Fr>`).
- **Other native fields.** The coprocessor is hardcoded to BN254 Fr at 256 bits. The Stage 4/5 kernels are generic over `F: Field`, but the tracer arithmetic helpers and `Stage1OuterRv64Data` are pinned to `ark_bn254::Fr`. Supporting BLS12-381 Fr (or any other native-aligned ≤256-bit prime field) requires parameterizing the tracer helpers and threading a generic `F` through Stage 1; the Stage 3/4/5 sumchecks and the R1CS constraint system itself need no change.
- **Pinned-slot SDK API.** The current SDK pays 7+7+1+12 = 27 cycles per binary op (Horner-load A, Horner-load B, FieldOp, advice-bound extract). Amortizing the extract over runs of in-field ops is future work.
- **Fully sparsifying remaining T-sized vectors.** `materialize_frd_inc` (cached, T-length), `FieldRegReplay.bytecode` (T-length), and the Stage 5 `frd_wa_at_field_reg_address` scratch (T-length) are all still dense. The K_FR×T factor space is never materialized.

## Evaluation

### Acceptance Criteria

- [x] `bn254-poseidon2 --backend inline` end-to-end prove + verify succeeds with `valid: true` (driver at `examples/bn254-poseidon2/src/main.rs`).
- [x] `cargo nextest run -p jolt-kernels` — all kernel tests pass (Stage 3/4/5 synthetic-witness round-trips plus the FR-specific sparse-state suite under `stage4::tests::field_reg`).
- [x] `cargo nextest run -p jolt-witness` — `field_reg` unit tests pass (limbs round-trip, `sub_limbs` borrow, `frd_inc` post-minus-pre, `limbs_to_field` LE assembly, empty-replay shape).
- [x] `cargo nextest run -p jolt-riscv` — `fr_access_flags_match_per_kind_contract` pins the FR-access classification table.
- [x] `cargo clippy --all-targets -- -D warnings` clean across `jolt-kernels`, `jolt-host`, `jolt-prover`, `jolt-witness`, `jolt-verifier`.
- [x] No K_FR×T dense materializer is reachable in production paths. Only `materialize_frd_inc` (T-length) remains and is cached once in `Stage45SparseTraceWitness.fr_frd_inc`, shared between Stage 4 and Stage 5.
- [x] Sparse FR replay buffers ≤ ~3 entries per FR-active cycle.
- [x] FINV(0) fails at the tracer rather than producing an invalid proof.
- [x] Drop-event soundness: dropping an FR event on a bc-active cycle is rejected by the Stage 6 bytecode-RAF input-claim check (the `frd_wa_diverges_from_bytecode_when_event_dropped` unit test documents the asymmetry that the Stage 6 anchor relies on).

### Testing Strategy

Unit tests (`--features host`):

- `crates/jolt-witness/src/field_reg.rs` — `FrLimbs` defaults & round-trip, `sub_limbs` borrow correctness, `materialize_frd_inc` post-minus-pre semantics, `limbs_to_field` little-endian assembly, empty-replay shape.
- `crates/jolt-riscv/src/kind.rs` — `fr_access_flags_match_per_kind_contract` locks the per-kind FR access flag table so any future ISA change must explicitly update it.
- `crates/jolt-kernels/src/stage4.rs::tests::field_reg` — `SparseFieldRegState` honest-replay population, frs1/frs2/frd alias coalescing, monotone-event enforcement, power-of-two `field_reg_count` precondition, bc-all-false event rejection, Stage 4 ↔ bytecode-driven FrdWa agreement on honest replays, and the malicious-replay divergence the Stage 6 anchor catches.
- `crates/jolt-kernels/src/stage4.rs::tests` — Stage 4 batched kernel proves+verifies synthetic witnesses, accepts sparse RAM addresses, accepts sparse register accesses, rejects a tampered eval.

End-to-end (`--features host`):

- `examples/bn254-poseidon2` — host driver running one Poseidon2 permutation over BN254 Fr, with `--backend inline|native` selecting between the FR coprocessor and software arkworks. Drives the full stack: SDK three-mode dispatch, tracer FieldOp emission, `FieldRegEvent` stream, R1CS slot population, Stage 3/4/5/6 sumchecks, Dory commitments, verifier round-trip.

Gaps:

- The Stage 5 `frd_wa_at_field_reg_address` aggregation is exercised only indirectly via the SDK e2e.
- ZK mode is not exercised against FR; the muldiv ZK e2e remains the project-wide ZK gate but does not execute any FR instruction.

### Performance

Measured at `log_T = 18` on the FR-active Poseidon2 workload:

| Metric         | inline (FR coprocessor) | native (software ark Fr) |
|----------------|------------------------:|-------------------------:|
| Guest cycles   |                  35,890 |                  252,969 |
| Prove time     |                  ~2.7 s |                   ~3.4 s |
| Peak RSS       |                ~2.6 GiB |                    ~same |

Memory shape: the host carries a single sparse `FieldRegReplay` (~few KB of events + a T-length bytecode shadow) instead of the K_FR×T dense buffers an early sketch needed. At `log_T = 18` this is ~9K sparse entries on a Poseidon2 permutation.

FR-inactive traces (e.g. muldiv) short-circuit every sparse code path via `replay.events.is_empty()` and produce zero-claim states; cycle-count parity vs the no-FR baseline is preserved.

## Design

### Architecture

Data flow, in execution order:

1. **Guest SDK** (`jolt-inlines/bn254-fr/src/sdk.rs`). Three compile-time paths:
   - *Host* (default, no RV target): delegates to `ark_bn254::Fr`.
   - *`compute_advice` pass on RV* (built automatically by the macro): computes via arkworks, writes the 4 result limbs to the advice tape using `VirtualHostIO`.
   - *Pass-2 RV*: emits a 27-cycle sequence per binary op — 7-cycle Horner load A + 7-cycle Horner load B + 1 FieldOp + 12-cycle advice-bound extract (4 `ADVICE_LD` + 7-cycle Horner reconstruction + FieldAssertEq).
2. **Macro** (`jolt-sdk/macros/src/lib.rs`). `#[jolt::provable(backend = "modular")]` builds two ELFs (default + `compute_advice` feature) via `build_with_features`, so the host replays the compute-advice ELF first to populate the advice tape, then proves the pass-2 ELF.
3. **Tracer** (`tracer/src/instruction/field_op.rs`, `field_assert_eq.rs`, `field_mov.rs`, `field_sll{64,128,192}.rs`). Opcode `0x0B` decodes on funct7: `0x40` → FieldOp/FieldAssertEq/FieldMov (funct3 selects); `0x41` → FieldSLL64/128/192. Execution mutates `Cpu.field_regs` and appends a `FieldRegEvent` on every write.
4. **jolt-host** (`crates/jolt-host/src/lib.rs`):
   - `fr_bytecode_from_trace` derives `FrCycleBytecode` per trace row via `JoltInstructionKind::fr_access_flags()`; slot indices are masked `& FIELD_REG_ADDR_MASK`.
   - `convert_fr_events` maps tracer events into `jolt_witness::field_reg::FieldRegEvent`.
   - `FieldRegReplay { num_cycles, bytecode, events }` is built once and passed to `Stage45SparseTraceWitness::with_field_reg_replay`, which eagerly caches `FrdInc` as `fr_frd_inc` for shared consumption by Stage 4/5.
   - `populate_r1cs_fr_slots` walks the replay's running FR-state and stamps `V_FIELD_RS1/RS2/RD_WRITE_VALUE` per cycle into the R1CS witness.
   - `populate_fr_cycle_fields` stamps `field_rs1`/`field_rs2`/`field_rd`/`is_field_op` on `Stage1Rv64Cycle` and `Stage3Cycle` from the same walk.
5. **Stage 3 `FieldRegClaimReduction`** (`crates/jolt-kernels/src/stage3.rs`). γ-batched eq sumcheck — proves `Σ_t eq(r, t) · (rd(t) + γ · rs1(t) + γ² · rs2(t))` where each factor is gated by `cycle.is_field_op`. Factor outputs are published as `FieldRdWriteValue` / `FieldRs1Value` / `FieldRs2Value` openings for Stage 4 / 5.
6. **Stage 4 `FieldRegRW`** (`crates/jolt-kernels/src/stage4.rs`, `SparseFieldRegState`). Input claim: `Σ_{k,j} eq(r_cycle, j) · (frd_wa(k,j) · (val(k,j) + frd_inc(j)) + γ · frs1_ra(k,j) · val(k,j) + γ² · frs2_ra(k,j) · val(k,j))`. The state walks per-cycle `SparseFieldRegEntry { row, col, val, prev_val, next_val, read_ra, frd_wa }` records — at most three entries per FR-active cycle (one frs1 read, one frs2 read merged on the same col, one frd write). `frs2_reads: Vec<(cycle, col)>` is held separately so the final `frs2_ra` evaluation can be recovered independently; the combined `read_ra` is decomposed as `frs1_ra = (combined − γ² · frs2_ra) · γ⁻¹`. When `current_trace_len == 1` the state materializes a tiny dense view of size `K_FR = 16` for the remaining address-phase rounds. Round-poly accumulation and entry binding shell out to Rayon `par_chunk_by` once the entry count crosses `DENSE_BIND_PAR_THRESHOLD`.
7. **Stage 5 `FieldRegValEvaluation`** (`crates/jolt-kernels/src/stage5.rs`). Degree-3 dense sumcheck over `frd_inc(t) · frd_wa_at_address(t) · lt(t)`, where `frd_wa_at_address` is a T-length aggregate keyed by the bytecode-derived write slot. `FrdInc` is consumed from the cache populated in Stage 4's pre-pass.
8. **Stage 6 bytecode-RAF anchor** (`crates/jolt-kernels/src/stage6.rs::bytecode_entry_stage_values`). Adds an FR stage group to the bytecode-RAF input claim:
   ```
   stage_fr = γ_fr⁰ · writes_frd · register_eq(frd, r_addr_fr)
            + γ_fr¹ · reads_frs1 · register_eq(frs1, r_addr_fr)
            + γ_fr² · reads_frs2 · register_eq(frs2, r_addr_fr)
   ```
   The verifier-side mirror in `crates/jolt-verifier/src/stages/common.rs` recomputes the same formula from bytecode metadata. This is the binding that catches dropped FR events.
9. **Verifier** mirrors live in `crates/jolt-verifier/src/stages/`. Stage 4's `expected_field_reg_rw` (in `stage4.rs`) is the canonical claim-reconstruction formula for both prover correctness checks and verifier acceptance.

### Alternatives Considered

- **Dense K_FR×T materialization for Stage 4**: would force host RAM ~520 MB at `log_T = 18` and kernel transient ~3 GB. Replaced with the sparse `SparseFieldRegState` representation.
- **New `OracleGeneration` variant for FR**: rejected. FR polynomials fit the existing `DenseTrace` (FieldRegInc) and `OneHotChunk` (FieldRegRa_d) shapes; virtual oracles (`FieldRegVal`, `Wa`, `RaRs1`, `RaRs2`) are declared as `Reference` in Stage 4/5 phases. Keeping the oracle taxonomy unchanged limited blast radius on MLIR plumbing.
- **Threading FR through Stage 1 outer Spartan as full-precision Fr field values**: would double the per-cycle witness footprint. Instead `Stage1Rv64Cycle` carries the four-limb natural form and the conversion to `F` happens at the Stage 3 sumcheck boundary.
- **Batched FR + integer Stage 4 sumcheck**: rejected because the integer Twist runs over `2^5 × T` and the FR Twist over `2^4 × T` — different log_K means awkward batching boundaries, and the FR-inactive zero short-circuit only works cleanly if the FR sumcheck is separable.
- **Sourcing the FR write slot from `event.frd` rather than `bc.frd`**: rejected. The event is uncommitted prover input; the bytecode row is committed via preprocessing. Anchoring the write slot on bytecode is what makes the Stage 6 RAF binding load-bearing for drop-event soundness.
- **Multi-cycle FAssertEq**: rejected for simplicity. The single-cycle constraint `V_FIELD_RS1 − V_FIELD_RS2 = 0` is sufficient; the SDK couples it with the 4×`ADVICE_LD` + 7-cycle Horner reconstruction so the advice limbs are bound natively.

## Documentation

No `book/` changes for this internal coprocessor — guest authors interact through `jolt_inlines_bn254_fr::Fr` whose method signatures (`add`, `sub`, `mul`, `inverse`, `from_limbs`, `to_limbs`) are documented inline in `jolt-inlines/bn254-fr/src/sdk.rs`. A future "native fields" chapter would be the right home if/when additional field coprocessors land.

## Open Follow-Ups

Tracked here for visibility; none block the "shipped" status:

1. **Sparsify the remaining T-sized vectors.** `FieldRegReplay.bytecode`, `fr_frd_inc`, and the Stage 5 `frd_wa_at_field_reg_address` scratch are each ~32 B/cycle. Aggregate ~25 MB at `log_T = 18`. Replacing with sparse `Vec<(cycle, F)>` would tighten the FR-inactive footprint.
2. **Dead-code prune.** The `FieldRegInc` / `FieldRegRa_*` oracle registration in `crates/bolt/src/protocols/jolt/oracles.rs` and `params.rs` is gated by `field_reg_d > 0` and unreachable in production today. Either wire `field_reg_d = 1` once committed-FR is ready, or remove the gated paths. `sub_limbs` is currently only exercised by unit tests; it remains pending a future use site.
3. **ZK BlindFold integration.** The three new sumcheck instances need `input_claim_constraint` / `output_claim_constraint` / `input_constraint_challenge_values` / `output_constraint_challenge_values` implementations. Blocked until BlindFold is ported to the modular stack.
4. **Pinned-slot SDK API.** Amortize the 12-cycle advice-bound extract over runs of in-field ops to drop per-op cost.
5. **Generalize to other native-aligned ≤256-bit prime fields.** Parameterize `tracer/src/instruction/field_arith_common.rs` and `Stage1OuterRv64Data` over `F: PrimeField`. The Stage 3/4/5 kernels are already generic. Estimated ~150 LOC plus a per-field inline SDK wrapper.
6. **Stage 6 `bytecode_entry_stage_values` `register_eq` precompute.** Per-entry cost is currently `O(log K_reg + log K_FR)` Fr muls for the 6 `register_eq` calls. Precomputing `EqPolynomial::evals(register_point)` once outside the bytecode loop drops it to a single array index per call — meaningful at ≥ 1 MB ELF sizes. Preexisting on the integer side; FR added the 3 extra calls.

## References

- `crates/jolt-witness/src/field_reg.rs`
- `crates/jolt-kernels/src/stage{3,4,5,6}.rs`
- `crates/jolt-host/src/lib.rs`
- `crates/jolt-r1cs/src/constraints/rv64.rs`
- `crates/jolt-riscv/src/kind.rs`
- `crates/jolt-verifier/src/stages/common.rs`
- `tracer/src/emulator/cpu.rs`, `tracer/src/instruction/field_*.rs`
- `jolt-inlines/bn254-fr/src/sdk.rs`, `jolt-sdk/macros/src/lib.rs`
- `examples/bn254-poseidon2/`
