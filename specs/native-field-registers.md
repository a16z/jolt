# Spec: BN254 Fr Native-Field Coprocessor

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Sagar Dhawan                   |
| Created     | 2026-05-17                     |
| Status      | shipped                        |
| PR          |                                |

## Summary

A native-field coprocessor that adds 16 × 256-bit BN254 Fr registers and 9 new RISC-V instructions to the modular Jolt zkVM, with three sumcheck instances (Stage 3 / 4 / 5) wired into the existing Twist register-checking pipeline. Guest programs that use the `jolt_inlines_bn254_fr::Fr` SDK type now run multi-thousand-cycle Fr arithmetic per op in software emulation (the integer ISA) AND get a single `FieldOp` cycle whose constraint witness is bound natively to the Fr field via the FR Twist. Software-emulated Fr arithmetic (e.g. arkworks on RV) is replaced end-to-end for FMUL/FADD/FSUB/FINV; FAssertEq closes the prover advice loop, and FieldMov/FieldSLL64/128/192 form the integer→FR bridge ABI for loading inputs and reconstructing outputs.

## Intent

### Goal

Prove BN254 Fr arithmetic at native cost: one FR register-file slot per Fr value, one `FieldOp` cycle per (mul, add, sub, inv) operation, and one constraint relation tying R1CS witnesses to a 16-slot FR Twist register file. The architectural boundaries introduced are:

- **Tracer**: `tracer::emulator::cpu::FieldRegEvent` (`tracer/src/emulator/cpu.rs:169`), `Cpu.field_regs: [[u64; 4]; 16]` (`cpu.rs:205`), `Cpu.field_reg_events: Vec<FieldRegEvent>` (`cpu.rs:208`), and 6 instruction files in `tracer/src/instruction/field_*.rs`.
- **ISA**: 9 instruction kinds — `FieldOp` (funct7=0x40, opcode=0x0B; funct3 selects FMUL/FADD/FSUB/FINV), `FieldAssertEq` (funct7=0x40 funct3=0x00), `FieldMov` (funct7=0x40 funct3=0x01), and `FieldSLL{64,128,192}` (funct7=0x41).
- **R1CS**: 13 new eq-constraint rows (rv64.rs rows 19–31) over slots `V_FIELD_RS1_VALUE=45`, `V_FIELD_RS2_VALUE=46`, `V_FIELD_RD_WRITE_VALUE=47` (`crates/jolt-r1cs/src/constraints/rv64.rs:84-86`). `NUM_R1CS_INPUTS=47`, `NUM_VARS_PER_CYCLE=50`, `NUM_EQ_CONSTRAINTS=32` (`rv64.rs:91-94`).
- **Witness types**: `jolt_witness::field_reg::{FrLimbs, FieldRegEvent, FrCycleBytecode, FieldRegReplay, limbs_to_field, sub_limbs}` in `crates/jolt-witness/src/field_reg.rs`. Compact, sparse-first: only `materialize_frd_inc<F>` (T-length) is retained as a dense materializer.
- **Sumcheck kernels** in `crates/jolt-kernels/`:
  - Stage 3 `FieldRegClaimReduction` (`stage3.rs:2420`) — γ-batched eq sumcheck reducing rd/rs1/rs2 R1CS values to FR write/read claims.
  - Stage 4 `FieldRegRW` (`stage4.rs:1629`, `SparseFieldRegState` at `stage4.rs:2041`) — sparse Hamming-shaped Twist read-write check over `K_FR = 16 × T`.
  - Stage 5 `FieldRegValEvaluation` (`stage5.rs:2916`) — degree-3 `frd_inc · frd_wa_at_address · lt` evaluation.
- **SDK**: `jolt_inlines_bn254_fr::Fr` (`jolt-inlines/bn254-fr/src/sdk.rs`) with three compile-time dispatch paths (host arkworks, `compute_advice` pass populating the advice tape, RV pass-2 emitting the FieldOp instruction sequence). `#[jolt::provable(backend = "modular")]` (`jolt-sdk/macros/src/lib.rs:154`) auto-builds the two-pass advice-tape ELF.

### Invariants

- **Replay/trace length parity**: `FieldRegReplay.num_cycles == trace_len` (asserted in `SparseFieldRegState::new` at `crates/jolt-kernels/src/stage4.rs:2079-2085`) and `replay.bytecode.len() == trace_len` (`stage4.rs:2098-2104`).
- **Cycle bound**: every `FieldRegEvent.cycle < trace_len` (`stage4.rs:2112-2118`).
- **Address mask**: FR addresses are masked `& 0xF` at *both* sides — producer (`fr_bytecode_from_trace` in `crates/jolt-host/src/lib.rs:476-478`) and consumer (`SparseFieldRegState::new` at `stage4.rs:2122,2135,2153`; `frd_wa_at_field_reg_address` at `stage5.rs:3000`).
- **Initial FR state is zero**: every slot starts at `F::zero()` at cycle 0 (`stage4.rs:2096`; cpu boot `tracer/src/emulator/cpu.rs:374`).
- **FINV(0) is unsatisfiable**: tracer panics in `field_op.rs:129` (`expect("FieldOp(FINV) on zero input; the SDK guards this via Fr::inverse() -> Option<Fr>")`); SDK guards via `Fr::inverse() -> Option<Fr>`.
- **Bytecode/event consistency**: every event-bearing cycle has at least one of `bytecode[cycle].reads_frs1/reads_frs2` or `event.rd_written` set. Cycles without an FR event have both flags clear and never read or write FR slots. The host populator (`populate_r1cs_fr_slots`, `crates/jolt-host/src/lib.rs:522-554`) and `SparseFieldRegState::new` walk this invariant in lockstep.
- **Bit ordering**: Stage 4 binds `r_cycle` LowToHigh inside `SparseFieldRegState` (`stage4.rs:2174`); `final_frs2_read_eval` reverses both the cycle and address halves of the bound point (`stage4.rs:2351-2353`) so `EqPolynomial::evals` receives MSB-first input matching `sparse_register_read_eval`.
- **Read claim decomposition**: `frs1_ra` is recovered from the combined eval as `(combined_read_ra − γ² · frs2_ra) · γ⁻¹` (`stage4.rs:2279`); the equality `combined_read_ra == γ · frs1_ra + γ² · frs2_ra` is asserted under `debug_assertions` (`stage4.rs:2282-2288`).
- **R1CS witness coupling**: `populate_r1cs_fr_slots` stamps `V_FIELD_RS1/RS2/RD_WRITE_VALUE` from the same `FieldRegReplay` that feeds Stage 3/4/5 (`crates/jolt-host/src/lib.rs:641`). Stage 3 input claim consumes those slots via `Stage3Cycle.{field_rs1, field_rs2, field_rd, is_field_op}` populated by `populate_fr_cycle_fields` (`crates/jolt-host/src/lib.rs:655`). The four expressions (`input_claim`/`output_claim`/`input_claim_constraint`/`output_claim_constraint`) for each FR sumcheck instance must stay in lockstep — Stage 4's `expected_field_reg_rw` (`stage4.rs:3882-3902`) is the verifier mirror of the prover's claim.
- **No new `OracleGeneration` variant**: FR oracles (`FieldRegInc` dense, `FieldRegRa_d` one-hot) reuse the existing `DenseTrace`/`OneHotChunk` shapes and are gated behind `params.field_reg_d > 0` (`crates/bolt/src/protocols/jolt/params.rs:87`). They are currently registered only when MLIR plumbing is active; today `field_reg_d = 0` (`params.rs:58`) and the FR Twist witness flows in-process via `Stage4FieldRegWitness` / `Stage5FieldRegValWitness` directly from `FieldRegReplay`.

### Non-Goals

- **ZK mode for the FR Twist sumchecks.** BlindFold integration (input/output constraint synchronization and Pedersen-committed round polynomials for the three new sumcheck instances) is not implemented in the modular stack. Building with `--features zk` exercises the existing Twist register-checking ZK path but the FR-specific instances are not wired in.
- **Multi-cycle FINV / FAssertEq.** Both are single-cycle today. FINV(0) is unsatisfiable and the prover refuses (panic in tracer); the SDK guards with `Option<Fr>`.
- **Variable-bit-width Fr.** The coprocessor is hardcoded to BN254 Fr at 256 bits. Other native fields are out of scope.
- **Pinned-slot SDK API.** The current SDK pays 7+7+1+12 = 27 cycles per binary op (load A Horner, load B Horner, FieldOp, advice-bound extract). Amortizing over runs of in-field ops is future work — `jolt-inlines/bn254-fr/src/sdk.rs:26-39` documents the cost.
- **Sparsifying remaining T-sized vectors.** `materialize_frd_inc` (T-length, in stage 4/5), bytecode (T-length, in `FieldRegReplay`), and `frd_wa_at_field_reg_address` (T-length scratch in stage 5) are all still dense. Follow-up; see "Open follow-ups" below.
- **Decoupling `FieldRegInc` from the dense `DenseTrace` family.** Today FR commitments piggyback on the existing per-cycle commitment plumbing; a dedicated FR-only commit path is not pursued.

## Evaluation

### Acceptance Criteria

All criteria are validated and reflect the shipped state:

- [x] `bn254-fr-poseidon2-sdk` end-to-end prove + verify succeeds with `valid: true` (driver at `examples/bn254-fr-poseidon2-sdk/src/main.rs`).
- [x] `cargo nextest run -p jolt-kernels` — all kernel tests pass (Stage 3/4/5 batched-kernel synthetic-witness round-trips remain green; see `stage4.rs:4492-4593` for the witness-shape verification tests).
- [x] `cargo nextest run -p jolt-witness` — `field_reg` unit tests pass (`crates/jolt-witness/src/field_reg.rs:164-247`: limbs round-trip, sub_limbs borrow, `frd_inc` post-minus-pre, `limbs_to_field` LE assembly, empty-replay shape).
- [x] `cargo clippy --features host -- -D warnings` clean across `jolt-kernels`, `jolt-host`, `jolt-prover`, `jolt-witness`.
- [x] No K_FR×T dense materializer is reachable in production paths. The legacy `materialize_field_reg_val` / `materialize_frs1_ra` / `materialize_frs2_ra` / `materialize_frd_wa` helpers were removed from `crates/jolt-witness/src/field_reg.rs`; only `materialize_frd_inc` (T-length) remains and is called by Stage 4 (`stage4.rs:2183`) and Stage 5 (`stage5.rs:2945`).
- [x] Sparse FR replay buffers ≤ ~3 entries per FR-active cycle (asserted by the producer construction in `SparseFieldRegState::new` at `stage4.rs:2094`: `Vec::with_capacity(replay.events.len() * 3)`).
- [x] FINV(0) traps at the tracer level rather than producing an invalid proof (`tracer/src/instruction/field_op.rs:128-130`).

### Testing Strategy

**Unit tests retained / added** (standard `--features host`):

- `crates/jolt-witness/src/field_reg.rs:164-247` — `FrLimbs` defaults & round-trip, `sub_limbs` borrow correctness, `materialize_frd_inc` post-minus-pre semantics, `limbs_to_field` little-endian assembly, empty-replay shape.
- `crates/jolt-kernels/src/stage4.rs:4480-4660` — Stage 4 batched kernel proves+verifies a synthetic witness, accepts sparse RAM addresses, accepts sparse register accesses, rejects a tampered eval, and round-trips `Stage45SparseTraceWitness::from_accesses`. These cover the non-FR Stage 4 paths the FR path piggybacks on (claim synchronization, transcript parity, point-shape normalization).
- `crates/jolt-kernels/src/stage4.rs:3882-3902`, `:2259-2370` — Stage 4 FR-specific verifier mirror and `final_frs2_read_eval` exercised by the e2e SDK test below.

**End-to-end** (standard `--features host`):

- `examples/bn254-fr-poseidon2-sdk` — A guest program performing one Poseidon2 permutation over BN254 Fr using the SDK. Drives the full stack: SDK three-mode dispatch (`jolt-inlines/bn254-fr/src/sdk.rs`), tracer FieldOp emission, `FieldRegEvent` stream, `populate_r1cs_fr_slots`, `populate_fr_cycle_fields`, Stage 3/4/5 FR sumchecks, Dory commitments, verifier round-trip. Validated at `log_T = 18`; see Performance section.

**Gaps**:

- No dedicated kernel unit test exercises `SparseFieldRegState::round_poly` against a synthetic event stream — the SDK e2e is currently the only coverage. Adding `stage4_field_reg_rw_proves_synthetic_events` is a follow-up.
- The `materialize_frd_inc` test covers semantics but not the Stage 5 `frd_wa_at_field_reg_address` aggregation; that's also indirect through the SDK e2e.
- ZK mode (`--features host,zk`) is not exercised against FR; muldiv ZK e2e remains the project-wide ZK gate but does not execute any FR instruction.

### Performance

Measured at `log_T = 18` on the FR-active Poseidon2 workload:

| Metric                 | jolt-inlines-bn254-fr (FR coprocessor) | arkworks-Fr-on-RV (software baseline) |
|------------------------|---------------------------------------:|--------------------------------------:|
| Prove time             |                              **2.72 s** |                                3.31 s |
| Peak RSS               |                            **2.58 GiB** |                                  ~same |

Memory-shape change:

- **Host-side**: the host now carries a single sparse `FieldRegReplay` (~few KB) instead of 5 dense K_FR×T buffers (`field_reg_val`, `frs1_ra`, `frs2_ra`, `frd_wa`, `frd_inc`). For `log_T = 18`, that is a reduction from ~520 MB of host-side scratch to ~9K sparse entries.
- **Kernel-side**: the transient K_FR×T materialization that the previous (Phase 4 first-pass) kernel performed inside Stage 4 has been eliminated — the sparse Hamming round-poly walks `entries: Vec<SparseFieldRegEntry<F>>` directly. Estimated kernel-side transient saving ~3 GB at `log_T = 18`.

There are no `jolt-eval` objectives tied specifically to FR today. The relevant cross-cutting objectives (Stage 4 prove time, peak RSS at `log_T = 18`) move in the FR-active direction; FR-inactive traces (e.g. `muldiv`) are functionally unchanged because `replay.events.is_empty()` short-circuits every sparse code path to `F::zero()`.

## Design

### Architecture

Data flow, in execution order:

1. **Guest SDK** (`jolt-inlines/bn254-fr/src/sdk.rs`): Three compile-time paths.
   - *Host* (default, no RV target): delegates to `ark_bn254::Fr`.
   - *`compute_advice` pass on RV* (built automatically by the macro): computes via arkworks, writes the 4 result limbs to the advice tape using `VirtualHostIO`.
   - *Pass-2 RV*: emits a 27-cycle sequence per binary op: 7-cycle Horner load A (FieldMov + FieldSLL64/128/192 + FieldAdd) + 7-cycle Horner load B + 1 FieldOp + 12-cycle advice-bound extract (4 `ADVICE_LD` + 7-cycle Horner reconstruction + FieldAssertEq).
2. **Macro** (`jolt-sdk/macros/src/lib.rs:154`): `#[jolt::provable(backend = "modular")]` builds two ELFs (default + `compute_advice` feature) via `build_with_features`, so the host can replay the compute-advice ELF first to populate the advice tape, then prove the pass-2 ELF.
3. **Tracer** (`tracer/src/instruction/field_op.rs`, `field_assert_eq.rs`, `field_mov.rs`, `field_sll{64,128,192}.rs`):
   - Decode arm at `tracer/src/instruction/mod.rs:1090-1114` splits opcode `0x0B` on funct7 — `0x40` → FieldOp/FieldAssertEq/FieldMov (funct3 selects); `0x41` → FieldSLL64/128/192.
   - Execution mutates `Cpu.field_regs` and appends a `FieldRegEvent { cycle_index, slot, old, new }` to `Cpu.field_reg_events` on every write (`cpu.rs:169-208`).
4. **jolt-host** (`crates/jolt-host/src/lib.rs`):
   - `fr_bytecode_from_trace` (`:459`) derives `FrCycleBytecode` per trace row from the decoded instruction kind. Address fields are masked `& 0xF`.
   - `convert_fr_events` (`:490`) maps tracer events into `jolt_witness::field_reg::FieldRegEvent`.
   - `FieldRegReplay { num_cycles, bytecode, events }` is built once (`:632`).
   - `populate_r1cs_fr_slots` (`:522`) walks the replay's running FR-state and stamps `V_FIELD_RS1/RS2/RD_WRITE_VALUE` per cycle into the R1CS witness so Stage 1 outer Spartan and Stage 3 claim reduction see the right values.
   - `populate_fr_cycle_fields` (`:561`) stamps `field_rs1`/`field_rs2`/`field_rd`/`is_field_op` on `Stage1Rv64Cycle` and `Stage3Cycle` from the same walk.
   - The replay is handed to Stage 4/5 via `Stage45SparseTraceWitness::with_field_reg_replay` (`:761`); Stage 3 picks up FR factors from `stage3_cycles_vec` directly.
5. **Stage 3 `FieldRegClaimReduction`** (`crates/jolt-kernels/src/stage3.rs:2420-2492`): γ-batched eq sumcheck — proves `Σ_t eq(r, t) · (rd(t) + γ · rs1(t) + γ² · rs2(t))` where each factor is gated by `cycle.is_field_op` (`stage3.rs:2485-2489`). Factor outputs are published as `FieldRdWriteValue` / `FieldRs1Value` / `FieldRs2Value` openings for Stage 4 / 5.
6. **Stage 4 `FieldRegRW`** (`crates/jolt-kernels/src/stage4.rs`, `SparseFieldRegState` at `:2041`):
   - Input claim: `Σ_{k,j} eq(r_cycle, j) · (frd_wa(k,j) · (val(k,j) + frd_inc(j)) + γ · (frs1_ra(k,j) · val(k,j) + γ · frs2_ra(k,j) · val(k,j)))` (`stage4.rs:3900-3902`).
   - Implementation walks per-cycle `SparseFieldRegEntry { row, col, val, prev_val, next_val, read_ra, frd_wa }` records (`stage4.rs:2058-2067`). At most three entries per FR-active cycle: one frs1 read (`read_ra = γ`), one frs2 read (`read_ra += γ²`, merged on same col), one frd write (`frd_wa = 1`).
   - `frs2_reads: Vec<(cycle, col)>` is held separately so the final `frs2_ra` evaluation can be recovered independently (`stage4.rs:2095, 2341-2369`). The combined `read_ra` at the end of the sumcheck is decomposed via `frs1_ra = (combined − γ² · frs2_ra) · γ⁻¹` (`stage4.rs:2279`).
   - When `current_trace_len == 1` the state materializes a tiny dense view (`materialize_dense`, `stage4.rs:2303`) of size `K_FR = 16` for the remaining address-phase rounds.
7. **Stage 5 `FieldRegValEvaluation`** (`crates/jolt-kernels/src/stage5.rs:2916`): degree-3 dense sumcheck over `frd_inc(t) · frd_wa_at_address(t) · lt(t)`, where `frd_wa_at_address` is the T-length aggregate `Σ_{events at t} address_eq[(ev.frd) & 0xF]` (`stage5.rs:2980-3003`). This is the address-bound write evaluation that Stage 4 outputs `FrdInc` and `FrdWa` consume to reconstruct `FieldRegVal` over the final point.
8. **Verifier** mirrors live in `crates/jolt-verifier/src/stages/stage{3,4,5}.rs`. Stage 4's `expected_field_reg_rw` (`crates/jolt-kernels/src/stage4.rs:3882-3902`) is the canonical claim-reconstruction formula for both prover correctness checks and verifier acceptance.

### Alternatives Considered

- **Dense K_FR×T materialization** (the original Phase 4 first-pass kernel): kept the kernel code identical in shape to the integer-register Twist but blew up host RAM by ~520 MB at `log_T = 18` and forced kernel-side transient allocations of ~3 GB. Replaced with the current sparse representation (`SparseFieldRegState`) once the sparse round-poly + per-cycle entries were proven equivalent against the dense path on small synthetic inputs.
- **New `OracleGeneration` variant for FR**: rejected. FR polynomials fit the existing `DenseTrace` (FieldRegInc) and `OneHotChunk` (FieldRegRa_d) shapes; virtual oracles (`FieldRegVal`, `Wa`, `RaRs1`, `RaRs2`) are declared as `Reference` in Stage 4/5 phases. Keeping the oracle taxonomy unchanged limited blast radius on MLIR plumbing.
- **Threading FR through Stage 1 outer Spartan as full-precision Fr field values**: would double the per-cycle witness footprint. Instead `Stage1Rv64Cycle` carries the four-limb natural form (`field_rs1/2/d: [u64; 4]`) and the conversion to `F` happens at the Stage 3 sumcheck boundary via `fr_limbs_to_field` (`stage3.rs:2468-2476`).
- **Multi-cycle FAssertEq**: rejected for simplicity. The single-cycle constraint `V_FIELD_RS1 − V_FIELD_RS2 = 0` is sufficient; the SDK couples it with the 4×`ADVICE_LD` + 7-cycle Horner reconstruction so the advice limbs are bound natively.

## Documentation

No `book/` changes required for this internal coprocessor — guest authors interact through `jolt_inlines_bn254_fr::Fr` whose method signatures (`add`, `sub`, `mul`, `inverse`, `from_limbs`, `to_limbs`) are documented inline in `jolt-inlines/bn254-fr/src/sdk.rs`. A future "native fields" chapter would be the right home if/when additional field coprocessors land.

## Execution

The shipped implementation followed the five-phase port plan previously captured in `specs/fr-v2-port-plan.md` (now deleted; superseded by this spec). Key deviations from the plan:

- The Phase-5b dense K_FR×T materializers landed first and were then replaced wholesale by `SparseFieldRegState` (Stage 4) and `frd_wa_at_field_reg_address` (Stage 5). Only `materialize_frd_inc` remains in `field_reg.rs`.
- The originally planned `FrCycleData` + `replay_field_regs` helpers were removed once the Stage 4 sparse path subsumed their use cases.
- `field_reg_d` stayed at 0 (`crates/bolt/src/protocols/jolt/params.rs:58`) — the FR oracle family (`FieldRegInc`, `FieldRegRa_*`) is registered on the MLIR side but not yet committed, since the in-process `FieldRegReplay` → Stage 4/5 path supplies the same witness without needing committed oracles.

## Open follow-ups

Tracked here for visibility; none block the "shipped" status:

1. **Sparsify the remaining T-sized vectors**:
   - `FieldRegReplay.bytecode: Vec<FrCycleBytecode>` (T-length, all zeros on non-FR cycles).
   - `SparseFieldRegState.frd_inc: Vec<F>` (T-length, all zeros on non-FR cycles).
   - Stage 5 `frd_wa_at_field_reg_address` output (T-length scratch).
   Each is ~32 B/cycle at `log_T = 18` = ~8 MB; aggregate is ~25 MB. Replacing with `Vec<(cycle, F)>` would tighten the FR-inactive zero footprint.
2. **Dead-code prune**: the `FieldRegInc` / `FieldRegRa_*` oracle registration code in `crates/bolt/src/protocols/jolt/oracles.rs` and `params.rs` is gated by `field_reg_d > 0` and currently unreachable in production (today's runs use `field_reg_d = 0`). Either wire `field_reg_d = 1` once committed-FR is ready, or remove the gated paths. `sub_limbs` is currently only exercised by unit tests; it remains in `field_reg.rs` pending a Phase-6 use site.
3. **ZK BlindFold integration**: the three new sumcheck instances need `input_claim_constraint` / `output_claim_constraint` / `input_constraint_challenge_values` / `output_constraint_challenge_values` implementations to participate in the modular BlindFold protocol. Until then, FR is standard-mode only.
4. **Pinned-slot SDK API**: amortize the 12-cycle extract over runs of in-field ops to drop per-op cost from 27 cycles toward the 13-cycle v1 baseline.
5. **Dedicated FR Stage 4 unit test**: today the SDK e2e is the only direct exercise of `SparseFieldRegState::round_poly`. A synthetic `stage4_field_reg_rw_proves_synthetic_events` test in `crates/jolt-kernels/src/stage4.rs#mod tests` would catch sparse-path regressions without rebuilding the guest ELF.

## References

- `specs/fr-v2-port-plan.md` (deleted; predecessor port plan)
- `specs/TEMPLATE.md`
- Source files cited inline above; primary entry points:
  - `crates/jolt-witness/src/field_reg.rs`
  - `crates/jolt-kernels/src/stage{3,4,5}.rs`
  - `crates/jolt-host/src/lib.rs:459-761`
  - `crates/jolt-r1cs/src/constraints/rv64.rs:84-604`
  - `tracer/src/emulator/cpu.rs:160-208`, `tracer/src/instruction/field_*.rs`
  - `jolt-inlines/bn254-fr/src/sdk.rs`, `jolt-sdk/macros/src/lib.rs:154`
  - `examples/bn254-fr-poseidon2-sdk/`
