# FieldReg Twist Integration Plan — Post-Deep-Read

| Field       | Value                                                                      |
|-------------|----------------------------------------------------------------------------|
| Created     | 2026-04-20                                                                 |
| Updated     | 2026-04-21 — Plans A/C/B all DONE; forward roadmap added                   |
| Status      | Protocol recipe proven end-to-end on refactor-crates. Productization next. |
| Sibling to  | `native-field-registers.md` (v2 spec)                                      |
| Supersedes  | All prior piecewise Phase 1/2/3 attempts                                   |

## Plan A — DONE (2026-04-20)

Shipped concretely:
- `Op::ScaleEval` + new `VerifierOp::ScaleEval` variant (`crates/jolt-compiler/src/module.rs`, `crates/jolt-verifier/src/verifier.rs`) — reconciles the zero-padded commitment MLE with the cycle-only FinalBind scalar.
- `ModuleBuilder::add_padded_poly` method (`crates/jolt-compiler/src/builder.rs`) — sets `committed_num_vars` for dense cycle-only polys.
- Standalone Twist (`crates/jolt-equivalence/tests/field_register_twist_standalone.rs`) now commits ALL 5 witness polys (Inc, Ra, Val, ReadValue, WriteValue) with proper opening at `[r_addr_BE ∥ r_cycle_BE]`.
- Batched-path single-instance wiring: `ch_batch` FiatShamir + `batch_challenges: vec![ch_batch]` on `VerifySumcheck` — aligns prover's `AbsorbInputClaim` + `Squeeze` with verifier's batched-path absorb+squeeze.
- 11/11 standalone tests green. 55/55 jolt-equivalence tests green (no regression).

Key gotchas encountered:
- `Op::AbsorbInputClaim` always absorbs claim; `VerifierOp::VerifySumcheck` only absorbs in batched path. Must use batched path even for single instances.
- Opening point must be BE-reversed per segment (`[r_addr_BE ∥ r_cycle_BE]`), NOT raw round_slots (LowToHigh).
- `ScaleEval` must fire on BOTH prover AND verifier — added matching `VerifierOp::ScaleEval`.

## Plan C — DONE (2026-04-20)

Shipped concretely:
- `FieldRegEvent` + `FieldRegConfig` + `DerivedSource::with_field_reg()` + compute methods (`field_reg_inc`, `field_reg_ra`, `field_reg_val`) in `crates/jolt-witness/src/derived.rs`. Mirrors `RamConfig` + `with_ram()` pattern.
- `PolySource::Derived` arm for `FieldRegInc | FieldRegRa | FieldRegVal | FieldRegEqCycle` in `polynomial_id.rs`.
- `FieldRegConfig::compute_read_value` / `compute_write_value` helpers for the Witness-source RV/WV polys (canonical single-source-of-truth).
- Test file uses `ProverData` + `OverrideProvider` (for adversarial mutations) instead of `StandaloneProvider`. Deleted `TwistWitness` and `StandaloneProvider`.
- 11/11 standalone tests green. 55/55 equivalence suite green. Clippy clean.

Anti-pattern from `memory_copy/feedback_no_test_bypass.md` resolved — FR witness now flows through the same production infrastructure RAM/Registers use.

## Hard Confirmations From 15-Agent Deep Read

1. **Main-protocol verifier is NOT partial.** Each `VerifySumcheck` op batches multiple `SumcheckInstance` entries. 6 prover `batched_sumchecks` contain ~20 total instances; all verified via in-op batching. `modular_self_verify` is the binding correctness gate (CLAUDE.md) and currently passes.

2. **`Op::ScaleEval { poly, factor_challenges: r_addr_be }`** is the fix for "dense cycle-only poly committed padded to K×T" opening mismatch. Multiplies the prover's FinalBind scalar by `∏(1 − r_addr_i)` so the claim matches the zero-padded MLE that the PCS computes at the full `[r_addr, r_cycle]` point. Applied before `Op::CollectOpeningClaimAt`.

3. **Opening point is big-endian reversed**, not raw LowToHigh. RAM Twist opens at `[r_addr_BE, r_cycle_BE]` (round_slots reversed per segment). My prior Plan A failed because I used raw round_slots.

4. **Sparse one-hot polys don't need ScaleEval** (they're genuinely K×T with real values). **Polys committed at native K×T (no padding) don't need ScaleEval either** — ScaleEval is specifically for the zero-padding correction.

5. **Layout `ra[slot*T + cycle]`** (cycle as LOW bits) matches the RAM Twist convention. No `Transpose` needed.

6. **Segmented reduce leaves buffers clean**: `interpolate_inplace` binds in place; final state is a 1-element scalar = `MLE(round_challenges)` in LowToHigh variable order.

7. **Anti-pattern**: `StandaloneProvider`'s direct `put()` bypasses proper `DerivedSource` infrastructure (violates `memory_copy/feedback_no_test_bypass.md`). Proper path: `FieldRegConfig` in `crates/jolt-witness/src/derived.rs`, mirroring `RamConfig`.

8. **Phase 2 stage-2 fold mechanics**: FR instance gets `first_active_round = stage2_max_rounds - (log_T + log_K_FR) = 45 - 9 = 36`. FR's 9 live rounds consume `round_slots[36..45]`. Claim scaled by `2^36` at the combined claim level — O(1) per inactive round, no perf penalty.

## Plan A — Standalone Twist with 5 Committed Polys (DO FIRST)

Scope: `crates/jolt-equivalence/tests/field_register_twist_standalone.rs` only. ~150 lines net.

Steps:
1. Declare Inc `Committed` at `num_vars=LOG_T`, `committed_num_vars=LOG_K+LOG_T` (padded).
2. Declare Ra, Val `Committed` at `num_vars=LOG_K+LOG_T` (native).
3. Unified `Op::Commit` over `[Inc, Ra, Val, ReadValue, WriteValue]` at K×T grid (num_vars = LOG_K+LOG_T for the main commit barrier; ReadValue/WriteValue may need separate commit at LOG_T).
4. After the main 2-phase sumcheck:
   - Derive `r_addr_slots` = last LOG_K slots of `round_slots` (phase-2 address binds).
   - Derive `r_cycle_slots` = first LOG_T slots of `round_slots` (phase-1 cycle binds).
   - Construct `r_addr_BE` = `r_addr_slots` reversed.
   - Construct `r_cycle_BE` = `r_cycle_slots` reversed.
   - `opening_point = [r_addr_BE ∥ r_cycle_BE]` as `Vec<ChallengeIdx>`.
5. `Op::ScaleEval { poly: Inc, factor_challenges: r_addr_BE.clone() }`.
6. `Op::CollectOpeningClaimAt` for Inc at `opening_point`, `committed_num_vars: Some(LOG_K+LOG_T)`.
7. Same for Ra, Val (no ScaleEval — native K×T, no padding correction needed).
8. ReadValue/WriteValue continue opening at rw_cycle_challenges (pre-sumcheck bind).
9. Verifier mirrors all: AbsorbCommitment in order, matching `CollectOpeningClaimAt` with same point_challenges.
10. Adversarial tests: mutate Inc[0], Ra[0], Val[0] — each should fail verify.

Acceptance: `twist_honest_accepts` + `twist_adversarial_inc_mutation_rejects` + `twist_adversarial_ra_mutation_rejects` + `twist_adversarial_val_mutation_rejects` + existing RV/WV adversarial tests all green.

## Plan C — Move Witness Out of Test Harness (DO SECOND)

Scope: `crates/jolt-witness/src/derived.rs`, `crates/jolt-compiler/src/polynomial_id.rs`, the test file. ~150 lines net.

Steps:
1. Add `FieldRegEvent { cycle, slot, old, new }` and `FieldRegConfig { k, initial_state, events }` to `crates/jolt-witness/src/derived.rs`.
2. Add `DerivedSource::with_field_reg(FieldRegConfig)` builder + internal accessor `field_reg()`.
3. Add compute methods `field_reg_inc`, `field_reg_ra`, `field_reg_val` (port `TwistWitness::from_events`).
4. Extend `DerivedSource::compute` match with `FieldRegInc/Ra/Val` → call appropriate method.
5. In `crates/jolt-compiler/src/polynomial_id.rs`, add `PolySource::Derived` arm for `FieldRegInc | FieldRegRa | FieldRegVal | FieldRegEqCycle`.
6. `ReadValue`/`WriteValue` stay `PolySource::Witness` — populate via `Polynomials::insert` using a helper that derives from the same event stream (single source of truth).
7. Replace `StandaloneProvider` in the test with proper `ProverData` + `FieldRegConfig`.
8. All Plan A tests remain green.

## Plan B — DONE (2026-04-21) — FR as 6th Stage-2 batched instance

### Status

All 6 Stage-2 instances pass. 11/11 standalone FR tests pass. 8/8 muldiv tests pass (including `modular_self_verify_with_fieldreg` and baseline `modular_self_verify`). Clippy clean on jolt-verifier/jolt-zkvm/jolt-compiler.

### Root-cause fixes in debug session

1. **SE_BASE off-by-2**: Adding `RecordEvals([field_reg_read_value, field_reg_write_value])` pre-sumcheck pushed FR RV/WV into `sp.evals[1..3]`, shifting the stage-2 batch evals to `sp.evals[3..]`. `SE_BASE` was left at 1. Fixed to 3 in `jolt_core_module_with_fieldreg.rs`. The failing test exposed it because FR's `output_check` reads `SE_FR_RA/VAL/INC`; baseline instances' `output_check` for RW coincidentally evaluated to 0 regardless of sp.evals[] index (Twist identity structure).
2. **`stage2_num_instances` regression**: Bumping `params.stage2_num_instances` from 5 → 6 broke the non-FR `jolt_core_module.rs` baseline (it uses the param to size batch-coefficient squeezes). Fix: reverted param to 5, hard-coded `const STAGE2_NUM_INSTANCES_FR: usize = 6` locally in the FR example.

### Debug workflow that worked

- eprintln FR_DEBUG gates at `jolt-verifier::CheckOutput` (per-instance output + sp.evals dump) and `jolt-zkvm::BatchAccumulateInstance`/`Evaluate`.
- Ran via `FR_DEBUG=1 cargo nextest run ... 2>/tmp/fr_debug.err 1>/tmp/fr_debug.out` — stderr redirect crucial (nextest captures stdout separately).
- Cross-referencing verifier's sp.evals[] with prover's `Evaluate` values revealed the index shift immediately.
- All debug instrumentation removed after fix.

---

---

## Phase 2b checkpoint (2026-04-21)

Post Plans A/C/B, Phase 2b landed the following:

### DONE
- **Task #55 — `FieldRegEvent` widened to `[u64;4]`.**
  `crates/jolt-witness/src/derived.rs`: `FieldRegEvent.{old,new}: u64 →
  FrLimbs = [u64;4]`, `FieldRegConfig.initial_state: Vec<u64> → Vec<FrLimbs>`.
  New `pub fn limbs_to_field<F: Field>(limbs: &FrLimbs) -> F` helper: LE
  byte-serialize + `F::from_bytes` (modular-reducing). `compute_read_value`,
  `compute_write_value`, `field_reg_inc`, `field_reg_val` all updated.
  Callers in standalone test + muldiv test updated.
- **Task #58 — PolyId rename.** `PolynomialId::FieldReg{Inc,Val,Wa,Ra,
  ReadValue,WriteValue,EqCycle}` already existed as dedicated variants; only
  a stale comment in the standalone test referencing `Ram*` aliases needed
  correction.
- **Task #62 — Non-empty-events E2E test (honest + 2 adversarial).**
  `crates/jolt-equivalence/tests/muldiv.rs`:
  - Refactored `run_jolt_zkvm_prover_with_fieldreg` into a
    `run_jolt_zkvm_prover_with_fieldreg_events(events: Vec<FieldRegEvent>)`
    helper with an empty-events convenience wrapper (so existing tests
    still pass).
  - `modular_self_verify_with_fieldreg_nonempty_events` — injects 3 events
    exercising non-trivial Fr values including a 4-limb value `[999, 999,
    999, 999]`. Runs full Jolt prove/verify pipeline, asserts acceptance.
  - `modular_self_verify_with_fieldreg_nonempty_events_inconsistent_event_rejects`
    — witness-consistency check: event with `old` diverging from running
    state panics in `compute_read_value` before sumcheck starts.
  - `modular_self_verify_with_fieldreg_nonempty_events_tampered_wv_rejects`
    — sumcheck-level check: honest events + committed WV mutated at one
    cycle; verifier rejects.
  - All 6 muldiv tests + 11 standalone FR tests green. Clippy clean.

**Outcome**: the v2 spec's architectural claim ("refactor sidesteps the
dual-path Bz asymmetry by construction") is now an empirically confirmed
result for the FR Twist under non-empty events. The full Stage-1/Stage-2
R1CS + FR Twist pipeline handles real 256-bit Fr values end-to-end with
honest-accepts + adversarial-rejects.

### Remaining open (refactor-only)

Dependency chain for "native field inlines end-to-end with a real guest":

1. **`#61` FieldOp ISA + tracer hook** — add a FieldOp R-type instruction to
   the refactor-crates tracer and emit `FieldRegEvent`s. Largest remaining
   item. Detailed next-session plan in `§Phase 2b — Task #61 implementation
   plan` below.
2. **`#57` BN254 Fr inline SDK** — depends on #61 to have the instructions to
   emit. Port limb-register ABI from v1.
3. **`#59` FieldOp arithmetic R1CS constraints** — depends on #61 for the
   R1CS column plumbing.
4. **`#49` Verified FADD/FSUB/FINV sequences** — depends on #57/#59.
5. **`#60` Smoke example + bench** — depends on #57 guest SDK.

Independent tracks:
- **`#52` Limb-to-Fr bridge (Phase 3)** — can be prototyped as a standalone
  Module with synthetic claims, decoupled from tracer work.
- **`#56` ZK mode** — large, orthogonal, blocked by `jolt-blindfold` crate
  substrate.

---

## Phase 2b — Task #61 implementation plan

Scope: add a FieldOp family of RISC-V instructions to
`/Users/sdhawan/Work/jolt-refactor-crates/tracer/` (shared tracer crate),
emitting `jolt_witness::derived::FieldRegEvent`s through a new tracer event
channel parallel to `RAMAccess`. Also register the instructions in
`crates/jolt-instructions/` and add the jolt-host bridge that materializes
`FieldRegConfig.events` from the cycle trace.

### Target ISA

| funct3 | Mnemonic | Semantics                                        |
|--------|----------|--------------------------------------------------|
| 0x02   | FMUL     | `field_regs[frd] = field_regs[frs1] * field_regs[frs2]` |
| 0x03   | FADD     | `field_regs[frd] = field_regs[frs1] + field_regs[frs2]` |
| 0x04   | FINV     | `field_regs[frd] = field_regs[frs1]^{-1}`         |
| 0x05   | FSUB     | `field_regs[frd] = field_regs[frs1] - field_regs[frs2]` |
| 0x06   | FMovI2F  | `field_regs[frd][limb] = x[rs1]` (limb from imm)  |
| 0x07   | FMovF2I  | `x[rd] = field_regs[frs1][limb]`                  |

Opcode: `0x0B` (custom-0). funct7: `0x40` for all field ops.

### Concrete steps

1. **`tracer/src/emulator/cpu.rs`**
   - Add `pub field_regs: [[u64; 4]; 16]` to `Cpu`, initialize to all-zeros
     in every constructor site (use `grep -n "fn new\|field_regs" cpu.rs`).
   - Add `pub field_reg_events: Vec<FieldRegEvent>` (define a local struct
     or re-export from `jolt-witness` via a dev-dep-free path).
   - `take_field_reg_events` accessor mirroring `EmulatorState` RAM draining.

2. **`tracer/src/instruction/field_op.rs`** (new) — one type `FieldOp` with
   a funct3 field distinguishing the four arithmetic ops. `RAMAccess = ()`.
   Custom `RISCVTrace::trace` override pushes a `FieldRegEvent` onto
   `cpu.field_reg_events` after `execute`. Execute dispatches on funct3:
   - `arkworks::bn254::Fr` for FMUL/FADD/FSUB/FINV (same as v1 `field_op.rs`).
   - Natural-form `[u64;4]` round-trip via `Fr::from_bigint` /
     `Fr::into_bigint`.

3. **`tracer/src/instruction/fmov_int_to_field_limb.rs`** (new) — FMovI2F
   as its own struct (different operand layout: reads integer register,
   writes one limb). Emits a FieldRegEvent only on the 4th cycle of an
   FMov group (when the full 256-bit value is assembled) — see v1 logic.

4. **`tracer/src/instruction/fmov_field_to_int_limb.rs`** (new) — FMovF2I;
   emits a read-only FieldRegEvent (`old == new`) on the first cycle of
   the group.

5. **`tracer/src/instruction/mod.rs`**
   - Register the new structs in `define_rv32im_enums!` (one line each).
   - Add a decode arm for opcode `0x0B` that splits by funct7/funct3 to
     dispatch to `FieldOp` / `FMovI2F` / `FMovF2I`.

6. **`crates/jolt-instructions/src/rv/field_op.rs`** (new) — 6 unit structs
   (`FMul`, `FAdd`, `FSub`, `FInv`, `FMovI2F`, `FMovF2I`) impl `Instruction`
   + `Flags`. `execute` is a stub since the pure-function model (u64→u64)
   doesn't fit Fr; the R1CS side uses the CircuitFlags + virtual columns to
   constrain.

7. **`crates/jolt-instructions/src/flags.rs`**
   - Add `CircuitFlags::IsFieldOp` (or per-op flags if the R1CS needs
     per-op distinction — defer that decision to task #59).

8. **`crates/jolt-instructions/src/instruction_set.rs`**
   - Append the 6 new `Box::new(...)` entries.

9. **`crates/jolt-host/src/`**
   - `tracer_cycle.rs`: expose `field_reg_event(&self) -> Option<FieldRegEvent>`
     on `CycleRow`, implement for the new `Cycle::FMul` etc. variants.
   - New `field_reg.rs` (or extension of `extract.rs`): fold the per-cycle
     Options into `(initial_state, events)` for `FieldRegConfig`.

10. **Test**: extend `modular_self_verify_with_fieldreg_nonempty_events` to
    optionally consume events produced by a tiny hand-rolled guest program
    instead of synthetic literals. Initial version can build the guest
    program via `Program::decode` + a hardcoded `.text` section emitting
    FMUL directly (no BN254 SDK yet — that's task #57).

### Risks / unknowns

- **CPU state constructors**: there are multiple `Cpu::new`-like sites across
  the tracer. Missing any of them leaves `field_regs` uninitialized.
- **Event ordering vs inline sequences**: if FMov is implemented as a 4-cycle
  inline sequence (like v1), the event must fire on the specific limb cycle
  where the 256-bit value is assembled, not every cycle.
- **jolt-host `CycleRow` trait**: adding a new method may require updating
  every `impl CycleRow` site in `tracer_cycle.rs` — survey with grep first.

Budget estimate: 1-2 focused engineer-days for the tracer side + jolt-host
bridge. Can be landed behind a feature flag or in its own branch to keep
`main` (of refactor-crates) stable.

## Test Gate (run after each plan)

```bash
cargo nextest run -p jolt-equivalence --test field_register_twist_standalone --cargo-quiet
cargo nextest run -p jolt-equivalence --test muldiv modular_self_verify --cargo-quiet
cargo nextest run -p jolt-equivalence --test muldiv modular_self_verify_with_fieldreg --cargo-quiet
cargo clippy -p jolt-compiler -p jolt-witness -p jolt-zkvm -p jolt-verifier -p jolt-equivalence --all-targets -- -D warnings
```

## Memory References

- `ARCHITECTURE.md`: ML-compiler philosophy (compiler = protocol, runtime = flat dispatcher).
- `TASKS.md`: current violation list for runtime→compiler lowering.
- `memory_copy/feedback_no_test_bypass.md`: anti-pattern warning for test-harness witness injection.
- `memory_copy/feedback_endianness_bugs.md`: vigilance for BE vs LE conventions.
