# v1 → v2 Cherry-Pick Inventory

| Field | Value |
|-------|-------|
| Created | 2026-04-24 |
| Status | Migration reference for `feat/field-coprocessor-simplified` v2 rebuild |
| Source branch | `feat/field-register-twist` (v1) |
| Target branch | New branch forked from `c7e0869d5` (last non-FR commit on refactor-crates) |

## Summary

29 commits on v1 add ~18k LOC across 72 files. Cherry-pick verdict:

- **Keep as-is:** 5 files (protocol-stable infrastructure)
- **Keep with edits:** 19 files (structurally right, strip v1-only bits)
- **Discard:** 8 files (exist only to serve v1 design)
- **Create new:** 5 files (v2 bridge instructions)

The largest artifact, `jolt_core_module_with_fieldreg.rs` (8988 LOC), is effectively a **near-total rewrite** — only the Registers-Twist-shaped scaffolding transfers. Every v1-bespoke block goes: bridge kernel, LimbSum Stage-5 reductions, FieldRegReadValue/WriteValue commitments, RdIncAtBridge aliases, BridgeValWeight* preprocessed polys.

## Keep-as-is (5 files)

| File | Rationale |
|------|-----------|
| `specs/bn254-fr-coprocessor.md` | v2 spec — ground truth |
| `jolt-inlines/bn254-fr/Cargo.toml` | Crate manifest, v2-agnostic |
| `jolt-inlines/bn254-fr/src/lib.rs` | FUNCT3_* constants for arithmetic ops unchanged in v2 (0x02..0x05). Strip I2F/F2I constants at edit time — 35 LOC file |
| `examples/bn254-fr-arkworks/guest/**` | Pure ark-bn254 reference guest; no coprocessor dep |
| `examples/bn254-fr-poseidon2-arkworks/guest/**` + `bn254-fr-poseidon2-external/guest/**` | ark-based Poseidon2 reference baselines — independent of coprocessor |

## Keep with edits (19 files)

### Tracer

**`tracer/src/emulator/cpu.rs`**
- Keep: `FIELD_REG_COUNT = 16`, `cpu.field_regs`, `cpu.field_reg_events`, clone impl additions, `FieldRegEvent { cycle_index, slot, old, new }`
- Strip: `op: Option<FieldOpPayload>` and `fmov: Option<FMovPayload>` event fields; delete `FieldOpPayload` and `FMovPayload` structs entirely

**`tracer/src/instruction/field_op.rs`**
- Keep: `FIELD_OP_OPCODE`, `BN254_FR_FUNCT7`, `FUNCT3_FMUL/FADD/FSUB/FINV`, the `FieldOp` struct, `execute`, `fr_from_limbs`/`fr_to_limbs`, `trace` outer shell
- Strip: `FieldOpPayload` construction in `trace`; `op: Some(payload)` event field. v2 event is `FieldRegEvent { cycle_index, slot: frd, old, new }` only
- Add: `FieldAssertEq` constant + decoder branch (funct3 0x06)

**`tracer/src/instruction/mod.rs`**
- Keep: `FieldOp` registration
- Strip: `FMovIntToFieldLimb`, `FMovFieldToIntLimb` enum variants + dispatch

**`tracer/src/lib.rs` + `tracer/src/emulator/mod.rs`**
- Strip re-exports of `FMovPayload`, `FieldOpPayload`, `FMov*Limb`
- Retain re-exports of `FieldRegEvent`, `FieldOp`, `FIELD_REG_COUNT`

**`tracer/Cargo.toml`** — keep ark-bn254/ark-ff deps (v2 still needs them)

### Witness / derived source

**`crates/jolt-witness/src/derived.rs`**
- Keep: `FrLimbs`, `limbs_to_field`, `FieldRegConfig { k, initial_state, events }`, `FieldRegEvent { cycle, slot, old, new }`, `with_field_reg` builder, `field_reg_inc()`, `field_reg_ra()`, `field_reg_val()`, and `PolynomialId::FieldRegInc/Ra/Val` match arms
- Strip: `FieldOpPayload`, `FMovPayload` types + event fields; `FIELD_OP_FUNCT3_FMOV_I2F/F2I`; `is_field_op_any()`, `field_op_b_gated()`, `is_field_op_no_inv()`, `limb_sum_a()`, `limb_sum_b()`, `limb_sum_range()`, `weight_of_rd_range()`, and all corresponding `PolynomialId::*` match arms
- Add: virtual readers for `FieldRs1Value`, `FieldRs2Value`, `FieldRdValue` (computed as `state[frs1/frs2/frd(c)]` replayed over cycles — only for standalone/test paths; in the main protocol these are virtual Twist openings)

**`crates/jolt-witness/src/preprocessed.rs`** — strip `BridgeValWeightA/B`, `BridgeAnchorA/B` population

**`crates/jolt-witness/src/provider.rs`** — strip `RdIncAtBridge*` alias forwarding

### Compiler

**`crates/jolt-compiler/src/polynomial_id.rs`**
- Keep: `FieldRegInc`, `FieldRegRa`, `FieldRegVal`, `FieldRegEqCycle` variants + descriptors (`Derived`, `committed: true` for Inc/Ra; virtual for Val/EqCycle)
- Add: `FieldRs1Value`, `FieldRs2Value`, `FieldRdValue` as **virtual** (no R1CS witness slot, no commitment — proven by FR Twist sumcheck)
- Strip: `FieldRegReadValue`, `FieldRegWriteValue`, `FieldOpOperandA/B`, `FieldOpResultValue`, `FieldRegReadLimb`, `FieldRegWriteLimb`, `LimbSumA/B`, `IsFieldOpAny`, `FieldOpBGated`, `IsFieldOpNoInv`, `BridgeValWeightA/B`, `BridgeAnchorA/B`, `BridgeValWeight`, `BridgeAnchorWeight`, `WeightAOfRd`, `WeightBOfRd`, `RdIncAtBridge`, `RdIncAtBridgeA`, `RdIncAtBridgeB`. Their `r1cs_variable_index` entries (slots 42–48) go

**`crates/jolt-compiler/src/params.rs`**
- Keep: `new_with_constraints` API, `num_r1cs_constraints` field, `fr_checking_{degree,rounds}`
- Strip: `bridge_checking_{degree,rounds}`; v2 `NUM_EQ_CONSTRAINTS` drops to ~24

**`crates/jolt-compiler/src/module.rs`** — keep `SegmentedConfig::outer_weight_poly`, `Op::BuildLinearCombination`, `VerifierOp::CollectOpeningClaimAt` (general-purpose)

**`crates/jolt-compiler/src/builder.rs`** — keep `add_padded_poly` unchanged

**`crates/jolt-compiler/src/compiler/emit.rs`** — keep `BuildLinearCombination` codegen additions

### Instructions

**`crates/jolt-instructions/src/flags.rs`**
- Keep: `IsFieldMul`, `IsFieldAdd`, `IsFieldSub`, `IsFieldInv`
- Strip: `IsFMovI2F`, `IsFMovF2I`
- Add: `IsFieldMov`, `IsFieldSLL64`, `IsFieldSLL128`, `IsFieldSLL192`, `IsFieldAssertEq`
- Update `NUM_CIRCUIT_FLAGS` from 20 → 23

### R1CS

**`crates/jolt-r1cs/src/constraints/rv64.rs`**
- Keep: `V_FLAG_IS_FIELD_MUL/ADD/SUB/INV` slots + Row 19 (FieldAdd), Row 20 (FieldSub), Rows 21–26 (FMUL/FINV V_PRODUCT routing)
- Strip: `V_FLAG_IS_FMOV_I2F/F2I`, `V_FIELD_REG_READ_LIMB/WRITE_LIMB`, `V_LIMB_SUM_A/B` slots; Rows 27 (FMov-I2F), 28 (FMov-F2I), 29 (LimbSumA), 30 (LimbSumB)
- Replace: `V_FIELD_OP_A/B/RESULT` → `V_FIELD_RS1_VALUE / V_FIELD_RS2_VALUE / V_FIELD_RD_VALUE`. **Virtual — no R1CS witness slot**. Rows reference them but their evaluations are bound at stage 3/4/5 by FR Twist chain
- Add: v2 rows for FieldMov, FieldSLL64, FieldSLL128, FieldSLL192, FieldAssertEq
- Adjust: `NUM_R1CS_INPUTS`, `NUM_EQ_CONSTRAINTS`, `NUM_VARS_PER_CYCLE`, `V_BRANCH`, `V_NEXT_IS_NOOP`

### jolt-host

**`crates/jolt-host/src/r1cs_witness.rs`**
- Keep: `w[V_FLAG_IS_FIELD_MUL/ADD/SUB/INV]` population from circuit flags
- Strip: `w[V_FLAG_IS_FMOV_I2F/F2I]`; `populate_limb_sum_columns`; `apply_field_op_events_to_r1cs` (the whole function, ~350 LOC); the `fieldreg_events` parameter on builders
- Add: `V_FLAG_IS_FIELD_MOV/SLL64/SLL128/SLL192/ASSERT_EQ` population

**`crates/jolt-host/src/tracer_cycle.rs`**
- Keep: `field_op_circuit_flags` / `field_op_is_fr` helpers (~lines 498–528)
- Strip: `FMov*Limb` branches
- Add: `FieldMov/SLL*/AssertEq` branches

**`crates/jolt-host/src/program.rs`** — keep `trace_with_field_reg_events` signature

**`crates/jolt-host/src/lib.rs`** — strip `apply_field_op_events_to_r1cs` re-export

### jolt-zkvm runtime

**`crates/jolt-zkvm/src/runtime/{handlers.rs, helpers.rs}`** — keep generic ops (`BuildLinearCombination` etc); strip anything referencing `V_LIMB_SUM_A/B` or `V_FIELD_OP_A/B` slot indices directly

### jolt-bench

**`crates/jolt-bench/src/programs.rs`** — keep `Poseidon2Sdk`/`Poseidon2Ark` variants and `inputs()`; target binaries replaced (new v2 SDK)

**`crates/jolt-bench/src/stacks/modular.rs`** — strip `apply_field_op_events_to_r1cs` call and `FieldRegConfig` event-passing for v1 bridge; keep `trace_with_field_reg_events` import and FR config attachment

## Discard (8 files)

| File | Reason |
|------|--------|
| `tracer/src/instruction/fmov_int_to_field_limb.rs` | v2 uses FieldMov + SLL chain |
| `tracer/src/instruction/fmov_field_to_int_limb.rs` | v2 uses Advice + reconstruction + FieldAssertEq |
| `crates/jolt-compiler/examples/jolt_core_module_with_fieldreg.rs` | 8988 LOC; architectural mismatch with v2 3-sumcheck chain — create from scratch |
| `examples/bn254-fr-smoke/{Cargo.toml, guest/**, src/main.rs}` | Guest uses v1 SDK (FMov-based) |
| `examples/bn254-fr-horner-sdk/**` | v1 SDK dep |
| `examples/bn254-fr-horner-arkworks/**` | Paired baseline for v1 horner-sdk; drop until v2 horner example exists |
| `examples/bn254-fr-poseidon2-sdk/**` | v1 SDK dep |
| `jolt-inlines/bn254-fr/src/sdk.rs` | Targets v1 FMov-I2F/F2I ABI (hand-encoded `i2f_word`/`f2i_word`). Whole file obsolete |

## Create new (5 files)

| File | Purpose |
|------|---------|
| `tracer/src/instruction/field_mov.rs` | `FieldMov rs1 → frd` (funct3 0x07): `field_regs[frd] = x[rs1]` as Fr |
| `tracer/src/instruction/field_sll64.rs` | `FieldSLL64 rs1 → frd` (funct3 0x08): `field_regs[frd] = x[rs1] · 2⁶⁴` |
| `tracer/src/instruction/field_sll128.rs` | funct3 0x09: `· 2¹²⁸` |
| `tracer/src/instruction/field_sll192.rs` | funct3 0x0A: `· 2¹⁹²` |
| `tracer/src/instruction/field_assert_eq.rs` | `FieldAssertEq frs1, frs2`: no write; emits R1CS-only event |

## Dependency gotchas

Files that look standalone but have hidden v1 coupling:

1. **`crates/jolt-equivalence/tests/field_register_twist_standalone.rs`** — first half (honest_cycles, run_prover, honest_accepts, adversarial_inc_mutation_rejects, tampered_commitment_rejects, override_provider_*) is the right template for v2; structurally mirrors Registers Twist test shape. But second half (`build_twist_module` line 485+ and six `twist_*` tests 1105+) bakes in `FieldRegReadValue`/`FieldRegWriteValue` as committed polys — those tests ONLY make sense in v1. **Don't port wholesale — rewrite to mutate virtual `FieldRs1Value`/`FieldRs2Value`/`FieldRdValue` openings.**

2. **`crates/jolt-equivalence/tests/muldiv.rs`** — honest `modular_self_verify_with_fieldreg` + `_nonempty_events` + `_fadd_payload_with_prologue_accepts` depend on `FieldOpPayload` being in the event. Port structure (FieldRegConfig attachment to DerivedSource) but rewrite every `FieldOpPayload { a, b, result }` construction. `bridge_tampered_field_op_a_rejects`, `fmov_f2i_tampered_read_limb_rejects`, `fmov_i2f_tampered_write_limb_rejects`, `audit_poc_*` ALL test v1 bridge rows 27–30 directly — **discard wholesale**, replace per spec with `modular_self_verify_with_fr_operand_tamper_rejects`.

3. **`crates/jolt-equivalence/tests/bn254_fr_smoke.rs`** — `trace_contains_expected_field_op_cycles`, `field_reg_events_payload_matches_fr_arithmetic`, `cycle_count_vs_arkworks`, `horner_64_cycle_count_vs_arkworks`, `poseidon2_cycle_count_vs_arkworks`. Cycle-count tests are salvageable templates once v2 SDK lands. The payload test depends on `FieldRegEvent.op`, which goes away — rewrite to check just `(slot, old, new)`.

4. **`crates/jolt-compiler/examples/jolt_core_module.rs`** — modified in v1 branch (5 lines); the base file is the Registers Twist template the v2 spec names as the structural template. Use **this** as ground truth, not `jolt_core_module_with_fieldreg.rs`. FR Twist instance added as a near-copy of the Registers Twist block in `build_stage3/4/5`, γ-batching `FieldRs1Value/Rs2Value/RdValue` through `FieldRegVal · FieldRegRa`.

5. **`crates/jolt-host/src/r1cs_witness.rs`** lines 280–562 (`apply_field_op_events_to_r1cs`) — signature and overlay pattern look reusable but body writes to slots that are all virtual in v2. **Don't port.** V2 populates no new R1CS witness columns beyond the five `V_FLAG_IS_FIELD_*` flags; that happens inline in `r1cs_cycle_witness`, not in a post-pass overlay.

6. **`crates/jolt-witness/src/derived.rs`** lines 657–726 (`limb_sum_range`, `weight_of_rd_range`) — look like reusable primitives but only called by LimbSum stage-5 reductions and bridge rows 29/30. **Delete.**

7. **`jolt-inlines/bn254-fr/src/sdk.rs`** — `binary_op`/`unary_op` inline-asm structure is a template. Register allocation (x10..x21) and hand-encoded instruction words target v1 FMov opcodes — **all encoding helpers must be rewritten** against v2 `FieldMov/SLL*` encodings per spec §ISA. The 7-cycle load / 12-cycle extract sequences replace the current 4+1+4 scheme.

## How to apply this inventory

From the v2 branch fork point (`c7e0869d5`):

```bash
git checkout -b feat/fr-coprocessor-v2 c7e0869d5

# For each "keep-as-is" or "keep-with-edits" file:
#   - For wholesale file moves: `git checkout feat/field-coprocessor-simplified -- <path>`
#     then apply the strip/edit instructions
#   - For small, isolated additions: re-type from scratch (faster than cherry-picking)

# "Discard" and "create new" paths don't consume any v1 code — just delete v1's
# version of the file if it existed, and write v2's fresh version.
```

No cherry-picking at commit level — the 29 v1 commits are too entangled with discarded concerns (FieldOpPayload infects many commits that also add keep-worthy infrastructure). File-level surgery is the right granularity.
