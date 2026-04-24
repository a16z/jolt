# BN254 Fr Native-Field Coprocessor (v2)

| Field      | Value                                                                 |
|------------|-----------------------------------------------------------------------|
| Author(s)  | sdhawan                                                               |
| Created    | 2026-04-24 (v2 — pivot)                                               |
| Status     | Design locked (see history at bottom); implementation pending on branch `feat/field-coprocessor-simplified` |
| Scope      | `refactor/crates` worktree at `jolt-refactor-crates/` only            |
| Base branch | `feat/field-register-twist` (v1 design preserved as reference)        |

## Summary

A BN254 Fr coprocessor adds native `Fr::{add,sub,mul,inv}` RISC-V instructions plus a 16 × 256-bit field-register file to the Jolt zkVM. **The coprocessor is a structural mirror of the existing integer Registers Twist** — same commitment shape, same sumcheck structure, same virtual-column pattern for operand values. The integer↔field bridge is a small family of arithmetic instructions (FieldMov + FieldSLL64/128/192 + FieldAdd) rather than dedicated transfer opcodes; limb decomposition happens only at the boundary of an Fr computation block, amortized over many in-field operations.

Design principle: **FR state is just another memory-checking instance**, structurally identical to integer registers. Every soundness-relevant value is bound via the existing Twist machinery; no bespoke bridging sumchecks or R1CS cross-check rows.

## Intent

### Goal

Deliver a sound, minimal BN254 Fr coprocessor: single-cycle FieldOp over a 16 × 256-bit field-register file, simple integer↔field bridging via Horner-style assembly, no operand-binding gap.

### Invariants

1. **FR Twist structure mirrors the Registers Twist.** Only `FieldRegInc` is PCS-committed; `FieldRegVal`, `FieldRs1Value`, `FieldRs2Value`, `FieldRdValue` are virtual, proven via the FR Twist sumcheck against the committed `FieldRegInc` (and the committed one-hot `FieldRegRa`).
2. **R1CS-input columns for field operands are virtual.** `FieldRs1Value(c)` = FR-Twist opening of `Val_fr` at slot `rs1(c)` on cycle `c`. No duplicated committed polys; no cross-check rows.
3. **Single access per cycle.** Each FieldOp emits one `FieldRegEvent`. FR state evolves one write per cycle (mirrors integer register Inc semantics).
4. **Natural-form `[u64;4]` representation for Fr values.** `F::from_natural_limbs(limbs)` at all FR-state-adjacent boundaries. No Montgomery reinterpretation.
5. **Integer↔field bridge is arithmetic.** No dedicated FMov opcodes. Bridge cycles emit ordinary FR writes and read from integer registers through the existing `Rs1Value` column. One R1CS row per bridge instruction.
6. **Acceptance is load-bearing.** Honest-accepts + mutated-rejects at every milestone.

### Non-goals

- 32-byte atomic memory loads (RAM Twist stays u64-grain).
- Multi-field coprocessor — BN254 Fr only.
- G1/G2/Gt / pairings.
- Compiler-managed FR-register allocation (see § *Future work* for amortization optimization).
- Recursive Jolt verifier.

## ISA

All field-coprocessor instructions use opcode `0x0B` (custom-0), funct7 `0x40`, funct3 as selector.

### Field arithmetic (reads and writes field registers only)

| funct3 | Mnemonic       | Semantics                                                  | R1CS constraint (sketch)                     |
|--------|----------------|------------------------------------------------------------|----------------------------------------------|
| 0x02   | `FieldMul`     | `field_regs[frd] = field_regs[frs1] × field_regs[frs2]`    | `FieldRs1Value · FieldRs2Value = FieldRdValue` (via V_PRODUCT reuse) |
| 0x03   | `FieldAdd`     | `field_regs[frd] = field_regs[frs1] + field_regs[frs2]`    | `FieldRs1Value + FieldRs2Value = FieldRdValue`                       |
| 0x04   | `FieldInv`     | `field_regs[frd] = field_regs[frs1]⁻¹`                     | `FieldRs1Value · FieldRdValue = 1` (FINV(0) is unsatisfiable)        |
| 0x05   | `FieldSub`     | `field_regs[frd] = field_regs[frs1] − field_regs[frs2]`    | `FieldRs1Value − FieldRs2Value = FieldRdValue`                       |
| 0x06   | `FieldAssertEq`| `assert field_regs[frs1] == field_regs[frs2]`              | `FieldRs1Value − FieldRs2Value = 0`                                  |

All gated by per-instruction circuit flags (`IsFieldMul`, `IsFieldAdd`, etc.).

### Integer→field bridge (reads integer register, writes field register)

| funct3 | Mnemonic       | Semantics                                                  | R1CS constraint                              |
|--------|----------------|------------------------------------------------------------|----------------------------------------------|
| 0x07   | `FieldMov`     | `field_regs[frd] = x[rs1]` as Fr                           | `Rs1Value = FieldRdValue`                    |
| 0x08   | `FieldSLL64`   | `field_regs[frd] = x[rs1] · 2⁶⁴` in Fr                     | `Rs1Value · 2⁶⁴ = FieldRdValue`              |
| 0x09   | `FieldSLL128`  | `field_regs[frd] = x[rs1] · 2¹²⁸` in Fr                    | `Rs1Value · 2¹²⁸ = FieldRdValue`             |
| 0x0A   | `FieldSLL192`  | `field_regs[frd] = x[rs1] · 2¹⁹²` in Fr                    | `Rs1Value · 2¹⁹² = FieldRdValue`             |

No symmetric dedicated field→int bridge is needed. See § *Field→integer extraction*.

### Semantic notes

- `frs1`, `frs2`, `frd` are the same 5-bit bytecode fields as standard R-type `rs1`/`rs2`/`rd`, low 4 bits used as slot index 0..15. Bytecode binding is free — `BytecodeReadRaf` already commits these fields.
- All FR constraints are in the proof field (BN254 Fr). Integer values (`Rs1Value`) naturally embed as Fr < 2⁶⁴, no reduction needed.
- `FieldRdValue` at each cycle is the FR-Twist-bound value at `field_regs[frd(c)]`. The instruction's constraint forces an update; the Twist's read-check sumcheck binds the value cryptographically.

## Architecture

### FR Twist — mirrors Registers Twist exactly

| Polynomial           | Role                              | Status    | Bound by                         |
|----------------------|-----------------------------------|-----------|----------------------------------|
| `FieldRegInc`        | Increment polynomial (K×T)        | Committed | Dory PCS                         |
| `FieldRegRa`         | Access-address one-hot (K×T)      | Committed (one-hot chunks) | Dory PCS           |
| `FieldRegVal`        | Running state (K×T)               | Virtual   | Twist sumcheck reduces to `Inc`  |
| `FieldRs1Value`      | Fr value read from slot `frs1(c)` | Virtual   | Twist sumcheck γ-batches with Val·Ra₁ |
| `FieldRs2Value`      | Fr value read from slot `frs2(c)` | Virtual   | Twist sumcheck γ-batches with Val·Ra₂ |
| `FieldRdValue`       | Fr value written to slot `frd(c)` | Virtual   | Twist sumcheck γ-batches with Val·Ra_d |

Matches the integer Registers Twist's committed/virtual split: only increments and one-hot addresses are committed; all value columns are virtual, proven by the sumcheck at Stage 2.

**Crucially: no `FieldRegReadValue` / `FieldRegWriteValue` as committed polynomials.** The v1 design over-committed these; their values are covered by `FieldRs1Value`/`FieldRs2Value`/`FieldRdValue` virtual openings, mirroring the Registers Twist.

### Soundness by construction

For any FieldOp or bridge-instruction constraint, both sides of the equation are either:
- Integer register values (`Rs1Value` — bound by Registers Twist), or
- Field register values (`FieldRs1Value` / `FieldRs2Value` / `FieldRdValue` — bound by FR Twist)

No R1CS column is prover-declared at the Stage-1 payload level. Tampering any operand either breaks the Stage-1 Az·Bz=Cz check or fails the corresponding Twist's Dory opening. There is no separate "operand binding" step — binding is inherent in the column definition.

### SDK ABI

The `jolt-inlines-bn254-fr` SDK loads `[u64;4]`-represented Fr values into field registers via Horner assembly, runs in-field arithmetic, and extracts back via advice reconstruction.

**Loading `a: [u64;4]` from `(x10, x11, x12, x13)` into `field_regs[frd]`:**
```
FieldMov     x10 → ft0           // ft0 = a[0]
FieldSLL64   x11 → ft1           // ft1 = a[1] · 2⁶⁴
FieldSLL128  x12 → ft2           // ft2 = a[2] · 2¹²⁸
FieldSLL192  x13 → ft3           // ft3 = a[3] · 2¹⁹²
FieldAdd     ft0, ft1 → ft4
FieldAdd     ft2, ft3 → ft5
FieldAdd     ft4, ft5 → field_regs[frd]
```
7 cycles per Fr load.

**Extracting `field_regs[frs1]` back into `(x10, x11, x12, x13)`:**
```
Advice       x10, x11, x12, x13              // prover supplies claimed limbs
FieldMov     x10 → ft0
FieldSLL64   x11 → ft1
FieldSLL128  x12 → ft2
FieldSLL192  x13 → ft3
FieldAdd     ft0, ft1 → ft4
FieldAdd     ft2, ft3 → ft5
FieldAdd     ft4, ft5 → ft6
FieldAssertEq  ft6, field_regs[frs1]
```
12 cycles per Fr extract.

**Canonical limb representation.** Fr is 254 bits; 4×u64 has 2 bits of slack. The SDK canonicalizes extracted limbs to the range `0..Fr−1` via a small range check on `x13`. Multiple u64[4] representations of the same Fr are NOT accepted — the extract instruction asserts `x13 < 2^(Fr_bits − 192)`.

### Prover pipeline

| Stage     | What runs                                                                                |
|-----------|------------------------------------------------------------------------------------------|
| Commit    | `FieldRegInc`, `FieldRegRa` one-hot chunks (mirrors integer Registers Twist commits)     |
| Stage 1 Spartan outer | R1CS `Az·Bz=Cz` — FieldOp rows + bridge-instruction rows                   |
| Stage 2 batched | Existing instances (RAM, Instruction, Bytecode RAF, Output) — FR Twist is NOT here |
| **Stage 3 FieldRegClaimReduction** | γ-batches `FieldRdValue + γ·FieldRs1Value + γ²·FieldRs2Value` openings from Stage 1 into a single sumcheck target. log_T rounds, degree 2. Mirrors integer `RegistersClaimReduction` at `jolt_core_module.rs:2746-2776, 2981-3061`. |
| **Stage 4 FieldRegReadWriteChecking** | The FR Twist proper — 2-phase segmented sumcheck. Phase 1 binds cycles (log_T rounds); Phase 2 binds slot address (log_K_FR = 4 rounds). Mirrors integer `RegistersReadWriteChecking` at `jolt_core_module.rs:3137-3604`. |
| **Stage 5 FieldRegValEvaluation** | Single-term degree-3 reduction of Val through committed Inc. log_T rounds. Mirrors integer `RegistersValEvaluation` at `jolt_core_module.rs:3693-3725, 4034-4040`. |
| Stage 6-7 | Claim reductions, Hamming booleanity (FR side follows integer pattern)                   |
| Stage 8   | Dory batched opening proofs                                                              |

**Important correction from v1:** the integer Registers Twist is a *chain* of three sumchecks across Stages 3/4/5 — not one Stage-2 instance. v1 incorrectly placed FR Twist as a single Stage-2 batched instance; v2 mirrors the integer chain exactly. The γ-batching pattern `rd_wv + γ·rs1_val + γ²·rs2_val` at `jolt_core_module.rs:3821-3857` is copied (three terms, not two) for the FR side.

## Amortization strategy (future work)

The per-Fr-crossing overhead (7 cycles load, 12 cycles extract) is expensive in isolation but amortized across a block of chained in-field operations. A Poseidon2 permutation (~400 Fr ops) that enters and exits the field register file once incurs ~19 boundary cycles for ~400 ops — **sub-5% amortized overhead**.

**Compiler optimization target:** the guest-side compiler lowers Rust-level Fr expression sequences into a single on-ramp / in-field-arithmetic / off-ramp block, keeping intermediates pinned in field registers across multiple source-level operations. Analogous to how the Rust compiler keeps intermediate values in integer registers rather than spilling to stack between arithmetic ops. This is aspirational for v2 — the SDK currently emits one on-ramp+FieldOp+off-ramp per source-level `Fr::mul`, leaving the optimization opportunity for later.

## Companion documents

| Path | Purpose |
|------|---------|
| `specs/fr-twist-mirror-plan.md` | Line-by-line mirror plan: how the FR Twist's 3-sumcheck chain (Stages 3/4/5) maps from integer Registers Twist. Use for Phase 4 implementation. |
| `specs/v2-implementation-plan.md` | Exact R1CS witness slot layout, row definitions (A/B/C formulas), PolynomialId additions/removals, tracer instruction specs, migration ordering. Use for Phases 2-3. |
| `specs/v1-cherry-pick-inventory.md` | File-by-file keep/edit/discard/create table. Use for Phase 1 teardown. |

## Implementation status

**Locked as of 2026-04-24.** Implementation pending on a fresh branch forked from `c7e0869d5` (last non-FR commit on refactor-crates). `feat/field-coprocessor-simplified` retained as v1 reference.

| Area | State |
|------|-------|
| Design | Locked (this spec) |
| Base branch | `feat/field-coprocessor-simplified` forked from `feat/field-register-twist` at commit `c01910e06` |
| v1 code (to be ripped out) | Present on base branch as reference — see § *Migration plan* |
| New code | Pending |

## Migration plan

Work on `feat/field-coprocessor-simplified`. Approach: **delete v1 infrastructure first, rebuild on the cleared ground**. Avoids maintaining dual paths during the restructure.

### Phase 1 — delete v1 complexity

1. Remove FMov-I2F/F2I instructions:
   - `tracer/src/instruction/fmov_{int_to_field,field_to_int}_limb.rs` (files)
   - `IsFMovI2F` / `IsFMovF2I` circuit flags
   - `V_FLAG_IS_FMOV_I2F`, `V_FLAG_IS_FMOV_F2I` R1CS slots
   - `V_FIELD_REG_READ_LIMB`, `V_FIELD_REG_WRITE_LIMB` R1CS slots
   - R1CS rows 27-28 (FMov gates)
   - `FMovPayload` / FieldRegEvent `fmov` field
2. Remove limb-sum bridge:
   - `V_LIMB_SUM_A`, `V_LIMB_SUM_B` R1CS slots
   - R1CS rows 29-30 (limb-sum bridge)
   - `populate_limb_sum_columns` in `jolt-host/src/r1cs_witness.rs`
   - Stage 5 `LimbSumAReduction` / `LimbSumBReduction` kernels in the fieldreg module
   - `PolynomialId::LimbSumA` / `LimbSumB` variants
3. Remove payload-declared FR operands:
   - `V_FIELD_OP_A`, `V_FIELD_OP_B`, `V_FIELD_OP_RESULT` R1CS slots (will return later with new semantics as virtual `FieldRs1/Rs2/RdValue`)
   - `FieldOpPayload` struct (`a`, `b`, `result` no longer needed)
   - `apply_field_op_events_to_r1cs` in `jolt-host/src/r1cs_witness.rs`
4. Collapse over-committed FR polys:
   - `PolynomialId::FieldRegReadValue` → virtual (or removed outright)
   - `PolynomialId::FieldRegWriteValue` → virtual (or removed outright)
   - Their commitments + opening proofs in the FR Twist schedule go away
   - Corresponding `commit_pairs` entries in `jolt_core_module_with_fieldreg.rs`

### Phase 2 — add v2 instructions

1. Tracer additions (`tracer/src/instruction/`):
   - `field_mov.rs` — `FieldMov rs1 → frd` (sets `field_regs[frd] = x[rs1]` as Fr)
   - `field_sll64.rs`, `field_sll128.rs`, `field_sll192.rs` — scaled versions
   - `field_assert_eq.rs` — `FieldAssertEq frs1, frs2` (no write, emits an R1CS-only event)
2. Circuit flags (`crates/jolt-instructions/src/flags.rs`):
   - `IsFieldMov`, `IsFieldSLL64`, `IsFieldSLL128`, `IsFieldSLL192`, `IsFieldAssertEq`

### Phase 3 — R1CS restructure

1. Add R1CS virtual-column slot constants and `PolynomialId` variants:
   - `FieldRs1Value`, `FieldRs2Value`, `FieldRdValue` as virtual (no R1CS witness slot — proven through FR Twist)
2. Add R1CS equality rows:
   - `FieldAdd`: `IsFieldAdd · (FieldRs1Value + FieldRs2Value − FieldRdValue) = 0`
   - `FieldSub`: analogous
   - `FieldAssertEq`: `IsFieldAssertEq · (FieldRs1Value − FieldRs2Value) = 0`
   - `FieldMov`: `IsFieldMov · (Rs1Value − FieldRdValue) = 0`
   - `FieldSLL64`: `IsFieldSLL64 · (Rs1Value · 2⁶⁴ − FieldRdValue) = 0`
   - ... analogous for SLL128, SLL192
   - `FieldMul`: routes through `V_PRODUCT` — `IsFieldMul · (V_PRODUCT − FieldRdValue) = 0` with `V_LEFT_INSTRUCTION_INPUT = FieldRs1Value`, `V_RIGHT_INSTRUCTION_INPUT = FieldRs2Value`
   - `FieldInv`: `IsFieldInv · (V_PRODUCT − 1) = 0` with `V_LEFT_INSTRUCTION_INPUT = FieldRs1Value`, `V_RIGHT_INSTRUCTION_INPUT = FieldRdValue`

### Phase 4 — FR Twist structural restructure (3-sumcheck chain mirroring integer registers)

1. **Stage 3 `FieldRegClaimReduction`.** New sumcheck mirroring integer `RegistersClaimReduction`. γ-batches `FieldRdValue(r_cycle_stage1) + γ·FieldRs1Value(r_cycle_stage1) + γ²·FieldRs2Value(r_cycle_stage1)` openings from Stage 1 into a single target. log_T rounds, degree 2.
2. **Stage 4 `FieldRegReadWriteChecking`.** New sumcheck mirroring integer `RegistersReadWriteChecking`. 2-phase segmented sumcheck: Phase 1 binds cycles (log_T rounds); Phase 2 binds FR slot address (log_K_FR = 4 rounds). Output claim opens `FieldRegVal(r_slot, r_cycle_stage4)`.
3. **Stage 5 `FieldRegValEvaluation`.** New sumcheck mirroring integer `RegistersValEvaluation`. Single-term degree-3, log_T rounds. Reduces `FieldRegVal(r_slot, r_cycle_stage4)` opening to committed `FieldRegInc` via the standard increment-over-LT identity.
4. Commit phase: `FieldRegInc` + `FieldRegRa` one-hot chunks (same as integer RdInc + RegRa pattern). NO commitment of `FieldRegReadValue`/`FieldRegWriteValue` — those v1 polynomial IDs are deleted in Phase 1.
5. Verifier mirrors: three new `SumcheckInstance` entries in the verifier schedule (one per new stage), using the same ClaimFactor templates as integer Registers Twist. `num_stages` bumps to accommodate.

See `specs/fr-twist-mirror-plan.md` for the line-by-line mirror mapping against `jolt_core_module.rs` integer Registers Twist.

### Phase 5 — SDK rewrite

1. `jolt-inlines-bn254-fr/src/sdk.rs`: rewrite `Fr::{add,sub,mul,inv}` inline asm blocks to emit FieldMov/SLL/Add on-ramp + FieldOp + FieldAssertEq off-ramp sequences.
2. Update `examples/bn254-fr-smoke/` and `examples/bn254-fr-poseidon2-*/` guests.

### Phase 6 — tests + benchmarks

1. Port existing honest tests (muldiv e2e, `modular_self_verify_with_fieldreg`, Poseidon2 cycle count, non-empty-events) to v2 ISA.
2. Add new adversarial test: tamper FieldRs1Value at any Stage-1 opening — must reject via FR Twist Dory opening.
3. Run Poseidon2 cycle count vs v1 to quantify the on-ramp/off-ramp overhead before compiler amortization. Target: within 30% of v1 unamortized; within 5% with a simple compiler optimization.

## Acceptance criteria

The v2 coprocessor is complete when all of the following hold:

1. All honest tests green: muldiv e2e (both `host` and `host,zk`), `modular_self_verify_with_fieldreg`, standalone FR Twist tests, Poseidon2 cycle count.
2. New adversarial test `modular_self_verify_with_fr_operand_tamper_rejects` — tampering any `FieldRs1Value`/`FieldRs2Value`/`FieldRdValue` Stage-1 opening fails verification via FR Twist Dory opening mismatch.
3. `jolt-bench muldiv --stack modular` prove_ms within +10% of current (736 ms baseline).
4. Poseidon2 guest cycle count within 30% of v1 unamortized; within 5% with basic SDK on-ramp/off-ramp amortization.
5. Clippy clean in both `host` and `host,zk` modes.
6. Cross-verify with jolt-core: broken intentionally (jolt-core has no FR coprocessor).
7. `bn254-fr-coprocessor.md` (this file) status updated from "design locked" to "shipped"; all v1 commits on `feat/field-register-twist` are retained as reference but the v2 branch replaces them on the merged path.

## Commands

```bash
cd /Users/sdhawan/Work/jolt-refactor-crates

# Full suite:
cargo nextest run --cargo-quiet

# Primary FR correctness gate:
cargo nextest run -p jolt-equivalence --test muldiv modular_self_verify_with_fieldreg --cargo-quiet

# FR Twist standalone (mirrors Registers Twist shape):
cargo nextest run -p jolt-equivalence --test field_register_twist_standalone --cargo-quiet

# Poseidon2 cycle count validation:
cargo nextest run -p jolt-equivalence --test bn254_fr_smoke poseidon2_cycle_count_vs_arkworks --cargo-quiet

# Perf bench:
./target/release/jolt-bench --program muldiv --stack modular --iters 5 --warmup 2

# Clippy in both modes:
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
```

## Naming reference

Slot/constraint constants live in `crates/jolt-r1cs/src/constraints/rv64.rs`. Polynomial identifiers in `crates/jolt-compiler/src/polynomial_id.rs`. Target layout after Phase 3:

```
Virtual R1CS inputs (proven via FR Twist sumcheck, not R1CS witness columns):
  FieldRs1Value    FieldRs2Value    FieldRdValue

Committed via Dory (FR Twist):
  FieldRegInc      FieldRegRa (one-hot chunked)

Virtual (FR Twist sumcheck outputs):
  FieldRegVal      FieldRegEqCycle

Circuit flags (integer-register-Twist style, per-instruction gates):
  IsFieldMul  IsFieldAdd  IsFieldSub  IsFieldInv  IsFieldAssertEq
  IsFieldMov  IsFieldSLL64  IsFieldSLL128  IsFieldSLL192
```

## Primary source files

| Path | Role |
|------|------|
| `crates/jolt-compiler/examples/jolt_core_module.rs` | Registers Twist template — FR Twist mirrors its structure |
| `crates/jolt-compiler/examples/jolt_core_module_with_fieldreg.rs` | FieldReg-extended module; to be rewritten on v2 lines |
| `crates/jolt-r1cs/src/constraints/rv64.rs` | R1CS matrix; Phase 3 adds new rows, removes v1 rows 27-30 |
| `crates/jolt-witness/src/derived.rs` | `FieldRegEvent` schema; simplified in v2 (no FMov/FieldOp payloads) |
| `crates/jolt-host/src/r1cs_witness.rs` | `apply_field_op_events_to_r1cs` + `populate_limb_sum_columns` DELETED in v2 |
| `tracer/src/instruction/field_op.rs` | Existing FieldOp implementation kept (semantics unchanged) |
| `tracer/src/instruction/field_{mov,sll*}.rs` | NEW in v2 (Phase 2) |
| `tracer/src/instruction/fmov_{int_to_field,field_to_int}_limb.rs` | DELETED in v2 (Phase 1) |
| `jolt-inlines/bn254-fr/` | SDK rewrite on Phase 5 lines |

## Design history

**v1 (2026-04-13 → 2026-04-24).** Original design on `feat/field-register-twist`. Introduced FMov-I2F/F2I as limb-level transfer instructions, `FieldOpPayload` as a prover-declared event payload, `V_LIMB_SUM_A/B` limb-sum bridge (rows 29-30), and over-committed `FieldRegReadValue`/`FieldRegWriteValue` polys. Closed one soundness gap (compensating tamper against the Stage-2 bridge sumcheck) via Plan P (limb-sum R1CS rows), but introduced a new operand-forgery gap: `V_FIELD_OP_A` was populated from the prover's event record, not bound to FR Twist state. The proposed fix (rows 31-33, ~600-900 LOC) would have bridged two redundant R1CS columns — **mzhu's feedback (2026-04-24):** those two columns shouldn't exist separately at all. This spec is the resulting pivot.

**Rejected prior designs (for posterity):**
- **Product-of-MLEs bridge (Plan Z/Z').** Killed by Schwartz-Zippel: product-of-MLEs ≠ MLE-of-product off the boolean hypercube. Multi-agent cross-review confirmed.
- **Stage-2 sumcheck bridge closing at fresh `r_cycle_bridge`.** Allowed compensating tamper (prover forges `LimbSumA += δ, FieldOpA += δ` with no cryptographic binding). Replaced in v1 by R1CS rows 29-30.
- **v1 rows 31-33 + `VerifierOp::AssertEqualEvals` cross-check.** Would have closed the operand-forgery gap, but by bridging two redundant columns. Obsoleted by this pivot — the redundancy itself is the bug.

**v2 principle:** any column representing "the Fr value at a register slot on a cycle" is virtual and proven via the FR Twist sumcheck, never via a direct Dory commitment + separate R1CS cross-check.
