# BN254 Fr Native-Field Coprocessor

| Field      | Value                                                                 |
|------------|-----------------------------------------------------------------------|
| Author(s)  | sdhawan                                                               |
| Created    | 2026-04-13 (v1), 2026-04-17 (v2), 2026-04-24 (consolidated)           |
| Status     | Coprocessor functional end-to-end (honest prove + verify on real guests); one soundness gap remaining around operand binding to FR Twist state |
| Scope      | `refactor/crates` worktree at `jolt-refactor-crates/` only            |
| Branch     | `feat/field-register-twist`                                           |
| Supersedes | `native-field-registers.md`, `native-field-registers-plan.md`, `fr-bridge-full-soundness-plan.md` |

## Summary

A BN254 Fr coprocessor adds native `Fr::{add,sub,mul,inv}` RISC-V instructions plus a 16×256-bit field-register file to the Jolt zkVM. Guest programs cross the integer/field boundary via per-limb moves (`FMovIntToFieldLimb` / `FMovFieldToIntLimb`). Target: ~17 traced cycles per `Fr::mul`, validated at ~13 cycles on real guests (~190× speedup vs ark-bn254 software baseline).

Design principle: **field-register state is just another read/write memory-checking instance**, structurally identical to RAM or integer registers. The FR Twist is authored as a `Module` in `jolt-compiler`'s `ModuleBuilder` / `Schedule` / `ClaimFormula` API — a sibling to the existing RAM and Registers Twists — then glued into the Jolt protocol as a Stage-2 batched instance. Soundness closure between the FR Twist and the integer Registers Twist is enforced by R1CS rows inside the existing Stage-1 Spartan outer sumcheck.

## Intent

### Goal

Deliver a BN254 Fr coprocessor on refactor-crates with a sound binding from integer-register limbs to Fr values. Single-cycle `FieldOp` over a 16×256-bit field-register file, limb-register SDK ABI.

### Invariants

1. **Natural-form representation.** FieldReg cells store natural-form `[u64; 4]`. R1CS uses `F::from_natural_limbs(limbs)` everywhere field-adjacent. No Montgomery reinterpretation.
2. **Single access per cycle.** Each Fr SDK source op decomposes into a sequence of single-cycle guest instructions. No Fr-coprocessor cycle emits more than one `FieldRegEvent`.
3. **FieldReg Twist consistency.** Every read of slot `f` at cycle `t` returns the last value written before `t`. The FR Twist Module proves this.
4. **Claim/constraint synchronization.** Enforced by the `ClaimFormula` / `SumcheckDef` pairing — both prover and verifier evaluate the same `Formula` against the same `Schedule`/`VerifierSchedule`.
5. **Acceptance is load-bearing.** Honest-accepts + mutated-rejects is the binary pass/fail for every incremental milestone.

### Non-goals

- 32-byte atomic memory loads (RAM Twist stays u64-grain).
- Multi-field coprocessor — BN254 Fr only.
- G1/G2/Gt operations or pairings.
- Rust-compiler-managed field-register allocation.
- Recursive Jolt verifier.

## Architecture

### Two parallel Twists + bridge

| Component              | Purpose                                                    | Committed polynomials                                             |
|------------------------|------------------------------------------------------------|-------------------------------------------------------------------|
| **Registers Twist**    | Integer registers x0..x31 (64-bit)                         | `RdInc`, `RdRa`, etc. (existing)                                  |
| **FieldReg Twist**     | Field registers field_regs[0..15] (Fr-valued)              | `FieldRegInc`, `FieldRegRa`, `FieldRegVal`, `FieldRegEqCycle`, `FieldRegReadValue`, `FieldRegWriteValue` |
| **Limb-sum bridge**    | `V_FIELD_OP_A == Σ 2^{64k} · x[10+k]` on FieldOp cycles    | Two new R1CS witness columns `V_LIMB_SUM_A/B`; no new commitments |
| **FieldOp arithmetic** | `V_FIELD_OP_A op V_FIELD_OP_B = V_FIELD_OP_RESULT`         | R1CS rows, no new commitments                                     |
| **FMov arithmetic**    | Limb transfer between int regs and Fr slots (per-limb)     | R1CS rows, no new commitments                                     |

### ISA

All field-coprocessor instructions use opcode `0x0B` (custom-0), funct7 `0x40`, funct3 as selector.

| funct3 | Mnemonic | Semantics                                                      |
|--------|----------|----------------------------------------------------------------|
| 0x02   | FMUL     | `field_regs[frd] = field_regs[frs1] * field_regs[frs2]`        |
| 0x03   | FADD     | `field_regs[frd] = field_regs[frs1] + field_regs[frs2]`        |
| 0x04   | FINV     | `field_regs[frd] = field_regs[frs1]^{-1}`                      |
| 0x05   | FSUB     | `field_regs[frd] = field_regs[frs1] - field_regs[frs2]`        |
| 0x06   | FMovI2F  | `field_regs[frd].limb[k] = x[rs1]`  (k from imm)               |
| 0x07   | FMovF2I  | `x[rd] = field_regs[frs1].limb[k]`  (k from imm)               |

Slot fields frs1/frs2/frd live in the same 5-bit rs1/rs2/rd bytecode fields as standard R-type; low 4 bits used as slot index (0..15).

### SDK ABI (load-bearing for bridge)

The `jolt-inlines-bn254-fr` SDK emits every `Fr::{add,sub,mul,inv}` as a **single inline-asm block**:

```
a.limbs[0..4]   →  a0..a3    (x10..x13)   live across FieldOp cycle
b.limbs[0..4]   →  a4..a7    (x14..x17)   live across FieldOp cycle
out.limbs[0..4] ←  a8..a11   (x18..x21)
```

8 FMov-I2F loads + FieldOp + 4 FMov-F2I stores in one `asm!` block — compiler cannot clobber x10..x17 between limb-load and FieldOp. At the FieldOp cycle, `x[10..13]` holds `a`'s limbs and `x[14..17]` holds `b`'s by construction.

### Prover pipeline (where FR fits)

| Stage     | What runs                                                                           |
|-----------|-------------------------------------------------------------------------------------|
| Commit    | All committed polys uploaded to Dory (adds `FieldRegInc/Ra/Val/ReadValue/WriteValue`) |
| **Stage 1** Spartan outer | R1CS `Az·Bz=Cz` via group-split uniskip + remaining. **Matrix rows 19-30 = FieldOp/FMov/limb-sum gates.** |
| **Stage 2** batched | 6 instances (RAM, Registers, Instruction, Bytecode RAF, Output, **FR Twist**)      |
| Stage 3-7 | Claim reductions (Ra, Val, Inc), Hamming booleanity                                  |
| Stage 8   | Dory batched opening proofs                                                          |

FR Twist is instance [5] of Stage 2. The limb-sum bridge is **not** a Stage 2 sumcheck — it's R1CS rows enforced by Stage 1.

## Implementation status

### Shipped

| Area | Description | Files |
|------|-------------|-------|
| **Standalone FR Twist** | FR Twist authored as a standalone `Module`: 16 slots × 256-bit, two-phase segmented sumcheck, 11/11 honest-and-adversarial tests. Lives in the test file below — no separate `examples/` binary. | `crates/jolt-equivalence/tests/field_register_twist_standalone.rs` |
| **Main-protocol integration** | FR Twist folded into the Jolt protocol as Stage-2 batched instance [5]; `modular_self_verify_with_fieldreg` green | `crates/jolt-compiler/examples/jolt_core_module_with_fieldreg.rs:2319-2341` |
| **256-bit event payload** | `FieldRegEvent.{old,new}` widened from `u64` to `[u64;4]`; `limbs_to_field` helper | `crates/jolt-witness/src/derived.rs:117-213` |
| **Dedicated polynomial IDs** | `PolynomialId::FieldReg*` variants added so FR and RAM Twists coexist without aliasing | `crates/jolt-compiler/src/polynomial_id.rs:28-42` |
| **ISA + tracer** | `FieldOp` + `FMov{I2F,F2I}` RISC-V encoding (opcode 0x0B, funct7 0x40, funct3 selector) with tracer event emission | `tracer/src/instruction/field_op.rs`, `tracer/src/instruction/fmov_int_to_field_limb.rs`, `tracer/src/instruction/fmov_field_to_int_limb.rs` |
| **Guest SDK** | `jolt-inlines-bn254-fr` crate — `Fr::{add,sub,mul,inv}` with single-asm-block ABI (x10..x17 live across FieldOp) | `jolt-inlines/bn254-fr/` |
| **FADD/FSUB gates** | R1CS rows 19-20: `V_FIELD_OP_A + V_FIELD_OP_B = V_FIELD_OP_RESULT` (and sub-variant) | `crates/jolt-r1cs/src/constraints/rv64.rs:404-426` |
| **FMUL/FINV gates** | R1CS rows 21-26: `A · B = Result` / `A · Result = 1` via `V_PRODUCT` reuse (no product-constraint count bump) | `crates/jolt-r1cs/src/constraints/rv64.rs:448-499` |
| **Uniskip widening** | Spartan outer uniskip widened to actually sample rows 19-26; baseline modules stay at 19 via `new_with_constraints` | `crates/jolt-compiler/src/params.rs:158-205` |
| **FMov integer-side gates** | R1CS rows 27-28 bind `V_RS1_VALUE == V_FIELD_REG_WRITE_LIMB` (I2F) and `V_RD_WRITE_VALUE == V_FIELD_REG_READ_LIMB` (F2I). Ties the integer-register side of FMov cycles but not the FR-Twist-state side — see *Soundness* below. | `crates/jolt-r1cs/src/constraints/rv64.rs:508-532` |
| **Real-guest smoke test** | `bn254-fr-smoke-guest` compiles to RISC-V ELF, traces correctly, `FieldRegEvent`s carry expected `FieldOpPayload` | `examples/bn254-fr-smoke/`, `crates/jolt-equivalence/tests/bn254_fr_smoke.rs` |
| **Non-empty-events E2E** | Full prover-verifier loop with 3 synthetic `FieldRegEvent`s including 4-limb values; honest accepts + two adversarial mutations reject | `crates/jolt-equivalence/tests/muldiv.rs:2174-2238` |
| **Limb-sum bridge (rows 29-30)** | `IsFieldOpAny · (V_LIMB_SUM_A − V_FIELD_OP_A) = 0` (and B-side with `IsFieldOpNoInv` guard = FMUL+FADD+FSUB, excluding FINV whose B-side operand is unused). `V_LIMB_SUM_A/B` populated from register-write history; reduced to committed `RdInc` via Stage 5 `LimbSumAReduction`. **Closes the compensating-tamper attack** against the original Stage-2 bridge. | `crates/jolt-r1cs/src/constraints/rv64.rs:550-570`, `crates/jolt-host/src/r1cs_witness.rs:215-257` |
| **Padded-constraint bump** | Shared R1CS matrix grew from 32 to 34 rows → `num_constraints_padded` 32 → 64 (constant `TOTAL_MATRIX_CONSTRAINTS_PADDED`). Measured +1.6% prove_ms on muldiv (noise range). | `crates/jolt-compiler/src/params.rs:61, 197-205` |

### Measured performance

| Metric                      | Value                                                   |
|-----------------------------|---------------------------------------------------------|
| Fr op cycle count (SDK)     | ~13 per `Fr::mul` (isolated, no surrounding work)       |
| ark-bn254 software baseline | ~2500 per `Fr::mul`                                     |
| Speedup                     | ~190× on Fr math                                        |
| `jolt-bench muldiv` prove_ms (pad=64 vs pad=32) | 736.0 vs 724.6 — **+1.6%** (noise range) |
| `jolt-bench muldiv` proof_bytes | 65,788 (identical pre/post)                        |

### Pending

| Area | Description | Blocker |
|------|-------------|---------|
| **Full operand-binding** | Rows 31-33 in `rv64.rs`: bind `V_FIELD_OP_A/B/RESULT` to FR Twist committed state at slots frs1/frs2/frd. Closes the operand-forgery attack described in *Soundness* below. | Design is in this spec; ~600-900 LOC, 1-2 sessions |
| **FMov limb-value binding** | Same pattern applied to rows 27-28: bind `V_FIELD_REG_READ_LIMB` / `V_FIELD_REG_WRITE_LIMB` to FR Twist limb values. Same root-cause gap as FieldOp. | Unblocked by operand-binding; ~1/3 the size |
| **Zero-knowledge mode** | BlindFold integration for all FR sumchecks (commitment-hiding prover, ZK verifier) | `jolt-blindfold` substrate incomplete; orthogonal to soundness work |

## Soundness

### What is sound

**Chain 1: Integer register side.** For FieldOp cycles, the rows 29-30 enforce `V_FIELD_OP_A == Σ 2^{64k} · Val_reg(10+k, cycle)`. `V_LIMB_SUM_A/B` reduces via Stage 5 `LimbSumAReduction` to committed `RdInc`, whose opening is verified by Dory PCS. Forging `V_FIELD_OP_A` forces a forged `RdInc` opening → PCS rejection.

**Chain 2: FR Twist state consistency.** `FieldRegInc/Ra/Val/ReadValue/WriteValue` are committed and prove Twist read-write consistency via Stage-2 instance [5]. Tampering any of these polys fails the Stage-2 sumcheck or its Dory opening.

**Chain 3: Arithmetic correctness.** R1CS rows 19-26 enforce `V_FIELD_OP_A op V_FIELD_OP_B = V_FIELD_OP_RESULT` for the four ops. Tampering the result mismatches operand arithmetic at Stage 1.

**Chain 4: FMov integer-side binding.** R1CS rows 27-28 enforce `V_RS1_VALUE == V_FIELD_REG_WRITE_LIMB` (I2F) and `V_RD_WRITE_VALUE == V_FIELD_REG_READ_LIMB` (F2I). Ties the integer register half of FMov cycles.

**Validated attacks that reject:**
- `audit_poc_compensating_tamper_solved` — compensating tamper of `V_FIELD_OP_A/B` + `V_LIMB_SUM_A/B` at Stage-2 openings. Rejected by the limb-sum bridge's Stage-1 R1CS enforcement.
- `modular_self_verify_with_fieldreg_fadd_tampered_result_rejects` — tampered FADD result. Rejected by Stage 1 outer.
- `modular_self_verify_with_fieldreg_nonempty_events_tampered_wv_rejects` — tampered committed `FieldRegWriteValue`. Rejected by Stage 2 FR Twist.
- `bridge_tampered_val_reg_rejects_via_registers_twist` — tampered integer `Val_reg` at x10..x13. Rejected by Registers Twist PCS opening.

### What is NOT sound — the remaining gap

**Operand-forgery via omitted FMov-I2F.**

`V_FIELD_OP_A` is populated from `FieldOpPayload.a` (prover-declared in the event record) at `jolt-host/src/r1cs_witness.rs:324-329`. **No R1CS row or sumcheck binds `V_FIELD_OP_A` to the FR Twist's committed `Val_fr(frs1, cycle)`.**

Concrete attack:

1. Honest program: `fmov_i2f(x20 → field_regs[0])` loads A into slot 0, then `FMUL frd, frs1=0, frs2=1` — honest result `field_regs[frd] = A·B`.
2. Malicious prover: omits the FMov-I2F event from the stream. FR Twist state at slot 0 stays 0.
3. Prover sets `FieldOpPayload.a = A'` (arbitrary Fr), sets x10..x13 to A's integer limbs.
4. Row 29 passes (`V_LIMB_SUM_A == V_FIELD_OP_A == A'` — both prover-chosen to match).
5. Rows 21-26 compute `V_FIELD_OP_RESULT = A'·B`, stored at slot frd.
6. Every check passes. Proof accepts with fraudulent operand.

**Root cause.** Rows 29-30 prove `V_FIELD_OP_A == integer-regs` but not `V_FIELD_OP_A == FR-Twist-state[frs1]`. The shipped limb-sum bridge is *one of two* required identities; the original Phase-3 design silently assumed an existing "R1CS gate" binding `V_FIELD_OP_A` to the FR Twist opening. No such gate exists.

**Same class affects FMov limb gates.** `V_FIELD_REG_READ_LIMB` / `V_FIELD_REG_WRITE_LIMB` are populated from `FMovPayload.limb` (prover-declared) with no FR-Twist-state binding. A prover can forge limb reads/writes independent of the FR Twist's committed state.

**Independent verification (2026-04-23).** A 10-agent adversarial cross-review of the proposed fix flagged this gap unanimously. Consensus: gap is real, naive single-row fixes are vacuous without a cross-check mechanism binding R1CS witness columns to the committed FR Twist polynomials.

### Proposed fix

Three new equality rows tying `V_FIELD_OP_A/B/RESULT` to FR Twist committed state. These are **inserted before the existing product constraints**, so the current rows 31-33 (Product / ShouldBranch / ShouldJump) shift to 34-36:

```
Row 31 (new eq): IsFieldOpAny   · (V_FIELD_OP_A      − V_FR_READ_VALUE_A)  = 0
Row 32 (new eq): IsFieldOpNoInv · (V_FIELD_OP_B      − V_FR_READ_VALUE_B)  = 0
Row 33 (new eq): IsFieldOpAny   · (V_FIELD_OP_RESULT − V_FR_WRITE_VALUE)   = 0
Row 34 (product, shifted): Product = LeftInstructionInput × RightInstructionInput
Row 35 (product, shifted): ShouldBranch = LookupOutput × Branch
Row 36 (product, shifted): ShouldJump = Jump × (1 − NextIsNoop)
```

`NUM_EQ_CONSTRAINTS: 31 → 34`. Matrix goes from 34 to 37 rows; pad stays at 64.

**Required infrastructure (NOT just R1CS rows):**

1. **Two new virtual-but-bound polynomials.** `FieldRegReadValueA` and `FieldRegReadValueB` — T-length, value at cycle c = pre-access `Val_fr(frs1(c), c)` and `Val_fr(frs2(c), c)` on FieldOp cycles, zero elsewhere. Must be bound to the committed `FieldRegVal` via an extended FR Twist read-check sumcheck.

2. **FR Twist sumcheck extension.** Current FR Twist at `jolt_core_module_with_fieldreg.rs:2319-2341` proves one read per cycle (at `e.slot = frd`). Extend to γ-batched form proving three reads per FieldOp cycle at (frs1, frs2, frd). Mirrors how integer Registers Twist γ-batches rs1/rs2/rd reads.

3. **Slot-address binding to bytecode.** The slot indices must come from bytecode's committed `RS1_INDEX`/`RS2_INDEX`/`RD_INDEX` fields (already committed as virtuals in `BytecodeReadRaf`). Without this, a malicious prover chooses which slot to read from.

4. **Cross-check mechanism.** R1CS witness columns (slots 49/50/51) must agree with committed `FieldRegReadValueA/B` + existing `FieldRegWriteValue` at the Stage-1 opening point. Two options:
   - **Option X:** Unified PolynomialId — the R1CS witness column IS the committed poly. Requires a new `PolySource::R1csCommitted` variant in the compiler.
   - **Option Y:** Separate storage + new `VerifierOp::AssertEqualEvals` asserting the two openings match at `stage1_cycle`.

**Why row 31 alone is insufficient without (1)-(3).** A naive implementation populating `V_FR_READ_VALUE_A` from the same event stream as `V_FIELD_OP_A` makes row 31 tautologically satisfied — both sides equal the prover-declared value. The cross-check to the committed FR Twist state is what makes row 31 meaningful.

### Scope of remediation

- **R1CS**: `NUM_EQ_CONSTRAINTS 31 → 34`, 3 new witness columns, group-split update (17/17 split; uniskip domain 17, next_pow2=32 → +5-8% outer Spartan cost).
- **FR Twist**: extended input_claim and output_check with γ-batched frs1/frs2 read terms.
- **Compiler**: new `PolySource` variant OR new VerifierOp.
- **Witness**: populate three new R1CS columns from FieldRegEvent stream + simulated FR state.
- **Verifier**: mirror FR Twist extension; cross-check opcode.
- **Tests**: PoC attack test (omit-FMov-I2F + forge operand) must reject. Existing honest tests must still pass.
- **Cross-verify with jolt-core**: breaks (acceptable per team guidance 2026-04-23).

Estimated: ~600-900 LOC across 6 files. 1-2 focused sessions.

**FMov gap.** Same pattern applied to rows 27-28 with limb-level binding. Roughly 1/3 the size of the FieldOp fix because FMov is one-limb-at-a-time. Separate sub-task.

## Design decisions

### Chosen

- **Natural-form `[u64;4]` for FieldReg cells.** `F::from_natural_limbs(limbs)` everywhere; no apparent-Montgomery.
- **Limb-register SDK ABI.** Fr values cross SDK boundary via integer registers, no RAM round-trip. Enables 13-cycles/mul.
- **16 × 256-bit FR file, single-cycle FieldOp.** One write + up to two reads per cycle; one `FieldRegEvent` per FieldOp cycle.
- **MSB-first limb load order** for FMovIntToFieldLimb — matches the Horner recurrence that the limb-to-Fr bridge uses.
- **Standalone → glue → real → bridge phase order.** Each layer has independent pass/fail.
- **Bridge as R1CS rows, not Stage-2 sumcheck.** Chosen after the original Stage-2 bridge was found unsound (compensating tamper at fresh `r_cycle_bridge`). R1CS rows enforce pointwise; openings reduce through existing committed polys.
- **Adversarial-rejects tests are mandatory acceptance criteria** at every phase.

### Out of scope

- **FLOAD / FSTORE / FLoadHorner ISA.** Memory-backed Fr unused with limb-register SDK ABI. Revisit only when a consumer demands it.
- **Compiler-managed FR-register allocation.** Would drop cycle count from ~13 toward theoretical ~5; depends on Rust-compiler work. Out of scope for current push.
- **FINV(0) = 0 handling.** R1CS FINV gate doesn't cover zero inversion. Guest must not invert zero.

### Rejected (with reasons)

- **Product-of-MLEs bridge.** An early attempt to bind FR operands via multiplicative coupling of two MLEs. Killed after cross-review confirmed the Schwartz-Zippel fallacy: product-of-MLEs ≠ MLE-of-product off the boolean hypercube.
- **Bridge as a separate Stage-2 sumcheck.** Correct but ~400 LOC of sumcheck plumbing. Superseded by the R1CS-row approach at rows 29-30, which reuses the existing Stage-1 Spartan outer sumcheck and reduces through already-committed polys.
- **"Port from main."** Main jolt-core has a different bytecode read-raf pattern that inspired the bytecode-slot-binding idea but is not directly portable — refactor-crates uses its own `BytecodeReadRaf` with different structure.

## Naming

Dedicated `PolynomialId` variants so RAM and FieldReg Twists coexist:

- `PolynomialId::FieldRegInc` — Twist Inc (committed, dense, cycle-major)
- `PolynomialId::FieldRegVal` — Twist Val (virtual, sparse K×T)
- `PolynomialId::FieldRegRa` — access address one-hot poly
- `PolynomialId::FieldRegEqCycle` — eq-table companion for the FR Twist sumcheck
- `PolynomialId::FieldRegReadValue` / `FieldRegWriteValue` — Fr values read/written per cycle (committed)
- `PolynomialId::IsFieldOpAny` / `IsFieldOpNoInv` — per-cycle indicator virtuals
- `PolynomialId::LimbSumA` / `LimbSumB` — R1CS witness columns for the limb-sum bridge
- `PolynomialId::FieldOpOperandA` / `FieldOpOperandB` / `FieldOpResultValue` — R1CS operand columns
- `PolynomialId::FieldRegReadLimb` / `FieldRegWriteLimb` — 64-bit limb columns for FMov cycles

Per-Module stages within the FR Twist:
- `FrRegReadWriteChecking` — 2-phase segmented sumcheck
- `FrRegValEvaluation` — Val MLE eval at Stage-2 binding point
- `FrRegClaimReduction` — claim reduction

R1CS constant slots (`crates/jolt-r1cs/src/constraints/rv64.rs`):

```
V_FIELD_OP_A         = 42    V_FIELD_REG_READ_LIMB  = 45    V_LIMB_SUM_A = 47
V_FIELD_OP_B         = 43    V_FIELD_REG_WRITE_LIMB = 46    V_LIMB_SUM_B = 48
V_FIELD_OP_RESULT    = 44                                   V_BRANCH     = 49
                                                            V_NEXT_IS_NOOP = 50
NUM_R1CS_INPUTS = 48, NUM_VARS_PER_CYCLE = 51
NUM_EQ_CONSTRAINTS = 31, NUM_PRODUCT_CONSTRAINTS = 3, total = 34 → pad 64
```

## References

### Primary files

| Path | Role |
|------|------|
| `crates/jolt-compiler/examples/jolt_core_module_with_fieldreg.rs` | Full fieldreg-extended module (prover + verifier schedule) |
| `crates/jolt-r1cs/src/constraints/rv64.rs` | R1CS matrix definition; rows 19-30 are FR-coprocessor rows |
| `crates/jolt-host/src/r1cs_witness.rs` | Per-cycle witness population + `apply_field_op_events_to_r1cs` + `populate_limb_sum_columns` |
| `crates/jolt-witness/src/derived.rs` | `FieldRegEvent`, `FieldRegConfig`, `DerivedSource::with_field_reg` |
| `crates/jolt-compiler/src/polynomial_id.rs` | `PolynomialId` enum + R1CS index map + descriptor |
| `crates/jolt-compiler/src/params.rs` | `ModuleParams` derivation, uniskip sizing |
| `crates/jolt-equivalence/tests/muldiv.rs` | E2E honest + adversarial tests through the full prover |
| `crates/jolt-equivalence/tests/field_register_twist_standalone.rs` | Phase-1 standalone tests (11 cases) |
| `crates/jolt-equivalence/tests/bn254_fr_smoke.rs` | Real-guest cycle-count validation |
| `tracer/src/instruction/field_op.rs` | FieldOp instruction implementation |
| `tracer/src/instruction/fmov_{int_to_field,field_to_int}_limb.rs` | FMov implementations |
| `jolt-inlines/bn254-fr/` | Guest-side SDK with asm! ABI |


## Commands

```bash
cd /Users/sdhawan/Work/jolt-refactor-crates

# Full refactor test suite:
cargo nextest run --cargo-quiet

# FR Twist standalone (Module-only, 11 tests):
cargo nextest run -p jolt-equivalence --test field_register_twist_standalone --cargo-quiet

# FR Twist through full protocol (end-to-end prove + verify):
cargo nextest run -p jolt-equivalence --test muldiv modular_self_verify_with_fieldreg --cargo-quiet

# Soundness gate (compensating-tamper rejection — regression signal for the limb-sum bridge):
cargo nextest run -p jolt-equivalence --test muldiv audit_poc_compensating_tamper_solved --cargo-quiet

# Real-guest cycle-count validation:
cargo nextest run -p jolt-equivalence --test bn254_fr_smoke --cargo-quiet

# Perf bench (muldiv, modular stack):
./target/release/jolt-bench --program muldiv --stack modular --iters 5 --warmup 2

# Clippy in both modes:
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
```

## Acceptance criteria

The coprocessor is soundness-complete when all of the following hold:

1. All existing honest tests pass (muldiv, FR self-verify, FMUL/FADD multi-limb).
2. `audit_poc_compensating_tamper_solved` still rejects — regression gate for the limb-sum bridge.
3. **New adversarial test**: omitted-FMov-I2F + forged operand attack rejects.
4. **New adversarial test**: forged FMov limb (read or write) rejects.
5. `jolt-bench muldiv modular` prove_ms within +10% of the current limb-sum-bridge state.
6. Clippy clean in both `host` and `host,zk` modes.
7. This spec's *What is sound* / *What is NOT sound* sections updated; operand-binding moves from *Pending* to *Shipped*.
