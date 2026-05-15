# Phase 4a — FieldRegClaimReduction Runbook

Self-contained execution plan for mirroring `RegistersClaimReduction` →
`FieldRegClaimReduction` in Stage 3. **Read this file end-to-end before
starting.** Every file path, line range, and identifier below has been
verified against the current tree.

---

## Where we are

Phase 1–3 of the FR coprocessor port are committed on branch
`spec/native-field-registers`. Worktree:
`/Users/sdhawan/Work/jolt-refactor-crates/.claude/worktrees/modular-sdk`.

Recent commits (`git log --oneline -10`):

```
c6a0a0e2f feat(phase-3a): FR Twist witness scaffolding
559bcc36f feat(phase-2a): 13 FR R1CS rows + 12 witness slots + row_bigcoeff
c5aa773e0 feat(phase-1g): bn254-fr inline SDK drop-in
8a1eb21f5 feat(phase-1f): jolt-riscv per-instruction structs for FR coprocessor
228e6594f feat(phase-1e): tracer FieldMov/FieldAssertEq/FieldSLL64-128-192
0cc429bc5 feat(phase-1d): thread FieldRegEvent stream through trace pipeline
cd298fa35 feat(phase-1c): tracer FieldOp instruction (FMUL/FADD/FSUB/FINV)
86e1d2f45 feat(phase-1b): tracer FieldReg CPU state
e67729a65 feat(phase-1a): add 9 BN254 Fr CircuitFlags (14 → 23)
92c79b443 docs(specs): FR coprocessor v2 port plan onto modular-sdk
```

Currently inert: `field_reg_d = 0` in
`crates/bolt/src/protocols/jolt/params.rs::JoltProtocolParams::new`. Phase
4a flips it to `1` and lights up the FR oracle family + Stage 3
FieldRegClaimReduction sumcheck.

Validation gates passing on the current tree:

- `cargo nextest run -p jolt-core muldiv --features host` ✅
- `cargo nextest run -p bolt --test commitment_ir` — 53/53 ✅
- `cargo nextest run -p jolt-witness` — 24/24 (incl. 5 new `field_reg::tests::*`) ✅
- `cargo clippy -p jolt-witness -p jolt-r1cs -p jolt-kernels -p bolt --all-targets -- -D warnings` ✅

---

## Prerequisite: Stage 1 R1CS oracle list — Phase 2 bookkeeping debt

Phase 2 extended `NUM_R1CS_INPUTS` from 35 to 47 in
`crates/jolt-r1cs/src/constraints/rv64.rs`, but the matching hardcoded
oracle-name list in Stage 1's MLIR builder was never updated:

`crates/bolt/src/protocols/jolt/phases/stage1.rs:22`:

```rust
const R1CS_INPUT_ORACLES: [&str; 35] = [ ... ];
```

This array is consumed at 6 sites in that file (search the symbol). It
controls:

1. Virtual oracle declarations (line ~965)
2. `ordered_oracles` of the outer-remaining batched relation (line ~973)
3. The relation's `output_count` attr (line ~1027)
4. The 35 opening-claim emit calls in the prover (line ~1247)
5. The `count` attr on the opening batch (line ~1279)
6. Verifier-side claim references (line ~1704)

Stage 1 currently doesn't open `V_FIELD_RS1_VALUE` (45),
`V_FIELD_RS2_VALUE` (46), or `V_FIELD_RD_WRITE_VALUE` (47) — Phase 4a
needs all three as upstream source claims for the new Stage 3 sumcheck.
The 9 `V_FLAG_IS_FIELD_*` columns (36–44) also belong in the list for
completeness; Phase 4c will reference some of them.

### Step 0: extend `R1CS_INPUT_ORACLES`

Mirror the canonical ordering of variable indices in
`crates/jolt-r1cs/src/constraints/rv64.rs` (constants
`V_LEFT_INSTRUCTION_INPUT = 1` through `V_FIELD_RD_WRITE_VALUE = 47`).
Replace the constant with all 47 entries:

```rust
const R1CS_INPUT_ORACLES: [&str; 47] = [
    // 1..=35 — unchanged from Phase 1 ordering:
    "LeftInstructionInput", "RightInstructionInput", "Product", "ShouldBranch",
    "PC", "UnexpandedPC", "Imm", "RamAddress",
    "Rs1Value", "Rs2Value", "RdWriteValue",
    "RamReadValue", "RamWriteValue",
    "LeftLookupOperand", "RightLookupOperand",
    "NextUnexpandedPC", "NextPC", "NextIsVirtual", "NextIsFirstInSequence",
    "LookupOutput", "ShouldJump",
    "OpFlagAddOperands", "OpFlagSubtractOperands", "OpFlagMultiplyOperands",
    "OpFlagLoad", "OpFlagStore", "OpFlagJump", "OpFlagWriteLookupOutputToRD",
    "OpFlagVirtualInstruction", "OpFlagAssert", "OpFlagDoNotUpdateUnexpandedPC",
    "OpFlagAdvice", "OpFlagIsCompressed", "OpFlagIsFirstInSequence",
    "OpFlagIsLastInSequence",
    // 36..=44 — Phase 2a Fr flag slots:
    "OpFlagIsFieldMul", "OpFlagIsFieldAdd", "OpFlagIsFieldSub",
    "OpFlagIsFieldInv", "OpFlagIsFieldAssertEq", "OpFlagIsFieldMov",
    "OpFlagIsFieldSLL64", "OpFlagIsFieldSLL128", "OpFlagIsFieldSLL192",
    // 45..=47 — Phase 2a Fr virtual operand slots:
    "FieldRs1Value", "FieldRs2Value", "FieldRdWriteValue",
];
```

**Verify before committing this:** the index order must match the
`V_*` constants in `crates/jolt-r1cs/src/constraints/rv64.rs:24–62`.
Off-by-one here silently corrupts the outer sumcheck's
`r1cs_input_evals` — `transcript_divergence` catches it but only after a
full proof run.

### Step 0 cascade — emit + kernel sides

The Stage 1 emit code in `crates/bolt/src/protocols/jolt/emit/rust/stage1.rs`
and the Stage 1 kernel in `crates/jolt-kernels/src/stage1.rs` build the
`r1cs_input_evals: [F; NUM_R1CS_INPUTS]` array directly off the R1CS key,
so they should already be size-correct (47). Verify by grepping:

```bash
grep -n "NUM_R1CS_INPUTS\|r1cs_input_evals\|35\b" crates/jolt-kernels/src/stage1*.rs
```

If anything still expects 35, fix it.

### Step 0 goldens regen + gate

```bash
source .bolt-dev-env
JOLT_UPDATE_GOLDENS=1 cargo nextest run -p bolt --test commitment_ir \
    generated_jolt_artifacts_have_uniform_crate_layout_and_import_rules \
    --cargo-quiet
cargo nextest run -p bolt --test commitment_ir --cargo-quiet --no-fail-fast
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
```

Expected: 53/53 commitment_ir + muldiv green. The 9 new flag oracles
and 3 new operand oracles render in Stage 1's outer-remaining sumcheck
but stay zero on every cycle (Phase 2's R1CS rows are still inert).
**Commit as `feat(phase-2a-cleanup): stage1 R1CS oracle list 35 → 47`.**

---

## Phase 4a — the actual port

### Template study (read first, do NOT edit)

These are the seven canonical RegistersClaimReduction touchpoints. Read
each end-to-end before writing any Fr mirror code.

#### Kernel (`crates/jolt-kernels/src/stage3.rs`)

| Anchor | Lines | What |
|---|---|---|
| `Stage3Relation::RegistersClaimReduction` | 29, 38, 48 | Enum variant + symbol map |
| `Stage3KernelAbi::RegistersClaimReduction` | 58, 67, 77 | ABI variant + name map |
| `Stage3ProverInstanceState::new` match | 1529 | Dispatch into `registers_state` |
| `SumOfProductsKind::Registers` | 1620 | State-kind tag |
| `fn registers_state` | 2127–2170 | **The actual prover kernel** |
| `fn register_factors` | 2391–2403 | Factor vector builder (RD/Rs1/Rs2 → Vec<F>) |
| `expected_registers` (verifier mirror) | 2513–2529 | **Verifier-side closed-form check** |
| `Stage3Relation::RegistersClaimReduction` in `expected_*` dispatcher | 2441 | Hooks verifier into the closed-form |
| Plan tables (kernel/ABI/relation/squeeze/program steps) | 3383, 3396, 3669–3771 | Static plan registry |
| Eval-name plumbing | 2153, 2158, 2163, 4170, 4175, 4180, 4213 | Output-name strings + claim hookup |

The math is dead-simple:
- **Prover**: γ-batch three column vectors `RdWriteValue + γ·Rs1Value + γ²·Rs2Value`, sum against `eq(r, RdWritePoint)`. Degree 2.
- **Verifier**: `eq(reverse(local_point), RdWritePoint) · (RdEval + γ·Rs1Eval + γ²·Rs2Eval)`.

#### Stage 3 MLIR builder (`crates/bolt/src/protocols/jolt/phases/stage3.rs`)

| Anchor | Lines | What |
|---|---|---|
| `REGISTERS_CLAIM_REDUCTION_DEGREE = 2` | 24 | Degree constant |
| `STAGE3_REGISTER_INPUTS = ["RdWriteValue", "Rs1Value", "Rs2Value"]` | 50 | Outputs |
| `trace_oracles.extend(STAGE3_REGISTER_INPUTS)` | 537 | Virtual-oracle declaration |
| `append_relation` for `jolt.stage3.registers_claim_reduction` | 598–609 | Relation op |
| `stage3.registers.gamma` field constant | 119 | Challenge squeeze |
| `stage3.registers.gamma2` field expr | 1067 | γ² derived from γ |
| Opening input for `RdWriteValue` (point source) | 1144–1173 | Wires Stage 1 → Stage 3 |
| Claim + instance for `registers_claim_reduction` | 1148, 1239–1242, 1248 | Sumcheck instance op |
| Eval outputs (`prefix: "stage3.registers_claim_reduction"`, outputs = `STAGE3_REGISTER_INPUTS`) | 1270–1272 | Final-round eval bindings |
| Batched output count | 1636 | `+ STAGE3_REGISTER_INPUTS.len()` |

#### Stage 3 emit (`crates/bolt/src/protocols/jolt/emit/rust/stage3.rs`)

Search `registers_claim_reduction` (5 hits). All are static-name match
arms in the prover/verifier code-gen.

### FR mirror — touchpoint-by-touchpoint

For each Registers touchpoint above, add a parallel FieldReg one. Use
these exact names (verified consistent with the FR spec at
`specs/fr-v2-port-plan.md`):

| Registers name | Fr mirror name |
|---|---|
| `Stage3Relation::RegistersClaimReduction` | `Stage3Relation::FieldRegClaimReduction` |
| `Stage3KernelAbi::RegistersClaimReduction` | `Stage3KernelAbi::FieldRegClaimReduction` |
| `"jolt.stage3.registers_claim_reduction"` | `"jolt.stage3.field_reg_claim_reduction"` |
| `"jolt_stage3_registers_claim_reduction"` | `"jolt_stage3_field_reg_claim_reduction"` |
| `STAGE3_REGISTER_INPUTS = ["RdWriteValue", "Rs1Value", "Rs2Value"]` | `STAGE3_FIELD_REG_INPUTS = ["FieldRdWriteValue", "FieldRs1Value", "FieldRs2Value"]` |
| `REGISTERS_CLAIM_REDUCTION_DEGREE = 2` | `FIELD_REG_CLAIM_REDUCTION_DEGREE = 2` |
| `"stage3.registers.gamma"` / `"stage3.registers.gamma2"` | `"stage3.field_reg.gamma"` / `"stage3.field_reg.gamma2"` |
| `"stage3.registers_claim_reduction.input"` | `"stage3.field_reg_claim_reduction.input"` |
| `"stage3.registers_claim_reduction.instance"` | `"stage3.field_reg_claim_reduction.instance"` |
| `"stage3.registers_claim_reduction.eval.{Rd|Rs1|Rs2}WriteValue"` | `"stage3.field_reg_claim_reduction.eval.Field{Rd|Rs1|Rs2}Value"` |
| `fn registers_state` | `fn field_reg_state` |
| `fn register_factors` | `fn field_reg_factors` |
| `fn expected_registers` | `fn expected_field_regs` |
| `SumOfProductsKind::Registers` | `SumOfProductsKind::FieldReg` |

### Critical differences from Registers

1. **Factor type**: Registers reads `cycle.rs1_value: u64` and casts via
   `F::from_u64`. FR operands are 256-bit (`FrLimbs`), so `Stage3Cycle`
   needs to carry `field_rs1: [u64; 4]`, `field_rs2: [u64; 4]`,
   `field_rd: [u64; 4]`. Bridge via:
   ```rust
   fn fr_limbs_to_field<F: Field>(limbs: [u64; 4]) -> F {
       let mut bytes = [0u8; 32];
       for (i, &limb) in limbs.iter().enumerate() {
           bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
       }
       F::from_le_bytes_mod_order(&bytes)
   }
   ```
   (Same pattern as `tracer/src/instruction/field_op.rs::fr_from_limbs`.)

2. **Stage3Cycle source**: Currently built in
   `crates/jolt-kernels/src/stage3.rs` via the trace conversion path.
   The FR data needs to flow from `replay_field_regs` (Phase 3 — already
   exists at `crates/jolt-witness/src/field_reg.rs`). Wire it through
   `Stage3ProverInputs` — likely add `field_reg_cycles: Option<&[FrCycleData]>`
   field. Verify the call site in `jolt-host/src/lib.rs` or
   `jolt-prover/src/prover.rs` (search `Stage3ProverInputs::new`).

3. **Eq point**: Registers uses `store.point("stage3.input.stage1.RdWriteValue")?`.
   For FR, use `store.point("stage3.input.stage1.FieldRdWriteValue")?`
   (the opening point Stage 1 publishes for that column — available
   after step 0 above).

4. **Gating**: The factor vectors must be zero on every non-FR cycle
   (mask by the OR of `IsFieldMul/Add/Sub/Inv/Mov/SLL*`). For inert
   testing (muldiv has no FR cycles) this means the factors are all
   zero, the sumcheck claim is zero, and the FieldReg sumcheck is
   trivially satisfied. **Do not skip the mask** — Phase 4b/c rely on it.

### Flip `field_reg_d`

Once the kernel + MLIR + emit code compiles, change
`crates/bolt/src/protocols/jolt/params.rs::JoltProtocolParams::new`:

```rust
let field_reg_d: usize = 0;  // ← remove
let field_reg_d = field_reg_log_k.div_ceil(log_k_chunk);  // ← restore
```

This automatically:
- Adds `FieldRegInc` + `FieldRegRa_0` to `main_witness_oracles` (the
  Phase 3 `if field_reg_d > 0` branches turn on)
- Bumps `num_committed` from 42 to 44
- Triggers FR oracle registration in `oracles::append_committed_oracles`

### Test fixture + asserts to update

When `field_reg_d` flips to 1, the following hardcoded test
assertions in `crates/bolt/tests/commitment_ir.rs` need bumping:

```
line 70:  "num_committed = 42"     → "num_committed = 44"
line 81:  ordered_oracles list     → insert "@FieldRegInc, " after "@RamInc, "
line 198: ordered_oracles list     → same
line 288: "num_committed must be 42" → "num_committed must be 44"
```

Plus the `CommitmentOracleInputs { ... }` struct literals at
lines ~3634 and ~4097 need `field_reg_inc` + `field_reg_indices` fields.
Re-emit the struct definition too — re-add the FR fields to the raw
string in
`crates/bolt/src/protocols/jolt/emit/rust/commitment.rs::emit_oracle_store_types`
(it was reverted in Phase 3 because oracles were inert; Phase 4a brings
them back).

### Verifier symmetry

Generated by Bolt. After kernel + MLIR + emit changes, run:

```bash
JOLT_UPDATE_GOLDENS=1 cargo nextest run -p bolt --test commitment_ir \
    generated_jolt_artifacts_have_uniform_crate_layout_and_import_rules
```

Then inspect `crates/jolt-prover/src/stages/stage3.rs` and
`crates/jolt-verifier/src/stages/stage3.rs` — both should now have
matching FieldRegClaimReduction code.

### Final gate

```bash
source .bolt-dev-env
cargo nextest run -p bolt --test commitment_ir --cargo-quiet --no-fail-fast
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo clippy -p jolt-witness -p jolt-r1cs -p jolt-kernels -p bolt \
    --message-format=short -q --all-targets -- -D warnings
```

All green → commit as `feat(phase-4a): Stage 3 FieldRegClaimReduction`.

---

## Constants reference (verified against current tree)

These will be inputs to Phase 4a code; verify before pasting:

```rust
// crates/jolt-r1cs/src/constraints/rv64.rs
pub const V_FLAG_IS_FIELD_MUL:        usize = 36;
pub const V_FLAG_IS_FIELD_ADD:        usize = 37;
pub const V_FLAG_IS_FIELD_SUB:        usize = 38;
pub const V_FLAG_IS_FIELD_INV:        usize = 39;
pub const V_FLAG_IS_FIELD_ASSERT_EQ:  usize = 40;
pub const V_FLAG_IS_FIELD_MOV:        usize = 41;
pub const V_FLAG_IS_FIELD_SLL64:      usize = 42;
pub const V_FLAG_IS_FIELD_SLL128:     usize = 43;
pub const V_FLAG_IS_FIELD_SLL192:     usize = 44;
pub const V_FIELD_RS1_VALUE:          usize = 45;
pub const V_FIELD_RS2_VALUE:          usize = 46;
pub const V_FIELD_RD_WRITE_VALUE:     usize = 47;
pub const NUM_R1CS_INPUTS:            usize = 47;
pub const NUM_VARS_PER_CYCLE:         usize = 50;
pub const NUM_EQ_CONSTRAINTS:         usize = 32;
pub const NUM_PRODUCT_CONSTRAINTS:    usize = 3;
pub const NUM_CONSTRAINTS_PER_CYCLE:  usize = 35;
```

```rust
// crates/jolt-witness/src/field_reg.rs
pub const FIELD_REG_COUNT: usize = 16;
pub const LOG_K_FR: usize = 4;
pub struct FrLimbs(pub [u64; 4]);
pub struct FrCycleData { rs1_pre, rs2_pre, rd_post, rd_index, rd_written }
pub fn replay_field_regs(events: Vec<FieldRegEvent>, trace_len: usize) -> Vec<FrCycleData>;
```

```rust
// crates/bolt/src/protocols/jolt/params.rs
pub field_reg_log_k: usize,  // = 4
pub field_reg_d: usize,      // Phase 3: 0; Phase 4a: 1
```

---

## What's NOT in scope for 4a

Explicitly deferred to Phase 4b / 4c:

- **Stage 4 FieldRegRW** (read-write checking, sparse phase-segmented
  sumcheck, `LOG_K_FR=4`, ScalarCapture transition). This is 4c.
- **Stage 5 FieldRegValEvaluation** (degree-3 `inc × eq_gather × LT`).
  This is 4b.
- Witness population from real FR-emitting guest programs. Phase 4a
  validates against muldiv (zero FR cycles); end-to-end FR proofs come
  in Phase 5 with the bn254-fr Poseidon2 example.

---

## Estimated effort

- Step 0 (stage1 oracles 35→47): ~30 min + 5 min goldens regen
- Step 1 (read templates): ~30 min
- Step 2 (kernel mirror in stage3.rs): ~45 min
- Step 3 (MLIR builder mirror in phases/stage3.rs): ~30 min
- Step 4 (emit mirror in emit/rust/stage3.rs): ~20 min
- Step 5 (flip `field_reg_d`, restore CommitmentOracleInputs FR fields): ~15 min
- Step 6 (goldens regen + test cycles): ~30 min (commitment_ir takes ~3 min/run, expect 5–8 runs)
- Step 7 (clippy + commit): ~10 min

Total: ~3 hours of focused work. Two commits expected:
1. `feat(phase-2a-cleanup): stage1 R1CS oracle list 35 → 47`
2. `feat(phase-4a): Stage 3 FieldRegClaimReduction`

---

## Resume command for the next session

> Execute Phase 4a per `specs/fr-v2-phase-4a-runbook.md`. Source the
> environment with `source .bolt-dev-env` first. Start with Step 0
> (extend `R1CS_INPUT_ORACLES` in `phases/stage1.rs` from 35 → 47),
> commit, then proceed through the runbook's FR mirror steps. Run the
> validation gates listed in the runbook after each substep.
