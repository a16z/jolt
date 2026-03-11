# Stage Parity Plan: Old jolt-core → New jolt-zkvm

**Date:** 2026-03-10
**Branch:** `refactor/crates`
**Approach:** Path A — restructure stages to replicate the old claim reduction chain

---

## Why Path A

The old pipeline's stages form a claim reduction chain where later stages consume
earlier stages' claims, converging to shared evaluation points. The end result is
that all committed polynomial opening claims land at a small number of related
points (suffixes of each other), collapsible via Lagrange normalization into a
single point → 1 Dory proof.

The new pipeline currently has independent stages, each producing claims at its
own random point. These are 7+ unrelated points that can't be collapsed. This is
both a correctness gap (8 missing sumcheck instances) and a structural gap (no
claim chaining).

Path A is preferred over Path B (sumcheck-based batch opening) because:
- The missing instances are soundness checks, not optimizations — they must exist regardless
- Point convergence is a natural consequence of RA decomposition, not extra work
- No additional proof size overhead (Path B adds ~log(N) sumcheck rounds)
- Matches the proven old protocol structure

---

## Instance Gap Map

| Old Instance | Old Stage | New Stage | Status |
|---|---|---|---|
| OuterUniSkip + OuterRemaining | S1 | S1 | ✓ |
| ProductVirtualUniSkip + Remainder | S2 | — | **MISSING** |
| RamReadWriteChecking | S2 | S4b | ✓ (moved) |
| InstructionLookupsClaimReduction | S2 | S3 | ✓ (moved) |
| RamRafEvaluation | S2 | S5 | ✓ (moved) |
| OutputCheck | S2 | S5 | ✓ (moved) |
| ShiftSumcheck | S3 | — | **MISSING** |
| InstructionInput | S3 | — | **MISSING** |
| RegistersClaimReduction | S3 | S3 | ✓ (moved) |
| RegistersReadWriteChecking | S4 | S4a | ✓ |
| RamValCheck | S4 | S4a | ✓ |
| InstructionReadRaf | S5 | — | **MISSING** |
| RamRaReduction | S5 | S3 | ✓ (simplified) |
| RegistersValEvaluation | S5 | — | **MISSING** |
| BytecodeReadRaf | S6 | — | **MISSING** |
| Booleanity (all RA polys) | S6 | — | **MISSING** (S6 only does Hamming) |
| RamHammingBooleanity | S6 | S6 | ✓ |
| RamRaVirtual | S6 | — | **MISSING** (S2 only does instruction) |
| InstructionRaVirtual | S6 | S2 | ✓ (moved) |
| IncClaimReduction | S6 | S3 | ✓ (moved) |
| AdviceClaimReduction (both phases) | S6–S7 | — | Deferred per spec |
| HammingWeightClaimReduction | S7 | S7 | ✓ |

**8 missing sumcheck instances** (not counting deferred advice).

---

## Old Pipeline: Full Stage Composition

### S1 — Spartan Outer R1CS
- **OuterUniSkip**: 1 uni-skip round over Z/cycle domain
- **OuterRemaining**: 1 + log_T rounds, degree 3
  - Identity: `Σ_x eq(τ, x) · (Az · Bz − Cz) = 0`
  - Produces: virtual poly openings (LookupOutput, PC, registers, RAM) at r_cycle

### S2 — Product Virtual + RAM RW + Claim Reductions + RAF + Output
Uni-skip first round, then 5-instance batch:
- **ProductVirtualUniSkip**: 1 uni-skip round
- **ProductVirtualRemainder**: log_T rounds, degree 3
  - Identity: continues binding cycle vars for 5 product-virtual R1CS constraints
- **RamReadWriteChecking**: log_K + log_T rounds, degree 3
  - Identity: `Σ_{k,j} eq(r, j) · ra(k,j) · (Val + γ·(inc + Val))`
- **InstructionLookupsClaimReduction**: log_T rounds, degree 2
  - Identity: `Σ_j eq(r, j) · (LookupOut + γ·LeftOp + γ²·RightOp + ...)`
- **RamRafEvaluation**: ~log_T + log_K rounds, degree 2
  - Identity: `Σ_k ra(k) · unmap(k) = raf_claim`
- **OutputCheck**: ~log_T + log_K rounds, degree 3
  - Identity: `Σ_k eq(r, k) · io_mask(k) · (Val_final − Val_io) = 0`

### S3 — Shift + Instruction Inputs + Registers Reduction
3 instances batched, log_T rounds each:
- **ShiftSumcheck**: degree 2
  - Identity: PC-shift invariant (PC at step j+1 = next PC from step j)
- **InstructionInput**: degree 3
  - Identity: instruction operand inputs at unified cycle point
- **RegistersClaimReduction**: degree 2
  - Identity: batches rd_wv + rs1_v + rs2_v register claims

### S4 — Registers RW + RAM Val Check
2 instances batched:
- **RegistersReadWriteChecking**: log_K_reg + log_T rounds, degree 3
  - Identity: register read/write value consistency
- **RamValCheck**: log_T rounds, degree 3
  - Identity: RAM value evaluation and final value check

### S5 — Instruction Read-RAF + RAM RA Reduction + Registers Val
3 instances batched:
- **InstructionReadRaf**: log_K_lookup + log_T rounds, degree n_virt + 2
  - Identity: instruction lookup read + RAF checking (multi-stage, prefix/suffix)
- **RamRaReduction**: log_T rounds, degree 2
  - Identity: consolidates 3 RAM RA claims (RAF, RW, val) at common r_address
- **RegistersValEvaluation**: log_T rounds, degree 3
  - Identity: `Val(r) = Σ_j inc(j) · wa(r_addr, j) · LT(r_cycle, j)`

### S6 — Bytecode RAF + Booleanity + Hamming Bool + RA Virtuals + Inc + Advice
6 core + 0–2 advice instances:
- **BytecodeReadRaf**: log_K_bc + log_T rounds, degree d+1
  - Identity: 5-stage batched read + RAF for bytecode
- **Booleanity**: log_k_chunk + log_T rounds, degree 3
  - Identity: `Σ_{k,j} eq · Σ_i γ_i · (ra_i² − ra_i) = 0` (all RA polys boolean)
- **RamHammingBooleanity**: log_T rounds, degree 3
  - Identity: `Σ_j eq · H(j) · (H(j) − 1) = 0`
- **RamRaVirtual**: log_T rounds, degree d+1
  - Identity: decomposes RAM RA claim into per-chunk committed poly claims
- **InstructionRaVirtual**: log_T rounds, degree n_committed_per_virtual + 1
  - Identity: decomposes instruction RA virtual claims into committed poly claims
- **IncClaimReduction**: log_T rounds, degree 2
  - Identity: reduces 4 RamInc/RdInc opening claims to 2 at one point
- **AdviceClaimReduction phase 1** (optional): varies, degree 2

### S7 — Hamming Weight + Advice Address
1 core + 0–2 advice instances:
- **HammingWeightClaimReduction**: log_k_chunk rounds, degree 2
  - Identity: fused Hamming weight + address unification for all RA polys
- **AdviceClaimReduction phase 2** (optional): varies, degree 2

### S8 — Dory Batch Opening
Not a sumcheck. RLC reduction + PCS::open at the unified point.

---

## New Pipeline: Current Stage Composition

### S1 — Uniform Spartan
- Outer sumcheck: degree 3, log2(total_rows_padded) rounds
- Inner sumcheck: degree 2, log2(total_cols_padded) rounds
- Outputs: (r_x, r_y, witness_opening_claim)

### S2 — RA Virtual (instruction only)
- 1 instance: `Σ_x eq(w,x) · Σ_i γ^i · Π_j ra_{i·m+j}(x)`
- Degree: m+1, num_vars: log_T
- Claims: one per committed RA chunk at S2_challenges

### S3 — Claim Reductions
- 4 instances (registers, instr lookups, RAM RA, increments), all degree 2
- Claims: at S3_challenges_reversed

### S4a — Register RW + RAM Val
- 2 instances, both degree 3
- Claims: at S4a_challenges

### S4b — RAM RW Checking
- 1 instance, degree 3
- Claims: at S4b_challenges

### S5 — RAM Output + RAF
- 2 instances, both degree 2
- Claims: at S5_challenges

### S6 — Hamming Booleanity (only)
- 1 instance, degree 3, zero-check
- Claims: H at S6_challenges

### S7 — Hamming Reduction
- 1 instance, degree 2
- Claims: at S7_challenges

### S8 — Batch Opening
- RLC reduction + PCS::open per distinct point group

---

## Claim Flow: Old vs New

### Old (chained — points converge)
```
S1 → (r_x, r_y) → virtual poly evals
S2 consumes S1 evals → r_stage2 (log_K + log_T vars)
S3 consumes S2 evals → r_stage3 (log_T vars, suffix of r_stage2)
S4 consumes S3 evals → r_stage4 (log_K_reg + log_T vars)
S5 consumes S4 evals → r_stage5 (log_K_lookup + log_T vars)
S6 consumes S2-S5 evals → r_stage6 (log_T vars = r_cycle)
    leaf claims: dense polys at r_cycle, RA polys at (r_addr_partial, r_cycle)
S7 consumes S6 RA claims → r_stage7 (log_k_chunk vars = r_address)
    leaf claims: all RA polys at (r_address, r_cycle) ← UNIFIED

Result: 3 related points (r_cycle ⊂ r_stage6_extended ⊂ (r_addr, r_cycle))
→ Lagrange normalization → 1 point → 1 Dory proof
```

### New (independent — points diverge)
```
S1 → (r_x, r_y)
S2 → claims at S2_challenges (RA polys)
S3 → claims at S3_challenges (dense polys)
S4 → claims at S4_challenges (reg/ram polys)
S5 → claims at S5_challenges (output/raf polys)
S6 → claims at S6_challenges (hamming poly)
S7 → claims at S7_challenges (hamming chunks)
S1 → claim at r_y (witness)

Result: 7+ unrelated random points → 7+ Dory proofs (or RLC per group)
```

---

## Confirmed Findings

### ProductVirtual is genuinely missing

The new `UniformSpartanProver::prove_dense_with_challenges` only runs outer +
inner sumcheck. The product virtual continuation (5 R1CS constraints:
Product, WriteLookupOutputToRD, WritePCtoRD, ShouldBranch, ShouldJump) is
fully absent. Without it, the Spartan R1CS proof is incomplete — there's an
unresolved claim over the product-virtual polynomials.

Source: `jolt-core/src/zkvm/spartan/product.rs` (946 lines).

### Kernel approach for new instances

All new instances use the IR-first pattern:

1. **Tags** in `jolt-ir/src/zkvm/tags.rs` — `poly::*` and `sumcheck::*` constants
2. **ClaimDefinition** in `jolt-ir/src/zkvm/claims/*.rs` — `ExprBuilder` formula
   with `OpeningBinding` / `ChallengeBinding` mappings
3. **KernelDescriptor** via `catalog::*` or `KernelShape::Custom` — drives compute
4. **Stage struct** implementing `ProverStage` using `KernelEvaluator`

Kernel shape selection:
- `eq · Σ c_i · p_i`: `catalog::eq_product()` (pre-compute LC) or
  `catalog::formula_descriptor(terms, n, d)` (kernel handles LC)
- `Σ γ^t · Π_k p_k`: `catalog::product_sum(D, n_virtual)` + `KernelEvaluator::with_toom_cook_eq`
- `eq · h · (h-1)`: `catalog::hamming_booleanity()`
- Arbitrary: `KernelShape::Custom` + `compile_with_challenges`

---

## Implementation Plan

**Approach**: Old stage numbering (S1=Spartan). Add instances and wire claim
chaining simultaneously — each instance is added into its correct old-pipeline
stage group, consuming prior_claims from the start.

### Step 1: S2 — ProductVirtual + RamRW + ClaimReductions + RAF + Output

The old S2 is the largest stage (5 instances + uni-skip). Current new pipeline
has these instances scattered across S3–S5. Restructure:

**1a. ProductVirtualRemainder** (NEW)
- Source: `jolt-core/src/zkvm/spartan/product.rs`
- What: log_T rounds, degree 3. Binds cycle vars for 5 product-virtual R1CS
  constraints. Takes r_cycle from S1 as tau_low, samples tau_high.
- Kernel: `KernelShape::Custom` (cubic: eq · Left · Right per constraint)
- Claims: produces virtual poly openings at r_stage2

**1b. Regroup existing instances into S2**
- Move RamReadWriteChecking (currently S4b) into S2
- Move InstructionLookupsClaimReduction (currently S3) into S2
- Move RamRafEvaluation (currently S5) into S2
- Move OutputCheck (currently S5) into S2
- All 5 instances batched with front-loaded batching (different num_vars/degrees)

**1c. ProductVirtualUniSkip** (NEW, optional optimization)
- 1 uni-skip round over the 5-element product domain
- Can defer — the remainder alone is correct, uni-skip is performance only

### Step 2: S3 — Shift + InstructionInput + RegistersClaimReduction

3 instances batched, all log_T rounds.

**2a. ShiftSumcheck** (NEW)
- Source: `jolt-core/src/zkvm/spartan/shift.rs`
- What: degree 2. PC-shift invariant: `PC(j+1) = NextPC(j)`
- Kernel: `catalog::eq_product()` with pre-computed LC of
  `EqPlusOne(r_outer, j) · (UnexpandedPC_shift + γ·PC_shift + γ²·IsNoop_shift)`
- Consumes: S2's virtual poly evals (PC, IsNoop) as input claims

**2b. InstructionInput** (NEW)
- Source: `jolt-core/src/zkvm/spartan/instruction_input.rs`
- What: degree 3. Instruction operand inputs at unified cycle point from S2.
- Kernel: `KernelShape::Custom` (eq · (RightInput + γ·LeftInput))
- Consumes: S2's instruction lookup claims

**2c. RegistersClaimReduction** (EXISTS in current S3)
- Already implemented. Just needs to move into this stage group and consume
  S2's register claims via prior_claims.

### Step 3: S4 — RegistersRW + RamValCheck

2 instances batched. Already exists as current S4a.

**3a. RegistersReadWriteChecking** (EXISTS)
**3b. RamValCheck** (EXISTS)
- Wire prior_claims: consume S3's register/RAM claims to derive eq points

### Step 4: S5 — InstructionReadRaf + RamRaReduction + RegistersValEval

3 instances batched. This is the most complex stage.

**4a. InstructionReadRaf** (NEW — most complex)
- Source: `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`
- What: log_K_lookup + log_T rounds, degree n_virt+2. Multi-stage batched
  read + RAF with prefix/suffix decomposition.
- Kernel: Likely `KernelShape::Custom` for the RAF phases, `ProductSum` for
  the product decomposition phases. May need multiple kernels composed.
- Consumes: S4's register claims

**4b. RamRaReduction** (EXISTS in current S3 as RAM RA claim reduction)
- Move from S3 into S5. Consolidates 3 RAM RA claims (RAF, RW, val).

**4c. RegistersValEvaluation** (NEW)
- Source: `jolt-core/src/zkvm/registers/val_evaluation.rs`
- What: log_T rounds, degree 3. `Val(r) = Σ_j inc(j) · wa(r_addr, j) · LT(r_cycle, j)`
- Kernel: `KernelShape::Custom` (cubic: eq · inc · wa, with LT baked into eq)
- Consumes: S4's register RW claims

### Step 5: S6 — BytecodeRaf + Booleanity + HammingBool + RA Virtuals + Inc

The old "mega-batch" stage. 6 core instances.

**5a. BytecodeReadRaf** (NEW)
- Source: `jolt-core/src/zkvm/bytecode/read_raf_checking.rs`
- What: log_K_bc + log_T rounds, degree d+1. 5-stage bytecode read + RAF.
- Kernel: Similar structure to InstructionReadRaf but for bytecode chunks.

**5b. Booleanity** (NEW — extend current S6)
- Source: `jolt-core/src/subprotocols/booleanity.rs`
- What: log_k_chunk + log_T rounds, degree 3. Zero-check that ALL RA polys
  (instruction + bytecode + RAM) are boolean: `Σ γ_i · (ra_i² − ra_i) = 0`
- Kernel: Generalization of `HammingBooleanity` to multiple RA polys.

**5c. RamHammingBooleanity** (EXISTS as current S6)
- Move into this stage group.

**5d. RamRaVirtual** (NEW)
- Source: `jolt-core/src/zkvm/ram/ra_virtual.rs`
- What: log_T rounds, degree d+1. Decomposes RAM RA claim into per-chunk
  committed poly claims. Same pattern as instruction RA virtual but for RAM.
- Kernel: `catalog::product_sum(ram_d, 1)` + `KernelEvaluator::with_toom_cook_eq`

**5e. InstructionRaVirtual** (EXISTS as current S2)
- Move from current S2 into S6. This is where it belongs in the old pipeline.

**5f. IncClaimReduction** (EXISTS in current S3)
- Move from S3 into S6. Reduces 4 RamInc/RdInc opening claims.

### Step 6: S7 — HammingWeight + (future) Advice

**6a. HammingWeightClaimReduction** (EXISTS as current S7)
- Already correct. Just needs to consume S6's RA claims via prior_claims
  to extend address dimension.

### Step 7: S8 — Point Normalization + Unified Opening

After S7, all committed poly claims land at related points. Add:

**PointNormalizationReduction** (new `OpeningReduction` impl in jolt-openings):
- `unified_point` = longest point (from RA polys: `(r_address, r_cycle)`)
- Dense claims: `eval *= eq(r_addr, 0)`, point extended to unified length
- RA claims: unchanged (already at unified point)

Then existing `RlcReduction` combines all claims at the single unified point → 1
Dory proof.

Supporting infrastructure:
- `ZeroPaddedSource` in jolt-poly: wraps `MultilinearPoly`, returns zeros beyond
  original length (for Dory streaming without materialization)
- Fix `combine_hints` in jolt-dory: auto-pad shorter hints with G1::identity()

### Step 8: Performance (P1/P2 from divergence report)

After structural correctness:
- Implement streaming RLC (`RlcSource`) instead of eager `rlc_combine`
- Restore Dory streaming evaluation (`DorySourceAdapter` equivalent)

---

## Execution Order

Build the claim chain from scratch, one old-pipeline stage at a time.
Each step adds instances and wires prior_claims simultaneously.

```
Step 1: S2 (ProductVirtual + regroup existing) ← HARDEST, do first
Step 2: S3 (Shift + InstrInput + RegClaimReduction)
Step 3: S4 (RegistersRW + RamVal) ← mostly exists
Step 4: S5 (InstrReadRaf + RamRaReduction + RegValEval) ← complex
Step 5: S6 (BytecodeRaf + Booleanity + HammingBool + RA virtuals + Inc)
Step 6: S7 (HammingWeight) ← mostly exists
Step 7: S8 (PointNormalization + unified opening)
Step 8: Performance optimizations
```

Each step is independently testable: synthetic witness → prove → verify.

---

## Fiat-Shamir Fixes (Already Done)

| Fix | File | Status |
|---|---|---|
| D1: Claim absorption before α | `jolt-sumcheck/src/batched.rs` | ✓ Done |
| D2: Opening claims flushed between stages | `jolt-zkvm/src/pipeline.rs`, `jolt-verifier/src/verifier.rs` | ✓ Done |
| D3: RLC claims absorbed before ρ | `jolt-openings/src/reduction.rs` | ✓ Done |

---

## Testing Strategy

- Each new instance gets a unit test (synthetic witness, prove → verify round-trip)
- Each step is tested with claim chaining: multi-instance stage, prior_claims wired
- After all steps: single Dory proof, full pipeline E2E with real-ish witness
- Final gate: `cargo nextest run -p jolt-zkvm muldiv` (when witness gen is wired)
