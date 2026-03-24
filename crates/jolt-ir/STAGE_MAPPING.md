# Exact jolt-core Stage Mapping

Reference for graph construction. Every vertex, every challenge squeeze, every input claim.

## S2: 5 instances

**Challenge squeeze order**:
1. `ProductVirtualUniSkipParams::new` → 1 scalar (tau_high)
2. `RamReadWriteCheckingParams::new` → 1 scalar (gamma)
3. `InstructionLookupsClaimReductionSumcheckParams::new` → 1 scalar (gamma)
4. `OutputSumcheckParams::new` → log_K scalars (r_address vector)

**Instance order** (in batched sumcheck):
1. RamReadWriteChecking — `log_K + log_T` rounds, degree 3
2. ProductVirtualRemainder — `log_T` rounds, degree varies
3. InstructionLookupsClaimReduction — `log_T` rounds, degree 2
4. RamRafEvaluation — `log_K` rounds, degree 2
5. OutputCheck — `log_K` rounds, degree 2

**Input claims**:
- RamRW: `rv + γ·wv` (from S1: RamReadValue, RamWriteValue)
- PVRemainder: `Σ L_i(τ_high)·base_evals[i]` (from S1 product constraint evals)
- InstrLookupsCR: `lo + γ·lop + γ²·rop + γ³·lip + γ⁴·rip` (from S1)
- RamRafEval: `raf_claim · 2^{phase3_rounds}` (from S1 RamAddress)
- OutputCheck: 0 (constant zero-check)

## S3: 3 instances

**Challenge squeeze order**:
1. `ShiftSumcheckParams::new` → 5 gamma powers
2. `InstructionInputParams::new` → 1 scalar (gamma)
3. `RegistersClaimReductionSumcheckParams::new` → 1 scalar (gamma)

**Instance order**:
1. Shift — `log_T` rounds, degree 2, EqPlusOne weighting
2. InstructionInput — `log_T` rounds, degree 3
3. RegistersClaimReduction — `log_T` rounds, degree 2

**Input claims**:
- Shift: `Σ γ^i · next_claim_i` (from S1: NextPC, NextUnexpandedPC, etc. + S2: NextIsNoop)
- InstrInput: `right + γ·left` (from S2: Right/LeftInstructionInput)
- RegistersCR: `rd_wv + γ·rs1 + γ²·rs2` (from S1: RdWriteValue, Rs1Value, Rs2Value)

## S4: 2 instances

**Challenge squeeze order**:
1. `RegistersReadWriteCheckingParams::new` → 1 scalar (gamma)
2. Domain separator + `RamValCheckSumcheckParams` → 1 scalar (gamma)

**Instance order**:
1. RegistersReadWriteChecking — `log_K + log_T` rounds, degree 3
2. RamValCheck — `log_T` rounds, degree 3

**Input claims**:
- RegistersRW: `rd_wv + γ·(rs1 + γ·rs2)` (from S3: RegistersCR evals, checked against InstrInput)
- RamValCheck: `(val_rw - init) + γ·(val_final - init)` (from S2: RamRW val + OutputCheck val_final + external init_eval)

## S5: 3 instances

**Challenge squeeze order**:
1. `InstructionReadRafSumcheckParams::new` → 1 scalar (gamma)
2. `RaReductionParams::new` → 1 scalar (gamma)
3. `RegistersValEvaluationSumcheckParams::new` → none (no squeeze)

**Instance order**:
1. InstructionReadRaf — `log_K + log_T` rounds, degree d+2
2. RamRaClaimReduction — `log_K` rounds, degree 2
3. RegistersValEvaluation — `log_T` rounds, degree 3

**Input claims**:
- InstrReadRaf: `rv + γ·lop + γ²·rop` (from S2: InstrLookupsCR evals)
- RamRaCR: `Σ γ^i · ra_i` (from S2: RamRafEval + RamRW + RamValCheck RA evals)
- RegistersValEval: `val_eval` (from S4: RegistersRW val eval)

## S6: 6 instances (+ optional advice)

**Challenge squeeze order**:
1. `BytecodeReadRafSumcheckParams::gen` → 1 scalar (gamma)
2. `HammingBooleanitySumcheckParams::new` → none
3. `BooleanitySumcheckParams::new` → 1 scalar (gamma)
4. `RamRaVirtualParams::new` → none
5. `InstructionRaSumcheckParams::new` → 1 scalar (gamma)
6. `IncClaimReductionSumcheckParams::new` → 1 scalar (gamma)

**Instance order**:
1. BytecodeReadRaf — `log_K + log_T` rounds, degree d+1
2. Booleanity — `log_T + log_K` rounds, degree 3
3. HammingBooleanity — `log_K` rounds, degree 3
4. RamRaVirtual — `log_T + log_K` rounds, Toom-Cook
5. InstructionRaVirtual — `log_T + log_K` rounds, Toom-Cook
6. IncClaimReduction — `log_T` rounds, degree 2

**Input claims**:
- BytecodeReadRaf: multi-stage batched claims from S1-S5
- Booleanity: 0 (zero-check)
- HammingBooleanity: 0 (zero-check)
- RamRaVirtual: `Σ γ^i · ra_i` (from S5 RamRaCR)
- InstrRaVirtual: `Σ γ^i · ra_i` (from S5 InstrReadRaf)
- IncCR: `v1 + γ·v2 + γ²·w1 + γ³·w2` (from S2: RamInc@RamRW, S4: RamInc@RamValCheck, S4: RdInc@RegistersRW, S5: RdInc@RegistersValEval)

## S7: 1 instance (+ optional advice)

**Challenge squeeze order**:
1. `HammingWeightClaimReductionParams::new` → 1 scalar (gamma)

**Instance order**:
1. HammingWeightClaimReduction — `log_K` rounds, degree 2

**Input claims**:
- HammingWeightCR: `Σ (γ^{3i}·hw_i + γ^{3i+1}·bool_i + γ^{3i+2}·virt_i)` (from S6: Booleanity + RA virtual + Hamming evals)
