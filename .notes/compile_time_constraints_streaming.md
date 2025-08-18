# Compile-time Constant Constraints and Streaming Evaluation Plan

This document proposes replacing dynamic, heap-allocated R1CS constraint construction with a compile-time constant constraint table and a streaming evaluation path. It also incorporates experimental typing and per-constraint custom evaluation data (FastSpec). The goals are to reduce allocations, eliminate dynamic branching in hot paths, and enable direct, low-overhead access to inputs (especially boolean flags) during SVO and prover precomputation.

## Goals
- Eliminate dynamic constraint building in the prover’s hot path by using a static constraints table.
- Evaluate A/B/C per row using a streaming provider (no full-length virtual witness materialization).
- Retain coarse types (AzType/BzType/CzType) and add optional FastSpec for custom, zero-branch evaluation of common patterns.
- Preserve existing APIs for key generation, verifier materialization, and as a fallback path.

## High-level Design
1. Represent each uniform constraint row as a const struct with:
   - Fixed-size linear combinations (LC) for `A`, `B`, `C` with ≤ 5 terms.
   - `AzType`, `BzType`, `CzType` lanes for correctness and downstream arithmetic.
   - Optional const FastSpec for A/B when a cheaper evaluation is known (e.g., boolean combos, var−var).
2. Replace the dynamic uniform-constraint builder with a static `UNIFORM_R1CS` table (`&'static [RowSpec]`).
3. Introduce an `InputProvider` abstraction to decouple evaluation from how inputs are sourced:
   - `MaterializedProvider` (adapts today’s `MultilinearPolynomial` arrays).
   - `StreamingProvider` (computes virtual inputs on-demand, per row, from `trace` and preprocessing).
4. Keep conversion utilities to materialize sparse matrices for keygen/verifier from the static table.

## Const-friendly Representations
- Fixed LC with up to 5 terms; no `Vec` at compile time.
- Const indices for inputs to avoid calling `to_index()` in const context.
- Const FastSpec variants with small fixed arrays.

```rust
// LC term: (input index in ALL_R1CS_INPUTS, coefficient)
#[derive(Clone, Copy)]
pub struct ConstTerm { pub input_index: usize, pub coeff: i128 }

// Fixed-capacity LC (N in 0..=5). const-friendly; no heap.
#[derive(Clone, Copy)]
pub struct ConstLC<const N: usize> {
    pub terms: [ConstTerm; N],
    pub has_const: bool,
    pub const_coeff: i128,
}

pub trait ConstLCAny {
    fn num_terms(&self) -> usize;
    fn term(&self, i: usize) -> Option<ConstTerm>;
    fn constant(&self) -> Option<i128>;
}

impl<const N: usize> ConstLCAny for ConstLC<N> {
    #[inline] fn num_terms(&self) -> usize { N }
    #[inline] fn term(&self, i: usize) -> Option<ConstTerm> { if i < N { Some(self.terms[i]) } else { None } }
    #[inline] fn constant(&self) -> Option<i128> { if self.has_const { Some(self.const_coeff) } else { None } }
}

// Optional per-row fast evaluators (const-friendly). No heap.
#[derive(Clone, Copy)]
pub enum AzFastSpecConst {
    // const_term + sum coeff_i * flag_i, flags are 0/1 u8
    BooleanLinSmall { const_term: i8, flags: [(usize, i8); 3], used: u8 },
    U64Var { idx: usize },
    U64AndSignVar { idx: usize },
}

#[derive(Clone, Copy)]
pub enum BzFastSpecConst {
    U64Diff { lhs: usize, rhs: usize },
    U64Var { idx: usize },
    U64AndSignVar { idx: usize },
    U64PlusImm { u64_idx: usize, imm_idx: usize },
    U128Var { idx: usize },
}

#[derive(Clone)]
pub struct TypedConstraintConst<A: ConstLCAny, B: ConstLCAny, C: ConstLCAny> {
    pub a: A,
    pub b: B,
    pub c: C,
    pub a_type: AzType,
    pub b_type: BzType,
    pub c_type: CzType,
    pub fast_a: Option<AzFastSpecConst>,
    pub fast_b: Option<BzFastSpecConst>,
}
```

### Const Indices
Define a single const map of `JoltR1CSInputs` to indices that mirrors `ALL_R1CS_INPUTS` ordering. Add a test that asserts equality at runtime to prevent drift.

```rust
pub mod inputs_idx {
    pub const LEFT_INSTRUCTION_INPUT: usize = 0;
    pub const RIGHT_INSTRUCTION_INPUT: usize = 1;
    pub const PRODUCT: usize = 2;
    // ... fill out to match ALL_R1CS_INPUTS exactly ...
}
```

### Static Table
`UNIFORM_R1CS` is a `&'static [TypedConstraintConst<...>]` where each row is a literal. Example (illustrative only):

```rust
pub const NUM_R1CS_CONSTRAINTS: usize = /* literal */;

pub static UNIFORM_R1CS: &[TypedConstraintConst<ConstLC<1>, ConstLC<2>, ConstLC<0>>] = &[
    TypedConstraintConst {
        a: ConstLC { terms: [ConstTerm { input_index: inputs_idx::LEFT_INSTRUCTION_INPUT, coeff: 1 }], has_const: false, const_coeff: 0 },
        b: ConstLC { terms: [
            ConstTerm { input_index: inputs_idx::LEFT_INSTRUCTION_INPUT, coeff: 1 },
            ConstTerm { input_index: inputs_idx::RS1_VALUE, coeff: -1 },
        ], has_const: false, const_coeff: 0 },
        c: ConstLC { terms: [], has_const: false, const_coeff: 0 },
        a_type: AzType::U5,
        b_type: BzType::U64AndSign,
        c_type: CzType::Zero,
        fast_a: Some(AzFastSpecConst::BooleanLinSmall { const_term: 0, flags: [(inputs_idx::LEFT_INSTRUCTION_INPUT, 1), (0,0), (0,0)], used: 1 }),
        fast_b: Some(BzFastSpecConst::U64Diff { lhs: inputs_idx::LEFT_INSTRUCTION_INPUT, rhs: inputs_idx::RS1_VALUE }),
    },
    // ... more rows ...
];
```

## Streaming Evaluation
Introduce an input abstraction to allow either materialized or on-demand access to inputs per row.

```rust
pub trait InputProvider<F: JoltField> {
    fn get_u8(&self, idx: usize, row: usize) -> u8;           // for flags / small u5
    fn get_u64(&self, idx: usize, row: usize) -> u64;         // for u64 lanes
    fn get_u64_and_sign(&self, idx: usize, row: usize) -> U64AndSign;
    fn get_i128(&self, idx: usize, row: usize) -> i128;       // for signed lanes
    fn get_u128(&self, idx: usize, row: usize) -> u128;       // for u128 lanes
}
```

- `MaterializedProvider`: wraps `&[MultilinearPolynomial<F>]` and forwards to typed getters (today’s behavior).
- `StreamingProvider`: computes all virtual inputs per row directly from `trace`/`preprocessing` without creating full-length vectors, and forwards committed columns to their materialized storage.

### Row Evaluation (fallback preserved)
- Try `fast_a/fast_b` when present and compatible with the declared lane.
- Otherwise, evaluate the small fixed LC by unrolling ≤ 5 terms into the typed lane (`AzType/BzType/CzType`).
- `Cz` uses the generic typed LC path; no special handling needed.

## Materialization for Key/Verifier
- Build sparse matrices from `UNIFORM_R1CS` (iterate fixed terms and constants) to produce `UniformR1CS` for keygen and verifier paths. This keeps the rest of the system unchanged.

## Migration Plan
1. Add `inputs_idx` and runtime tests that assert equality with `ALL_R1CS_INPUTS`.
2. Introduce `ConstLC`, `TypedConstraintConst`, and const FastSpec enums.
3. Transcribe `constraints.rs` into `UNIFORM_R1CS` via helper macros that mirror today’s `constrain_*` APIs but emit const rows.
4. Implement `InputProvider`, `MaterializedProvider`; integrate evaluation into the prover precompute path.
5. Optional: implement `StreamingProvider` for virtual inputs to avoid full witness expansion.
6. Add a feature flag `const_r1cs` to toggle between the static and dynamic paths; ensure bit-for-bit identical outputs in tests.

## Testing Strategy
- Unit tests verifying that `inputs_idx::*` match `JoltR1CSInputs::to_index()`.
- Property tests comparing const-evaluated rows (both fast and fallback) with the existing LC evaluators on random rows within type bounds.
- End-to-end tests ensuring `UNIFORM_R1CS` and the dynamic builder produce identical constraints and prover outputs.

## Performance Notes
- Removing dynamic allocation for constraints and using unrolled LC eliminates iterator overhead and branches in hot loops.
- FastSpec allows direct boolean arithmetic on flags (0/1 u8) and direct passthrough/diff of common operands.
- Streaming virtual inputs reduces memory traffic by avoiding full-length polynomial allocation for values derivable from `trace` on-the-fly.

## Risks & Caveats
- Input index drift: must be guarded by runtime tests against `ALL_R1CS_INPUTS`.
- LC arity cap: default to 5; bump if future constraints exceed this.
- Authoring ergonomics: prefer macros to reduce manual errors when defining const rows.
- Committed columns still need materialization for PCS; streaming applies primarily to virtual inputs.

## Constraint Analysis (UPDATED)

### Actual Constraint Count: 28 constraints
Detailed breakdown of `JoltRV32IMConstraints::uniform_constraints`:

1. **constrain_eq_conditional** calls: 18 constraints
   - LeftOperandIsRs1Value → LeftInstructionInput == Rs1Value
   - LeftOperandIsPC → LeftInstructionInput == UnexpandedPC  
   - !(LeftOperandIsRs1Value || LeftOperandIsPC) → LeftInstructionInput == 0
   - RightOperandIsRs2Value → RightInstructionInput == Rs2Value
   - RightOperandIsImm → RightInstructionInput == Imm
   - !(RightOperandIsRs2Value || RightOperandIsImm) → RightInstructionInput == 0
   - Load → RamReadValue == RamWriteValue
   - Load → RamReadValue == RdWriteValue
   - Store → Rs2Value == RamWriteValue
   - AddOperands → RightLookupOperand == LeftInstructionInput + RightInstructionInput
   - SubtractOperands → RightLookupOperand == LeftInstructionInput - RightInstructionInput + 0x10000000000000000
   - MultiplyOperands → RightLookupOperand == Product
   - !(AddOperands || SubtractOperands || MultiplyOperands || Advice) → RightLookupOperand == RightInstructionInput
   - Assert → LookupOutput == 1
   - WriteLookupOutputToRD → RdWriteValue == LookupOutput
   - WritePCtoRD → RdWriteValue == UnexpandedPC + 4 - 2*IsCompressed
   - ShouldJump → NextUnexpandedPC == LookupOutput
   - ShouldBranch → NextUnexpandedPC == UnexpandedPC + Imm
   - !(ShouldBranch || Jump) → NextUnexpandedPC == UnexpandedPC + 4 - 4*DoNotUpdateUnexpandedPC - 2*IsCompressed + 2*CompressedDoNotUpdateUnexpPC
   - InlineSequenceInstruction → NextPC == PC + 1

2. **constrain_if_else** calls: 2 constraints
   - (Load || Store) ? (Rs1Value + Imm) : 0 == RamAddress
   - (AddOperands || SubtractOperands || MultiplyOperands) ? 0 : LeftInstructionInput == LeftLookupOperand

3. **constrain_prod** calls: 8 constraints
   - RightInstructionInput * LeftInstructionInput == Product
   - Rd * WriteLookupOutputToRD == WriteLookupOutputToRD
   - Rd * Jump == WritePCtoRD
   - Jump * (1 - NextIsNoop) == ShouldJump
   - Branch * LookupOutput == ShouldBranch
   - IsCompressed * DoNotUpdateUnexpandedPC == CompressedDoNotUpdateUnexpPC

### Maximum Linear Combination Terms
Analysis of A/B/C terms across all 28 constraints:

**A terms (max 4 terms):**
- Single flags: 1 term
- Complex boolean: up to 4 terms (1 - flag1 - flag2 - flag3 - flag4)

**B terms (max 5 terms):**
- Single variables: 1 term
- Variable differences: 2 terms  
- Complex PC update: 5 terms (UnexpandedPC + 4 - 4*flag1 - 2*flag2 + 2*product)

**C terms (max 1 term):**
- Zero: 0 terms
- Single variable: 1 term

**Overall maximum: 5 terms** (in B term of PC update constraint)

## Simplified Plan (NO TYPES OR FASTSPEC)

### Phase 1: Basic Const Infrastructure
1. Create `inputs_idx` module with const indices matching `ALL_R1CS_INPUTS`
2. Implement basic `ConstLC<N>` and `ConstTerm` structs (no types, no FastSpec)
3. Add runtime verification tests for index consistency

### Phase 2: Static Constraint Table
1. Manually unroll all 28 constraints into static `UNIFORM_R1CS` table
2. Cap LC terms at 5 to handle the most complex constraint
3. Generate `&'static [ConstraintConst]` with literal Az/Bz/Cz terms

### Phase 3: Integration & Testing
1. Add `const_r1cs` feature flag
2. Implement basic constraint evaluation (no streaming, no FastSpec)
3. Bit-for-bit comparison tests between dynamic and static paths

## Milestones
- M1: Const data structures + `inputs_idx` + verification tests
- M2: Manual constraint unrolling into static table (all 28 constraints)
- M3: Basic evaluation integration with feature flag
- M4: Comprehensive testing and validation
