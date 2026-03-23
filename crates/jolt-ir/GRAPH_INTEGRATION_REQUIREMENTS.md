# Protocol Graph: Integration Requirements from jolt-zkvm/jolt-verifier

These requirements come from analyzing how the prover (jolt-zkvm) and verifier
(jolt-verifier) will consume the protocol graph. They should be incorporated
into the graph types during implementation.

## 1. Witness Hint on Vertices (Computation Metadata)

The protocol graph captures **what** to prove. The **how** (which evaluator to
use) is a computation concern owned by the kernel backend and consumed by
jolt-zkvm.

The kernel backend (jolt-compute / jolt-cpu) should provide a mapping from
vertex properties (formula shape, degree, multi-phase structure) to the
appropriate computation strategy. jolt-zkvm then dispatches based on this.

The graph doesn't need a `VertexKind` enum — the vertex's structural
properties (formula, degree, num_vars, produced claim count) are sufficient
for the backend to determine the computation strategy. For example:

- Formula is a linear combination of polynomials → `claim_reduction_witness`
- Formula has RA product structure (product of d chunks) → `RaVirtualCompute`
- Vertex spans multiple variable groups (address + cycle) → `SegmentedEvaluator`
- Formula is `h² − h` → `booleanity_witness`

However, multi-phase structure (address→cycle phase transitions) IS a
structural property that the graph should capture, since it affects num_vars
per phase and the challenge point composition. This should be on the vertex:

```rust
struct SumcheckVertex {
    // ... existing fields ...
    /// If the vertex has multiple variable-binding phases (e.g., address then
    /// cycle), list them here. Single-phase vertices have one entry.
    phases: Vec<Phase>,
}

struct Phase {
    num_vars: NumVars,
    /// Which variable group this phase binds (for point composition).
    variable_group: VariableGroup,
}

enum VariableGroup {
    Cycle,
    Address,
    Combined, // single-phase over full (address || cycle) domain
}
```

## 2. Standardized Polynomial Tags

Replace opaque `u64` tags with a typed `PolynomialId` enum in jolt-ir that
both the graph and jolt-zkvm can reference. This enum is the canonical
namespace for all Jolt polynomials:

```rust
/// Canonical polynomial identifiers. Used by the protocol graph and
/// by jolt-zkvm to look up polynomial data in PolynomialTables.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum PolynomialId {
    // Committed
    RamInc,
    RdInc,
    InstructionRa(usize),   // parameterized by chunk index
    BytecodeRa(usize),
    RamRa(usize),
    TrustedAdvice,
    UntrustedAdvice,

    // Virtual (from R1CS)
    RamReadValue,
    RamWriteValue,
    RamAddress,
    RamVal,
    RamValFinal,
    RdWriteValue,
    Rs1Value,
    Rs2Value,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    LookupOutput,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,
    HammingWeight,
    UnexpandedPc,
    Imm,

    // Trace-derived
    IsRdNotZero,
    WriteLookupToRdFlag,
    JumpFlag,
    BranchFlag,
    NextIsNoop,
    LeftIsRs1,
    LeftIsPc,
    RightIsRs2,
    RightIsImm,
    NextUnexpandedPc,
    NextPc,
    NextIsVirtual,
    NextIsFirstInSequence,
}
```

This replaces `poly::RAM_INC = 100u64` etc. The `PolynomialTables` in
jolt-zkvm will implement `Index<PolynomialId>` to look up table data.

The existing `u64` tags in `jolt_ir::zkvm::tags` should be deprecated in
favor of this enum.

## 3. Claims Reference ClaimIds Directly (No Indirection)

The `ClaimFormula`'s `OpeningBinding` should reference `ClaimId`s directly,
not `(PolynomialId, SumcheckId)` pairs. This eliminates the resolution step:

```rust
struct OpeningBinding {
    /// Variable index in the expression.
    var_id: u32,
    /// Which claim's evaluation value this variable reads.
    claim_id: ClaimId,
}
```

When evaluating the formula, the prover/verifier looks up `eval_cache[claim_id]`
and passes it as `openings[var_id]`. No intermediate resolution needed.

The graph construction populates these `ClaimId`s when connecting vertices.
The `ClaimId` is known at graph build time because the graph explicitly
tracks which vertex produces which claims.

## 4. Public Polynomials as Weighting

Each sumcheck vertex uses a "public polynomial" (visible to both prover and
verifier) for weighting. Name these explicitly:

```rust
enum PublicPolynomial {
    /// Standard equality polynomial: eq(r, x).
    Eq,
    /// Successor equality polynomial: eq+1(r, x). Used by shift.
    EqPlusOne,
    /// Less-than polynomial: LT(r, x). Used by RegistersValEval.
    Lt,
    /// Precomputed from prior challenge values. The vertex's formula
    /// encodes how to compute it; the prover materializes it.
    Derived,
}
```

Add `pub weighting: PublicPolynomial` to `SumcheckVertex`. The prover uses
this to build the appropriate weighting table. The verifier uses it to
evaluate the weighting polynomial at the challenge point for the output check.

## 5. Per-Stage Challenge Squeeze Specification

Each stage squeezes challenges from the transcript before its sumcheck runs.
These are the γ values, eq points, and domain separators that the sumcheck
instances need. The order must match between prover and verifier.

Add a `pre_squeeze: Vec<ChallengeSpec>` field to `Stage`:

```rust
enum ChallengeSpec {
    /// Squeeze one scalar from the transcript.
    Scalar { label: &'static str },
    /// Squeeze a vector of `dim` scalars.
    Vector { label: &'static str, dim: NumVars },
    /// Squeeze one scalar and compute `n` powers: [1, γ, γ², ..., γ^{n-1}].
    GammaPowers { label: &'static str, count: SymbolicExpr },
}
```

Both prover and verifier walk `stage.pre_squeeze` to derive challenges in
the same order. The `label` is for debugging/logging only — Fiat-Shamir
binding comes from the transcript state, not the label.

This replaces the hand-matched squeeze code in `stages.rs` and `verify.rs`.

## 6. Spartan as a Sumcheck Stage

Spartan (S1) is modeled as a stage containing sumcheck vertices, not as a
special case outside the graph. Specifically:

- The outer sumcheck is a `SumcheckVertex` with a uni-skip first round
  (captured via a flag or phase structure on the vertex).
- The inner sumcheck is another vertex in the same stage.
- The stage produces challenge vectors `r_x` (outer) and `r_y` (inner).
- Virtual polynomial evaluations at `r_cycle = Slice(r_y, 0..log_T)` are
  the produced claims.

The uni-skip is a computation optimization — the vertex's structural
properties (degree, num_vars, formula) are the same whether or not uni-skip
is used. The prover chooses to use uni-skip based on the vertex's degree
and domain size.

```rust
// Spartan stage contains two vertices
Stage {
    id: S1,
    vertices: vec![outer_sumcheck_vertex, inner_sumcheck_vertex],
    challenge_point: ChallengePoint { num_vars: log_rows + log_cols },
    pre_squeeze: vec![/* Spartan-specific preamble */],
}
```

The produced claims are virtual polynomial evaluations (RamReadValue,
RamWriteValue, etc.) at `Slice(Challenges(S1), 0..log_T)`. These flow
to S2/S3 vertices as consumed claims.

## Assessment and Design Decisions

### Req 1 — Multi-phase: Accepted with refinement

`Combined` should not be the default. Single-phase vertices should use
`phases: vec![Phase { num_vars, variable_group: Cycle }]` — one entry, not
a `Combined` variant. `Combined` implies something structural (binding over the
full address || cycle domain in a single pass), which is rare. Most vertices
bind only cycle variables.

### Req 2 — PolynomialId enum: Accepted

Highest-value change alongside Req 5. Eliminates entire class of tag-mismatch
bugs. The enum should include `SpartanWitness` — it IS a committed polynomial
(appears in the commitment strategy, enters the transcript) even though it is
NOT opened as a separate PCS claim (its columns are embedded in the Dory
commitment matrix; the single opening at the unified point covers it
implicitly).

### Req 3 — ClaimId-direct bindings: Accepted

Clean extension of the existing `OpeningBinding` in `jolt-ir/src/claim.rs`.
The `ClaimId` is known at graph build time since the graph explicitly tracks
vertex → produced claims. No design concerns.

### Req 4 — PublicPolynomial: Accepted, Derived stays opaque

The `Derived` variant should remain opaque in the graph. In S6's `V_inc_cr`,
the weighting is a linear combination of eq tables at different points
(`eq(ep_s2, j) + γ · eq(ep_s4, j)`) — more structured than a generic
"derived," but that structure is a computation concern. The graph's structural
concern is that the vertex has a weighting; whether it's `Eq` or `Derived`
affects how the verifier evaluates it, but the claim flow is identical either
way. If we later need verifier-side dispatch on `Derived` sub-types, we can
refine then.

### Req 5 — ChallengeSpec: Accepted with type refinement

`GammaPowers { count: usize }` should use `count: NumVars` (or a similar
symbolic type) since the count can depend on config — e.g., `3 * D_total`
for S7's gamma powers. Using `usize` would force eager resolution at graph
build time, which is fine for now but loses the symbolic parameterization
that the rest of the graph preserves.

### Req 6 — Spartan as sumcheck stage: Accepted, uni-skip stays off-graph

Breaking V_spartan into outer + inner sumcheck vertices gives full visibility
into Spartan's internal claim flow. The uni-skip optimization stays off the
graph — it's a computation choice that doesn't affect claim structure. The
verifier's sumcheck verify function already handles uni-skip transparently
(the first round polynomial has a different shape, but the verification
equation is the same). If we later want to analyze uni-skip as a structural
property, we can add it as annotation on the vertex.

### PointNormalization stays as a vertex

Req 6 implies `Sumcheck` should be the only non-terminal vertex kind (since
Spartan becomes sumcheck vertices). But `PointNormalization` is not a
sumcheck — it's a verifier-side algebraic identity. It should remain as a
vertex rather than becoming an implicit claim transformation, because:
1. It's explicit and auditable in the graph
2. The verifier must know about it (applies Lagrange scaling)
3. It appears in the claim completeness check (consumes short-point claims,
   produces unified-point claims)

## Summary

| # | Requirement | Where | Impact | Decision |
|---|---|---|---|---|
| 1 | Multi-phase structure on vertices | `SumcheckVertex.phases` | Prover dispatches evaluator | Accepted; drop `Combined` default |
| 2 | Typed `PolynomialId` enum | `jolt_ir::PolynomialId` | Replaces opaque u64 tags | Accepted; include `SpartanWitness` |
| 3 | `ClaimId`-direct formula bindings | `OpeningBinding.claim_id` | No resolution step in eval | Accepted |
| 4 | `PublicPolynomial` on vertices | `SumcheckVertex.weighting` | Prover/verifier build/eval weight | Accepted; `Derived` stays opaque |
| 5 | `ChallengeSpec` on stages | `Stage.pre_squeeze` | Transcript parity by construction | Accepted; `count` → `NumVars` |
| 6 | Spartan as sumcheck stage | `Stage` with sumcheck vertices | Unified graph traversal | Accepted; uni-skip off-graph |
