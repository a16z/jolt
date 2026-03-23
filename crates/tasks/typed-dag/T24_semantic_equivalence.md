# T24: Semantic Equivalence with jolt-core

**Status**: `[ ]` Not started
**Depends on**: Protocol graph (jolt-ir), graph_prover.rs, graph_verify.rs
**Blocks**: E2E muldiv with real Spartan
**Estimated scope**: Large

## Objective

Achieve exact semantic equivalence between the graph-driven pipeline and
the jolt-core reference implementation. Every sumcheck instance must be
present, every claimed_sum must match, every output formula check must
pass, and the Fiat-Shamir transcript must be identical.

## Gap List (ordered by priority)

### Critical (breaks soundness)

#### G1: Output formula check missing in verifier
**File**: `jolt-verifier/src/graph_verify.rs`
**Problem**: After `BatchedSumcheckVerifier::verify()` returns `(final_eval, challenges)`,
the verifier does NOT check `Σ_j α^j · pad_j · w(eq_point, eval_point) · g(evals_j) == final_eval`.
This means any evaluations the prover provides are accepted unchecked for virtual polys.
**Fix**: After unpacking evals, iterate over each vertex in the stage. For each sumcheck vertex:
1. Compute `w_eval` by evaluating the weighting polynomial (`Eq`, `EqPlusOne`, `Lt`) at the eval point.
2. Compute `g_eval` by evaluating `vertex.formula.definition` with the unpacked evals as openings
   and stage challenges as challenge values.
3. Accumulate `α^j · pad_j · w_eval · g_eval` and compare against `final_eval`.

The `formula.definition.evaluate()` already exists. The challenge values come from
`squeeze_stage_challenges`. The `α` and padding factors come from the batched sumcheck
protocol (same as `BatchedSumcheckVerifier` internally computes).

#### G2: S1 virtual evals not in proof
**File**: `jolt-verifier/src/graph_verify.rs`, `jolt-zkvm/src/graph_prover.rs`
**Problem**: The verifier needs Spartan virtual evals (RamReadValue, RamWriteValue, etc.)
to compute S2/S3 input claims. Currently zeroed.
**Fix**: The prover already evaluates them and stores in the eval cache. Add them to the
proof as a dedicated field `spartan_virtual_evals: Vec<F>` on `JoltProof`, OR include
them in the S1 `StageProof.evals`. The verifier reads them from the proof, populates
the eval cache, and downstream `InputClaim::Formula` evaluations use them.

The Spartan witness opening proof (PCS) will ultimately verify these evals. For now,
the verifier trusts them from the proof.

### Correctness (wrong values)

#### G3: BruteForce claimed_sum returns zero
**File**: `jolt-zkvm/src/graph_prover.rs`
**Problem**: `InputClaim::BruteForce` → `F::zero()`. Affects PV, RA virtual, shift,
and any vertex whose claimed_sum can't be derived from a formula over prior evals.
**Fix**: For each BruteForce vertex, compute `Σ_x w(r, x) · g(x)` over the full
polynomial table. This requires:
1. Materializing the weighting polynomial table (eq, eq+1, etc.)
2. Evaluating the formula's expression at each hypercube point
3. Summing

This is O(N) per vertex — same as jolt-core does for these vertices (they're NOT
O(1) in jolt-core either — `input_claim()` for PV involves Lagrange evaluation,
and RA virtual sums over the full domain).

Implementation: In `build_witness()`, we already materialize the weight table and
collect the poly tables. The brute-force sum is `Σ_j w[j] · g(poly_tables, j)`.
Compute it alongside the witness construction.

#### G4: Derived weighting returns unit
**File**: `jolt-zkvm/src/graph_prover.rs`
**Problem**: `PublicPolynomial::Derived` → `vec![1; N]`. Wrong for IncCR and RamValCheck.
**Fix**:
- **IncCR**: weight = `eq(r_s2, x) + γ·eq(r_s4, x)` for ram_inc, and `γ²·eq(r_s4, x) + γ³·eq(r_s5, x)` for rd_inc.
  These eq points come from the eval cache's stored points for prior stages.
  The vertex's `deps` reference the prior claims whose points we need.
- **RamValCheck**: weight = `eq(r, x) · (LT(r', x) + γ)` where r is the RW challenge point
  and γ is the val_check gamma.

For both: the vertex's `deps` claims carry the prior stage points. The eval cache stores
points per stage. The graph's `ChallengeLabel` on the vertex tells us which gamma to use.

Implementation: When `weighting == Derived`, inspect the vertex's produced claims and
deps to determine the weight construction formula. This is vertex-specific logic that
MUST exist somewhere — either as a callback or a match on produced PolynomialIds.

#### G5: Multi-phase vertices dispatched as single-phase
**File**: `jolt-zkvm/src/graph_prover.rs`
**Problem**: RamRW, RegistersRW, InstructionReadRaf, BytecodeReadRaf, Booleanity all have
`phases: [Phase { Cycle }, Phase { Address }]` but `build_witness()` ignores phases entirely.
**Fix**: When `sv.phases.len() > 1`, use `SegmentedEvaluator` instead of `formula_witness`:
1. Build a `KernelEvaluator` for phase 1 (first variable group)
2. Define a transition callback that takes phase 1 challenges and builds the phase 2 evaluator
3. Wrap in `SegmentedEvaluator`
4. The batched sumcheck sees it as a normal `SumcheckCompute`

The `SegmentedEvaluator` already exists in `evaluators/segmented.rs`.

### Missing graph vertices

#### G6: RegistersValEval (S5)
**File**: `jolt-ir/src/protocol/build.rs`
**Problem**: No vertex for RegistersValEval in `build_s5()`. jolt-core's S5 has this instance.
**Formula**: `Σ_j LT(r_cycle, j) · inc(j) · wa(j) = val_claim`
**Fix**: Add `add_composition()` in `build_s5()` with:
- Weighting: `PublicPolynomial::Lt`
- Produces: claims on `RdInc`, `RdWa` at S5 challenge point
- Formula: `registers_val_evaluation()` from `claims::registers`
- InputClaim: `Formula` reading from S4's RegistersRW claims

#### G7: InstructionReadRaf (S5)
**File**: `jolt-ir/src/protocol/build.rs`
**Problem**: No vertex for InstructionReadRaf. jolt-core's S5 has this instance.
**Formula**: Complex multi-phase RA product with RAF evaluation.
**Fix**: Add vertex with:
- Weighting: `PublicPolynomial::Eq`
- Phases: `[Phase { Address }, Phase { Cycle }]` (multi-phase)
- Produces: claims on `InstructionRa(0..d)` at S5 challenge point
- Degree: `d_instr + 2`
- Formula: from `claims::instruction` or a new definition

#### G8: BytecodeReadRaf (S6)
**File**: `jolt-ir/src/protocol/build.rs`
**Problem**: No vertex for BytecodeReadRaf. jolt-core's S6 has this instance.
**Formula**: Multi-stage folded bytecode lookup with RAF.
**Fix**: Add vertex with:
- Weighting: `PublicPolynomial::Eq`
- Phases: `[Phase { Address }, Phase { Cycle }]` (multi-phase)
- Produces: claims on `BytecodeRa(0..d)` at S6 challenge point
- Degree: `d_bc + 1`

#### G9: S6 RA virtual instances (cycle-bound)
**File**: `jolt-ir/src/protocol/build.rs`
**Problem**: jolt-core's S6 has InstructionRaVirtual and RamRaVirtual operating in the
cycle-only domain. These are different from S2's RA virtual (which uses Spartan's r_cycle).
S6's RA virtual uses the Booleanity challenge point.
**Fix**: Add vertices in `build_s6()` with:
- Same formula as S2's RA virtual
- But at S6's challenge point (not S2's)
- Produces: claims on RA polys at S6 point

### Stage placement

#### G10: RamRaCR placement
**File**: `jolt-ir/src/protocol/build.rs`
**Problem**: RamRaCR is in S7 in our graph. In jolt-core it's in S5.
**Fix**: Move RamRaCR to `build_s5()`. This changes the point topology —
S5's challenge point becomes the point for RAM RA reduction claims,
which flow to S7 HammingWeightCR.

### Performance (correct but slow)

#### G11: RA virtual uses formula_witness instead of RaVirtualCompute
**File**: `jolt-zkvm/src/graph_prover.rs`
**Problem**: The generic `build_witness()` builds a `formula_witness` for RA virtual
vertices. This evaluates the product formula term-by-term. jolt-core uses `RaVirtualCompute`
with Toom-Cook grid evaluation which is much faster for high fan-in products.
**Fix**: In `build_witness()`, detect RA virtual vertices (degree = d+1, single product
term over d polynomials) and dispatch to `RaVirtualCompute` instead.

#### G12: No uni-skip
**File**: `jolt-zkvm/src/graph_prover.rs`
**Problem**: Spartan outer (S1) and ProductVirtual (S2) use univariate skip in jolt-core.
Our prover doesn't. This doesn't affect correctness but changes the transcript (different
first-round polynomial shape) so the verifier would reject proofs if it expects uni-skip.
**Fix**: For vertices where uni-skip applies, compute the analytic first-round polynomial
and set it via `KernelEvaluator::set_first_round_override()`.

#### G13: Shift EqPlusOne point combination
**File**: `jolt-zkvm/src/graph_prover.rs`
**Problem**: jolt-core's shift sumcheck combines `eq+1(r_outer, j)` and `eq+1(r_product, j)`
tables (outer from S1, product from S2 PV). Our graph vertex has a single eq+1 point.
**Fix**: The shift vertex's weighting polynomial should use the combined eq+1 table.
The graph should either:
- Carry both source points in the `PublicPolynomial::EqPlusOne` variant
- Or use `Derived` weighting with explicit construction

## Execution Order

1. **G2** (S1 evals in proof) + **G1** (output formula check) — these unblock verify round-trip
2. **G3** (brute-force claimed_sum) + **G4** (derived weighting) — these unblock correct proving
3. **G6-G10** (missing vertices + stage placement) — graph completeness
4. **G5** (multi-phase dispatch) — correct witness for multi-phase vertices
5. **G11-G13** (performance + uni-skip) — optimization + transcript parity

## Acceptance Criteria

- [ ] All 18+ sumcheck vertices present in the graph
- [ ] Verifier checks output formula for every stage
- [ ] S1 virtual evals carried in proof and verified
- [ ] BruteForce claimed_sums computed from tables
- [ ] Derived weightings precomputed correctly
- [ ] Multi-phase vertices use SegmentedEvaluator
- [ ] prove_from_graph → verify_from_graph round-trip passes for synthetic trace
- [ ] prove_from_graph → verify_from_graph round-trip passes for muldiv guest
