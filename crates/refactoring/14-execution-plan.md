# Execution Plan: Compiler-Driven Pipeline

Concrete phased plan with file-level scope. Each phase has a validation checkpoint.
Phases are ordered by dependency; parallelism opportunities noted.

## Scope Summary

| Action | Crate | Lines |
|--------|-------|-------|
| **New crate** | jolt-r1cs | ~1,200 |
| **Rewrite** | jolt-ir protocol/ (build.rs, validate.rs, types.rs) | ~4,100 → ~2,000 |
| **Add** | jolt-compute (make_witness, sparse, challenge decoupling) | +~150 |
| **Rewrite** | jolt-zkvm prover.rs + delete evaluators/ | 4,186 → ~200 |
| **Rewrite** | jolt-verifier verify.rs + proof.rs | ~700 → ~400 |
| **Delete** | jolt-spartan prover/verifier/proof | ~1,693 |
| **Keep** | jolt-ir expr/claim/composition, jolt-sumcheck, jolt-cpu, jolt-metal, jolt-witness, jolt-poly, jolt-field, jolt-crypto, jolt-transcript, jolt-openings, jolt-dory | ~unchanged |

**Net: ~3,400 lines deleted, ~1,400 lines new, ~2,600 lines rewritten.**

---

## Phase 0: Extract jolt-r1cs (independent, parallelizable)

### Goal
Pure R1CS data crate. No protocol logic. No sumcheck. No transcript.

### Create: crates/jolt-r1cs/

**Move from jolt-spartan:**

| Source | Destination | Lines | Content |
|--------|-------------|-------|---------|
| `jolt-spartan/src/r1cs.rs` | `jolt-r1cs/src/r1cs.rs` | 202 | `R1CS<F>` trait, `SimpleR1CS<F>` |
| `jolt-spartan/src/key.rs` | `jolt-r1cs/src/key.rs` | 151 | `SpartanKey<F>` (dense MLE storage) |
| `jolt-spartan/src/uniform_key.rs` | `jolt-r1cs/src/uniform_key.rs` | 599 | `UniformSpartanKey<F>` (per-cycle sparse matrices) |
| `jolt-spartan/src/error.rs` | `jolt-r1cs/src/error.rs` | 41 | `SpartanError` enum |

**New in jolt-r1cs:**

| File | Lines | Content |
|------|-------|---------|
| `src/lib.rs` | ~30 | Module exports |
| `src/edge_transform.rs` | ~200 | `combined_partial_evaluate(r1cs, r_x) -> Vec<F>` — extracted from uniform_prover.rs |

**Dependencies:** `jolt-field`, `jolt-poly`, `serde`

**Does NOT depend on:** `jolt-sumcheck`, `jolt-transcript`, `jolt-openings`, `jolt-compute`

### Also move: uni_skip.rs → jolt-sumcheck

Move `jolt-spartan/src/uni_skip.rs` (311 lines) into `jolt-sumcheck` as a utility
module. It's a sumcheck optimization (analytical first round for zero-checks), not
Spartan-specific.

### Validation

```bash
cargo clippy -p jolt-r1cs --message-format=short -q -- -D warnings
cargo nextest run -p jolt-r1cs --cargo-quiet
# Existing jolt-spartan tests migrated to jolt-r1cs
```

---

## Phase 1: Backend factory (independent, parallelizable with Phase 0)

### Goal
`ComputeBackend` gains `make_witness` + `SumcheckWitness` + sparse primitives.
Challenges decoupled from kernel compilation.

### Modify: jolt-compute/src/traits.rs

**Add associated types:**

```rust
type SumcheckWitness<F: Field>: SumcheckCompute<F> + Send;
type SparseBuffer<F: Field>: Send + Sync;
```

**Add methods:**

```rust
// Witness factory — replaces build_witness() in jolt-zkvm
fn make_witness<F: Field>(
    &self,
    kernel: &Self::CompiledKernel<F>,
    inputs: Vec<Self::Buffer<F>>,
    eq: EqInput<'_, Self, F>,
    challenges: &[F],
) -> Self::SumcheckWitness<F>;

// Sparse primitives — replaces SparseRwEvaluator in jolt-zkvm
fn upload_sparse<F: Field>(&self, entries: &[(usize, Vec<F>)]) -> Self::SparseBuffer<F>;
fn sparse_reduce<F: Field>(
    &self,
    entries: &Self::SparseBuffer<F>,
    eq: &Self::Buffer<F>,
    kernel: &Self::CompiledKernel<F>,
    challenges: &[F],
    num_evals: usize,
) -> Vec<F>;
fn sparse_bind<F: Field>(
    &self,
    entries: &mut Self::SparseBuffer<F>,
    challenge: F,
    order: BindingOrder,
);
```

**Change `compile_kernel` signature:**

```rust
// Remove compile_kernel_with_challenges. Challenges are runtime, not compile-time.
fn compile_kernel<F: Field>(&self, formula: &CompositionFormula) -> Self::CompiledKernel<F>;
// Challenges passed to make_witness() and pairwise_reduce() at runtime.
```

### Modify: jolt-cpu/src/backend.rs

**Implement new associated types:**

| Type | CPU impl |
|------|----------|
| `SumcheckWitness<F>` | `CpuSumcheckWitness<F>` — new struct |
| `SparseBuffer<F>` | `Vec<SparseEntry<F>>` |

**Move from jolt-zkvm evaluators/ into jolt-cpu:**

| Source | Destination | Lines | Content |
|--------|-------------|-------|---------|
| `jolt-zkvm/src/evaluators/kernel.rs` | `jolt-cpu/src/witness.rs` | ~600 | `CpuSumcheckWitness` (was `KernelEvaluator`) |
| `jolt-zkvm/src/evaluators/mles_product_sum.rs` | already in `jolt-cpu/src/toom_cook.rs` | 0 | Already deduplicated |

`CpuSumcheckWitness<F>` owns:
- Input buffers (`Vec<Vec<F>>`)
- Compiled kernel (`CpuKernel<F>`)
- Reconstruction state (StandardGrid or ToomCook — internal, not exposed)
- Eq state (tensor/flat/none — internal)
- Binding order
- Round counter

Implements `SumcheckCompute<F>` via:
- `round_polynomial()` → `backend.pairwise_reduce(...)` + reconstruct
- `bind(challenge)` → `backend.interpolate_pairs_batch_inplace(...)`

### Modify: jolt-cpu/src/lib.rs

**Remove `compile_with_challenges`**. The public API becomes:

```rust
pub fn compile<F: Field>(formula: &CompositionFormula) -> CpuKernel<F>;
```

Challenge resolution moves into `CpuSumcheckWitness::new()` which takes challenges
as a runtime parameter.

### Modify: jolt-metal/src/device.rs (later, not blocking)

Same pattern: `MetalSumcheckWitness<F>`, `MetalSparseBuffer<F>`. Can be stubbed
initially (delegate to CPU fallback via HybridBackend).

### Modify: jolt-compute/src/hybrid.rs

Add `HybridSumcheckWitness`, `HybridSparseBuffer` that delegate to primary or
fallback based on buffer sizes.

### Validation

```bash
cargo clippy -p jolt-compute --message-format=short -q -- -D warnings
cargo clippy -p jolt-cpu --message-format=short -q -- -D warnings
cargo nextest run -p jolt-cpu --cargo-quiet
# Port existing evaluator tests from jolt-zkvm to jolt-cpu
```

---

## Phase 2: Spartan dissolution (depends on Phase 0)

### Goal
Spartan is no longer special. Its sumchecks are regular graph vertices in the
existing `build_jolt_protocol()`. S1 disappears as a concept — the outer/inner
sumchecks are just vertices in stages.

### Modify: jolt-ir/src/protocol/build.rs

**Replace `build_spartan()` (~90 lines)** with explicit vertices:

1. **Outer sumcheck vertex**: `formula = Az·Bz - Cz`, `claimed_sum = Constant(0)`,
   `weighting = Eq`, `num_vars = log_rows`
2. **Product virtual remainder vertex**: handles uni-skip continuation
3. **EdgeTransform vertex**: calls `jolt_r1cs::combined_partial_evaluate(r1cs, r_x)`
   to materialize the inner polynomial
4. **Inner sumcheck vertex**: `formula = combined_row · witness`,
   `claimed_sum = linear(az_eval, bz_eval, cz_eval)`, `num_vars = log_cols`

**Add new `ClaimDefinition` functions** in `jolt-ir/src/zkvm/claims/spartan.rs`:

```rust
pub fn r1cs_outer() -> ClaimDefinition { ... }      // Az·Bz - Cz
pub fn r1cs_inner() -> ClaimDefinition { ... }      // combined_row · witness
```

**Add new vertex types** to `jolt-ir/src/protocol/types.rs`:

```rust
pub struct EdgeTransformVertex {
    pub id: VertexId,
    pub transform: TransformKind,
    pub inputs: Vec<InputBinding>,
    pub outputs: Vec<OutputBinding>,
}

pub enum TransformKind {
    CombinedPartialEvaluate,
    // Future: sparse→dense materialization, etc.
}
```

### Modify: jolt-ir/src/polynomial_id.rs

**Add R1CS virtual polynomial IDs:**

```rust
pub enum PolynomialId {
    // ... existing variants ...
    Az,
    Bz,
    Cz,
    CombinedRow,
}
```

### Modify: jolt-zkvm/src/prover.rs

**Remove S1 special case** (lines 222-254). The outer/inner sumchecks now flow
through the generic stage loop. The prover needs a `PolynomialSource` impl that
can provide Az, Bz, Cz (computed from R1CS × witness at runtime).

### Modify: jolt-verifier/src/verify.rs

**Remove S1 hardcoding** (lines 200-221). The Spartan sumcheck verifications now
flow through the generic stage loop. Remove `verify_spartan()` call.

### Modify: jolt-verifier/src/proof.rs

**Remove `spartan_proof` and `spartan_evals` fields.** Spartan sumcheck proofs
are now in `stage_proofs`. Spartan evals are now in `stage_proofs[*].evals`.

### Modify: jolt-verifier/src/key.rs

**Remove `spartan_key` from `JoltVerifyingKey`.** The R1CS key is now part of the
`PolynomialSource` or passed separately.

### Delete: jolt-spartan prover/verifier

| File | Lines | Action |
|------|-------|--------|
| `prover.rs` | 858 | DELETE |
| `uniform_prover.rs` | 450 | DELETE |
| `verifier.rs` | 209 | DELETE |
| `uniform_verifier.rs` | 116 | DELETE |
| `proof.rs` | 60 | DELETE |

Total: **1,693 lines deleted.**

jolt-spartan crate either deleted entirely (if all remaining code moved to
jolt-r1cs + jolt-sumcheck) or reduced to a thin re-export crate.

### Validation

```bash
cargo nextest run -p jolt-zkvm muldiv --cargo-quiet --features host
cargo nextest run -p jolt-zkvm muldiv --cargo-quiet --features host,zk
# This is the critical checkpoint. Spartan dissolution must not break E2E.
```

---

## Phase 3: Protocol types + compiler (depends on Phase 2)

### Goal
Replace `build_jolt_protocol()` with `jolt_protocol()` + `jolt_scheduling_hints()`
+ `compile()`. The protocol is pure math. Scheduling is data. The compiler validates
and assembles.

### Rewrite: jolt-ir/src/protocol/types.rs (~441 lines → ~500 lines)

**New types (replace existing ProtocolGraph/Staging/CommitmentStrategy):**

```rust
// Level 0: Protocol
pub struct PolynomialIdentity {
    pub name: &'static str,
    pub formula: CompositionFormula,
    pub polynomials: Vec<PolynomialId>,
    pub claimed_sum: IdentityClaim,
    pub domain: DomainSpec,
    pub weighting: Weighting,
    pub produces: Vec<PolynomialId>,
}

// Level 1: Claim Graph (mechanically expanded)
pub struct ClaimGraph {
    pub nodes: Vec<ClaimNode>,
    pub edges: Vec<ClaimEdge>,
}

pub enum ClaimNode {
    Sumcheck(SumcheckClaim),
    Commitment(CommitmentClaim),
    Opening(OpeningClaim),
    PointNorm(PointNormClaim),
}

// Level 2: Staged Graph (compiler output)
pub struct StagedGraph {
    pub stages: Vec<Stage>,
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
}

pub enum Vertex {
    Sumcheck(SumcheckVertex),
    Commit(CommitVertex),
    Opening(OpeningVertex),
    Rlc(RlcVertex),
    PointNormalization(PointNormVertex),
    EdgeTransform(EdgeTransformVertex),
}

// Scheduling hints
pub struct SchedulingHints {
    pub stage_assignment: HashMap<IdentityId, StageId>,
    pub batch_groups: HashMap<StageId, Vec<Vec<IdentityId>>>,
    pub commitment_groups: Vec<Vec<PolynomialId>>,
    pub opening_groups: Vec<Vec<PolynomialId>>,
    pub identity_meta: HashMap<IdentityId, IdentityMeta>,
}
```

**Keep from existing types.rs:**
- `ClaimId`, `VertexId`, `StageId`, `CommitmentGroupId` (newtype IDs)
- `Polynomial`, `Claim`, `SymbolicPoint`
- `ChallengeSpec`, `ChallengeLabel`

### Rewrite: jolt-ir/src/protocol/build.rs (~2,270 lines → ~600 lines)

**Replace `build_jolt_protocol()` with three functions:**

```rust
/// Pure protocol: ~10 identity groups. (~200 lines)
pub fn jolt_protocol(config: &ProtocolConfig) -> Vec<PolynomialIdentity>

/// Scheduling hints that reproduce current S1-S7 staging. (~200 lines)
pub fn jolt_scheduling_hints(config: &ProtocolConfig) -> SchedulingHints

/// ProtocolConfig stays as-is.
pub struct ProtocolConfig { ... }
```

`jolt_protocol()` collects existing claim definitions from `zkvm/claims/*.rs` —
those files stay unchanged. Each identity bundles:
- The `ClaimDefinition` (already exists)
- The polynomial bindings (currently in build_sN functions)
- The claimed sum (currently in InputClaim construction)
- The weighting and domain

`jolt_scheduling_hints()` extracts the stage assignments, batch groups, and
commitment groups that are currently interleaved in the procedural builder.

### New: jolt-ir/src/protocol/compiler.rs (~800 lines)

**The compiler pipeline:**

```rust
pub fn compile(
    protocol: &[PolynomialIdentity],
    hints: &SchedulingHints,
) -> Result<StagedGraph, CompileError>
```

**Internal passes:**

```rust
// Pass 1: Protocol → Claim Graph (mechanical, ~150 lines)
fn expand_to_claim_graph(
    protocol: &[PolynomialIdentity],
    hints: &SchedulingHints,
) -> ClaimGraph

// Pass 2: Claim Graph → Staged Claim Graph (hint-guided, ~100 lines)
fn assign_stages(
    graph: &ClaimGraph,
    hints: &HashMap<IdentityId, StageId>,
) -> Result<StagedClaimGraph, CompileError>

// Pass 3: Batching (hint-guided, ~100 lines)
fn batch_sumchecks(
    graph: &StagedClaimGraph,
    hints: &HashMap<StageId, Vec<Vec<IdentityId>>>,
) -> Result<BatchedGraph, CompileError>

// Pass 4: Commitment grouping (hint-guided, ~80 lines)
fn group_commitments(
    graph: &BatchedGraph,
    hints: &[Vec<PolynomialId>],
) -> Result<GroupedGraph, CompileError>

// Pass 5: Opening reduction (hint-guided, ~80 lines)
fn reduce_openings(
    graph: &GroupedGraph,
    hints: &[Vec<PolynomialId>],
) -> Result<ReducedGraph, CompileError>

// Pass 6: Edge transform insertion (automatic, ~80 lines)
fn insert_edge_transforms(graph: &ReducedGraph) -> StagedGraph
```

Each pass validates its hint against graph constraints before applying.

### Rewrite: jolt-ir/src/protocol/validate.rs (~1,373 lines → ~400 lines)

Simplify: most validation moves into the compiler passes (each pass validates its
own invariants). `validate.rs` becomes a final sanity check on the `StagedGraph`:
- Acyclicity
- Stage independence (no intra-stage Fiat-Shamir deps)
- Completeness (every committed poly has an opening)
- All claims consumed

### Validation

```bash
cargo clippy -p jolt-ir --message-format=short -q -- -D warnings
cargo nextest run -p jolt-ir --cargo-quiet
# Compiler must produce a StagedGraph that matches the old ProtocolGraph
# (structural equivalence test)
```

---

## Phase 4: Wire prover + verifier (depends on Phase 1, 3)

### Goal
Prover and verifier consume `StagedGraph` from the compiler. Generic graph executor
replaces all per-stage code.

### Rewrite: jolt-zkvm/src/prover.rs (~631 lines → ~200 lines)

**New `prove_from_graph()`:**

```rust
pub fn prove_from_graph<F, PCS, B>(
    graph: &StagedGraph,
    kernels: &KernelMap<B, F>,
    source: &impl PolynomialSource<F>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut impl Transcript,
) -> Proof<F, PCS>
```

Single `match vertex` loop. ~60 lines of match arms. No per-stage code.
No `build_witness()`. Uses `backend.make_witness()`.

**New `PolynomialSource` trait:**

```rust
pub trait PolynomialSource<F: Field> {
    fn get(&self, id: PolynomialId) -> &[F];
    fn materialize(&self, id: PolynomialId) -> Vec<F>;
}
```

`UnifiedSource` wraps: committed store + trace polys + R1CS virtual polys.

### Delete: jolt-zkvm/src/evaluators/ (8 files, 3,555 lines)

| File | Lines | Replacement |
|------|-------|-------------|
| `kernel.rs` | 1,155 | `CpuSumcheckWitness` in jolt-cpu (Phase 1) |
| `ra_virtual.rs` | 58 | Absorbed into `make_witness` |
| `segmented.rs` | 741 | Multi-phase → separate graph vertices (Phase 2) |
| `sparse_rw.rs` | 455 | `backend.sparse_reduce()` (Phase 1) |
| `ra_poly.rs` | 437 | Stays in jolt-witness or moves to jolt-cpu |
| `catalog.rs` | 232 | Formulas in jolt-ir claim definitions |
| `mles_product_sum.rs` | 310 | Already in jolt-cpu/toom_cook.rs |
| `mod.rs` | 12 | Deleted |

### Rewrite: jolt-zkvm/src/tables.rs (~529 lines → ~200 lines)

Simplify to implement `PolynomialSource<F>`. Wraps `WitnessStore` + `TracePolynomials`.

### Rewrite: jolt-verifier/src/verify.rs (~660 lines → ~400 lines)

**New `verify_from_graph()`:**

Consumes `StagedGraph`. Same generic loop structure as prover. No S1 hardcoding.
No separate `verify_spartan()` call.

### Rewrite: jolt-verifier/src/proof.rs (~41 lines → ~30 lines)

**Flat proof structure:**

```rust
pub struct Proof<F, PCS> {
    pub config: ProverConfig,
    pub commitments: Vec<PCS::Output>,
    pub sumcheck_proofs: Vec<SumcheckProof<F>>,
    pub evals: Vec<F>,
    pub opening_proofs: Vec<PCS::Proof>,
}
```

Cursor-based consumption: `ProofCursor` walks arrays in graph topological order.

### Validation

```bash
cargo nextest run -p jolt-zkvm muldiv --cargo-quiet --features host
cargo nextest run -p jolt-zkvm muldiv --cargo-quiet --features host,zk
# This is THE critical checkpoint. Full E2E must pass in both modes.
```

---

## Phase 5: Cleanup (depends on Phase 4)

### Delete jolt-spartan crate

All useful code extracted to jolt-r1cs (Phase 0) and jolt-sumcheck (uni_skip).
Prover/verifier deleted in Phase 2. Nothing remains.

### Delete old protocol types

Remove any backward-compat shims in jolt-ir that bridged old ProtocolGraph → new
StagedGraph.

### BlindFold adaptation (deferred)

BlindFold currently uses `SpartanProver::prove_relaxed()`. In the new model, the
relaxed Spartan proof is also graph vertices — same pattern, different formula
(includes relaxation error term `u·(Az·Bz - Cz) + E(x)`). Get standard mode
working first, then layer ZK.

---

## Parallelism Map

```
Phase 0 (jolt-r1cs extraction)  ─────┐
Phase 1 (backend factory)       ─────┤ parallel
                                     │
Phase 2 (Spartan dissolution)   ─────┘ depends on Phase 0
                                     │
Phase 3 (protocol + compiler)   ─────  depends on Phase 2
                                     │
Phase 4 (wire prover/verifier)  ─────  depends on Phase 1 + 3
                                     │
Phase 5 (cleanup)               ─────  depends on Phase 4
```

**Critical path: 0 → 2 → 3 → 4**
**Parallel: 0 ‖ 1**, then **2** merges.

---

## Validation Strategy

Every phase validates with:
1. `cargo clippy -p <crate> --message-format=short -q -- -D warnings`
2. `cargo fmt -q`
3. `cargo nextest run -p <crate> --cargo-quiet`

Critical E2E checkpoints:
- **After Phase 2** (Spartan dissolved): `muldiv --features host` must pass
- **After Phase 4** (full pipeline): `muldiv --features host` must pass
- **After Phase 4** (ZK mode): `muldiv --features host,zk` must pass

---

## Lines of Code Summary

| Phase | Created | Rewritten | Deleted | Net |
|-------|---------|-----------|---------|-----|
| 0 | ~1,200 (jolt-r1cs) | 0 | 0 | +1,200 |
| 1 | ~800 (CpuSumcheckWitness + trait additions) | 0 | 0 | +800 |
| 2 | ~400 (new Spartan vertices + edge transforms) | ~200 (prover/verifier S1 code) | ~1,693 (jolt-spartan prover/verifier) | -1,093 |
| 3 | ~800 (compiler.rs) | ~4,100 (build.rs, validate.rs, types.rs) | 0 | -3,300 → +800 |
| 4 | ~200 (PolynomialSource, ProofCursor) | ~1,800 (prover, verify, proof) | ~3,555 (evaluators/) | -3,355 |
| 5 | 0 | 0 | ~500 (jolt-spartan remnants, shims) | -500 |
| **Total** | **~3,400** | **~6,100** | **~5,748** | **~-2,348** |

The codebase shrinks by ~2,300 lines while gaining a compiler pipeline,
backend factory, sparse primitives, and clean protocol/scheduling separation.
