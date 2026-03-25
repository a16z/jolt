# Architecture Sins

Abstraction boundary violations and encapsulation failures in the IR → compute → prover
stack. Each document identifies a sin, explains why it matters, and proposes a fix.
Analysis 11 provides the unifying model that resolves or simplifies most sins.

## The Unifying Insight (Analysis 11)

Every computation in a SNARK prover follows the same pattern:

```
resolve inputs → dispatch to device → small result back to host → Fiat-Shamir → propagate
```

**The Fiat-Shamir boundary IS the host-device sync boundary.** A vertex is one device
dispatch whose result must be read back to host before the next computation's challenges
are known. This is exactly the ML compiler execution model (XLA/TVM/Triton).

A vertex is a protocol-level declaration: **formula + input_bindings + claimed_sum**.
Algorithm (dense/sparse/uni-skip), eq variant, round count, kernel selection — all
derived by the backend from the vertex's formula and input metadata. The frontend
declares WHAT, the backend decides HOW.

The prover is a generic graph executor:

```rust
for vertex in graph.topological_order() {
    let inputs = graph.resolve_inputs(vertex, &cache);
    let result = backend.dispatch(vertex, inputs);  // backend derives algorithm from vertex + data
    let challenges = transcript.absorb_and_challenge(&result);
    cache.store(vertex.id, result, challenges);
}
```

## Sin Index

| # | Title | Status | Core Issue |
|---|-------|--------|------------|
| [00](00-kernel-shape-taxonomy.md) | KernelShape conflates "what" with "how" | **Fix** | Frontend makes backend optimization decisions |
| [01](01-eq-special-casing.md) | Eq special-cased everywhere | **Simplified** | Eq variant derived by backend from formula structure |
| [02](02-compile-descriptor-untyped.md) | compile_descriptor untyped | **Simplified** | Vertex IS the descriptor, no ExecutionPlan needed |
| [03](03-graph-doesnt-own-layout.md) | Graph doesn't own layout | **Absorbed** | Graph IS the execution plan |
| [04](04-two-polynomial-sources.md) | Two polynomial sources | **Unblocked** | Spartan resolved → PolynomialSource unifies access |
| [05](05-ir-compute-boundary.md) | IR/compute boundary unclear | **Fix** | Formula in IR, kernels in backend |
| [06](06-sumcheck-compute-roundtrip.md) | SumcheckCompute heap alloc | **No action** | Conscious tradeoff |
| [07](07-evaluator-role.md) | Evaluator does two jobs | **Simplified** | Evaluator = trivial dispatch wrapper |
| [08](08-sparse-in-backend.md) | Sparse not in backend | **Fix** | sparse_reduce as backend primitive |
| [09](09-phased-orchestration.md) | Phase sequencing in code | **Dissolved** | Phases are vertices, graph is sequencer |
| [10](10-spartan-from-first-principles.md) | Spartan special-cased | **Dissolved** | R1CS in jolt-r1cs, sumchecks are graph vertices |
| [11](11-unified-execution-model.md) | **Unified execution model** | **New** | ML-compiler model: dispatch → result → Fiat-Shamir |

## Implementation Order

### Phase 1: Foundation (IR + Backend primitives)

| Step | Sin | What changes |
|------|-----|-------------|
| 1a | 0 | `CompositionFormula` (SoP) replaces `KernelShape` in jolt-ir |
| 1b | 5 | Pattern recognition moves from IR to backend (`compile_kernel` does it) |
| 1c | 1 | Eq detection moves to backend (derived from formula structure), single `pairwise_reduce` with `EqInput` |
| 1d | 8 | `sparse_reduce`, `sparse_bind` added to `ComputeBackend` trait |
| 1e | — | `uniskip_reduce` added to `ComputeBackend` trait (new, from Analysis 11) |

### Phase 2: Graph + Data (vertex model + crate extraction)

| Step | Sin | What changes |
|------|-----|-------------|
| 2a | 3 | Vertex carries: formula + input_bindings + claimed_sum. Backend derives the rest. Graph IS the plan |
| 2b | 10 | `jolt-r1cs` extracted (UniformR1cs, MLE eval, combined_partial_evaluate) |
| 2c | 4 | `PolynomialSource` unifies committed + virtual + R1CS polynomial access |
| 2d | 10 | Spartan vertices wired into graph (uni-skip → dense outer → materialize → dense inner → opening) |

### Phase 3: Prover (executor simplification)

| Step | Sin | What changes |
|------|-----|-------------|
| 3a | 7 | Evaluator reduced to trivial dispatch (one vertex, one `backend.dispatch()` call) |
| 3b | 2 | `compile_descriptor` deleted — vertex is self-describing |
| 3c | 9 | Phased evaluator deleted — phases are vertices |
| 3d | 10 | S1 special case deleted — same generic loop for all stages |
| 3e | 11 | Commitments + openings become graph vertices — entire proof is one graph walk |

### No action needed

| Sin | Why |
|-----|-----|
| 6 | Conscious tradeoff, heap alloc is noise vs O(N) bind cost |

## Target State

```
jolt-ir (pure protocol description)
  CompositionFormula    ← SoP structure, field-agnostic
  ProtocolGraph         ← DAG of vertices, each declaring WHAT not HOW
  Vertex types:
    Sumcheck            ← formula + input_bindings + claimed_sum
    Commit              ← polynomial id(s)
    Opening             ← polynomial + symbolic point + eval claim
    RLC                 ← general batching: claims, commitments, any homomorphic structure
    PointNormalization  ← rescale eval by Lagrange factor
    EdgeTransform       ← buffer materialization between vertices (no transcript)

jolt-compute (backend derives HOW from WHAT)
  Setup (AOT):
    compile_kernel(formula) → one per unique formula (~15-25 total)
    Challenges are symbolic — runtime parameters, not baked in
  Dispatch (per-vertex):
    pairwise_reduce(kernel, inputs, challenges)    → dense
    sparse_reduce(kernel, entries, challenges)      → sparse
    uniskip_reduce(inputs, tau_1)                   → analytical first-round
  Backend inspects formula + input metadata → picks algorithm, eq variant, grid

jolt-r1cs (constraint data, extracted from jolt-spartan)
  UniformR1cs           ← sparse per-cycle matrices (was UniformSpartanKey)
  evaluate_mle()        ← matrix MLE evaluation at a point
  combined_partial_evaluate() ← edge transform: materialize M(r_x, ·)

jolt-zkvm (graph executor, monomorphized over PCS + Backend)
  prove_from_graph<PCS, B>(graph, kernels, source, backend, pcs_setup, transcript)
    for vertex in graph.topological_order() {
        match vertex {
            Sumcheck  → kernel[id] + runtime buffers + challenges → round loop
            Commit    → PCS::commit(poly, setup) → transcript
            Opening   → PCS::open(poly, point, eval, setup) → transcript
            RLC       → squeeze challenge, reduce claims/commitments
            PointNorm → scalar rescale
            Edge      → materialize buffers for successor
        }
    }
  No per-stage code. No special cases. No evaluator complexity.
```
