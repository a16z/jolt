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

## Active Documents

| # | Title | Status |
|---|-------|--------|
| [11](11-unified-execution-model.md) | **Unified execution model** | Foundational insight |
| [13](13-frontend-lowering-pipeline.md) | Compiler-driven pipeline | Active spec |
| [14](14-execution-plan.md) | Execution plan (rewrite) | Active plan |

## Resolved Sins (in `done/`)

Sins 00-10 and 12 have been addressed by the progressive lowering pipeline
(identity.rs, compiler.rs, protocol_def.rs in jolt-ir). Moved to `done/`.

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
