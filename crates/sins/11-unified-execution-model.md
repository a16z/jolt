# Analysis 11: Unified Execution Model — Every Vertex is Dispatch → Result → Fiat-Shamir

## The Insight

Every computation in a SNARK prover follows the same pattern:

1. **Resolve inputs** — challenges, buffers, metadata from predecessor vertices
2. **Dispatch to device** — execute a kernel (sumcheck reduce, MSM, NTT, sparse walk)
3. **Small result back to host** — round polynomial, commitment, evaluation, claim scalar(s)
4. **Fiat-Shamir** — hash the result into the transcript, derive challenges
5. **Propagate** — challenges and results flow to dependent vertices

This is identical to the execution model in ML compilers (XLA, TVM, Triton):

| ML compiler | SNARK prover |
|-------------|-------------|
| Tensor operation | Cryptographic operation |
| GPU kernel dispatch | Backend kernel dispatch |
| Host sync for control flow / dynamic shapes | Host sync for Fiat-Shamir |
| Kernel fusion across sync-free regions | Kernel fusion across transcript-free regions |
| Memory planning (tensor lifetimes) | Buffer planning (polynomial lifetimes) |

**The Fiat-Shamir boundary IS the host-device sync boundary.** This is the invariant
that defines vertex granularity: a vertex is one device dispatch whose result must be
read back to host before the next computation's challenges are known.

## What a Vertex Is

A vertex is a **protocol-level declaration** — it describes WHAT to compute, not HOW.

### Frontend (jolt-ir / protocol graph) — shared by prover and verifier:

| Field | What it is |
|-------|-----------|
| `formula` | `CompositionFormula` — the SoP expression |
| `input_bindings` | Which polynomials feed this vertex (by `PolynomialId`) |
| `claimed_sum` | How the input claim is derived from predecessor outputs |

That's it. No `algorithm`, no `eq_variant`, no `num_rounds`.

### Backend-derived (prover-only, verifier doesn't care):

| Concern | How it's determined |
|---------|-------------------|
| Algorithm (Dense / Sparse / UnivariateSkip) | Derived from polynomial metadata — if inputs are sparse, use `sparse_reduce`; if claimed_sum is zero (zero-check), backend can apply uni-skip |
| Eq variant (TensorEq / Eq / None) | Derived from formula structure — backend detects eq factors |
| Round count | Implicit from input polynomial size: `log2(len)` |
| Kernel | AOT-compiled by backend from formula via pattern matching |

The backend sees the vertex's formula + input metadata and derives everything it needs.
The verifier sees the same vertex and knows how to check the proof without knowing the
execution strategy.

## Vertex Types

The graph has six vertex types. All follow the same execution pattern (resolve → dispatch →
result → transcript → propagate), but differ in what they compute and what result they
produce.

| Vertex | Inputs | Dispatch | Result | Transcript interaction |
|--------|--------|----------|--------|----------------------|
| **Sumcheck** | formula + poly buffers + claimed_sum | Backend kernel (dense/sparse/uni-skip) | Round polys, challenges, produced evals | Absorb round polys, derive per-round challenges |
| **Commit** | polynomial buffer(s) | PCS::commit (MSM) | Commitment(s) | Absorb commitment |
| **Opening** | polynomial + point + eval | PCS::open | Proof elements | Absorb proof elements |
| **RLC** | claims and/or commitments + transcript challenges | Host scalar/group math | Reduced claims/commitments | Squeeze challenges, absorb reduced values |
| **PointNormalization** | eval + padding source | Host scalar math | Rescaled eval | None (pure cache update) |
| **Edge transform** | predecessor outputs + data sources | Host or device materialization | Buffers for successor vertex | None (no transcript) |

### RLC as a general-purpose reduction vertex

RLC is not specific to opening claims. It's a general batching vertex that can reduce:
- **Opening claims**: multiple `(poly, point, eval)` → fewer claims via random linear combination
- **Commitments**: multiple commitments → batched commitment via RLC of group elements
- **Any homomorphic structure**: anything where `rlc(a, b; r) = a + r·b` preserves the
  verification equation

The vertex declares what it reduces and over what structure. The executor dispatches the
appropriate scalar/group math.

## AOT Kernel Compilation + Backend Factory

All sumcheck kernels are compiled at setup time, before proving begins. The compiled
kernel encapsulates the execution strategy (algorithm, reconstruction, eq handling, grid).
At runtime, the backend pairs a compiled kernel with data to produce a `SumcheckWitness`
— an opaque `SumcheckCompute` impl that jolt-sumcheck's round loop drives.

This eliminates `KernelEvaluator` entirely. The backend owns both compilation and the
runtime witness type.

### The backend trait:

```rust
trait ComputeBackend {
    type CompiledKernel<F: Field>: Send + Sync;
    type SumcheckWitness<F: Field>: SumcheckCompute<F>;

    /// AOT: compile formula structure into a kernel.
    /// Encapsulates: algorithm (dense/sparse/uni-skip), reconstruction strategy
    /// (Toom-Cook/standard), eq handling, grid choice.
    /// Challenges are symbolic — runtime parameters, not baked in.
    fn compile_kernel<F: Field>(&self, formula: &CompositionFormula) -> Self::CompiledKernel<F>;

    /// Runtime: pair a pre-compiled kernel with polynomial buffers and runtime
    /// challenge values. Returns a ready-to-use SumcheckCompute impl.
    /// Replaces both build_witness() and KernelEvaluator.
    fn make_witness<F: Field>(
        &self,
        kernel: &Self::CompiledKernel<F>,
        inputs: Vec<Self::Buffer<F>>,
        challenges: &[F],
    ) -> Self::SumcheckWitness<F>;

    // ... existing primitives (pairwise_reduce, interpolate_pairs, upload, etc.)
    // These are now internal to SumcheckWitness, not called by the orchestrator.
}
```

### Who owns setup:

```
jolt-ir (owns CompositionFormula on each graph vertex)
    ↓ consumed by
jolt-compute (ComputeBackend::compile_kernel takes &CompositionFormula)
    ↓ implemented by
jolt-cpu-kernels / jolt-metal (pattern-match formula → kernel + reconstruction strategy)
    ↓ orchestrated by
jolt-zkvm (walks graph at setup, calls compile_kernel per vertex, stores kernel map)
```

The backend is a consumer of jolt-ir formula types. `jolt-compute` depends on `jolt-ir`
for `CompositionFormula`. Concrete backends (jolt-cpu-kernels, jolt-metal) depend on
`jolt-compute` + `jolt-ir`.

### Setup phase (jolt-zkvm, once before proving):

```rust
fn setup_kernels<F, B: ComputeBackend>(
    graph: &ProtocolGraph,
    backend: &B,
) -> HashMap<VertexId, B::CompiledKernel<F>> {
    graph.sumcheck_vertices()
        .map(|v| (v.id, backend.compile_kernel(&v.formula)))
        .collect()
}
```

### Proving phase (jolt-zkvm, per sumcheck vertex):

```rust
// The orchestrator never touches kernel internals.
let inputs = source.resolve(vertex.input_bindings);
let challenges = cache.get_runtime_challenges(vertex);
let mut witness = backend.make_witness(&kernels[&vertex.id], inputs, &challenges);

// jolt-sumcheck owns the round loop — calls witness.round_polynomial() + witness.bind()
let (proof, point) = SumcheckProver::prove(&claim, &mut witness, transcript);
```

### What this eliminates:

| Before | After |
|--------|-------|
| `KernelEvaluator` in jolt-zkvm | `B::SumcheckWitness<F>` (backend-owned) |
| `InterpolationMode` enum | Baked into `CompiledKernel` at setup |
| `ToomCookState` | Internal to backend's `SumcheckWitness` impl |
| `Reconstructor` trait | Unnecessary — reconstruction internal to `SumcheckWitness` |
| `build_witness()` 160-line 4-arm match | `backend.make_witness(kernel, inputs, challenges)` |
| `compile_descriptor()` in jolt-ir | `backend.compile_kernel(formula)` |

On CPU: `CompiledKernel` wraps a closure capturing formula structure; `SumcheckWitness`
holds buffers + kernel ref and calls `pairwise_reduce` internally.
On GPU: `CompiledKernel` wraps a pipeline state; challenge values go in a uniform buffer
at dispatch time. No shader recompilation during proving.

Total unique kernels across Jolt: ~15-25. All compiled once at setup.

Total unique kernels across all of Jolt: ~15-25. All compiled once at setup.

## Generic PCS Dispatch

The prover function is monomorphized over `PCS: CommitmentScheme`. Commit and Opening
vertices dispatch through the PCS trait — the graph declares WHAT (which polynomial,
which point), the monomorphized executor knows HOW (which PCS impl).

```rust
fn prove_from_graph<PCS: CommitmentScheme, B: ComputeBackend>(
    graph: &ProtocolGraph,
    kernels: &KernelMap<B>,
    source: &impl PolynomialSource<PCS::Field>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut impl Transcript,
) {
    for vertex in graph.topological_order() {
        match vertex {
            Vertex::Sumcheck { formula, input_bindings, claimed_sum } => {
                let inputs = source.resolve(input_bindings);
                let kernel = &kernels[&vertex.id];
                let challenges = cache.get_challenges(vertex);
                // sumcheck round loop via jolt-sumcheck
                // round_polynomial() calls backend.pairwise_reduce(kernel, inputs, challenges)
                // transcript absorbs round poly, derives challenge, backend binds
            }

            Vertex::Commit { polynomial } => {
                let poly = source.get(polynomial);
                let (commitment, _) = PCS::commit(poly, pcs_setup);
                transcript.append(&commitment);
                cache.store_commitment(vertex.id, commitment);
            }

            Vertex::Opening { polynomial, point, eval } => {
                let poly = source.get(polynomial);
                let point = cache.resolve_point(point);
                let eval = cache.get_eval(eval);
                let proof = PCS::open(&poly, &point, eval, pcs_setup, &mut transcript);
                cache.store_proof(vertex.id, proof);
            }

            Vertex::Rlc { inputs, structure } => {
                // General-purpose RLC: batch claims, commitments, or both
                let challenge: F = transcript.challenge();
                let reduced = rlc_reduce(inputs, challenge, &cache);
                cache.store(vertex.id, reduced);
            }

            Vertex::PointNormalization { consumes, produces, padding } => {
                let lagrange = compute_lagrange(cache.resolve_point(padding));
                cache.set(produces, cache.get(consumes) * lagrange);
            }

            Vertex::EdgeTransform { transform, inputs } => {
                // combined_partial_evaluate, phase transition, etc.
                let result = transform.execute(inputs, &cache, source);
                cache.store_buffers(vertex.id, result);
            }
        }
    }
}
```

No per-stage code. No special cases. The graph + kernels + PCS monomorphization = the
complete proof.

## Consequences for the Architecture

### Spartan dissolves

Spartan is not infrastructure — it's a recipe. The recipe says: "take R1CS, interpret
matrices as multilinears, run sumcheck vertices over them, open the witness."

In the graph:
```
[Commit: witness] → [Sumcheck: outer zero-check] → [EdgeTransform: materialize M(r_x,·)]
    → [Sumcheck: inner product] → [RLC: batch claims] → [Opening: z(r_y)]
```

The backend sees the outer vertex has a zero-check claim and sparse R1CS inputs — it picks
uni-skip for round 1, sparse or dense reduce for the rest. No `jolt-spartan` prover. No
S1 special case. The R1CS data lives in `jolt-r1cs`.

### Phased execution dissolves

Multi-phase sumchecks (RAM read-write: sparse → sparse → dense) become multiple vertices:
```
[Sumcheck: cycle vars] → [Sumcheck: addr vars] → [Sumcheck: remaining vars]
```

The backend picks sparse or dense per vertex based on the input annotations. No `PhasePlan`,
no `TransitionSpec`, no phase tracking in the evaluator. The graph IS the phase sequence.
Edge transforms handle buffer materialization between vertices.

### The evaluator becomes trivial

The evaluator's only job: for ONE vertex, call the pre-compiled kernel with runtime data
and return the result. No reconstruction logic, no mode switching, no phase management.
The backend's `compile_kernel` already decided the strategy at setup time.

### Commitments, openings, and RLC are vertices

Currently these happen "outside" the graph. In this model they're first-class vertices,
making the entire proof — from first commitment to last opening — one graph walk.

RLC vertices are general-purpose batching: they can reduce opening claims (scalar RLC),
commitments (group-element RLC), or any homomorphic structure. This covers both the
current `RlcReduction` for openings and potential commitment batching.

### Edge transforms

Some edges carry more than challenges. Between vertices, we sometimes need to materialize
data:

- **`combined_partial_evaluate`** (Spartan): takes r_x from outer sumcheck, produces dense
  polynomial for inner sumcheck from R1CS key
- **RAM phase transitions**: materializes dense tables from bound sparse entries

These are **edge transforms** — computations that produce buffers for the next vertex.
They don't interact with the transcript (no Fiat-Shamir), so they're not vertices.
They execute between vertices as part of input resolution.

## Verifier Model

The verifier walks the **same graph** as the prover. Same vertex types, different work
per vertex. The verifier never touches the compute backend — no kernels, no buffers,
no device dispatch. It's pure scalar math + transcript replay + PCS verification.

### Per-vertex verifier behavior:

| Vertex | Verifier action |
|--------|----------------|
| **Sumcheck** | Read round polys from proof → verify sumcheck PIOP → check output formula against cached evals |
| **Commit** | Read commitment from proof → append to transcript |
| **Opening** | Read opening proof from proof → PCS::verify(commitment, point, eval, proof) |
| **RLC** | Squeeze same challenges → derive reduced claims/commitments (same as prover) |
| **PointNormalization** | Same scalar math as prover |
| **EdgeTransform** | Skip entirely — prover-only buffer materialization |

### Verifier flow:

```rust
fn verify_from_graph<PCS: CommitmentScheme>(
    graph: &ProtocolGraph,
    proof: &Proof<F, PCS>,
    vk: &VerifyingKey<F, PCS>,
    transcript: &mut impl Transcript,
) -> Result<(), VerifyError> {
    let mut cache = EvalCache::new(graph);
    let mut cursor = proof.cursor();

    for vertex in graph.topological_order() {
        match vertex {
            Vertex::Commit { polynomial } => {
                let commitment = cursor.next_commitment();
                transcript.append(&commitment);
                cache.store_commitment(vertex.id, commitment);
            }

            Vertex::Sumcheck { formula, input_bindings, claimed_sum } => {
                let claimed_sum = formula.evaluate_claim(&cache);
                let round_polys = cursor.next_round_polys();
                let (final_eval, challenges) =
                    SumcheckVerifier::verify(&claim, &round_polys, transcript)?;

                let evals = cursor.next_evals(vertex.num_produced_claims());
                for &e in &evals { transcript.append(&e); }
                cache.store_evals(vertex, &evals);
                cache.store_point(vertex.id, challenges);

                let expected = formula.evaluate_output(&evals, &cache);
                if expected != final_eval {
                    return Err(VerifyError::EvaluationMismatch);
                }
            }

            Vertex::RLC { inputs } => {
                let r: F = transcript.challenge();
                cache.store(vertex.id, rlc_reduce(&inputs, r, &cache));
            }

            Vertex::Opening { polynomial, point, eval } => {
                let commitment = cache.get_commitment(polynomial);
                let point = cache.resolve_point(point);
                let eval = cache.get_eval(eval);
                let proof = cursor.next_opening_proof();
                PCS::verify(&commitment, &point, eval, &proof, &vk.pcs_setup, transcript)?;
            }

            Vertex::PointNormalization { consumes, produces, padding } => {
                cache.set(produces, cache.get(consumes) * compute_lagrange(padding));
            }

            Vertex::EdgeTransform { .. } => {} // prover-only
        }
    }
    Ok(())
}
```

### Flat proof structure

The proof becomes a stream consumed in graph topological order — no named fields for
Spartan vs stages vs openings:

```rust
struct Proof<F, PCS> {
    commitments: Vec<PCS::Output>,
    round_polys: Vec<SumcheckProof<F>>,
    evals: Vec<F>,
    opening_proofs: Vec<PCS::Proof>,
}
```

A cursor walks through each array as the verifier walks the graph. No indexing by stage
number, no special fields — the graph structure implicitly defines the proof layout.

### What changes from today:

| Current | New |
|---------|-----|
| S1 Spartan special-cased via `verify_spartan()` | Same Sumcheck vertex handler |
| `proof.spartan_evals` read separately | Evals read from cursor like everything else |
| Spartan witness opening special-cased | Opening vertex in the graph |
| Commitments verified after all stages | Commit vertices in graph order |
| RLC called once at end for openings | RLC vertices wherever needed |
| `JoltProof` with 7 named fields | Flat stream consumed by cursor |

### Verifying key simplifies too:

```rust
struct VerifyingKey<F, PCS> {
    graph: ProtocolGraph,          // the protocol definition
    r1cs: UniformR1cs<F>,          // for checking Spartan output formula (matrix MLE evals)
    pcs_setup: PCS::VerifierSetup, // for PCS::verify
}
```

## What Remains Domain-Specific

Only three things:

1. **The graph definition** — which vertices exist, how they're connected, what formulas
   they carry. This IS the protocol. For Jolt, it encodes Spartan + RAM checking +
   instruction lookups + claim reductions + commitments + openings + RLC reductions.

2. **The kernels** — AOT-compiled from formulas by the backend at setup time. The backend
   pattern-matches `CompositionFormula` and picks the best kernel. Runtime dispatch is
   just data + challenges through pre-compiled kernels.

3. **Data sources** — `jolt-r1cs` for constraint matrices, `jolt-witness` for trace tables,
   `PolynomialSource` as the unified interface.

## Implementation Order

This unified model subsumes and clarifies several earlier sins:

| Sin | Status under unified model |
|-----|--------------------------|
| 00 (KernelShape) | **Unchanged** — CompositionFormula is the vertex's formula |
| 01 (Eq special-casing) | **Simplified** — eq variant derived by backend from formula structure |
| 02 (compile_descriptor) | **Simplified** — vertex IS the descriptor (formula + inputs + claim) |
| 03 (graph owns layout) | **Absorbed** — the graph IS the execution plan |
| 04 (polynomial sources) | **Unblocked** — Spartan resolved, PolynomialSource feeds vertex inputs |
| 05 (IR/compute boundary) | **Clarified** — IR owns formula + input bindings, backend owns everything else |
| 06 (SumcheckCompute roundtrip) | **Unchanged** — conscious tradeoff |
| 07 (evaluator role) | **Simplified** — evaluator is trivial dispatch, no reconstruction |
| 08 (sparse in backend) | **Unchanged** — sparse_reduce is a backend algorithm |
| 09 (phased orchestration) | **Dissolved** — phases are vertices, graph is the sequencer |
| 10 (Spartan) | **Dissolved** — Spartan is vertices in the graph + jolt-r1cs data |

### Revised fix order

```
Phase 1: Foundation
  Sin 0  — CompositionFormula replaces KernelShape
  Sin 5  — IR/compute boundary: formula + input bindings in IR, everything else in backend
  Sin 1  — Eq detection moves to backend (derived from formula structure)

Phase 2: Backend
  Sin 8  — Sparse as first-class backend primitive
  New    — UnivariateSkip as backend primitive (derived from zero-check claim)
  New    — AOT compilation: compile_kernel takes CompositionFormula, challenges are runtime

Phase 3: Graph
  Sin 3  — Graph vertices: Sumcheck, Commit, Opening, RLC, PointNorm, EdgeTransform
  Sin 10 — Spartan dissolved into graph vertices + jolt-r1cs
  Sin 4  — PolynomialSource unifies data access
  New    — Commit/Opening/RLC as first-class vertices, PCS dispatch via monomorphization

Phase 4: Prover
  Sin 7  — Evaluator becomes trivial dispatch of pre-compiled kernel
  Sin 2  — compile_descriptor eliminated (vertex IS the descriptor)
  Sin 9  — Phased execution eliminated (vertices ARE phases)

Phase 5: New crate
  jolt-r1cs — extracted from jolt-spartan (R1CS types, uniform key, MLE evaluation)
  jolt-spartan — deleted or reduced to a standalone recipe crate for external consumers
```

## Decisions

### Vertex model
- A vertex is a protocol-level declaration: `formula + input_bindings + claimed_sum`
- Algorithm, eq variant, round count, kernel — all derived by the backend from vertex + data
- No `num_rounds` — implicit from polynomial size
- No `algorithm` — derived from input sparsity annotations + claim structure
- No `eq_variant` — derived from formula structure by backend
- Fiat-Shamir boundary = host-device sync boundary = vertex boundary
- Six vertex types: Sumcheck, Commit, Opening, RLC, PointNormalization, EdgeTransform

### Backend factory model
- `KernelEvaluator` eliminated — replaced by `B::SumcheckWitness<F>` associated type
- `compile_kernel(formula)` at setup (AOT) → `CompiledKernel` encapsulates algorithm + reconstruction + eq + grid
- `make_witness(kernel, inputs, challenges)` at runtime → `SumcheckWitness` implements `SumcheckCompute`
- Challenges are runtime parameters, not baked-in constants
- `build_witness()` 4-arm match eliminated — one `make_witness()` call
- `InterpolationMode`, `ToomCookState`, `Reconstructor` — all internal to backend

### Setup ownership
- jolt-ir owns the graph with `CompositionFormula` on each vertex
- jolt-compute defines `ComputeBackend` trait, depends on jolt-ir for `CompositionFormula`
- jolt-cpu-kernels / jolt-metal implement `ComputeBackend`, pattern-match formulas
- jolt-zkvm walks graph at setup, calls `backend.compile_kernel()` per vertex, stores kernel map
- The backend is a consumer of jolt-ir formula types

### Protocol dispatch
- RLC is general-purpose: reduces claims (scalar RLC), commitments (group RLC), or any homomorphic structure
- PCS dispatch via monomorphization: `prove_from_graph<PCS, B>` — graph declares WHAT, monomorphized executor knows HOW
- jolt-sumcheck keeps the round loop (protocol logic) — calls `witness.round_polynomial()` + `witness.bind()`

### Verifier
- Walks the same graph as the prover — same vertex types, different per-vertex work
- Never touches the compute backend — no kernels, buffers, or device dispatch
- Proof is a flat stream consumed in graph topological order by a cursor
- No special fields for Spartan vs stages vs openings — graph structure defines proof layout
- VerifyingKey = graph + R1CS + PCS setup

### Dissolved concerns
- Phased execution → phases are vertices
- Spartan → R1CS data in jolt-r1cs, sumchecks are graph vertices
- Edge transforms are not vertices — they don't touch the transcript
- The prover is a generic graph executor — no per-stage code
- Frontend/backend split: IR declares WHAT (formula, inputs, claim), backend decides HOW (algorithm, kernel, grid)
