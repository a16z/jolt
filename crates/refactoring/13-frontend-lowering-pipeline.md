# Analysis 13: Compiler-Driven Pipeline

The protocol (polynomial identities) is the top-level IR. A **compiler** derives
staging, batching, commitment grouping, and all execution decisions. Fiat-Shamir
is the sole sequential constraint.

## Design Principle: Progressive Lowering with Expansion

Every lowering pass **expands** the representation. The input is small and declarative;
the output is large and operational. No pass loses information — each adds decisions
that the previous level left unspecified.

```
Protocol          ~10 polynomial identities (pure math)
    ↓ mechanically expand
Claim Graph       ~50-100 nodes (one per sumcheck/opening/commitment claim)
    ↓ compiler: hint-guided or cost-model-driven optimization passes
Staged Graph      ~100-200 nodes (grouped into Fiat-Shamir epochs, batched)
    ↓ compiler: backend-specific lowering
Computation Graph ~1000-5000 ops (kernels, buffer moves, reductions)
    ↓ compiler: device scheduling
Device Schedule   thread groups, memory transfers, kernel launches
```

Each level is a complete, self-contained representation. The compiler can be
inspected, debugged, and tested at every level independently.

## Bootstrapping Strategy: Hinted Compiler

Building the full cost-model-driven optimization passes is a large project. The
architecture separates concerns FIRST, optimizes LATER:

**Phase A (now)**: Every compiler pass accepts explicit **scheduling hints** that
reproduce today's hand-tuned `build_jolt_protocol()` decisions. The pipeline is
fully generic — hints are data, not code. The protocol is pure math.

**Phase B (later)**: Replace hints one pass at a time with cost-model-driven
optimization. Each pass becomes: "use the hint if provided, otherwise derive."

This gives:
- Clean separation of protocol from scheduling (the architecture win)
- Full pipeline exercised and tested from day 1
- Zero performance regression (hints reproduce current behavior exactly)
- Incremental path to fully automatic optimization

```rust
/// Pure protocol — no scheduling concerns.
fn jolt_protocol(config: &ProtocolConfig) -> Vec<PolynomialIdentity> { ... }

/// Scheduling hints — temporary, reproduces today's hand-tuned decisions.
/// Eventually: Option<SchedulingHints>, then deleted entirely.
fn jolt_scheduling_hints(config: &ProtocolConfig) -> SchedulingHints { ... }

/// Compiler: protocol + hints → staged graph.
/// Validates hints against Fiat-Shamir correctness (rejects unsound schedules).
fn compile(
    protocol: &[PolynomialIdentity],
    hints: &SchedulingHints,
) -> StagedGraph { ... }
```

### SchedulingHints

The hints carry exactly the decisions currently baked into `build_jolt_protocol()`,
extracted into a separate data structure:

```rust
pub struct SchedulingHints {
    /// Which identities share a Fiat-Shamir epoch.
    /// Key: identity name. Value: stage index.
    pub stage_assignment: HashMap<IdentityId, StageId>,

    /// Which sumchecks within a stage batch together.
    /// Each inner Vec is one batch group (shared RLC challenge).
    pub batch_groups: HashMap<StageId, Vec<Vec<IdentityId>>>,

    /// Which polynomials commit together and in what transcript order.
    pub commitment_groups: Vec<Vec<PolynomialId>>,

    /// Opening reduction groups (claims that RLC together).
    pub opening_groups: Vec<Vec<PolynomialId>>,

    /// Per-identity metadata the compiler cannot yet derive.
    /// Weighting (Eq/EqPlusOne/Lt), multi-phase structure, etc.
    pub identity_meta: HashMap<IdentityId, IdentityMeta>,
}

pub struct IdentityMeta {
    /// How the eq polynomial weights this sumcheck.
    pub weighting: Option<WeightingHint>,
    /// Multi-phase decomposition (e.g., RAM RW: sparse-cycle, sparse-addr, dense).
    pub phases: Option<Vec<PhaseHint>>,
    /// Challenge labels for formula slots.
    pub challenge_labels: Vec<&'static str>,
}
```

### Hint validation

The compiler MUST validate hints against Fiat-Shamir correctness:

```rust
fn validate_hints(
    claim_graph: &ClaimGraph,
    hints: &SchedulingHints,
) -> Result<(), HintError> {
    // 1. Every identity in the protocol has a stage assignment.
    // 2. No intra-stage Fiat-Shamir dependency:
    //    if B depends on A's challenge output, stage(A) < stage(B).
    // 3. Batch groups contain only same-domain sumchecks.
    // 4. Every committed polynomial appears in exactly one commitment group.
    // 5. Every opening claim appears in exactly one opening group.
}
```

A hint that violates Fiat-Shamir ordering is rejected with a clear error — not
silently accepted. The human can hint freely but cannot break soundness.

### Migration path

```
Phase A: compile(protocol, hints)              — hints required, reproduce current perf
Phase B: compile(protocol, Some(hints))        — hints optional, compiler can derive
Phase C: compile(protocol, None) + autotuner   — hints deleted, cost model drives all
```

At each phase, the staged graph output is identical in structure. Downstream code
(prover, verifier) never changes — only the compiler internals evolve.

## Level 0: Protocol (~10 identities)

The protocol is pure math. It specifies WHAT the SNARK proves, with zero
operational concerns. This is the only hand-written layer.

```rust
/// A polynomial identity that the SNARK must prove.
/// Example: "For all x in {0,1}^n: eq(tau,x) · [f(x)·g(x) - h(x)] = 0"
pub struct PolynomialIdentity {
    /// Unique name — used as key in scheduling hints.
    pub name: &'static str,
    /// The composition applied to the polynomials.
    pub formula: CompositionFormula,
    /// Which polynomials participate (maps formula Input(i) to a polynomial).
    pub polynomials: Vec<PolynomialId>,
    /// What the sum equals (usually 0 for a zero-check, or a claim from another identity).
    pub claimed_sum: IdentityClaim,
    /// Domain: which variables are summed over.
    pub domain: DomainSpec,
    /// Which eval claims this identity produces when reduced.
    /// Mechanically determined: one per input polynomial.
    pub produces: Vec<PolynomialId>,
}

pub enum IdentityClaim {
    /// Sum is zero (zero-check). Example: booleanity, R1CS satisfaction.
    Zero,
    /// Sum equals a claim produced by another identity.
    Predecessor(ClaimRef),
    /// Sum is a constant known at graph construction time.
    Constant(i64),
}

pub enum DomainSpec {
    /// Sum over {0,1}^n where n = log2(trace_length).
    TraceLength,
    /// Sum over {0,1}^n where n is determined by a specific polynomial's size.
    PolynomialSize(PolynomialId),
    /// Sum over {0,1}^n where n is a symbolic expression.
    Symbolic(SymbolicSize),
}
```

### What a protocol looks like

Jolt's protocol is ~10 identity groups:

```rust
fn jolt_protocol(config: &ProtocolConfig) -> Vec<PolynomialIdentity> {
    vec![
        // 1. R1CS satisfaction (Spartan)
        //    ∀x: eq(τ,x) · [Az(x)·Bz(x) - Cz(x)] = 0
        r1cs_satisfaction(),

        // 2. Instruction lookup RA (Twist/Shout)
        //    ∀x: eq(τ,x) · Σ γ^i · ∏_j ra_ij(x) = claimed_sum
        instruction_ra(config.d_instr),

        // 3. Bytecode read checking
        bytecode_ra(config.d_bc),

        // 4. RAM read-write checking
        //    ∀x: eq(τ,x) · [ra·(val_w - val_r) + γ·ra·(inc + val_r)] = 0
        ram_read_write(),

        // 5. Register read-write checking
        register_read_write(),

        // 6. Hamming booleanity
        //    ∀x: eq(τ,x) · h(x) · (h(x) - 1) = 0
        hamming_booleanity(),

        // 7. RA booleanity
        //    ∀x: eq(τ,x) · Σ γ^i · (ra_i² - ra_i) = 0
        ra_booleanity(config),

        // 8. Claim reductions (one per claim group)
        //    ∀x: eq(r,x) · Σ γ^i · poly_i(x) = Σ γ^i · eval_i
        claim_reductions(config),

        // 9. RAM output checking / RAF evaluations
        ram_output(),
        raf_evaluations(config),
    ]
}
```

Each function returns one or more `PolynomialIdentity`. No stages, no batching,
no commitment groups — just the math.

### What the protocol does NOT contain

| Concern | Protocol says | Compiler/hints decide |
|---------|-------------|----------------------|
| Staging / ordering | nothing | which identities share a Fiat-Shamir epoch |
| Batching | nothing | which sumchecks share challenges and batch |
| Commitment grouping | nothing | which polynomials commit together |
| Opening reduction | nothing | how to batch opening claims (RLC strategy) |
| Algorithm | nothing | dense / sparse / uni-skip per sumcheck |
| Kernel shape | nothing | Toom-Cook / specialized / general SoP |
| Memory layout | nothing | buffer lifetimes, materialization order |
| Weighting | nothing | Eq / EqPlusOne / Lt (hint or derived from formula) |
| Multi-phase | nothing | sparse→dense phase structure (hint or derived from input metadata) |

## Level 1: Claim Graph (~50-100 nodes)

**Mechanical derivation** from the protocol. No heuristics. Each polynomial identity
expands into its constituent claims: sumcheck claims, commitment claims, and opening
claims.

```rust
pub enum ClaimNode {
    /// A sumcheck to prove: Σ_x formula(inputs(x)) = claimed_sum.
    Sumcheck(SumcheckClaim),
    /// A polynomial commitment: PCS::commit(poly).
    Commitment(CommitmentClaim),
    /// A polynomial opening: PCS::open(poly, point, eval).
    Opening(OpeningClaim),
    /// A point normalization: rescale eval by Lagrange factor.
    PointNorm(PointNormClaim),
}

pub struct SumcheckClaim {
    /// Which identity this claim derives from.
    pub identity: IdentityId,
    pub formula: CompositionFormula,
    pub input_bindings: Vec<PolynomialId>,
    pub claimed_sum: ClaimSource,
    /// Which polynomial evaluations this sumcheck produces.
    pub produces: Vec<EvalClaim>,
}
```

### Derivation rules

The expansion is deterministic:

1. Each `PolynomialIdentity` → one `SumcheckClaim` (or a small tree if multi-phase)
2. Each polynomial referenced by a sumcheck that is committed → one `CommitmentClaim`
3. Each sumcheck produces eval claims on its inputs → one `OpeningClaim` per committed input
4. Eval claims on virtual polynomials → recursive sumcheck claims (claim reduction)

The claim graph has edges: `SumcheckClaim → EvalClaim → OpeningClaim`, and
`SumcheckClaim → SumcheckClaim` for claim reductions.

### What the claim graph captures

- **All claims** that must be proved for soundness
- **Data dependencies** between claims (which evals feed which claimed sums)
- **Polynomial sharing** (multiple sumchecks reference the same polynomial)

### What the claim graph does NOT capture

- Stage boundaries (Fiat-Shamir grouping)
- Batching decisions (which sumchecks share challenges)
- Commitment ordering
- Algorithm choices

This is the "unscheduled" representation — analogous to an ML compiler's
platform-independent IR before operator fusion and scheduling passes.

## Level 2: Staged Graph (~100-200 nodes)

The staged graph is the claim graph with scheduling decisions applied. In the
hinted-compiler model, each pass consults hints; in the fully automatic model,
each pass uses a cost model.

### Compiler passes

The compiler transforms the claim graph into the staged graph via 5 passes.
Each pass is a pure function: `Graph → Graph`. Passes compose.

Each pass has the same structure:

```rust
fn pass(graph: &InputGraph, hints: Option<&PassHints>) -> OutputGraph {
    if let Some(h) = hints {
        // Validate hint against graph constraints
        validate(graph, h)?;
        // Apply hint
        apply_hint(graph, h)
    } else {
        // Derive from cost model (Phase B+)
        optimize(graph, &cost_model)
    }
}
```

#### Pass 1: Stage assignment (Fiat-Shamir grouping)

**Input**: Claim graph with data-dependency edges.
**Output**: Claim graph with `stage_id` annotations on every node.

The fundamental constraint: if node B depends on a challenge derived from node A's
transcript contribution, then `stage(A) < stage(B)`.

**Hint mode** (Phase A): `hints.stage_assignment` maps each identity to a stage.
The compiler validates that no intra-stage Fiat-Shamir dependency exists and rejects
invalid assignments.

**Automatic mode** (Phase B+): longest-path layering on the dependency DAG. Nodes
with no Fiat-Shamir predecessors → stage 0. All others → `max(predecessor_stage) + 1`.
This is the minimum-depth schedule.

```rust
fn assign_stages(
    graph: &ClaimGraph,
    hints: Option<&HashMap<IdentityId, StageId>>,
) -> Vec<(NodeId, StageId)> {
    match hints {
        Some(h) => {
            // Validate: no B depends on A's challenges within same stage
            validate_stage_assignment(graph, h)?;
            // Apply directly
            apply_stage_hints(graph, h)
        }
        None => {
            // Longest-path layering
            compute_minimum_depth_schedule(graph)
        }
    }
}
```

#### Pass 2: Batching (sumcheck grouping)

**Input**: Staged claim graph.
**Output**: Batched staged graph where sumchecks within a stage are grouped.

Sumchecks in the same stage that operate on the same variable domain can share a
random linear combination challenge and run as a single batched sumcheck. This
reduces proof size (one set of round polynomials instead of N) at the cost of
wider kernels.

**Hint mode**: `hints.batch_groups` specifies which identities batch together per stage.
**Automatic mode**: group by domain compatibility, optimize with cost model
(`proof_size_savings > prover_time_overhead`).

#### Pass 3: Commitment grouping

**Input**: Batched staged graph with commitment claims.
**Output**: Graph with commitment claims grouped into commitment vertices.

**Hint mode**: `hints.commitment_groups` specifies polynomial groups and transcript order.
**Automatic mode**: group by availability, PCS compatibility, and memory constraints.

#### Pass 4: Opening reduction

**Input**: Graph with opening claims.
**Output**: Graph with RLC vertices that batch openings at common points.

**Hint mode**: `hints.opening_groups` specifies which opening claims reduce together.
**Automatic mode**: group by evaluation point, insert RLC vertex per group.

#### Pass 5: Edge transform insertion

**Input**: Reduced graph.
**Output**: Final staged graph with EdgeTransform vertices for buffer materialization.

**Always automatic** — no hint needed. Where a sumcheck consumes a virtual polynomial
that requires materialization (e.g., Spartan's `combined_partial_evaluate` output),
the compiler inserts an EdgeTransform vertex.

### Stage structure after all passes

```rust
pub struct StagedGraph {
    pub stages: Vec<Stage>,
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
}

pub struct Stage {
    pub id: StageId,
    pub vertices: Vec<VertexId>,
    pub pre_squeeze: Vec<ChallengeSpec>,
}
```

Six vertex types (from Analysis 11):

```rust
pub enum Vertex {
    Sumcheck(SumcheckVertex),
    Commit(CommitVertex),
    Opening(OpeningVertex),
    Rlc(RlcVertex),
    PointNormalization(PointNormVertex),
    EdgeTransform(EdgeTransformVertex),
}
```

### Why this structure, even with hints

Even when every decision comes from hints (Phase A), the architecture provides:

1. **Auditability**: the protocol is readable as pure math. The scheduling is
   readable as a separate data structure. Today they're interleaved in 1900 lines.
2. **Testability**: you can test protocol correctness independently from
   scheduling correctness. You can swap hints and see different staged graphs.
3. **Extensibility**: adding a new identity means writing one function that returns
   a `PolynomialIdentity` and one hint entry for stage/batch placement. No surgery
   on a monolithic graph builder.
4. **Backend decoupling**: the staged graph is the same regardless of CPU/GPU.
   Backend differences appear only at Level 3 (computation graph).

## Level 3: Computation Graph (~1000-5000 ops)

Backend-specific lowering. Each vertex in the staged graph expands into concrete
operations.

### Sumcheck vertex expansion

A single sumcheck vertex expands to:

```
1 × compile_kernel(formula)           — AOT, at setup time
1 × make_witness(kernel, inputs)      — runtime, binds data to kernel
N × round_polynomial()                — N = num_vars rounds
N × bind(challenge)                   — one per round
1 × final_evaluations()               — extract eval claims
```

For a sumcheck with `num_vars = 20`, this is ~42 ops from one vertex.

### Commit vertex expansion

```
1 × upload_polynomial(poly)           — host → device transfer
1 × MSM(poly, generators)             — the actual commitment
1 × download_commitment()             — device → host
1 × transcript.append(commitment)     — Fiat-Shamir
```

### Opening vertex expansion

```
1 × PCS::open(poly, point, eval)      — may internally be many ops (Dory: 2 log(n) rounds)
```

### What the computation graph captures

- Every buffer allocation and deallocation
- Every kernel dispatch with concrete parameters (grid size, thread count)
- Every host-device transfer
- The complete memory timeline

This is the level where profiling and optimization happen. It's analogous to
XLA's HLO after all fusion passes, or TVM's lowered TIR.

## Level 4: Device Schedule

The final representation, specific to a (backend, hardware) pair. Not a data
structure the compiler produces explicitly — it's the sequence of system calls
the runtime makes.

**CPU**: thread pool task graph. Each kernel dispatch → rayon parallel iterator.
Buffer allocations are Rust `Vec`s. No explicit scheduling needed beyond rayon's
work-stealing.

**Metal**: command buffer construction. Kernel dispatches → MTLComputeCommandEncoder.
Buffer allocations → MTLBuffer. Thread group sizes tuned to the formula's arithmetic
intensity and the GPU's SIMD width. Barriers between stages.

**CUDA**: stream-based execution. Kernel launches → cudaLaunchKernel. Memory
allocations → cudaMalloc / memory pool. Concurrent dispatches within a stage use
separate streams. cudaStreamSynchronize at stage boundaries.

## Fiat-Shamir as the Sole Sequential Constraint

### What it forces

1. **Stage boundaries are irreducible**: stage k+1 cannot begin until stage k's
   transcript interaction completes. This is soundness — not a performance tradeoff.

2. **Host readback per stage**: round polynomials must be hashed (on host) before
   challenges can be squeezed. GPU results must be downloaded at every stage boundary.

3. **Challenge freshness**: every challenge depends on all prior transcript entries.

### What it permits

1. **Everything within a stage is parallel**: no ordering constraint between vertices
   in the same stage. The compiler exploits this for batching and concurrent dispatch.

2. **Buffer pipelining**: stage k+1's data uploads can overlap with stage k's
   transcript processing (the buffers don't depend on new challenges).

3. **Speculative execution**: if `claimed_sum = 0` (zero-check), the backend can
   begin without waiting for the claimed sum from a predecessor.

### The key insight

Staging is not a protocol concept — it's an optimization concept. The protocol
specifies polynomial identities. The compiler groups them into stages to minimize
sequential depth while respecting Fiat-Shamir. A different cost model (or a different
backend with different latency characteristics) could produce a different staging
of the same protocol.

## Kernel Compilation

The backend's `compile_kernel` is the bridge between the symbolic protocol and
concrete execution. It runs at setup time (once per circuit) and produces opaque
compiled kernels.

### What happens inside compile_kernel

```
CompositionFormula
    │
    ├─ Pattern: pure product of D inputs, disjoint groups?
    │   → ToomCook kernel (O(D log D), eval_prod_D specialized)
    │   → Reconstruction: ToomCook grid {1,...,D-1,∞} + Gruen recovery
    │   → EqHandling: TensorSplit (factored out)
    │
    ├─ Pattern: all terms arity ≤ 1?
    │   → PreCombine + degree-2 kernel
    │   → Reconstruction: standard grid {0, 2}
    │   → EqHandling: Flat
    │
    ├─ Pattern: x²-x (booleanity)?
    │   → Specialized degree-3 kernel
    │   → Reconstruction: standard grid {0, 2, 3}
    │   → EqHandling: TensorSplit or Flat (backend decides)
    │
    └─ General SoP
        → Codegen / stack-VM for the SoP expression
        → Reconstruction: standard grid {0, 2, ..., degree}
        → EqHandling: Flat
```

All ~15-25 unique formulas compile once. The compiled kernel encapsulates algorithm,
reconstruction strategy, eq handling, and eval grid. None of this leaks to the caller.

### Per-backend compilation

**CPU**: Rust closure capturing the formula structure, specialized via match arms.
Monomorphization gives zero dispatch overhead.

**Metal**: MSL compute shader generated from term structure. Challenge slots are
uniform buffer parameters (reusable across challenge values).

**CUDA**: PTX kernel. Challenge slots are kernel arguments.

## Runtime Execution

The runtime executor is a simple loop over the staged graph. It does not make
optimization decisions — those were all made by the compiler.

```rust
fn prove<PCS, B: ComputeBackend>(
    graph: &StagedGraph,
    kernels: &KernelMap<B, PCS::Field>,
    source: &impl PolynomialSource<PCS::Field>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut impl Transcript,
) -> Proof<PCS::Field, PCS> {
    let mut cache = ProverCache::new(graph);
    let mut proof = ProofBuilder::new();

    for stage in &graph.stages {
        for &vid in &stage.vertices {
            match &graph.vertices[vid] {
                Vertex::Sumcheck(sv) => {
                    let inputs = resolve_inputs(sv, source, &cache, backend);
                    let challenges = cache.resolve_challenges(sv);
                    let claimed_sum = cache.resolve_claimed_sum(&sv.claimed_sum);
                    let kernel = &kernels[&sv.formula.structural_hash()];
                    let mut witness = backend.make_witness(kernel, inputs, &challenges);
                    let claim = SumcheckClaim {
                        num_vars: witness.num_vars(),
                        degree: witness.degree(),
                        claimed_sum,
                    };
                    let (sp, point) = SumcheckProver::prove(&claim, &mut witness, transcript);
                    let evals = witness.final_evaluations();
                    for &e in &evals { transcript.append_scalar(e); }
                    cache.store_point(sv.id, point);
                    cache.store_evals(sv, &evals);
                    proof.push_sumcheck(sp, evals);
                }
                Vertex::Commit(cv) => {
                    let poly = source.get(cv.polynomial);
                    let commitment = PCS::commit(poly, pcs_setup);
                    transcript.append(&commitment);
                    cache.store_commitment(cv.id, commitment.clone());
                    proof.push_commitment(commitment);
                }
                Vertex::Opening(ov) => {
                    let poly = source.get(ov.polynomial);
                    let point = cache.resolve_point(&ov.point);
                    let eval = cache.get_eval(ov.eval);
                    let opening_proof = PCS::open(poly, &point, eval, pcs_setup, transcript);
                    proof.push_opening(opening_proof);
                }
                Vertex::Rlc(rv) => {
                    let challenge = transcript.challenge();
                    let reduced = rv.reduce(&cache, challenge);
                    cache.store(rv.id, reduced);
                }
                Vertex::PointNormalization(pn) => {
                    let lagrange = cache.compute_lagrange(&pn.padding_source);
                    for (&c, &p) in pn.consumes.iter().zip(pn.produces.iter()) {
                        cache.set_eval(p, cache.get_eval(c) * lagrange);
                    }
                }
                Vertex::EdgeTransform(et) => {
                    let buffers = et.execute(source, &cache, backend);
                    cache.store_buffers(et.id, buffers);
                }
            }
        }

        // Stage boundary: squeeze challenges for next stage
        let challenges = transcript.challenge_vec(stage.pre_squeeze.len());
        cache.store_stage_challenges(stage.id, challenges);
    }

    proof.finalize()
}
```

The executor is ~60 lines. It makes zero decisions. All intelligence is in the
compiler (staged graph construction) and the backend (kernel compilation + witness).

## Verification

The verifier walks the **same staged graph** with the same topological order and
the same transcript interactions. The difference: it reads proof elements instead
of computing them.

```
Prover                           Verifier
──────                           ────────
source.get(poly) → buffer        (skip)
backend.make_witness → dispatch  cursor.next_sumcheck_proof() → verify rounds
PCS::commit(poly)                cursor.next_commitment() → store
PCS::open(poly, point)           PCS::verify(commitment, point, proof)
transcript.append(result)        transcript.append(result)  ← SAME
transcript.challenge()           transcript.challenge()     ← SAME
```

The verifier never touches the compute backend. It only needs: the staged graph,
the proof, and the PCS verifier key.

## ML Compiler Correspondence

```
ML Compiler                    SNARK Compiler
─────────────                  ──────────────
Model definition (PyTorch)     Protocol (polynomial identities)
Computation graph (ONNX/HLO)   Claim graph (sumcheck/opening/commit claims)
Fusion + scheduling passes     Stage assignment + batching + commitment grouping
  (or: user-placed pragmas)      (or: scheduling hints)
Fused computation graph        Staged graph (Fiat-Shamir epochs)
Op lowering (TIR/LLVM IR)      Kernel compilation + buffer planning
Device codegen (PTX/MSL)       Backend dispatch (CPU closures / GPU shaders)
Runtime executor (TVM RPC)     prove() graph walk
```

The key difference from ML compilers: SNARK provers have an additional sequential
constraint (Fiat-Shamir) that is strictly stronger than data dependency. Two ops
might have no data dependency but still require sequential execution because op B's
challenges depend on op A's transcript contribution.

The compiler makes this explicit: stage boundaries ARE Fiat-Shamir boundaries ARE
host-device sync points. Within a stage, everything is parallel.

The hints are analogous to user-placed scheduling pragmas in ML compilers (TVM's
`te.schedule`, XLA's sharding annotations). They guide the compiler without
changing the semantics. The compiler validates them and rejects unsound schedules.

## Resolved Design Decisions

These decisions were resolved during architecture discussion:

### Claim reductions: hinted v1, derived later

Claim reductions (registers CR, instruction lookups CR, RAM RA CR, etc.) exist to
reduce multiple eval claims down to terminal PCS opening proofs — eval proofs are
expensive, so the protocol batches them via sumcheck. In v1, claim reductions are
specified via hints (which claims reduce together, in which stage). Later, the
compiler derives them automatically from the rule: "eval claims on virtual polynomials
at common points → reduction sumcheck."

### Spartan dissolution is a prerequisite

Spartan cannot remain an opaque vertex. Before building the compiler pipeline, the
current S1 special-case must be dissolved into explicit graph vertices:

- **Outer sumcheck**: formula = `Az·Bz - Cz`, claimed_sum = 0 (zero-check)
- **EdgeTransform**: `combined_partial_evaluate(R1CS, r_x)` materializes the inner input
- **Inner sumcheck**: formula = `combined_row · witness`, claimed_sum from outer
- **46 virtual polynomial evaluations**: explicit eval vertices at the inner challenge point

This requires:
- `jolt-r1cs` crate extraction (R1CS data structures, `combined_partial_evaluate`)
- New `PolynomialId` variants for R1CS-derived virtual polynomials (Az, Bz, Cz, combined_row)
- The inner sumcheck's challenge point `r_y` becomes the `r_cycle` that all other
  identities consume — this is a Fiat-Shamir dependency edge in the claim graph

### Weighting is part of the identity (it's math)

The weighting polynomial (Eq, EqPlusOne, Lt) is part of the polynomial identity's
mathematical meaning: `Σ eq(τ,x)·f(x) = 0` is a different identity from
`Σ lt(τ,x)·f(x) = 0`. The identity specifies the weighting TYPE. The eq POINT
(which challenge vector τ refers to) is determined by the stage assignment.

```rust
pub enum Weighting {
    /// Standard eq(τ, x) weighting.
    Eq,
    /// Dual-point: eq(τ, x) + eq(τ', x) (shift identity).
    EqPlusOne,
    /// Less-than polynomial: lt(τ, x) = Σ_{y<x} eq(τ, y).
    Lt,
    /// Explicit public polynomial (e.g., Lagrange basis).
    Derived,
}
```

Algorithm selection (dense/sparse/uni-skip, TensorSplit) is derived by the backend
from the formula structure and input metadata. This is NOT on the identity.

### First lowering expands to Fiat-Shamir limit

The protocol specifies identities at a high level (e.g., "RAM read-write checking"
is one identity). The claim graph expansion pass expands EVERYTHING to the
Fiat-Shamir limit of data dependencies:

- Multi-phase sumchecks (RAM RW: sparse-cycle → sparse-addr → dense) become
  separate claim nodes, one per phase, because each phase may need its own
  challenge point
- Claim reductions become explicit nodes
- Every eval claim becomes an explicit edge

The protocol author writes `ram_read_write()` as one identity. The expansion
pass produces 3 claim nodes (one per phase) + edges + EdgeTransform vertices
for sparse→dense materialization. Phase structure is conveyed via hints in v1.

### Input/output formulas are symbolically fixed

All formulas (input claims, output compositions, challenge slots) are fully
determined at compile time (graph construction). Only the concrete challenge
VALUES are runtime. This means:
- The compiler can statically verify formula consistency
- Kernel compilation is AOT (no JIT)
- The verifier can reconstruct formulas without the witness

### Backend factory is a parallel workstream

`make_witness` + `SumcheckWitness` associated type (Sin 7) and sparse backend
primitives (Sin 8) are needed but orthogonal to the compiler pipeline. The staged
graph executor can initially use the current `build_witness()` and `KernelEvaluator`
internally, driven by the compiler-produced staged graph. The backend factory
replaces these internals without changing the staged graph structure.

## Implementation Sequence

### Phase 0: Spartan dissolution (prerequisite)

**Before the compiler pipeline can exist, Spartan must not be special.**

#### 0a. Extract jolt-r1cs

Move from jolt-spartan:
- `R1CS` trait, `SimpleR1CS`
- `UniformSpartanKey` → `UniformR1cs`
- `combined_partial_evaluate` (becomes an EdgeTransform operation)
- Matrix MLE evaluation

New crate depends on: jolt-field, jolt-poly. Does NOT depend on jolt-sumcheck.

#### 0b. Decompose Spartan into graph vertices

Replace the opaque S1 vertex with explicit vertices in `build_jolt_protocol()`:

1. **Outer sumcheck vertex**: `formula = Az·Bz - Cz`, `claimed_sum = Constant(0)`,
   `weighting = Eq`, `num_vars = log_rows`
2. **Uni-skip first round**: backend detects `claimed_sum = 0` and applies analytically
3. **Product virtual remainder vertex**: handles the uni-skip → dense transition
4. **EdgeTransform vertex**: `combined_partial_evaluate(r1cs, r_x)` materializes
   `combined_row` polynomial for the inner sumcheck
5. **Inner sumcheck vertex**: `formula = combined_row · witness`,
   `claimed_sum = f(az_eval, bz_eval, cz_eval)`, `num_vars = log_cols`
6. **46 eval vertices**: inner sumcheck challenge point `r_y[:log_T]` = `r_cycle`,
   evaluate all virtual polynomials at this point

Wire into existing `prove_from_graph()` generic loop. Delete `SpartanProver` calls.

**Validates**: `cargo nextest run -p jolt-core muldiv --features host`

#### 0c. Backend factory: make_witness + sparse primitives

Parallel with 0a-0b. Add to `ComputeBackend`:

```rust
type SumcheckWitness<F: Field>: SumcheckCompute<F> + Send;

fn make_witness<F: Field>(
    &self,
    kernel: &Self::CompiledKernel<F>,
    inputs: Vec<Self::Buffer<F>>,
    challenges: &[F],
) -> Self::SumcheckWitness<F>;

type SparseBuffer<F: Field>: Send + Sync;

fn upload_sparse<F: Field>(&self, entries: &[(usize, Vec<F>)]) -> Self::SparseBuffer<F>;
fn sparse_reduce<F: Field>(...) -> Vec<F>;
fn sparse_bind<F: Field>(...);
```

Move `KernelEvaluator`, `InterpolationMode`, `ToomCookState` from jolt-zkvm into
jolt-cpu-kernels as internal state of `CpuSumcheckWitness`.

**Validates**: existing sumcheck correctness tests via make_witness path

### Phase 1: Protocol extraction

#### 1a. Define PolynomialIdentity type

In jolt-ir, add:

```rust
pub struct PolynomialIdentity {
    pub name: &'static str,
    pub formula: CompositionFormula,
    pub polynomials: Vec<PolynomialId>,
    pub claimed_sum: IdentityClaim,
    pub domain: DomainSpec,
    pub weighting: Weighting,
    pub produces: Vec<PolynomialId>,
}
```

#### 1b. Extract protocol declarations

Write `jolt_protocol(config) -> Vec<PolynomialIdentity>` by collecting the existing
claim definitions (instruction.rs, ram.rs, registers.rs, spartan.rs, etc.) with
their polynomial bindings and claimed sums. ~200 lines. Pure math, no scheduling.

#### 1c. Extract scheduling hints

Write `jolt_scheduling_hints(config) -> SchedulingHints` by extracting the stage
assignments, batch groups, commitment groups, and opening groups from the current
`build_jolt_protocol()`. ~200 lines. Pure data, no graph construction.

### Phase 2: Claim graph expansion

#### 2a. Define ClaimGraph types

`ClaimNode` enum, `ClaimEdge` types, `ClaimGraph` struct. In jolt-ir.

#### 2b. Implement expansion pass

`expand_to_claim_graph(protocol, hints) -> ClaimGraph`:

1. Each identity → SumcheckClaim (expand multi-phase identities per hint)
2. Each committed polynomial → CommitmentClaim
3. Each sumcheck output on virtual poly → recursive SumcheckClaim (claim reduction, per hint)
4. Each sumcheck output on committed poly → OpeningClaim
5. Compute Fiat-Shamir dependency edges

Phase hints control multi-phase expansion and claim reduction placement.

### Phase 3: Hinted compiler passes

#### 3a. Stage assignment pass

`assign_stages(claim_graph, hints) -> StagedClaimGraph`

Validate: no intra-stage Fiat-Shamir dependency. Apply hint assignments.

#### 3b. Batching pass

`batch_sumchecks(staged_graph, hints) -> BatchedGraph`

Validate: batch groups contain same-domain sumchecks. Apply hint groups.

#### 3c. Commitment grouping pass

`group_commitments(graph, hints) -> GroupedGraph`

Validate: every committed polynomial in exactly one group. Apply hint groups.

#### 3d. Opening reduction pass

`reduce_openings(graph, hints) -> ReducedGraph`

Insert RLC vertices per hint groups.

#### 3e. Edge transform insertion

`insert_edge_transforms(graph) -> StagedGraph`

Always automatic. Insert EdgeTransform where virtual polynomials need materialization.

### Phase 4: Wire to prover/verifier

#### 4a. Compiler entry point

```rust
pub fn compile(
    protocol: &[PolynomialIdentity],
    hints: &SchedulingHints,
) -> Result<StagedGraph, CompileError>
```

Chains: expand → assign_stages → batch → group_commitments → reduce_openings → edge_transforms.
Validates at every step.

#### 4b. Replace graph source in prover/verifier

Change `prove_from_graph()` and `verify_from_graph()` to consume `StagedGraph`
from the compiler instead of `ProtocolGraph` from the monolithic builder.

#### 4c. Bit-exact verification

The compiler-produced staged graph must produce identical Fiat-Shamir transcripts
as the current hand-built graph. Verified by muldiv E2E in both host and host,zk modes.

### Phase 5: Cleanup

#### 5a. Delete build_jolt_protocol()

Protocol + hints + compiler replaces it entirely.

#### 5b. Delete SpartanProver/SpartanVerifier

Replaced by standard graph vertices + jolt-r1cs data.

#### 5c. Delete KernelEvaluator from jolt-zkvm

Replaced by `B::SumcheckWitness<F>` from backend factory.

#### 5d. Flat proof structure

Replace `JoltProof` named fields with flat arrays consumed by cursor.

## Parallelism Map

```
Phase 0a (jolt-r1cs) ──────────────────┐
Phase 0b (Spartan dissolution) ────────┤
Phase 0c (backend factory) ────────────┤ can parallelize
                                       │
Phase 1 (protocol extraction) ─────────┘ depends on 0b
Phase 2 (claim graph expansion) ──────── depends on 1
Phase 3 (hinted compiler passes) ─────── depends on 2
Phase 4 (wire to prover/verifier) ────── depends on 3, 0c
Phase 5 (cleanup) ────────────────────── depends on 4
```

Critical path: `0b → 1 → 2 → 3 → 4`
Parallel: `0a`, `0c` run alongside critical path.

## Summary

```
Protocol (~10 identities)
  │  Hand-written. Pure math. Weighting is specified (it's math).
  │  No stages, batching, commitment groups, or algorithm choices.
  │
  ↓ mechanical expansion (deterministic, hint-guided for phases/reductions)
  │
Claim Graph (~50-100 nodes)
  │  All claims enumerated. Fiat-Shamir dependency edges computed.
  │  Multi-phase identities expanded. Claim reductions generated.
  │  No scheduling decisions.
  │
  ↓ compiler passes (hint-guided now, cost-model-driven later)
  │  1. Stage assignment    — hint or longest-path layering
  │  2. Batching            — hint or domain-compatible grouping
  │  3. Commitment grouping — hint or availability + memory analysis
  │  4. Opening reduction   — hint or point-based grouping
  │  5. Edge transforms     — always automatic
  │
Staged Graph (~100-200 nodes)
  │  Complete execution plan. Six vertex types.
  │  All scheduling decisions applied and validated.
  │
  ↓ backend-specific lowering (make_witness, compile_kernel)
  │
Computation Graph (~1000-5000 ops)
  │  Concrete kernels, buffer allocations, transfers.
  │  Profiling and optimization target.
  │
  ↓ runtime execution
  │
Device Schedule
     Thread groups, kernel launches, barriers.
     Backend-specific. Not explicitly represented.
```

The protocol author writes ~10 identities. The hint author (or future cost model)
specifies scheduling. The compiler validates and assembles. The executor runs.

### Prerequisite: Spartan dissolution + jolt-r1cs + backend factory

These are parallel workstreams that must complete before the compiler pipeline
can fully replace the current architecture. Spartan dissolution is on the critical
path; jolt-r1cs extraction and backend factory can parallelize alongside it.
