# Jolt Crate Rewrite

---

## Why

```
┌─────────────────────────────────────────────────────────────┐
│                    jolt-core (monolith)                      │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────┐  │
│  │ Sumcheck │──│ Spartan │──│  Dory   │──│  Witnesses   │  │
│  │          │  │         │  │         │  │              │  │
│  │ knows    │  │ knows   │  │ knows   │  │ knows        │  │
│  │ about    │  │ about   │  │ about   │  │ about        │  │
│  │ Dory,    │  │ field   │  │ sumcheck│  │ Dory commit, │  │
│  │ witness  │  │ layout, │  │ rounds, │  │ prover       │  │
│  │ layout   │  │ Dory    │  │ witness │  │ pipeline     │  │
│  └─────────┘  └─────────┘  └─────────┘  └──────────────┘  │
│                                                             │
│  Everything depends on everything. One config hardcoded.    │
└─────────────────────────────────────────────────────────────┘
```

| Problem | Consequence |
|---------|------------|
| Algorithm implementations leak into each other | Can't change sumcheck without touching Spartan, Dory, witnesses |
| Abstractions encode *one* configuration | Adding a second PCS (Hachi) or backend (Metal) means huge diff, not plugging in |
| Verifier drags in prover + rayon | Can't embed verifier in constrained environments (mobile, on-chain, WASM) |

**Goal: express a *family* of proving systems, not just one.**

---

## Crate Map

### Primitives, no protocol awareness

| Crate | Role |
|-------|------|
| `jolt-field` | Field trait + BN254 scalar field |
| `jolt-transcript` | Fiat-Shamir (Blake2b, Keccak) |
| `jolt-crypto` | Group/pairing traits, Pedersen — zero arkworks leakage |
| `jolt-poly` | Polynomial types (Dense, Compact, Eq, Multilinear) |
| `jolt-profiling` | Tracing instrumentation |

### Protocol, no compute awareness

| Crate | Role |
|-------|------|
| `jolt-openings` | PCS trait + claim types + opening reduction |
| `jolt-sumcheck` | Sumcheck engine (batched, streaming, uni-skip) |
| `jolt-r1cs` | R1CS constraint data (sparse matrices, MLE eval) — extracted from old jolt-spartan |
| `jolt-blindfold` | ZK layer (Pedersen-committed sumcheck + Nova folding) |
| `jolt-instructions` | RISC-V instruction set definitions |

### PCS Backends

| Crate | Role |
|-------|------|
| `jolt-dory` | Dory commitment scheme |
| `jolt-hyperkzg` | HyperKZG commitment scheme |
| `Hachi?` | 
| `Hashes?` | 

### Compiler (protocol definition → dispatch plan)

| Crate | Role |
|-------|------|
| `jolt-compiler` | Protocol spec (L0) → Fiat-Shamir schedule (L1) → execution plan (L2). Zero deps — pure data. |

### Compute (how to execute kernels)

| Crate | Role |
|-------|------|
| `jolt-compute` | Backend-agnostic trait: `compile_kernel(formula)` + `make_witness(kernel, data)` |
| `jolt-cpu` | CPU backend (Rayon, SIMD, Toom-Cook) |
| `jolt-metal` | Metal GPU backend |

### Orchestration (puts it all together)

| Crate | Role |
|-------|------|
| `jolt-host` | Guest ELF compilation, RISC-V trace execution |
| `jolt-witness` | Trace → multilinear evaluation tables |
| `jolt-zkvm` | Prover: walks L2 execution plan, dispatches to compute backend |
| `jolt-verifier` | Verifier: walks L1 staged protocol, replays Fiat-Shamir (no rayon, no compute) |
| `jolt-wrapper` | Verifier wrapping: symbolic exec + gnark codegen |

---

## Dependency Graph

```
                     ┌─────────────────────────────────┐
                     │          jolt-zkvm (prover)      │
                     │  generic over PCS + Backend      │
                     └───┬──────────┬──────────┬────────┘
                         │          │          │
              ┌──────────▼┐   ┌────▼─────┐  ┌─▼────────────┐
              │jolt-witness│   │ verifier │  │ jolt-compute  │
              │            │   │          │  │ (backend trait)│
              │trace→tables│   │no compute│  └───┬──────────┘
              └──────┬─────┘   │no rayon  │      │
                     │         └────┬─────┘  ┌───┼──────┐
              ┌──────▼─────┐       │        ┌▼─┐ ┌▼───┐ ┌▼────┐
              │  jolt-host │       │        │cpu│ │mtl │ │cuda │
              │(trace exec)│       │        └──┘ └────┘ └─────┘
              └────────────┘       │
                                   │
               ┌───────────────────▼───────────────────┐
               │          Protocol layer                │
               │                                        │
               │  blindfold → sumcheck → openings       │
               │               r1cs (data only)         │
               └───────────────────┬───────────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                  │
          ┌──────▼────┐    ┌──────▼──────┐    ┌──────▼──────┐
          │ jolt-poly │    │ PCS backends│    │ jolt-crypto │
          └─────┬─────┘    │ dory,       │    └──────┬──────┘
                │          │ hyperkzg    │           │
                │          └─────────────┘    ┌──────▼──────┐
                │                             │ transcript  │
                └──────────┬──────────────────┴──────┘
                           │
                     ┌─────▼─────┐
                     │ jolt-field │  (leaf)
                     └───────────┘

 Separately (zero or minimal deps):

 ┌────────────────┐              ┌──────────────┐
 │ jolt-compiler  │─────────────▶│ jolt-compute │  (compute needs formula types)
 │ (zero deps)    │              │ (+ field)    │
 │ L0 → L1 → L2  │              └──────────────┘
 └────────────────┘
 ┌──────────────┐
 │ jolt-wrapper │  (field, transcript — gnark codegen)
 └──────────────┘
```

```
Key insights:
- jolt-compiler has zero dependencies — pure data structures, usable by anything
- Protocol crates never import jolt-cpu or jolt-metal
- Verifier never imports jolt-compute — it only needs L1 (staged protocol)
- Compute backends only pulled in by jolt-zkvm (prover) at the top
```

---

## End-to-End Data Flow

```
                              ┌────────────────┐
                              │ jolt-compiler  │
                              │                │
                              │ L0 (protocol)  │
                              │   ↓ compile    │
                              │ L1 (stages)  ──┼───────────────────────────────┐
                              │   ↓ lower      │                               │
                              │ L2 (ops)  ─────┼──────────────┐                │
                              └────────────────┘              │                │
                                                              ▼                ▼
 ┌───────┐     ┌───────────┐   ┌────────────┐  ┌──────────────┐   ┌─────┐  ┌──────────┐
 │ .rs   │────▶│ jolt-host │──▶│jolt-witness│─▶│  jolt-zkvm   │──▶│     │─▶│ jolt-    │
 │ guest │     │           │   │            │  │  (prover)    │   │proof│  │ verifier │
 └───────┘     │ RISC-V    │   │ trace →    │  │              │   └─────┘  │          │
               │ emulator  │   │ MLE tables │  │ walks L2 ops │            │walks L1  │
               └───────────┘   └────────────┘  │ dispatches to│            │stages    │
                                               │ backend      │            │no compute│
                                               └──────────────┘            └──────────┘
```

---

## The Compiler Pipeline (centerpiece)

### The core insight

Every computation in a SNARK prover follows the same pattern:

```
resolve inputs → dispatch to device → small result back to host → Fiat-Shamir → propagate
```

**The Fiat-Shamir boundary IS the host-device sync boundary.**
This is exactly the ML compiler execution model (XLA / TVM / Triton).

| ML compiler | SNARK prover |
|-------------|-------------|
| Tensor operation | Cryptographic operation |
| GPU kernel dispatch | Backend kernel dispatch |
| Host sync for control flow | Host sync for Fiat-Shamir |
| Kernel fusion across sync-free regions | Kernel fusion across transcript-free regions |
| Memory planning (tensor lifetimes) | Buffer planning (polynomial lifetimes) |

### Progressive Lowering (L0 → L1 → L2)

```
    L0: Protocol                           (pure math — a DAG of claims)
    ─────────────────────────────────────────────────────────────────────
    Vertices: Sumcheck, Evaluate
    Expressions: sum-of-products over Poly / Challenge / Claim handles
    ~10 identities define ALL of Jolt

    Builder API:
      let eq  = proto.poly("eq", &[n], PublicPoly::Eq);
      let az  = proto.poly("Az", &[n], Virtual);
      let bz  = proto.poly("Bz", &[n], Virtual);
      let cz  = proto.poly("Cz", &[n], Virtual);
      let outer = proto.sumcheck(eq * (az * bz - cz), zero, &[n]);

         │  compiler: validate → analyze → stage → batch → reduce
         │  cost-driven: searches for staging that minimizes proof size
         ▼

    L1: Staged Protocol                    (Fiat-Shamir schedule)
    ─────────────────────────────────────────────────────────────────────
    Stages (FS rounds), Batches (α-weighted groups), Reductions, Opening RLC
    → Consumed by verifier: knows proof layout, never sees kernels or buffers

         │  compiler: plan execution → emit ops + kernels
         ▼

    L2: Execution Plan                     (concrete dispatch plan)
    ─────────────────────────────────────────────────────────────────────
    Ops: Dispatch, Bind, Materialize, DeriveChallenge, Transcript, Upload, Free
    KernelDefs: composition expr extracted from each sumcheck vertex
    Buffer lifecycle tracked: Upload → use → Free
    → Consumed by prover: never reasons about protocol structure

         │  ComputeBackend: compile_kernel(composition) → AOT
         ▼

    Hardware                               (run it)
    ─────────────────────────────────────────────────────────────────────
    CPU:   CpuKernel (Rayon + SIMD closure)
    Metal: MSL shader → MTLComputePipelineState
    CUDA:  PTX kernel (future)
```

### What dissolves

```
Before (jolt-core)                      After (compiler + compute)
─────────────────                       ──────────────────────────
Spartan prover/verifier (1,693 LOC)     → sumcheck vertices in L0 + jolt-r1cs data
Per-stage hand-written prover code      → prover walks L2 ops, one generic loop
Per-stage hand-written verifier code    → verifier walks L1 stages, one generic loop
KernelEvaluator + 4-arm match          → backend.make_witness(kernel, data)
Phased execution (sparse→dense)         → multiple L0 vertices, backend picks algorithm
7 named fields in JoltProof             → flat stream consumed by cursor in L1 order
Protocol definition interleaved         → L0 is pure math, scheduling is separate hints
  with scheduling decisions               to the compiler
```

### ComputeBackend trait (simplified)

```rust
trait ComputeBackend {
    type CompiledKernel<F>: Send + Sync;
    type SumcheckWitness<F>: SumcheckCompute<F>;

    /// AOT at setup: L2 KernelDef → compiled kernel.
    /// ~15-25 unique kernels for all of Jolt.
    /// Backend inspects formula → picks algorithm, reconstruction, grid.
    fn compile_kernel<F>(&self, composition: &Expr) -> Self::CompiledKernel<F>;

    /// Runtime per-vertex: pair compiled kernel + data → SumcheckCompute impl
    /// that jolt-sumcheck's round loop drives.
    fn make_witness<F>(
        &self,
        kernel: &Self::CompiledKernel<F>,
        inputs: Vec<Self::Buffer<F>>,
        challenges: &[F],
    ) -> Self::SumcheckWitness<F>;
}
```

```
Compiler (L0) declares WHAT: formula + input bindings + claimed sum.
Compiler (L2) emits KernelDefs: one composition Expr per sumcheck vertex.
Backend decides HOW: algorithm (dense/sparse/uni-skip), reconstruction, grid.
Same L2 plan compiles to CPU closures or Metal shaders.
```

---

## Prover consumes L2 (ExecutionPlan)

The compiler already decided everything. The prover just walks and executes.

```rust
fn prove<PCS, B: ComputeBackend>(
    plan: &ExecutionPlan,          // L2: from jolt-compiler
    backend: &B,
    source: &impl PolynomialSource,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut impl Transcript,
) -> Proof {
    // AOT: compile all kernels at setup (~15-25 total)
    let kernels: Vec<_> = plan.kernels.iter()
        .map(|k| backend.compile_kernel(&k.composition))
        .collect();

    let mut slots = SlotMap::new();   // runtime values (scalars, buffers, challenges)

    // One flat loop over ops. No branching on protocol structure.
    for op in &plan.ops {
        match op {
            Dispatch { kernel, inputs, output } => {
                let witness = backend.make_witness(
                    &kernels[kernel],
                    slots.resolve_buffers(inputs),
                    &slots.resolve_challenges(op),
                );
                let (proof, evals) = sumcheck::prove(&witness, transcript);
                slots.store(*output, evals);
            }

            Bind { buffer, challenge } =>
                backend.bind(&mut slots[buffer], slots[challenge]),

            Materialize { transform, inputs, output } =>
                slots.store(*output, transform.execute(&slots, source)),

            DeriveChallenge { output } =>
                slots.store(*output, transcript.challenge()),

            Transcript { value } =>
                transcript.append(&slots[value]),

            Upload { poly, output } =>
                slots.store(*output, backend.upload(source.get(poly))),

            Free { slot } =>
                slots.free(*slot),
        }
    }
}
```

---

## Verifier consumes L1 (StagedProtocol)

The verifier doesn't need L2 — it never dispatches kernels. It replays Fiat-Shamir
over the staged protocol structure.

```rust
fn verify<PCS>(
    staged: &StagedProtocol,       // L1: from jolt-compiler
    proof: &Proof,
    vk: &VerifyingKey<PCS>,
    transcript: &mut impl Transcript,
) -> Result<(), VerifyError> {
    let mut cache = EvalCache::new();
    let mut cursor = proof.cursor();   // flat stream, consumed in stage order

    for stage in &staged.stages {
        // Each stage = one Fiat-Shamir round
        for batch in &stage.batches {
            // Batched sumcheck verify — cheap, just field ops
            let round_polys = cursor.next_round_polys();
            let (final_eval, challenges) = sumcheck::verify(
                &batch.claimed_sum(&cache), &round_polys, transcript,
            )?;

            // Check composition evaluates correctly at the binding point
            let evals = cursor.next_evals(batch.num_produced_claims());
            let expected = batch.composition.evaluate(&evals, &cache);
            if expected != final_eval { return Err(EvaluationMismatch); }

            cache.store_evals(batch, &evals);
            cache.store_point(batch, challenges);
        }

        // Reductions within stage
        for reduction in &stage.reductions {
            let r = transcript.challenge();
            cache.store(reduction.id, rlc_reduce(&reduction.claims, r, &cache));
        }
    }

    // Final opening RLC + PCS verify
    let opening_rlc = staged.opening_rlc.reduce(&cache, transcript);
    PCS::verify_batch(&opening_rlc, &cursor.next_opening_proof(), &vk.pcs, transcript)?;

    Ok(())
}
```

```
Key:
- Prover walks L2 (ops) — never sees protocol structure, just dispatch instructions
- Verifier walks L1 (stages) — never imports jolt-compute, jolt-cpu, or rayon
- Proof is a flat stream — no named fields, graph structure defines proof layout
- The compiler is the bridge: same L0 protocol lowers to L1 (verifier) and L2 (prover)
```

---

## Status

| Layer | State | Notes |
|-------|-------|-------|
| Foundation (field, transcript, crypto, poly) | Done | Stable, benchmarked |
| Protocol (sumcheck, r1cs, blindfold, openings) | Done | Extracted from jolt-core |
| PCS (dory, hyperkzg) | Done | Pluggable behind `CommitmentScheme` trait |
| Compiler (L0→L1→L2) | WIP | Types + lowering passes in progress |
| Compute backends | WIP | CPU done, Metal in progress |
| Witness | Done | Generic TraceSource / WitnessSink |
| Prover (jolt-zkvm) | WIP | Moving to L2-driven executor |
| Verifier | Done | L1-driven, no prover deps |
| E2E muldiv test | Passing | Both standard and ZK mode |
