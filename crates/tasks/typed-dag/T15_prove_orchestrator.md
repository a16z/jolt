# T15: Prove Orchestrator

**Status**: `[ ]` Not started
**Depends on**: T07–T14 (all stages + PCS opening)
**Blocks**: T16 (Verifier DAG), T17 (E2E)
**Crate**: `jolt-zkvm`
**Estimated scope**: Medium (~150 lines)

## Objective

Wire all stage functions into the top-level `prove()` function.
This IS the DAG — the explicit sequence of function calls.

## Deliverables

### 1. `prove()` function

```rust
pub fn prove<PCS, B>(
    tables: &PolynomialTables<Fr>,
    config: &ProverConfig,
    transcript: &mut impl Transcript<Challenge = Fr>,
    backend: &Arc<B>,
) -> JoltProof<Fr>
where
    PCS: CommitmentScheme<Field = Fr> + AdditivelyHomomorphic,
    B: ComputeBackend,
{
    let commitments = commit_polynomials::<PCS>(tables, config);

    let s1 = prove_spartan(tables, config, transcript, backend);
    let s2 = prove_stage2(&s1, tables, config, transcript, backend);
    let s3 = prove_stage3(&s1, &s2, tables, config, transcript, backend);
    let s4 = prove_stage4(&s2, &s3, tables, config, transcript, backend);
    let s5 = prove_stage5(&s2, &s4, tables, config, transcript, backend);
    let s6 = prove_stage6(&s2, &s4, &s5, tables, config, transcript, backend);
    let s7 = prove_stage7(&s5, &s6, tables, config, transcript, backend);
    let opening = prove_opening::<PCS>(&s6, &s7, tables, &commitments, transcript);

    JoltProof { s1, s2, s3, s4, s5, s6, s7, opening, commitments }
}
```

### 2. `JoltProof` struct

```rust
pub struct JoltProof<F: Field> {
    pub s1: SpartanOutput<F>,
    pub s2: Stage2Output<F>,
    pub s3: Stage3Output<F>,
    pub s4: Stage4Output<F>,
    pub s5: Stage5Output<F>,
    pub s6: Stage6Output<F>,
    pub s7: Stage7Output<F>,
    pub opening: DoryProof,      // PCS::Proof
    pub commitments: Vec<...>,   // PCS::Output
}
```

### 3. `commit_polynomials()` function

Commits all committed polynomials in the correct order:
1. RamInc, RdInc (dense)
2. InstructionRa[0..d], BytecodeRa[0..d], RamRa[0..d] (sparse)

Returns commitments + hints for the PCS opening step.

### 4. Entry point from host layer

Wire `prove()` into the host-level API that takes a trace and produces
a proof. This replaces the current `prove_pipeline()`.

## Notes

- The DAG wiring is trivially correct if all stage output types are right.
- The orchestrator should be simple — the complexity lives in the stages.
- Consider `tracing::instrument` on each stage call for profiling.
- Memory: stage outputs should be dropped as soon as no downstream stage
  needs them. The borrow checker enforces this naturally.

## Acceptance Criteria

- [ ] `prove()` compiles with all stages wired
- [ ] `JoltProof` serializable
- [ ] Commitment order matches verifier expectations
- [ ] `cargo clippy -p jolt-zkvm` passes
