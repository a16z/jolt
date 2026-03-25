# Implementation Plan: Unified Execution Model

No backwards compatibility concerns. Aggressive rewrite.

## Phase 1: IR Foundation

### 1a. CompositionFormula replaces KernelShape (Sin 0)

**Crate**: jolt-ir

- Delete `KernelShape` enum (`ProductSum`, `EqProduct`, `HammingBooleanity`, `Custom`)
- Delete `KernelDescriptor` struct (shape + degree + tensor_split)
- Delete `EvalGrid`, `EqHandling`, `TensorSplit` from jolt-ir
- Introduce `CompositionFormula`:
  ```rust
  pub struct CompositionFormula {
      /// Sum of product terms. Each term: coefficient × product of input indices.
      pub terms: Vec<ProductTerm>,
      /// Number of input polynomials.
      pub num_inputs: usize,
      /// Number of symbolic challenge slots (resolved at runtime).
      pub num_challenges: usize,
  }
  pub struct ProductTerm {
      pub coeff: Coefficient,
      pub factors: Vec<Factor>,
  }
  pub enum Factor {
      Input(usize),
      Challenge(usize),
  }
  pub enum Coefficient {
      One,
      MinusOne,
      Literal(i64),
  }
  ```
- `ClaimDefinition::compile_descriptor()` → delete entirely. The formula IS the descriptor.
- `ClaimDefinition` keeps its `Expr`-based symbolic representation for building formulas,
  but adds `fn to_composition_formula() -> CompositionFormula` that normalizes to SoP.
- All tests that use `KernelShape` switch to `CompositionFormula`.

**Depends on**: nothing
**Validates**: `cargo clippy -p jolt-ir`, unit tests

### 1b. Clean IR/compute boundary (Sin 5)

**Crates**: jolt-ir, jolt-compute

- jolt-ir exports: `CompositionFormula`, `ProtocolGraph`, `Vertex`, `PolynomialId`, `ClaimDefinition`
- jolt-ir does NOT export: anything related to kernel shapes, eval grids, eq handling,
  optimization hints
- `jolt-compute::ComputeBackend::compile_kernel` changes signature:
  ```rust
  fn compile_kernel<F: Field>(&self, formula: &CompositionFormula) -> Self::CompiledKernel<F>;
  ```
  Takes `CompositionFormula` (from jolt-ir) instead of `KernelDescriptor`.
- Delete `jolt_ir::KernelDescriptor` — consumers use `CompositionFormula` directly.

**Depends on**: 1a
**Validates**: `cargo clippy -p jolt-compute -p jolt-ir`

### 1c. Eq unification (Sin 1)

**Crate**: jolt-compute

- Collapse 3 methods into 1:
  ```rust
  fn pairwise_reduce<F: Field>(
      &self,
      inputs: &[&Self::Buffer<F>],
      eq: EqInput<'_, Self, F>,
      kernel: &Self::CompiledKernel<F>,
      challenges: &[F],
      num_evals: usize,
      order: BindingOrder,
  ) -> Vec<F>;
  ```
  where:
  ```rust
  enum EqInput<'a, B: ComputeBackend, F: Field> {
      None,
      Buffer(&'a B::Buffer<F>),
      Tensor { outer: &'a B::Buffer<F>, inner: &'a B::Buffer<F> },
  }
  ```
- Delete `pairwise_reduce_unweighted`, `tensor_pairwise_reduce`.
- Backend impl detects eq variant from `EqInput` and dispatches internally.
- Backend's `compile_kernel` pattern-matches formula to detect eq structure
  and decides which `EqInput` variant to expect — but this is internal.

**Depends on**: 1b
**Validates**: `cargo clippy -p jolt-compute -p jolt-cpu-kernels`, kernel unit tests

## Phase 2: Backend Factory

### 2a. Sparse as backend primitive (Sin 8)

**Crate**: jolt-compute, jolt-cpu-kernels

- Add to `ComputeBackend`:
  ```rust
  type SparseBuffer<F: Field>: Send + Sync;

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
- CPU impl: `SparseBuffer<F> = Vec<SparseEntry<F>>` where entries are sorted by position.
  This is what `SparseRwEvaluator` does today — moved behind the trait.

**Depends on**: 1c
**Validates**: sparse reduce unit tests, RAM read-write correctness

### 2b. Backend factory: make_witness + SumcheckWitness (Sin 7)

**Crate**: jolt-compute, jolt-cpu-kernels

- Add associated type and factory method:
  ```rust
  type SumcheckWitness<F: Field>: SumcheckCompute<F> + Send;

  fn make_witness<F: Field>(
      &self,
      kernel: &Self::CompiledKernel<F>,
      inputs: Vec<Self::Buffer<F>>,
      eq: EqInput<'_, Self, F>,
      challenges: &[F],
  ) -> Self::SumcheckWitness<F>;
  ```
- CPU impl: `CpuSumcheckWitness<F>` — holds buffers + kernel ref + internal
  reconstruction state (Toom-Cook or standard, determined from compiled kernel).
  Implements `SumcheckCompute<F>`.
- Move `InterpolationMode`, `ToomCookState`, `toom_cook_reconstruct` from
  jolt-zkvm/evaluators/kernel.rs into jolt-cpu-kernels as internal state of
  `CpuSumcheckWitness`.

**Depends on**: 2a, 1c
**Validates**: all sumcheck correctness tests via make_witness path

### 2c. AOT compilation with runtime challenges

**Crate**: jolt-compute, jolt-cpu-kernels

- `compile_kernel` takes `&CompositionFormula` only — no challenge values.
- `make_witness` (and `pairwise_reduce`) take `challenges: &[F]` as runtime arguments.
- CPU impl: kernel closure captures formula structure, challenge slots resolved at
  dispatch time via indexing into the challenges slice.
- Verify: no calls to `compile_kernel_with_challenges` remain. All kernel compilation
  happens at setup time.

**Depends on**: 2b
**Validates**: muldiv E2E test

## Phase 3: Graph Model

### 3a. Expand vertex types in ProtocolGraph (Sin 3)

**Crate**: jolt-ir

- Redefine `Vertex` enum:
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
  Where `SumcheckVertex` carries only: `formula: CompositionFormula`,
  `input_bindings: Vec<PolynomialId>`, `claimed_sum: InputClaim`.
  No `degree`, `num_vars`, `phases`, `weighting` — backend derives these.
- `CommitVertex`: `polynomial: PolynomialId`
- `OpeningVertex`: `polynomial: PolynomialId`, `point: SymbolicPoint`, `eval: ClaimId`
- `RlcVertex`: `inputs: Vec<RlcInput>` where `RlcInput` can be claims or commitments
- `EdgeTransformVertex`: `transform: TransformKind`, `inputs/outputs`
- Build the graph with Commit vertices before sumcheck vertices, Opening vertices
  after, RLC vertices between openings and PCS verify.

**Depends on**: 1a, 1b
**Validates**: graph construction tests, `cargo clippy -p jolt-ir`

### 3b. Extract jolt-r1cs (Sin 10)

**New crate**: jolt-r1cs

- Move from jolt-spartan:
  - `R1CS` trait, `SimpleR1CS`
  - `UniformSpartanKey` → `UniformR1cs`
  - `combined_partial_evaluate_uniform` → `combined_partial_evaluate`
  - Matrix MLE evaluation (`evaluate_local_mles`, `evaluate_full_mles`)
- Move from jolt-spartan: `ir_r1cs.rs` bridge (R1csEmission implements R1CS)
- jolt-r1cs depends on: jolt-field, jolt-poly, jolt-ir (for R1csEmission bridge)
- jolt-r1cs does NOT depend on: jolt-sumcheck, jolt-transcript, jolt-compute

**Depends on**: nothing (can parallelize with other phases)
**Validates**: `cargo clippy -p jolt-r1cs`, existing Spartan unit tests migrated

### 3c. Wire Spartan as graph vertices (Sin 10)

**Crates**: jolt-ir (graph construction), jolt-zkvm (prover)

- Add Spartan outer + inner sumcheck vertices to the protocol graph where S1 currently
  is special-cased.
- Outer vertex: formula = `Az * Bz - Cz`, claimed_sum = Constant(0) (zero-check),
  input_bindings = [Az, Bz, Cz] (virtual polynomials from R1CS × witness).
- EdgeTransform between outer and inner: `combined_partial_evaluate` from jolt-r1cs.
- Inner vertex: formula = `combined_row * witness`, claimed_sum from predecessor.
- Opening vertex: `z(r_y)` via standard PCS.
- Delete `jolt-spartan`'s prover code (`SpartanProver`, `UniformSpartanProver`,
  `OuterSumcheckCompute`, `InnerSumcheckCompute`).
- Delete S1 special case from `prove_from_graph` in jolt-zkvm.

**Depends on**: 3a, 3b, 2b
**Validates**: muldiv E2E test — this is the critical correctness check

### 3d. PolynomialSource unification (Sin 4)

**Crate**: jolt-zkvm (or new jolt-witness extension)

- Introduce:
  ```rust
  trait PolynomialSource<F> {
      fn get(&self, id: PolynomialId) -> &[F];
      fn materialize(&self, id: PolynomialId) -> Vec<F>;
  }
  ```
- `UnifiedSource` wraps: committed store + trace polys + R1CS virtual polys.
- Delete all `is_committed() && != SpartanWitness` branching.
- The source knows how to provide any polynomial by ID — committed from store,
  virtual from trace, R1CS from jolt-r1cs.

**Depends on**: 3c (Spartan resolved)
**Validates**: all E2E tests

### 3e. Commit + Opening + RLC as graph vertices

**Crates**: jolt-ir (graph construction), jolt-zkvm (prover), jolt-verifier

- Add Commit vertices before sumcheck stages.
- Add Opening vertices after claim reductions.
- Add RLC vertices between opening claims and PCS proofs.
- RLC vertex is general: can batch opening claims (scalar), commitments (group), or both.
- Delete commitment code from `prove_from_graph` that's outside the loop.
- Delete PCS opening code that's after the loop.
- Everything goes through the graph walk.

**Depends on**: 3a
**Validates**: muldiv E2E test

## Phase 4: Prover + Verifier Rewrite

### 4a. Prover: generic graph executor

**Crate**: jolt-zkvm

- `prove_from_graph` becomes:
  ```rust
  fn prove_from_graph<PCS, B>(
      graph, kernels, source, backend, pcs_setup, transcript
  ) -> Proof<F, PCS>
  ```
  Single `match vertex` loop — no per-stage code, no S1 special case.
- Delete `build_witness()` entirely — replaced by `backend.make_witness()`.
- Delete `KernelEvaluator`, `InterpolationMode`, `ToomCookState` from jolt-zkvm.
- Delete `SparseRwEvaluator`, `PhasedEvaluator`, `SegmentedEvaluator` from jolt-zkvm.
- Setup function: walk graph, `backend.compile_kernel()` per sumcheck vertex → kernel map.

**Depends on**: 2b, 3a, 3c, 3d, 3e
**Validates**: muldiv E2E in both `host` and `host,zk` modes

### 4b. Flat proof structure

**Crate**: jolt-verifier (proof types)

- Replace `JoltProof` (7 named fields) with:
  ```rust
  struct Proof<F, PCS> {
      commitments: Vec<PCS::Output>,
      sumcheck_proofs: Vec<SumcheckProof<F>>,
      evals: Vec<F>,
      opening_proofs: Vec<PCS::Proof>,
  }
  ```
- Cursor-based consumption: `ProofCursor` walks arrays in graph topological order.
- Prover writes into the same flat structure.
- Delete `StageProof`, `spartan_proof`, `spartan_evals` fields.

**Depends on**: 4a
**Validates**: muldiv E2E (prover produces, verifier consumes same format)

### 4c. Verifier: generic graph executor

**Crate**: jolt-verifier

- `verify_from_graph` becomes the single `match vertex` loop.
- No `verify_spartan()` — Spartan is just Sumcheck vertices.
- No per-stage code.
- Consumes `Proof` via cursor in graph order.
- VerifyingKey: `graph + r1cs + pcs_setup`.
- Delete `verifier::verify_spartan()`, Spartan verifier wrappers.

**Depends on**: 4b
**Validates**: muldiv E2E

## Phase 5: Cleanup

### 5a. Delete jolt-spartan prover/verifier

- Keep only what moved to jolt-r1cs.
- Delete: `SpartanProver`, `SpartanVerifier`, `UniformSpartanProver`, `UniformSpartanVerifier`,
  `OuterSumcheckCompute`, `InnerSumcheckCompute`, `RelaxedOuterSumcheckCompute`,
  `proof.rs`, `uni_skip.rs` (uni-skip logic moves to backend).
- jolt-spartan crate either deleted entirely or kept as a thin recipe crate for external
  consumers (`jolt-r1cs` + `jolt-sumcheck` + PCS = standalone Spartan SNARK).

### 5b. Delete dead code in jolt-zkvm

- Delete `evaluators/kernel.rs` (KernelEvaluator)
- Delete `evaluators/sparse.rs` (SparseRwEvaluator) if it exists
- Delete phased/segmented evaluator code
- Delete `witness_builder` code that's replaced by `PolynomialSource`
- Delete anything that references `KernelShape`, `KernelDescriptor`, `compile_descriptor`

### 5c. BlindFold adaptation

- BlindFold currently uses `SpartanProver::prove_relaxed()`. In the new model, the
  relaxed Spartan proof is also graph vertices — same pattern, different formula
  (includes the relaxation error term `u·(Az·Bz - Cz) + E(x)`).
- This is deferred — get standard mode working first, then layer ZK.

## Validation Strategy

Every phase validates with:
1. `cargo clippy` on affected crates (with `--message-format=short -q -- -D warnings`)
2. `cargo fmt`
3. `cargo nextest run -p <crate>` for unit tests

Critical E2E checkpoints:
- **After 3c** (Spartan wired as graph vertices): `muldiv --features host` must pass
- **After 4a** (prover rewrite): `muldiv --features host` must pass
- **After 4c** (verifier rewrite): `muldiv --features host` must pass
- **After 5c** (BlindFold): `muldiv --features host,zk` must pass

## Parallelism Opportunities

These can run concurrently:
- 1a + 3b (CompositionFormula + jolt-r1cs extraction — independent crates)
- 2a + 3a (sparse backend + graph vertex types — independent crates)
- 4b + 5a (flat proof structure + jolt-spartan deletion — independent)

Critical path:
```
1a → 1b → 1c → 2b → 2c → 3c → 4a → 4c
                 ↗ 2a ↗         ↗ 4b ↗
          3b ──────────↗
          3a ──────────↗
```

Estimated: 5 sequential phases, ~15 parallelizable work items.
