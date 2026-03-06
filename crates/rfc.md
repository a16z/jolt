
## Why this RFC?
Jolt audits are expected to begin in April-May. It is very important that the codebase is in good shape before the audits. In doing so, we: (1) make it easier to define scope of audits (2) prevent unnecessary re-auditing in the future and (3) make the audit process itself pleasant.

We also need to modernize the codebase to be AI friendly. This is more or less proportional to just having good codebase practices, but with emphasis on imposing structure, smaller files, linters, and providing plenty of context to the AI.

## Audit-readiness criteria
The codebase should be (1) well encapsulated (2) cohesive and (3) modular. Below we analyze Jolt's architecture, identify structural problems, and propose a path to fixing them before audits.

## Analysis

#### Dependency tree of `jolt-core`

```
transcripts        (no deps)
    |
    v
field / utils / guest
    |
    v
poly
  - commitment/ (dory, kzg, hyperkzg, mock)
  - opening_proof.rs
    |
    v
msm
    |
    v
subprotocols       (sumcheck, streaming, uniskip, booleanity)
    |
    v
zkvm
  - r1cs/
  - spartan/
  - ram/
  - registers/
  - bytecode/
  - instruction/  
  - claim_reductions/
  - prover.rs     
  - verifier.rs 
    |
    v
external: ark-* (a16z fork), dory-pcs, tracer, common
```

---

## Findings

#### (1) Coupling of PIOP to Opening Proofs — **ADDRESSED (redesigned)**
- ~~Commitment phase and opening proof data structures / traits assume a dory-like commitment scheme~~
- ~~Heavily coupling to the standard homomorphism technique~~
- `jolt-crypto` defines backend-agnostic group and commitment abstractions: `JoltGroup` (additive group with MSM), `PairingGroup` (bilinear maps), `JoltCommitment` (vector commitment trait). `Pedersen<G>` provides a blanket `JoltCommitment` impl for any `JoltGroup`. BN254 concrete types (`Bn254G1`, `Bn254G2`, `Bn254GT`) wrap arkworks behind `#[repr(transparent)]` newtypes with zero API leakage, gated behind a `bn254` feature flag (default on).
- `jolt-openings` defines PCS traits: `CommitmentScheme: Commitment` (+ open/verify) → `AdditivelyHomomorphic` (+ combine) → `StreamingCommitment` (+ incremental). No batching baked into PCS — batching is a reduction strategy.
- `OpeningReduction` trait separates claim transformation from proving. `RlcReduction` provided for homomorphic PCS. Protocol-specific reductions (e.g., sumcheck-based) implemented by `jolt-zkvm`.
- No accumulators, no statefulness. Claims are plain data, collected by the caller.

#### (2) Dory Globals — **ADDRESSED**
- ~~`dory_globals.rs` uses 9 `static mut OnceLock<usize>` variables with `unsafe` blocks~~
- `jolt-dory` uses instance-local `DoryParams` passed at construction. Zero global mutable state. Each `DoryScheme` instance carries its own configuration.

#### (3) Arkworks coupling — **PARTIALLY ADDRESSED**
- `jolt-field` wraps `ark_bn254::Fr` behind `#[repr(transparent)]` and never exposes arkworks types in its public API
- `serde` used instead of `CanonicalSerialize`/`CanonicalDeserialize` in all public APIs
- Challenge types are Jolt-native (no arkworks in trait bounds)
- **Remaining:** Still using the a16z fork. See [migration.md](./migration.md) for the plan to switch to upstream v0.5.0.

#### (4) Unclear boundaries between traits and utility interfaces — **ADDRESSED**
- ~~`JoltField` trait has 40+ methods mixing: arithmetic, serialization, RNG, conversion~~
- `jolt-field` cleanly separates: `Field` (core arithmetic), `UnreducedOps` / `ReductionOps` (deferred reduction), `Challenge` (Fiat-Shamir challenges), `OptimizedMul` (fast-path multiplication), `FMAdd` / `BarrettReduce` / `MontgomeryReduce` (accumulation)
- `SmallScalar` eliminated — replaced by `Polynomial<T>` with `bind_to_field<F: From<T>>()` in `jolt-poly`
- MSM code confined to `jolt-dory/src/optimizations/`

#### (5) Coupling of optimization parameters / techniques to backend primitives — **PARTIALLY ADDRESSED**
- `UnreducedOps`, `ReductionOps`, `FMAdd`, `BarrettReduce`, `MontgomeryReduce` are now separate traits (not baked into `Field`)
- `jolt-spartan` defines `FirstRoundStrategy` enum for univariate skip (pluggable, not hard-coded)
- **Remaining:** Accumulator const-generic limb counts still exist in `jolt-field`. Streaming sumcheck still coupled to specific poly representations.

#### (6) Arkworks fork is a mess — **PARTIALLY ADDRESSED**
- `jolt-optimizations` crate moved into `jolt-dory/src/optimizations/` (GLV, batch addition, vector ops)
- `SignedBigInt` types moved into `jolt-field/src/signed/`
- **Remaining:** Still using the fork for `ark-ff`, `ark-ec`, `ark-bn254`, `ark-serialize`. See [migration.md](./migration.md) phases 2-6 for the plan.

#### (7) Spartan bespoke-ness — **ADDRESSED (clarified scope)**
- ~~`zkvm/spartan/outer.rs` dark arts~~
- `jolt-spartan` is a standalone generic Spartan SNARK crate with `R1CS<F>` trait. It serves BlindFold (verifier R1CS after Nova folding), recursive verification, and any other "prove R1CS via Spartan" use case.
- The main zkVM outer sumcheck remains as custom code in `jolt-zkvm` — its specialized evaluation path (lazy `UniformSpartanKey`, two constraint groups, univariate skip with Jolt-specific domain sizes, streaming rounds) is too tightly coupled to Jolt's uniform constraint structure for a generic Spartan crate. It uses `jolt-sumcheck` for the protocol and `jolt-ir` for constraint definitions, but with its own evaluation logic.
- Constants are scoped. `FirstRoundStrategy` enum replaces hard-coded domain size constants.
- Decoupled from streaming sumcheck internals.

#### (8) Lack of sum-check testing — **ADDRESSED**
- `jolt-sumcheck` has 27 test functions covering: basic sumcheck, eq-product, wrong claimed sum, single variable, batched (2-claim, 3-claim), wrong round count, degree bound, determinism, streaming, mixed degree/num_vars, challenge slicing, transcript label mismatch, tampered coefficients, multi-backend (Blake2b + Keccak)
- Integration tests in `tests/integration.rs`
- Criterion benchmarks in `benches/sumcheck_prove.rs`

#### (9) Bespoke opening point reduction logic — **ADDRESSED (redesigned)**
- ~~`OpeningPoint<const E: Endianness, F>` with const generic for endianness~~
- `jolt-openings` defines `OpeningReduction<PCS>` trait: a stateless claim transformation (many claims → fewer claims). Output type equals input type, so reductions compose. `RlcReduction` groups by point and combines via `AdditivelyHomomorphic::combine`.
- `jolt-sumcheck` defines `SumcheckReduction<F>` trait: a claim reduction performed via sumcheck (many claims → fewer claims + `SumcheckProof`). Implementations provide protocol-specific witness construction via `build_witnesses()`. Used for advice polynomial reduction (two composed phases: cycle vars → address vars). Output claims carry partially-bound polynomials in `evaluations`, enabling phase composition.
- No DAG structure, no endianness generics, no accumulators. Claims are plain data (`ProverClaim`, `VerifierClaim`), collected by the caller in `Vec`s.

#### (10) Duplicated constants across tracer and jolt-inlines — **PARTIALLY ADDRESSED**
- `jolt-instructions` centralizes all opcodes in `opcodes.rs` (68 unique opcode constants)
- **Remaining:** Tracer and jolt-inlines may still duplicate some opcode values. Need to make them import from `jolt-instructions`.

#### (11) End-to-end tests — **PLANNED (strategy defined)**
- A 5-level testing strategy is defined in [zkvm_spec.md](./zkvm_spec.md) §6:
  - **Level 1:** IR expression unit tests (each `ClaimDefinition` evaluated on known inputs)
  - **Level 2:** Witness/`SumcheckCompute` unit tests (compute round polys, check degree/sum)
  - **Level 3:** Per-stage integration tests (single stage prove+verify with mock commitments)
  - **Level 4:** Multi-stage pipeline tests (stages 1–7 in sequence, real transcript)
  - **Level 5:** Full e2e tests (trace → proof → verify for small RISC-V programs)
- Infrastructure for auto-generating e2e tests from `examples/` deferred to post-`jolt-zkvm`

#### (12) Fragmented symbolic IR for sumcheck expressions — **PARTIALLY ADDRESSED**

Every sumcheck instance in Jolt has a mathematical expression that defines how its claim composes from polynomial openings and challenges. Today, this formula is written **four separate times** in four incompatible formats:

1. **Imperative Rust** — `SumcheckInstanceParams::input_claim()` (`jolt-core/src/subprotocols/sumcheck_verifier.rs:54`) computes the scalar value at runtime. Hand-written per-sumcheck. Any change here must be mirrored in (2).

2. **BlindFold sum-of-products** — `SumcheckInstanceParams::input_claim_constraint()` / `output_claim_constraint()` (`jolt-core/src/subprotocols/sumcheck_verifier.rs:59,65`) returns `OutputClaimConstraint` (defined in `jolt-core/src/subprotocols/blindfold/output_constraint.rs`) as `ProductTerm` / `ValueSource` structs for the BlindFold R1CS. Hand-written per-sumcheck. Must be kept in perfect sync with (1) — the CLAUDE.md calls this out as a **critical invariant**.

3. **MleAst symbolic field** — `zklean-extractor/src/mle_ast.rs` implements `JoltField` by recording operations as AST nodes in a global mutable arena. Running the verifier with `F = MleAst` captures the full computation for Lean4 extraction. Also extended by the gnark transpiler (PR [#1322](https://github.com/a16z/jolt/pull/1322)) to generate Groth16 circuits. Re-derives claim formulas from scratch by symbolic execution.

4. **ClaimExpr tree** — `SumcheckFrontend::input_output_claims()` (`jolt-core/src/subprotocols/sumcheck_claim.rs:279-281`) returns yet another AST (`ClaimExpr<F>` with `Constant`, `Var`, `Add`, `Mul`, `Sub` nodes) used by `zklean-extractor/src/sumchecks.rs` for Lean4 claim extraction. Separate from (2) and (3).

**Consequences:**
- **Sync hazard:** Any modification to a sumcheck claim formula requires updating up to 4 implementations. The BlindFold invariant (`input_claim()` ↔ `input_claim_constraint()`) is the most dangerous — desynchronization causes R1CS unsatisfiability that only surfaces in ZK mode e2e tests.
- **No shared IR:** BlindFold cannot consume `ClaimExpr`, and `ClaimExpr` cannot produce `OutputClaimConstraint`. MleAst cannot be used by BlindFold. Each system is a silo.
- **Global mutable state:** MleAst uses a `static OnceLock<RwLock<Vec<Node>>>` arena, violating the crate design principle of no global state.
- **New backends require new IRs:** Adding GPU kernel compilation, recursion circuits, or additional formal verification targets would require yet another representation of the same formulas.

**Resolution:** A new `jolt-ir` crate provides a single expression IR that is the **source of truth** for all sumcheck claim formulas. Developers write each formula once; all backends (runtime evaluation, BlindFold R1CS, Lean4, circuit transpilation, GPU kernels) are derived from that IR via a visitor pattern. See spec §4.10.

Additionally, `jolt-ir` replaces the compile-time `LC` / `lc!` / `r1cs_eq_conditional!` constraint authoring system used by the main zkVM's Spartan outer sumcheck (`jolt-core/src/zkvm/r1cs/`). The 12-variant `LC` enum and custom macros are replaced by `ExprBuilder` arithmetic, which produces readable degree-2 expressions that are factored into bilinear pairs at init time for the fused evaluator. This means the R1CS backend in `jolt-ir` serves both BlindFold (sparse matrix emission for `jolt-spartan`) and the main zkVM (bilinear factorization for the lazy evaluator). See spec §4.10 "Replacing compile-time R1CS constraints".

**Current status:** `jolt-ir` is complete with all five backends (evaluate, R1CS, Lean4, circuit transpilation) plus integration tests, fuzz targets, and benchmarks. `jolt-wrapper` subsumes the former `jolt-gnark` crate and provides `GnarkAstEmitter` for gnark/Go code generation plus full symbolic execution / AST capture pipeline. `KernelDescriptor` with `ProductSum` and `Custom` shapes is implemented; `jolt-cpu-kernels` compiles descriptors to `CpuKernel<F>`. **Remaining:** Downstream integration (task 06) — migrate `jolt-zkvm` and `jolt-spartan` to consume `jolt-ir` claim definitions, replacing the dual-implementation pattern.

#### (13) CPU-only prover with no compute backend abstraction — **ADDRESSED**

- `jolt-compute` defines `ComputeBackend` trait with typed buffer management and parallel primitives (`interpolate_pairs`, `pairwise_reduce`, `product_table`). Protocol-agnostic — no mention of sumcheck. `CpuBackend` compiles to identical code after monomorphization (zero overhead).
- `jolt-cpu-kernels` compiles `jolt-ir::KernelDescriptor` → `CpuKernel<F>` closures with specialization for D=4,8,16 + generic + Custom `Expr` stack-machine.
- `KernelDescriptor` in `jolt-ir` has two shapes: `ProductSum` (explicit fast-path, ~80% of prover time) and `Custom` (escape hatch for bespoke compositions). Both CPU and GPU backends use delayed modular reduction internally. Buffers generic over element type for compact polynomial storage.
- GPU backend crates (`jolt-metal`, `jolt-cuda`, `jolt-webgpu`) will compile `KernelDescriptor` into target-specific kernels at setup time.
- `jolt-zkvm` witness implementations will be generic over `B: ComputeBackend`.

See [backend_agnostic.md](./backend_agnostic.md) for the full design and spec §4.9 for the crate specification.

---

## Plan to address

Extract into new crates under `crates/`. While doing so address each observation above as it affects each crate.

| Crate | Status | API surface | Depends on |
|-------|--------|-------------|------------|
| `jolt-transcript` | **Done** | `Transcript` trait, `AppendToTranscript`, Blake2b/Keccak impls | — |
| `jolt-field` | **Done** | `Field`, `UnreducedOps`, `ReductionOps`, `Challenge`, `OptimizedMul`, accumulation traits, signed bigints | — |
| `jolt-poly` | **Done** | `MultilinearPolynomial`, `Polynomial<T>`, `EqPolynomial`, `UnivariatePoly`, `CompressedPoly`, `IdentityPolynomial` | `jolt-field` |
| `jolt-crypto` | **Done** | `JoltGroup`, `PairingGroup`, `JoltCommitment`, `Pedersen<G>`, `PedersenSetup<G>`, `Bn254`, `Bn254G1`, `Bn254G2`, `Bn254GT` | `jolt-field`, `jolt-transcript` |
| `jolt-openings` | **Done** | `CommitmentScheme`, `AdditivelyHomomorphic`, `StreamingCommitment`, `OpeningReduction`, `RlcReduction`, claim types, RLC utilities | `jolt-crypto`, `jolt-field`, `jolt-poly`, `jolt-transcript` |
| `jolt-sumcheck` | **Done** | `SumcheckProver`, `SumcheckVerifier`, `BatchedSumcheckProver/Verifier`, `StreamingSumcheckProver`, `SumcheckCompute`, `SumcheckReduction` | `jolt-field`, `jolt-poly`, `jolt-transcript`, `jolt-openings` |
| `jolt-spartan` | **Done** | `R1CS` trait, `SimpleR1CS`, `SpartanKey`, `SpartanProver`, `SpartanVerifier`, `FirstRoundStrategy` | `jolt-sumcheck`, `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript` |
| `jolt-instructions` | **Done** | `Instruction`, `LookupTable`, `JoltInstructionSet`, `TableId`, `LookupQuery`, 68 RV64IMAC instructions | `jolt-field` |
| `jolt-dory` | **Done** | `DoryScheme`, `DoryParams`, commitment/proof types, streaming, optimizations | `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript`, `dory-pcs` |
| `jolt-ir` | **Done** | `Expr`, `ExprBuilder`, `ClaimDefinition`, `ExprVisitor`, `CircuitEmitter`, normalization passes, evaluate/R1CS/Lean4/circuit backends | `jolt-field` |
| `jolt-wrapper` | **Done** | Verifier wrapping: `SymbolicField`, `AstBundle`, `AstEmitter`, `GnarkAstEmitter` — symbolic execution + pluggable codegen (gnark/Groth16, future backends). Subsumes former `jolt-gnark`. | `jolt-field`, `jolt-ir` |
| `jolt-zkvm` | **In progress** | zkvm sumchecks (ram, registers, bytecode, claim reductions), prover/verifier | `jolt-sumcheck`, `jolt-openings`, `jolt-spartan`, `jolt-instructions`, `jolt-ir`, `jolt-compute` |
| `jolt-compute` | **Done** | `ComputeBackend`, `Scalar`, `CpuBackend` — backend-agnostic buffer management and parallel primitives | `jolt-field` |
| `jolt-cpu-kernels` | **Done** | Compiles `KernelDescriptor` → `CpuKernel<F>`. Specialized D=4,8,16 + generic + Custom Expr stack-machine | `jolt-compute`, `jolt-ir`, `jolt-field` |
| `jolt-metal` | **Future** | Metal GPU compute backend (`ComputeBackend` impl) | `jolt-compute`, `jolt-field` |
| `jolt-metal-kernels` | **Future** | Compiles `KernelDescriptor` → Metal compute pipelines | `jolt-compute`, `jolt-ir`, `jolt-field` |
| `jolt-cuda` | **Future** | CUDA GPU compute backend (`ComputeBackend` impl) | `jolt-compute`, `jolt-field` |
| `jolt-cuda-kernels` | **Future** | Compiles `KernelDescriptor` → CUDA PTX modules | `jolt-compute`, `jolt-ir`, `jolt-field` |
| `jolt-webgpu` | **Future** | WebGPU compute backend (`ComputeBackend` impl) | `jolt-compute`, `jolt-field` |
| `jolt-webgpu-kernels` | **Future** | Compiles `KernelDescriptor` → WebGPU compute shaders | `jolt-compute`, `jolt-ir`, `jolt-field` |

**Note:** `jolt-math` (originally proposed) was dropped. Its functionality was absorbed into `jolt-field` (accumulation traits) and `jolt-poly` (polynomial-level utilities).


## Crate design conventions

The extracted crates converge on a set of design principles and structural patterns documented below. New crates must follow these conventions; existing crates should be brought into compliance.

### Directory layout

Every crate under `crates/` follows this canonical structure:

```
crates/<name>/
├── Cargo.toml
├── README.md          # Public API reference, feature flags, license
├── src/
│   ├── lib.rs         # Crate root: module declarations, curated re-exports
│   ├── ...            # One file per responsibility
│   └── tests.rs       # Unit tests (optional; #[cfg(test)] inline modules also acceptable)
├── tests/             # Integration tests exercising the public API only
├── benches/           # Criterion benchmarks for performance-critical paths
└── fuzz/              # cargo-fuzz targets (Cargo.toml + fuzz_targets/)
```

| Component | Required | Notes |
|-----------|----------|-------|
| `README.md` | Yes | Lists public types/traits/functions, feature flags, and license. Must stay in sync with `lib.rs` exports. |
| `benches/` | When crate has hot paths | Criterion benchmarks. At minimum, benchmark any function that runs inside a sumcheck inner loop or per-polynomial operation. |
| `tests/` | When cross-crate integration matters | Tests that exercise the crate purely through its public API, ensuring re-exports and trait contracts work end-to-end. |
| `fuzz/` | When crate accepts external input | `cargo-fuzz` targets for serialization round-trips, arithmetic invariants, and edge-case exploration. |

#### Current status

| Crate | README | benches | tests/ | fuzz |
|-------|--------|---------|--------|------|
| jolt-field | Yes | Yes | Yes | Yes |
| jolt-crypto | Yes | Yes | Yes | Yes |
| jolt-poly | Yes | Yes | — | Yes |
| jolt-sumcheck | Yes | Yes | Yes | — |
| jolt-openings | Yes | Yes | — | — |
| jolt-spartan | Yes | — | — | — |
| jolt-transcript | Yes | Yes | Yes | Yes |
| jolt-ir | Yes | Yes | Yes | Yes |
| jolt-wrapper | Yes | — | Yes | — |
| jolt-dory | Yes | Yes | Yes | Yes |
| jolt-compute | Yes | Yes | Yes | — |
| jolt-cpu-kernels | Yes | — | Yes | — |
| jolt-instructions | Yes | — | — | — |

### Design principles

#### 1. Traits define the contract; structs own the data

Traits are the crate's public API boundary. Concrete structs implement traits but should not leak implementation details. Consumers depend on trait bounds, not concrete types.

- **Trait methods use associated types** so implementations can choose their own return types (e.g. `type Bound` on `MultilinearPolynomial`, `type Commitment` on `CommitmentScheme`).
- **Data structs are generic over the trait hierarchy** — the three canonical type parameters are `F: Field`, `PCS: CommitmentScheme`, and `T: Transcript`.
- **No `dyn Trait` in public APIs** unless type erasure is explicitly required (e.g. `VerifierOpeningAccumulator` stores `Box<dyn Any>` for commitment type erasure, but exposes a typed `accumulate<C>` method).

#### 2. Encapsulation and no API leakage

- **Re-export only what consumers need** from `lib.rs`. Internal helpers, macros, and intermediate types stay private.
- **Backend types do not leak.** For example, `jolt-field` wraps `ark_bn254::Fr` behind `#[repr(transparent)]` and never exposes arkworks types in its public API. `jolt-dory` wraps `dory-pcs` types behind newtype wrappers.
- **Feature flags gate optional modules**, not core functionality. The `test-utils` feature on `jolt-openings` enables a `mock` module; the core API is unconditional.
- **Workspace lints are inherited** via `[lints] workspace = true` — no per-crate lint overrides unless justified.

#### 3. Minimal, justified dependencies

- Internal crates form a strict DAG: `jolt-transcript` → `jolt-field` → `jolt-poly` / `jolt-ir` → `jolt-openings` → `jolt-sumcheck` (`+jolt-openings` for `SumcheckReduction`) → `jolt-spartan` (`+jolt-ir`) / `jolt-dory`. `jolt-wrapper` depends on `jolt-field` + `jolt-ir`. `jolt-compute` depends only on `jolt-field`. Note: `jolt-sumcheck` depends on `jolt-openings` (for claim types used by `SumcheckReduction`) but does NOT depend on `jolt-ir` — it is a generic protocol crate. Only `jolt-spartan` and `jolt-zkvm` consume `jolt-ir`.
- Each crate depends only on the internal crates it directly uses. No transitive dependency shortcuts.
- External dependencies are workspace-managed. Dev-only crates (`rand`, `rand_chacha`, `criterion`) go in `[dev-dependencies]` and are annotated for `cargo-machete` when needed.

#### 4. Stateless provers and verifiers

Protocol actors (`SumcheckProver`, `SumcheckVerifier`, `SpartanProver`, `SpartanVerifier`) are **zero-field unit structs** with only associated functions. All state is passed as arguments. This eliminates lifetime entanglement and makes the protocol flow explicit.

#### 5. Comprehensive testing at every level

- **Unit tests** (`#[cfg(test)]` modules): Cover core logic, edge cases, arithmetic invariants, and serialization round-trips. Every public type has at least a construction + round-trip test.
- **Integration tests** (`tests/` directory): Exercise the crate through its public API only, catching re-export and trait-contract regressions.
- **Negative tests**: Verify that invalid inputs, tampered proofs, and constraint violations are properly rejected.
- **Determinism tests**: Verify that proofs and transcripts are reproducible from the same inputs.
- **Backend-agnostic tests**: Protocol tests run against multiple transcript backends (Blake2b, Keccak) and commitment schemes (Mock, Dory) to ensure generic code works across instantiations.

#### 6. Benchmarks for hot paths

Performance-critical crates include Criterion benchmarks in `benches/`. Focus areas:
- Polynomial bind and evaluate (the sumcheck inner loop)
- EqPolynomial table construction
- RLC combination
- Transcript operations (append, challenge)
- Commitment scheme operations (commit, prove, verify)

Benchmark names should be descriptive and parameterized (e.g. `bind_in_place/n=16`, `rlc_combine/k=8_n=2^18`).

#### 7. Fuzz targets for robustness

Crates that handle serialized data or perform arithmetic with external inputs include `cargo-fuzz` targets. Current focus:
- Serialization round-trips (serialize → deserialize → re-serialize produces identical bytes)
- Arithmetic invariant checking (associativity, commutativity, identity elements)
- Edge-case discovery for field operations and polynomial construction

#### 8. Documentation standards

Per the CI `comments-review` policy:

**Delete:**
- Section separators (`// ===`, `// ---`)
- Doc comments restating the item name
- Obvious comments on self-evident methods
- Commented-out code
- TODOs without issue links

**Require:**
- Doc comments on all public types, traits, and functions in library crates
- `// SAFETY:` comments that explain *why* invariants hold (not just that they do)
- Mathematical notation in doc comments where it clarifies the algorithm (LaTeX via `$...$`)
- WHY comments for non-obvious design decisions

#### 9. Error handling

- Library crates use `thiserror` for error enums with structured variants (not string messages).
- Errors compose via `#[from]` for sub-protocol errors (e.g. `SpartanError` wraps `SumcheckError` and `OpeningsError`).
- `panic!` / `assert!` only for programming errors (invariant violations), never for input validation.
- Panics are documented in `# Panics` doc sections.

#### 10. Serialization

- All proof and commitment types derive `Serialize + Deserialize` via `serde`.
- Use `#[serde(bound = "")]` on generic structs to avoid unnecessary trait bounds.
- Arkworks `CanonicalSerialize`/`CanonicalDeserialize` is confined to the backend crates (`jolt-field`, `jolt-dory`) and does not leak into public APIs.

---

## Benefits

### Pre-audit
- Carefully define and iterate on APIs before committing
- Clear, enforceable boundaries between modules
- Isolated unit testing per crate
- Smaller audit surface per crate
- Independent versioning and release

### AI
- Smaller files with single responsibility are easier for LLMs to reason about (each file has to be smaller than 20k, and preferable 10k tokens)
- Explicit dependencies via `Cargo.toml` give AI clear context boundaries (definitely needs to be adapted, but can be inspired from https://x.com/notnotstorm/status/2015898852482863382?s=20)
- Well-defined traits act as natural documentation for AI code generation
- Isolated crates allow AI to make changes without risk of distant side effects
- Better test coverage provides AI with examples and correctness checks
- Can use pedantic clippy / more static code analysis

### Future features
- **Recursion:** `jolt-ir`'s unified IR means the gnark/Groth16 transpiler (PR [#1322](https://github.com/a16z/jolt/pull/1322)) consumes the same `Expr` as BlindFold and the standard verifier — no separate symbolic execution pass needed for claim formulas. For wrapping the Jolt verifier in Spartan (recursive verification), the verifier's computation is expressed as `Expr`s, normalized to R1CS via the R1CS backend, and proved with `jolt-spartan`'s generic Spartan.
- **Lattices/hashes:** Decoupling the PIOP from the rest of the code allows easier integration of new primitives
- **ZK:** BlindFold consumes `jolt-ir`'s sum-of-products normalization instead of hand-written `OutputClaimConstraint` structs. `jolt-blindfold` orchestrates committed sumcheck (via `CommittedRoundHandler`, generic over `JoltCommitment`), Nova folding, and Spartan proof over the verifier R1CS (via `jolt-spartan`). **Open question:** Should `Pedersen<G>` from `jolt-crypto` implement `CommitmentScheme` from `jolt-openings` (for BlindFold round polynomial commitments)? The dependency direction prevents this in jolt-crypto itself — a `PedersenPCS` adapter in jolt-blindfold or jolt-openings may be needed.
- **GPU/hardware acceleration:** `jolt-ir`'s kernel IR enables a sumcheck kernel compiler that targets CPU, Metal, CUDA, or WebGPU from a single expression definition
- **Formal verification:** `jolt-ir`'s Lean4 backend replaces `MleAst`'s global-arena approach with structured, instance-local expression trees
- **Constraint authoring:** The compile-time `LC`/`lc!`/`r1cs_eq_conditional!` macro system is replaced by `ExprBuilder` arithmetic, giving readable constraint definitions while preserving evaluation performance via bilinear factorization at init time
