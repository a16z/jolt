
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

#### (1) Coupling of PIOP to Opening Proofs — **ADDRESSED**
- ~~Commitment phase and opening proof data structures / traits assume a dory-like commitment scheme~~
- ~~Heavily coupling to the standard homomorphism technique~~
- `jolt-openings` now defines a generic three-tier trait hierarchy: `CommitmentScheme` (base), `HomomorphicCommitmentScheme` (additive, e.g. Dory/KZG), `StreamingCommitmentScheme` (chunked). Lattice and hash schemes can implement `CommitmentScheme` without the homomorphic extension.

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

#### (7) Spartan bespoke-ness — **ADDRESSED**
- ~~`zkvm/spartan/outer.rs` dark arts~~
- `jolt-spartan` is a standalone crate with generic `R1CS<F>` trait. Any R1CS system can use it, not just Jolt.
- Constants are scoped to the crate. `FirstRoundStrategy` enum replaces hard-coded domain size constants.
- Decoupled from streaming sumcheck internals.

#### (8) Lack of sum-check testing — **ADDRESSED**
- `jolt-sumcheck` has 27 test functions covering: basic sumcheck, eq-product, wrong claimed sum, single variable, batched (2-claim, 3-claim), wrong round count, degree bound, determinism, streaming, mixed degree/num_vars, challenge slicing, transcript label mismatch, tampered coefficients, multi-backend (Blake2b + Keccak)
- Integration tests in `tests/integration.rs`
- Criterion benchmarks in `benches/sumcheck_prove.rs`

#### (9) Bespoke opening point reduction logic — **ADDRESSED**
- ~~`OpeningPoint<const E: Endianness, F>` with const generic for endianness~~
- `jolt-openings` accumulators are clean, stateless collectors. `accumulate()` → `reduce_and_prove()`/`reduce_and_verify()`. Point grouping is automatic (hash-based). RLC reduction is a separate utility (`rlc_combine`, `rlc_combine_scalars`).
- No DAG structure, no endianness generics. Simple, flat API.

#### (10) Duplicated constants across tracer and jolt-inlines — **PARTIALLY ADDRESSED**
- `jolt-instructions` centralizes all opcodes in `opcodes.rs` (68 unique opcode constants)
- **Remaining:** Tracer and jolt-inlines may still duplicate some opcode values. Need to make them import from `jolt-instructions`.

#### (11) End-to-end tests — **NOT YET ADDRESSED**
- Still requires boilerplate
- Infrastructure for auto-generating e2e tests from `examples/` not yet built
- Will be addressed during `jolt-zkvm` crate development

---

## Plan to address

Extract into new crates under `crates/`. While doing so address each observation above as it affects each crate.

| Crate | Status | API surface | Depends on |
|-------|--------|-------------|------------|
| `jolt-transcript` | **Done** | `Transcript` trait, `AppendToTranscript`, Blake2b/Keccak impls | — |
| `jolt-field` | **Done** | `Field`, `UnreducedOps`, `ReductionOps`, `Challenge`, `OptimizedMul`, accumulation traits, signed bigints | — |
| `jolt-poly` | **Done** | `MultilinearPolynomial`, `Polynomial<T>`, `EqPolynomial`, `UnivariatePoly`, `CompressedPoly`, `IdentityPolynomial` | `jolt-field` |
| `jolt-openings` | **Done** | `CommitmentScheme`, `HomomorphicCommitmentScheme`, `StreamingCommitmentScheme`, accumulators, RLC | `jolt-field`, `jolt-poly`, `jolt-transcript` |
| `jolt-sumcheck` | **Done** | `SumcheckProver`, `SumcheckVerifier`, `BatchedSumcheckProver/Verifier`, `StreamingSumcheckProver`, `SumcheckWitness` | `jolt-field`, `jolt-poly`, `jolt-transcript` |
| `jolt-spartan` | **Done** | `R1CS` trait, `SimpleR1CS`, `SpartanKey`, `SpartanProver`, `SpartanVerifier`, `FirstRoundStrategy` | `jolt-sumcheck`, `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript` |
| `jolt-instructions` | **Done** | `Instruction`, `LookupTable`, `JoltInstructionSet`, `TableId`, `LookupQuery`, 68 RV64IMAC instructions | `jolt-field` |
| `jolt-dory` | **Done** | `DoryScheme`, `DoryParams`, commitment/proof types, streaming, optimizations | `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript`, `dory-pcs` |
| `jolt-zkvm` | **In progress** | zkvm sumchecks (ram, registers, bytecode, claim reductions), prover/verifier | `jolt-sumcheck`, `jolt-openings`, `jolt-spartan`, `jolt-instructions` |

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
| jolt-poly | Yes | Yes | — | Yes |
| jolt-sumcheck | Yes | Yes | Yes | — |
| jolt-openings | Yes | Yes | — | — |
| jolt-spartan | Yes | — | — | — |
| jolt-transcript | Yes | Yes | Yes | Yes |
| jolt-dory | Yes | — | — | Yes |
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

- Internal crates form a strict DAG: `jolt-transcript` → `jolt-field` → `jolt-poly` → `jolt-sumcheck` / `jolt-openings` → `jolt-spartan` / `jolt-dory`.
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
- **Recursion:** Clearer APIs make it easier to compose different types over the same traits, extend prover/verifier, and simplify the transpiler / wrapping project.
- **Lattices/hashes:** Decoupling the PIOP from the rest of the code allows easier integration of new primitives
- **ZK:** Blindfold can be a separate crate
