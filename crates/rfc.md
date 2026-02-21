
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

#### (1) Coupling of PIOP to Opening Proofs
- Commitment phase and opening proof data structures / traits assume a dory-like commitment scheme
- Heavily coupling to the standard homomorphism technique
- We should generalize this code to support lattices (different homomophism and commitment types) + hashes eventually (no homomorphism)

#### (2) Dory Globals
- `dory_globals.rs` uses 9 `static mut OnceLock<usize>` variables with `unsafe` blocks
- Global mutable state for: `GLOBAL_T`, `MAX_NUM_ROWS`, `NUM_COLUMNS` (× 3 contexts)
- `DoryContext` enum (Main, TrustedAdvice, UntrustedAdvice) with atomic switching
- Context guard pattern for RAII but fundamentally still global state
- Just a huge pain in the ass. We need to get rid of it.

#### (3) Arkworks coupling
- Using a16z fork (`dev/twist-shout`) with custom optimizations
- `JoltField` trait wraps arkworks but leaks through everywhere
- Challenge field implementations directly use arkworks Montgomery forms
- `CanonicalSerialize`/`CanonicalDeserialize` from arkworks baked into trait bounds (should use `Serde` instead)
- Changing field backend would require touching 100+ files
- Going to be very difficult to swap field for lattices / hashes (unreduced arithmetic, accumulation trait)

#### (4) Unclear boundaries between traits and utility interfaces
- `JoltField` trait has 40+ methods mixing: arithmetic, serialization, RNG, conversion
- `SmallScalar` trait in utils vs field operations overlap
- MSM code should probably not exist in Jolt / should live only in arkworks (other than a trait perhaps)
- No clear separation between "core math" vs "protocol-specific" operations

#### (5) Coupling of optimization parameters / techniques to backend primitives
- `JoltField::Unreduced<const N>` has a generic const for number of limbs
- Montgomery/Barrett reduction traits baked into `JoltField`
- `Acc5U`, `Acc6S`, `Acc7S`, `Acc8S` accumulators hard-coded for specific limb counts
- Streaming sumcheck tied to specific polynomial representations
- Univariate skip optimization hard-coded with domain size constants (`OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE`)
- Hard to experiment with alternative optimization strategies

#### (6) Arkworks fork is a mess
- Using `a16z/arkworks-algebra` branch `dev/twist-shout`
- Patches in workspace `Cargo.toml` for: `ark-bn254`, `ark-ff`, `ark-ec`, `ark-serialize`
- CI broken
- Additional `jolt-optimizations` crate from same fork
- Not upstream-able, maintenance burden
- Ideally: sync with upstream and maintain any necessary `jolt-optimizations` in suitable crates in Jolt itself

#### (7) Spartan bespoke-ness
- `zkvm/spartan/outer.rs` dark arts (andrew: it's not too dark, just needs some cleanup)
- Constants like `OUTER_FIRST_ROUND_POLY_DEGREE_BOUND`, `OUTER_UNIVARIATE_SKIP_DEGREE` from r1cs
- Streaming sumcheck state shares memory with Spartan-specific structures
- Would be hard to use Spartan implementation for non-Jolt R1CS (andrew: I think it's okay)

#### (8) Lack of sum-check testing
- Only 2 files have sumcheck-related tests (`read_raf_checking.rs`, `hamming_weight.rs`)
- `subprotocols/sumcheck.rs` has a test module but it's basic
- No systematic unit tests for streaming sumcheck
- No property-based testing for sumcheck correctness
- e2e testing is the primary method, largely due to the coupling between sum-check code and state management

#### (9) Bespoke opening point reduction logic
- `OpeningPoint<const E: Endianness, F>` with const generic for endianness
- `ProverOpeningAccumulator` and `VerifierOpeningAccumulator` is sort of the last state object we have in Jolt
- RLC (random linear combination) logic interleaved with opening point tracking
- Sort of bespokely handled the DAG structure to most efficiently reduce to a single point. Would be best to come up with a lightweight abstraction in case the PIOP changes (ideally without introducing additional state management)


#### (10) Duplicated constants across tracer and jolt-inlines
- Some inlines may invoke virtual instructions via `asm!`, e.g. sha2 uses REV8W and secp256k1 uses VirtualAssertEq
- Currently, the opcode/funct3 associated with those instructions is duplicated across tracer and jolt-inlines
- Could lead to footguns in the future

#### (11) End-to-end tests
- Each end-to-end test requires a lot of boilerplate
- In the past, small discrepancies between the end-to-end test and the `jolt::provable` proc-macro codepath has led to inconsistent behavior
- We should introduce infrastructure to easily generate an end-to-end test for each guest program in `examples/`
- Additionally, should allow easy parameterization/sweeps over prover configuration options for better test coverage

---

## Plan to address

Extract into new crates under `crates/`. While doing so address each observation above as it affects each crate. The table below focuses on `jolt-core`

| Crate | API surface | Depends on |
|-------|-------------|------------|
| `jolt-transcript` | `Transcript` trait, `AppendToTranscript`, Blake2b/Keccak impls | — |
| `jolt-field` | `Field` trait (add, mul, inv, from_bytes, to_bytes), challenge field types | `jolt-transcript` |
| `jolt-math` | `Math` trait (log2, pow2), bit interleave/uninterleave, accumulators | `jolt-field`? |
| `jolt-poly` | `MultilinearPolynomial`, `EqPolynomial`, `DensePolynomial`, binding ops | `jolt-field` |
| `jolt-openings` | `CommitmentScheme` trait, `OpeningAccumulator`, batch reduction, RLC | `jolt-field`, `jolt-poly`, `jolt-transcript` |
| `jolt-dory` | Dory impl of `CommitmentScheme`, setup, instance-local params (no globals) | `jolt-openings`, `dory-pcs` |
| `jolt-sumcheck` | `SumcheckProver`, `SumcheckVerifier`, streaming variant, round polys, traits | `jolt-field`, `jolt-poly`, `jolt-transcript` |
| `jolt-spartan` | `R1CS` trait, `SpartanProver`, `SpartanVerifier`, uniform key (andrew: i think this can be merged into the zkvm) | `jolt-sumcheck`, `jolt-openings` |
| `jolt-zkvm` | zkvm sumchecks (ram, registers, bytecode, claim reductions) | `jolt-sumcheck`, `jolt-openings`, `jolt-spartan` |


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
