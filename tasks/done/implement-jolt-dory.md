# impl-jolt-dory: Clean-room implementation of jolt-dory

**Scope:** crates/jolt-dory/

**Depends:** impl-jolt-openings

**Verifier:** ./verifiers/scoped.sh /workdir jolt-dory

**Context:**

Implement the `jolt-dory` crate — the Dory polynomial commitment scheme, implementing `CommitmentScheme`, `HomomorphicCommitmentScheme`, and `StreamingCommitmentScheme` from `jolt-openings`. Wraps the external `dory-pcs` crate. All parameters are instance-local (no globals).

**This is a clean-room rewrite of the wrapper layer.** Study `jolt-core/src/poly/dory/` for how Dory is currently integrated. The `dory-pcs` crate itself is not being rewritten — this crate wraps it behind the new trait hierarchy.

**Dependencies:** `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript`, `dory-pcs`.

### Reference material

The old code lives in:
- `jolt-core/src/poly/dory/mod.rs` — `DoryCommitmentScheme` struct wrapping dory-pcs
- `jolt-core/src/poly/dory/dory_globals.rs` — 9 `static mut OnceLock<usize>` variables with unsafe blocks (THIS MUST DIE)

Also read the Jolt Book section on Dory: https://jolt.a16zcrypto.com/

### Critical design constraint: NO GLOBALS

The old `dory_globals.rs` uses `static mut` with `OnceLock` for `GLOBAL_T`, `MAX_NUM_ROWS`, `NUM_COLUMNS` across 3 contexts (Main, TrustedAdvice, UntrustedAdvice). This is replaced by instance-local parameters.

### Public API contract

```rust
/// Dory polynomial commitment scheme.
/// All configuration is instance-local — no global state.
pub struct DoryScheme {
    params: DoryParams,
}

/// Instance-local Dory parameters (replaces the old globals).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoryParams {
    pub t: usize,
    pub max_num_rows: usize,
    pub num_columns: usize,
}

impl DoryScheme {
    pub fn new(params: DoryParams) -> Self;
}

impl CommitmentScheme for DoryScheme {
    type Field = ark_bn254::Fr; // via jolt-field arkworks impl
    type Commitment = DoryCommitment;
    type Proof = DoryProof;
    type ProverSetup = DoryProverSetup;
    type VerifierSetup = DoryVerifierSetup;

    fn protocol_name() -> &'static str { "dory" }
    // ... all trait methods, delegating to dory-pcs
}

impl HomomorphicCommitmentScheme for DoryScheme {
    type BatchedProof = DoryBatchedProof;
    // ... batch operations via dory-pcs
}

impl StreamingCommitmentScheme for DoryScheme {
    type PartialCommitment = DoryPartialCommitment;
    // ... chunked commitment via dory-pcs
}
```

### Wrapper types

All dory-pcs types that appear in the public API get wrapper types with `Serialize`/`Deserialize` (via serde, not arkworks `CanonicalSerialize`):

```rust
#[derive(Clone, Debug)]
pub struct DoryCommitment(/* inner dory-pcs type */);

#[derive(Clone)]
pub struct DoryProof(/* inner dory-pcs type */);

#[derive(Clone)]
pub struct DoryBatchedProof(/* inner */);

#[derive(Clone)]
pub struct DoryProverSetup(/* inner */);

#[derive(Clone)]
pub struct DoryVerifierSetup(/* inner */);
```

Each wrapper implements `Serialize`/`Deserialize` via custom impls that delegate to arkworks serialization internally but present a serde interface externally.

### Implementation notes

- `DoryScheme::new(params)` stores the params. When `setup_prover`/`setup_verifier` are called, the params are used instead of reading globals.
- Threading params through is the key difference from the old code. Every dory-pcs call that previously read globals now receives params explicitly.
- If dory-pcs itself reads globals internally, the wrapper must set them before each call and restore them after. This is a temporary workaround until dory-pcs is also refactored. Document this clearly.
- The `parallel` feature flag should gate rayon usage (Dory proving is the primary consumer of parallelism).

### File structure

```
jolt-dory/src/
├── lib.rs
├── scheme.rs           # DoryScheme, impl CommitmentScheme
├── params.rs           # DoryParams (instance-local)
├── commitment.rs       # DoryCommitment, DoryProof wrapper types
├── streaming.rs        # StreamingCommitmentScheme impl
├── serde_bridge.rs     # Custom Serialize/Deserialize for arkworks wrapper types
└── error.rs
```

**Acceptance:**

- `DoryScheme` implements all three trait tiers (CommitmentScheme, Homomorphic, Streaming)
- No `static mut`, no `OnceLock`, no `unsafe` for global state
- `DoryParams` passed at construction, threaded through all operations
- Commit → prove → verify round-trip works for small polynomials
- Batch prove → batch verify works for multiple polynomials
- Streaming commit matches non-streaming commit for the same polynomial
- All wrapper types implement serde `Serialize`/`Deserialize`
- `parallel` feature flag works
- No file exceeds 500 lines
- Rustdoc on all public items
- `cargo clippy` clean
- Unit tests for round-trip, batching, streaming, and parameter threading
