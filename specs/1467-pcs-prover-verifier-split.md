# Spec: PCS Trait Prover/Verifier Split + Fused Batched Openings

| Field       | Value                                                |
|-------------|------------------------------------------------------|
| Author(s)   | @quangvdao                                           |
| Created     | 2026-04-22                                           |
| Status      | proposed                                             |
| PR          | TBD (off `refactor/crates`)                          |
| Branch      | `quang/pcs-prover-verifier-split`                    |
| Supersedes  | parts of #1461 (parks the `VerifierBackend` direction)|

## Summary

Four coupled changes to `crates/jolt-openings`:

1. **Split each PCS trait along the prover/verifier role axis.**
   `CommitmentScheme`, `AdditivelyHomomorphic`, and `ZkOpeningScheme`
   each become a `…Verifier` base trait and a no-suffix prover trait
   that extends the verifier base. `StreamingCommitment` stays
   monolithic (purely prover-side). Result: `jolt-verifier` no longer
   pulls in `Polynomial` / `OpeningHint` / `ProverSetup` /
   `SetupParams` through any trait bound it touches.

2. **Add a verifier-owned setup constructor.**
   `CommitmentSchemeVerifier` gains `type VerifierSetupParams` and
   `fn verifier_setup(params) -> VerifierSetup`. Consumers that
   derive the verifier setup from public inputs alone (wasm
   verifier, on-chain verifier, trusted-setup loader) call this
   without ever materializing a `ProverSetup`.

3. **Fuse the batched-opening protocol on the core trait.**
   `CommitmentSchemeVerifier` gains `type BatchProof` + `fn verify_batch`;
   `CommitmentScheme` gains `fn prove_batch`. These are the only
   opening-protocol entry points on the core PCS traits. The
   previous `OpeningReduction` trait and the `OpeningVerification`
   trait (from #1461's interim design) are folded in.

4. **Push single-claim `open`/`verify` down to the homomorphic
   sub-traits.** `open` moves to `AdditivelyHomomorphic`; `verify`
   moves to `AdditivelyHomomorphicVerifier`. They are documented as
   per-group primitives consumed by `homomorphic_prove_batch` /
   `homomorphic_verify_batch`. Hachi-style schemes implement only
   `CommitmentScheme(Verifier)` and never see single-claim methods,
   avoiding a stub-or-panic tax. Tests / benches that exercise
   single-claim opens against Mock / HyperKZG / Dory keep working
   unchanged because those schemes still expose the methods on the
   homomorphic sub-traits.

No new top-level traits are introduced. The Jolt runtime IR
(`Op`, `VerifierOp`) gains one fused op pair (`Op::ProveBatch` +
`VerifierOp::VerifyOpenings`) replacing the prior
reduce-then-open / reduce-then-verify pairs. `Op::BindOpeningInputs`
stays as a separate IR step (rationale below).

## Intent

### Goal

Land the semantics of #1461's PCS-side changes (single fused
batched-opening protocol per scheme) without #1461's
`FieldBackend` / `CommitmentBackend` / `Tracing` abstractions, while
also paying off accumulated bloat in the PCS trait surface so that
verifier-only crates (current `jolt-verifier`, future wasm /
on-chain verifiers) do not import prover types.

### Non-goals

- No changes to backend types (`FieldBackend`, `CommitmentBackend`,
  `Tracing<PCS>`); those are dropped relative to #1461 in
  response to maintainer feedback ("expand the IR, not the
  abstractions"). The verifier remains generic only over `(F,
  PCS, ProofTranscript)` as it was on `refactor/crates`.
- No reorganization of `jolt-crypto::Commitment` /
  `VectorCommitment`. Those stay above `CommitmentSchemeVerifier`
  unchanged.
- No new ZK-batched-opening trait. `ZkOpeningScheme` keeps its
  single-claim `open_zk` / `verify_zk` shape; batch-ZK is a
  follow-up.

### Motivation

Four pressures align:

1. **Verifier-side bloat.** Today every `jolt-verifier` bound
   reads `PCS: CommitmentScheme<Field = F>` and inherits the full
   prover surface: `ProverSetup`, `OpeningHint`, `Polynomial`,
   `SetupParams`, `commit`, `open`, `setup`. Audit of
   `crates/jolt-verifier/src/{verifier.rs,proof.rs,key.rs}`
   confirms none of these are referenced; the trait bound is
   over-broad. Removing the leakage is a prerequisite for any
   verifier-only build (wasm bundle, on-chain verifier, embedded
   verifier).

2. **Verifier-derived setup.** A real verifier needs to build
   `VerifierSetup` from public inputs alone (URS file for HyperKZG,
   structured generators for Dory). Today the only constructor on
   the trait is `verifier_setup(prover_setup: &ProverSetup)`,
   which is the prover-side projection (useless to a verifier
   that never holds a `ProverSetup`). We need an honest verifier
   constructor.

3. **PCS batching shape from #1461 needs to land.** Hachi-style
   lattice schemes don't decompose into "reduce then open"; they
   need a single fused `prove_batch` / `verify_batch` API. #1461
   captured this as a separate `OpeningVerification` trait, but
   now that we're already touching the trait family, the natural
   home is on `CommitmentScheme(Verifier)` itself.

4. **Single-claim methods don't belong on the core trait.** In
   production today single-claim `open`/`verify` are called from
   exactly one site each: the body of `homomorphic_prove_batch` /
   `homomorphic_verify_batch` (`crates/jolt-openings/src/homomorphic.rs`).
   They are RLC-helper internals masquerading as part of the core
   PCS API. Hachi-style fused-batch schemes don't have a native
   single-claim notion: forcing them to implement `open`/`verify`
   leaves only bad options (`unimplemented!()` panics, or
   wrap-batch-of-one-and-unwrap). Pushing the methods down to the
   `AdditivelyHomomorphic*` sub-traits matches their actual
   semantic role (per-group RLC primitives) and frees the core
   trait of the Hachi tax.

### Invariants

1. **Verifier-side bound diet.** After this change, every type
   parameter declaration in `crates/jolt-verifier/` and
   `crates/jolt-zkvm/src/{prove,proving_key,preprocessing}.rs`
   that today reads `PCS: CommitmentScheme<…>` will read either
   `PCS: CommitmentSchemeVerifier<Field = F>` (verifier code) or
   `PCS: CommitmentScheme<Field = F>` (prover code, same name as
   today, now extends the verifier trait). No verifier-side bound
   transitively pulls in `Polynomial` / `OpeningHint` /
   `ProverSetup` / `SetupParams`.

2. **Source-of-truth setup methods.** `verifier_setup` on
   `CommitmentSchemeVerifier` and `setup` on `CommitmentScheme`
   produce the same `VerifierSetup` value when fed the public
   portion of the same parameters. PCS-level tests assert this
   equality.

3. **Prover/verifier transcript parity.** `prove_batch` and
   `verify_batch` drive the Fiat-Shamir transcript through
   byte-identical sequences for any matching pair of input
   claims. Per-PCS parity tests
   (`crates/{jolt-openings,jolt-hyperkzg,jolt-dory}/tests/verification_parity.rs`)
   cross-check this with `Blake2bTranscript`.

4. **Single batched-opening op pair in the IR.** The compiler
   emits one `Op::ProveBatch` per opening accumulator flush and
   one `VerifierOp::VerifyOpenings` per matching verify step.
   Followed (where applicable) by exactly one
   `Op::BindOpeningInputs` / `VerifierOp::BindOpeningInputs`.
   Counted by op-class instrumentation; muldiv e2e enforces.

5. **Equivalence to `refactor/crates` semantics.** The Jolt
   transcript bytes for the muldiv e2e (`cargo nextest run -p
   jolt-equivalence transcript_divergence`) match
   `refactor/crates` byte-for-byte after this PR. Internal trait
   reorg does not change observable Fiat-Shamir behavior.

## Trait Surface After This PR

Lives in `crates/jolt-openings/src/schemes.rs`. All other content
of that file (imports, `OpeningsError`, etc.) unchanged.

### `CommitmentSchemeVerifier` (new base trait, verifier-only surface)

Only fused batched verification + post-batch transcript bind.
Single-claim `verify` lives on `AdditivelyHomomorphicVerifier` (see below).

```rust
pub trait CommitmentSchemeVerifier: Commitment + Clone + Send + Sync + 'static {
    type Field: Field;
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;
    type Proof:      Clone + Debug + Send + Sync + Serialize + DeserializeOwned;
    type BatchProof: Clone + Debug + Send + Sync + Serialize + DeserializeOwned;

    /// Public inputs the verifier needs to derive its setup
    /// (URS file, max num_vars, etc.). Distinct from the prover's
    /// `SetupParams`, and does not require toxic waste / full URS.
    type VerifierSetupParams;

    fn verifier_setup(params: Self::VerifierSetupParams) -> Self::VerifierSetup;

    fn verify_batch(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        batch_proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;

    /// Post-batch transcript bind for downstream Fiat-Shamir.
    /// Default: no-op. Dory overrides; Mock / HyperKZG do not.
    fn bind_opening_inputs(
        _t: &mut impl Transcript<Challenge = Self::Field>,
        _point: &[Self::Field],
        _eval: &Self::Field,
    ) {
    }
}
```

### `CommitmentScheme` (renamed semantics, prover extension)

Same name as today's monolithic trait, now strictly an extension
of `CommitmentSchemeVerifier`. Prover crates keep writing
`PCS: CommitmentScheme<Field = F>`, so call-site churn is minimal.
Single-claim `open` lives on `AdditivelyHomomorphic` (see below).

```rust
pub trait CommitmentScheme: CommitmentSchemeVerifier {
    type ProverSetup: Clone + Send + Sync;
    type Polynomial:  MultilinearPoly<Self::Field> + From<Vec<Self::Field>>;
    type OpeningHint: Clone + Send + Sync + Default;
    type SetupParams;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup);

    /// Project the prover setup down to the verifier setup. Used
    /// by single-machine prover/verifier roundtrips that already
    /// hold a `ProverSetup` and want to avoid re-running
    /// `verifier_setup` from public inputs.
    fn project_verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup;

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    fn prove_batch<T: Transcript<Challenge = Self::Field>>(
        claims: Vec<ProverClaim<Self::Field>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut T,
    ) -> (Self::BatchProof, Vec<Self::Field>);
}
```

### `AdditivelyHomomorphicVerifier` + `AdditivelyHomomorphic`

`combine` / `combine_hints` are the pure RLC primitives.
`verify` / `open` are the per-group single-claim primitives that
`homomorphic_verify_batch` / `homomorphic_prove_batch` invoke
once per opening point after RLCing within the group. Tests and
benches MAY also call them directly to exercise the single-claim
path against homomorphic schemes.

```rust
pub trait AdditivelyHomomorphicVerifier: CommitmentSchemeVerifier
where
    Self::Output: HomomorphicCommitment<Self::Field>,
{
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output;

    /// Verify one combined claim. Per-group primitive used by
    /// [`homomorphic_verify_batch`].
    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;
}

pub trait AdditivelyHomomorphic: AdditivelyHomomorphicVerifier + CommitmentScheme {
    fn combine_hints(
        _hints: Vec<Self::OpeningHint>,
        _scalars: &[Self::Field],
    ) -> Self::OpeningHint {
        Self::OpeningHint::default()
    }

    /// Open one combined polynomial at one point. Per-group
    /// primitive used by [`homomorphic_prove_batch`].
    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof;
}
```

### `ZkOpeningSchemeVerifier` + `ZkOpeningScheme`

```rust
pub trait ZkOpeningSchemeVerifier: CommitmentSchemeVerifier {
    type HidingCommitment: Clone + Debug + Eq + Send + Sync + 'static
        + Serialize + DeserializeOwned + AppendToTranscript;

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval_commitment: &Self::HidingCommitment,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;
}

pub trait ZkOpeningScheme: ZkOpeningSchemeVerifier + CommitmentScheme {
    type Blind: Clone + Send + Sync;

    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind);
}
```

### `StreamingCommitment` (unchanged shape)

Bound becomes the renamed (still-named) `CommitmentScheme`. No
method changes.

```rust
pub trait StreamingCommitment: CommitmentScheme {
    type PartialCommitment: Clone + Send + Sync;
    fn begin(setup: &Self::ProverSetup) -> Self::PartialCommitment;
    fn feed(partial: &mut Self::PartialCommitment, chunk: &[Self::Field], setup: &Self::ProverSetup);
    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output;
}
```

### Trait Family Summary

```
                    CommitmentSchemeVerifier
                          (verifier base)
                    ─ verifier_setup
                    ─ verify_batch
                    ─ bind_opening_inputs
                            │
            ┌───────────────┼─────────────────┐
            │               │                 │
            │      AddHomVerifier       ZkOpeningSchemeVerifier
            │      ─ combine            ─ verify_zk
            │      ─ verify             ─ HidingCommitment
            │               │                 │
            │               │                 │
        CommitmentScheme    │                 │
       (prover extension)   │                 │
       ─ setup              │                 │
       ─ project_verifier_  │                 │
           setup            │                 │
       ─ commit             │                 │
       ─ prove_batch        │                 │
            │               │                 │
            ├───────────────┤                 │
            │               │                 │
            │     AdditivelyHomomorphic       │
            │     ─ combine_hints             │
            │     ─ open                      │
            │               │                 │
            │   ┌───────────┘                 │
            │   │                             │
            │   │           ZkOpeningScheme  ─┘
            │   │           ─ open_zk
            │   │           ─ Blind
            │   │
       StreamingCommitment
       (Dory only, unchanged)
       ─ begin / feed / finish
```

Verifier-only consumers cross *only* the top row (and the
verifier-side homomorphic / ZK extensions). Prover-only types
(`Polynomial`, `OpeningHint`, `ProverSetup`, `Blind`) live
strictly below the dashed line implied by the
`CommitmentScheme` / `…Verifier` boundary.

## Claim Types

`crates/jolt-openings/src/claims.rs`:

```rust
/// Prover-side opening claim: polynomial, evaluation point, and value.
#[derive(Clone, Debug)]
pub struct ProverClaim<F: Field> {
    pub polynomial: Polynomial<F>,
    pub point: Vec<F>,
    pub eval: F,
}

/// Verifier-side opening claim: commitment, point, and value.
/// Generic over `F` and `PCS: CommitmentSchemeVerifier<Field = F>`
/// (no backend type parameter; delta vs #1461).
#[derive(Clone, Debug)]
pub struct OpeningClaim<F: Field, PCS: CommitmentSchemeVerifier<Field = F>> {
    pub commitment: PCS::Output,
    pub point: Vec<F>,
    pub eval: F,
}
```

(`VerifierClaim<F, C>` is removed; `OpeningClaim<F, PCS>` subsumes
it now that the verifier-side trait is small enough to put in the
bound.)

`ProverClaim` rename / extension to `ProverOpeningClaim` carrying
slot/commitment metadata for packed schemes (Hachi mega-poly) is
deferred. Note inherited from #1461; same TODO comment.

## Helper Functions

`crates/jolt-openings/src/homomorphic.rs` holds the helpers (the
file groups both prover- and verifier-side bodies for
additively-homomorphic schemes; the name tracks the
`AdditivelyHomomorphic` / `AdditivelyHomomorphicVerifier` trait
pair they target). Two free helpers, no backend parameters, plain
`Transcript`:

```rust
pub fn homomorphic_prove_batch<PCS, T>(
    claims: Vec<ProverClaim<PCS::Field>>,
    hints: Vec<PCS::OpeningHint>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> (Vec<PCS::Proof>, Vec<PCS::Field>)
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>;

pub fn homomorphic_verify_batch<PCS, T>(
    claims: Vec<OpeningClaim<PCS::Field, PCS>>,
    batch_proof: &[PCS::Proof],
    setup: &PCS::VerifierSetup,
    transcript: &mut T,
) -> Result<(), OpeningsError>
where
    PCS: AdditivelyHomomorphicVerifier,
    PCS::Output: HomomorphicCommitment<PCS::Field> + AppendToTranscript,
    T: Transcript<Challenge = PCS::Field>;
```

`homomorphic_verify_batch`'s bound on `AdditivelyHomomorphicVerifier`
(rather than `AdditivelyHomomorphic`) is the concrete payoff of
the trait split: the verifier-side helper does not need to know
about `OpeningHint`. Mock / HyperKZG / Dory `verify_batch` bodies
are one-line delegations to this helper.

## Per-PCS Impl Block Layout

Each PCS writes the impl blocks below. The split is mechanical
(no logic changes); methods relocate from the (deleted) monolithic
`CommitmentScheme` impl into the new homes shown.

| Trait | Mock / HyperKZG / Dory contents |
|---|---|
| `CommitmentSchemeVerifier` | `verifier_setup`, `verify_batch` (one-line delegation to `homomorphic_verify_batch::<Self, _>(...)`), `bind_opening_inputs` (default for Mock/HyperKZG; overridden for Dory) |
| `CommitmentScheme` | `setup`, `project_verifier_setup`, `commit`, `prove_batch` (one-line delegation to `homomorphic_prove_batch::<Self, _>(...)`) |
| `AdditivelyHomomorphicVerifier` | `combine`, `verify` (the body that lives in today's `CommitmentScheme::verify`) |
| `AdditivelyHomomorphic` | `combine_hints`, `open` (the body that lives in today's `CommitmentScheme::open`) |
| `ZkOpeningSchemeVerifier` | `verify_zk` + `HidingCommitment` (Mock + Dory; HyperKZG already has) |
| `ZkOpeningScheme` | `open_zk` + `Blind` (Mock + Dory) |
| `StreamingCommitment` | Dory only; `begin`/`feed`/`finish` unchanged |

**Key observation:** `verify_batch` / `prove_batch` bodies are
each one line for Mock / HyperKZG / Dory because the helpers do
all the work. The actual logic (single-claim open/verify) lives
in `AdditivelyHomomorphic*::open` / `verify`, which is exactly
where the homomorphic helpers reach for it. No new code; just
relocated method bodies.

**Hachi (forward-looking, not in this PR):** implements only
`CommitmentSchemeVerifier` + `CommitmentScheme`, with native
fused `prove_batch` / `verify_batch` bodies. Never sees `open` /
`verify` / `combine` / `combine_hints`.

## IR Changes

The Jolt runtime IR (`crates/jolt-compiler/src/module.rs`) does
not gain new shapes from this PR; it has already been adjusted
on `quang/verifier-backend-v2` to use the fused
`Op::ProveBatch` + `VerifierOp::VerifyOpenings` pair. We carry
that forward.

| IR op | Status |
|---|---|
| `Op::ProveBatch` | unchanged from `quang/verifier-backend-v2` |
| `VerifierOp::VerifyOpenings` | unchanged shape, body delegates to `PCS::verify_batch` (was `PCS::verify_batch_with_backend`) |
| `Op::BindOpeningInputs` | unchanged; **stays separate** (see rationale) |
| `VerifierOp::CollectOpeningClaim` | unchanged |
| `Op::ReduceOpenings`, `Op::Open` | already removed on v2; remain removed |

`JoltProof::opening_proof: PCS::BatchProof` already in place from
v2; we keep it.

### Why `Op::BindOpeningInputs` stays separate

`bind_opening_inputs` absorbs the opening point and joint eval
into the Fiat-Shamir transcript so subsequent stages
(BlindFold R1CS, recursion) draw challenges that depend on the
just-verified opening. For Mock / HyperKZG it's a default no-op;
for Dory it appends two labelled records
(`dory_opening_point`, `dory_opening_eval`); see
`crates/jolt-dory/src/scheme.rs:220-231`.

Keeping it as its own IR op is preferred over folding it into
`verify_batch`:

1. **Independent protocol concerns.** "Verify the proof" and
   "absorb point/eval into transcript for downstream" are two
   different obligations. Mock/HyperKZG happen to combine them;
   Dory splits them. Folding either over-constrains Mock/HyperKZG
   with redundant ops or sneaks in implicit post-verify side
   effects.
2. **IR-level visibility.** Op counters, profilers, future
   ISA-level audits and (potentially) recursion lowering all see
   the bind explicitly.
3. **Parameterized by compiler decisions.**
   `Op::BindOpeningInputs { point_challenges: Vec<ChallengeIdx> }`
   pins where in the transcript schedule the opening point came
   from (a compiler-time decision). Encoding it in the IR keeps
   the prover/verifier transcript schedule visible.
4. **Multiple binds stay easy.** A future scheme that wants two
   binds (one per downstream stage) just emits two ops.
5. **Honors "expand the IR, not the abstractions."**

The cost is one extra IR op type and one return-slot field on
`prove_batch` (the `Vec<Self::Field>` of binding evals). Worth it.

## Call-site Migration

| File | Change |
|---|---|
| `crates/jolt-verifier/src/verifier.rs` | bound `PCS: CommitmentSchemeVerifier<Field = F>` (was `CommitmentScheme + OpeningVerification`) |
| `crates/jolt-verifier/src/proof.rs` | same bound |
| `crates/jolt-verifier/src/key.rs` | same bound |
| `crates/jolt-zkvm/src/{prove,proving_key,preprocessing,runtime/*}` | bound stays `CommitmentScheme<Field = F>`; runtime helpers don't call `PCS::open` directly so no `AdditivelyHomomorphic` bound needed at the runtime layer |
| `crates/jolt-{openings,hyperkzg,dory}/tests/verification_parity.rs` | replace `Native<F>` / `Tracing<PCS>` calls with plain `Blake2bTranscript`; two test files instead of three (drop the backend-specific one) |
| `crates/jolt-{dory,hyperkzg}/tests/commit_open_verify.rs`, `crates/jolt-dory/benches/dory.rs`, `#[cfg(test)]` blocks calling `XScheme::open` / `XScheme::verify` | NO change. Those methods are now on the `AdditivelyHomomorphic*` impls of the same concrete types, so the call syntax `DoryScheme::open(...)` / `DoryScheme::verify(...)` resolves identically. |
| `crates/jolt-bench/src/stacks/modular.rs` | bound updates |

Files removed (relative to `quang/verifier-backend-v2`):

- `crates/jolt-openings/src/backend.rs` (FieldBackend / CommitmentBackend / ScalarOrigin / CommitmentOrigin)
- `crates/jolt-verifier-backend/` (entire crate: Tracing, AstGraph, Native, viz, replay)
- `crates/jolt-verifier-backend/tests/verification_parity.rs`

## Testing Strategy

1. **Per-PCS parity tests.** `crates/{jolt-openings,jolt-hyperkzg,jolt-dory}/tests/verification_parity.rs`
   construct claims, call `prove_batch` with `Blake2bTranscript`,
   call `verify_batch` against an independently-constructed
   transcript, assert byte-identical transcript state at the end.
   Coverage: single claim, shared-point batch, distinct-point
   batch, mixed-group, empty batch.
2. **Verifier-only setup roundtrip.** Per-PCS: derive
   `VerifierSetup` two ways (`verifier_setup(public_params)`
   vs `project_verifier_setup(&prover_setup)` after
   `setup(setup_params)`) and assert equality of the resulting
   `VerifierSetup` (modulo deserialization roundtrip if needed).
3. **muldiv e2e in both modes.**
   `cargo nextest run -p jolt-equivalence muldiv` and
   `cargo nextest run -p jolt-equivalence transcript_divergence`
   gate every change. Transcript divergence catches IR / Fiat-
   Shamir mistakes; muldiv catches semantic regressions.
4. **Verifier-bound audit.** Add a `compile_fail` test in
   `jolt-verifier` that tries to use `PCS::ProverSetup` /
   `PCS::Polynomial` / `PCS::OpeningHint` from a generic context
   bound only on `CommitmentSchemeVerifier`. Locks in invariant 1.
5. **Clippy in both feature modes.** Standard
   `cargo clippy --all-targets -- -D warnings` plus the workspace
   feature variants used by the two existing modes.

## Rollout

Single PR off `refactor/crates`. Branch `quang/pcs-prover-verifier-split`.

1. Add `CommitmentSchemeVerifier` (verifier base) with
   `verifier_setup` / `verify_batch` / `bind_opening_inputs`.
   Add `VerifierSetupParams` associated type. Rename existing
   `verifier_setup` (the projection) to `project_verifier_setup`.
2. Convert `CommitmentScheme` to an extension trait
   (`: CommitmentSchemeVerifier`) with `setup` /
   `project_verifier_setup` / `commit` / `prove_batch` only.
   `open` / `verify` are removed from this trait.
3. Add `AdditivelyHomomorphicVerifier`
   (`: CommitmentSchemeVerifier`) with `combine` and `verify` (the
   body that lived in today's `CommitmentScheme::verify`).
4. Convert `AdditivelyHomomorphic` to extend
   `AdditivelyHomomorphicVerifier + CommitmentScheme`, add
   `combine_hints` (moved here from today's `AdditivelyHomomorphic`)
   and `open` (the body that lived in today's
   `CommitmentScheme::open`).
5. Add `ZkOpeningSchemeVerifier`
   (`: CommitmentSchemeVerifier`) with `verify_zk` +
   `HidingCommitment`; convert `ZkOpeningScheme` to extend it +
   `CommitmentScheme` with `open_zk` + `Blind`.
6. Update homomorphic helpers: `homomorphic_verify_batch` bounds on
   `AdditivelyHomomorphicVerifier`, calls `PCS::verify` (now resolved
   on the homomorphic verifier trait); `homomorphic_prove_batch`
   bounds on `AdditivelyHomomorphic`, calls `PCS::open` (now
   resolved on the homomorphic prover trait). Drop the
   `_with_backend` suffixes; helpers take plain transcripts.
7. Update `OpeningClaim` to drop the backend type param and bound
   on `CommitmentSchemeVerifier`.
8. Drop `OpeningReduction` / `OpeningVerification` traits entirely.
   Drop `crates/jolt-openings/src/backend.rs`.
9. Drop `crates/jolt-verifier-backend/` (entire crate).
10. Mechanically split each per-PCS `impl CommitmentScheme for X`
    into the impl blocks listed in the per-PCS table (pure method
    relocation, no logic changes).
11. Update IR consumers (`jolt-verifier`, `jolt-zkvm`,
    `jolt-bench`, tests). Bound updates only, no logic changes.
    Tests calling `XScheme::open` / `::verify` directly do not
    need editing (method resolution handles the trait relocation).
12. Re-write the three `verification_parity.rs` files to use plain
    `Blake2bTranscript` (no Native / no Tracing).
13. Run the full gate (clippy + per-crate nextest +
    `jolt-equivalence` muldiv + transcript_divergence). Commit.

PR #1461 stays parked as a reference for the discarded backend
direction. Spec doc 1461 stays untouched (records what landed and
why we stepped back). This PR is the forward direction.

## Alternatives Considered

### Lifecycle split (Setup / Commit / Open as three traits)

Considered but rejected. The `Setup` / `Commit` / `Open` stages
are **type-coupled by the math**: `commit` produces
`(Output, OpeningHint)` consumed by `open`; `open` produces
`Proof` consumed by `verify`; `verify` consumes
`Output + Proof + VerifierSetup`. Downstream call sites always
want the leaf trait (e.g. `Open`), so a lifecycle split collapses
to "write three `impl` blocks per PCS for one bound at the call
site". Pure cosmetic shuffle. Arkworks-pc, halo2, gnark all keep
the analogous trait monolithic for the same reason. The role
split (verifier vs prover) carves orthogonally and pays at call
sites.

### Keep single-claim `open`/`verify` on `CommitmentScheme(Verifier)`

Considered but rejected. Putting `open` / `verify` on the core
PCS traits forces every implementer (including Hachi-style fused
schemes that have no native single-claim notion) to provide
bodies. The available options for Hachi are
`unimplemented!()` (panic-prone in production) or
`Self::prove_batch(vec![one_claim], ...).0` (wasteful, semantically
backwards: deriving the supposed building block from the API
that's allegedly built on top of it). Both are bad. The chosen
design (single-claim methods on `AdditivelyHomomorphic*`)
matches actual semantics: in production these methods are called
from exactly one site each, the bodies of `homomorphic_*_batch`,
which are themselves the only consumers of `AdditivelyHomomorphic*`.
See follow-up "Drop single-claim entirely" for the inverse design.

### Drop single-claim `open`/`verify` entirely

Considered but rejected. With no `open`/`verify` in the trait
family, `homomorphic_prove_batch` / `homomorphic_verify_batch`
either (a) take per-call closures from each PCS or (b) get
inlined into each PCS's `prove_batch` / `verify_batch` body.
Option (a) gives ugly bound signatures (`OpenFn: Fn(&Polynomial,
&[F], F, &ProverSetup, Option<OpeningHint>, &mut T) -> Proof`)
and per-call closure overhead. Option (b) duplicates ~30-40 LOC of
grouping + RLC + transcript framing across Mock / HyperKZG / Dory.
Both lose the existing per-PCS test infrastructure
(`tests/commit_open_verify.rs`, `benches/dory.rs`, in-impl
`#[cfg(test)]` blocks) which call `XScheme::open` / `::verify`
directly. The chosen design (move down to `AdditivelyHomomorphic*`)
keeps the helpers simple, keeps tests/benches working without
edits, and still spares Hachi from implementing methods it
doesn't have.

### Fold prove_batch / verify_batch into existing single-claim methods

Considered but rejected (predates this spec; was the original
state on `refactor/crates`). Forces non-homomorphic schemes
(Hachi) to fake a single-claim API or fail at the trait level.

### Type pair (`CommitmentSchemeProver::Verifier = …`) instead of super-trait

Considered but rejected. Decouples prover and verifier types more
aggressively but every reference to a verifier-side type from the
prover becomes
`<PCS as CommitmentSchemeProver>::Verifier::Output` or similar.
Hairy associated paths everywhere. The super-trait approach
inherits types directly with zero ceremony.

### Separate Hachi-style `BatchedOpeningScheme` trait

Considered but rejected. We would end up with
`CommitmentSchemeVerifier` for single-claim verify and
`BatchedOpeningSchemeVerifier` for `verify_batch`. Two traits
tracking exactly the same `Self` set with mutually-redundant
bounds. Folding both prove/verify shapes into the same trait
keeps every PCS in one impl-block-pair regardless of whether it
batches via decomposition (homomorphic) or via fusion (lattice).

### Keep `OpeningReduction` as a separate trait

Considered but rejected. `OpeningReduction` was a vestige of an
intermediate "reduce, then verify per group" design that doesn't
fit Hachi-style schemes. Once we have `prove_batch` /
`verify_batch` on `CommitmentScheme(Verifier)`, there is no need
for an intermediate "reduce" surface: homomorphic schemes do the
RLC inside their `prove_batch` body via the helper, and
non-homomorphic schemes never had a "reduce" to expose. Same
reasoning as the v2 collapse, just applied without the
`OpeningVerification` intermediate.
