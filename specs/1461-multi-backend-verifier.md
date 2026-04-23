# Spec: Multi-Backend Verifier (FieldBackend + CommitmentBackend + Tracing AST)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @quangvdao                     |
| Created     | 2026-04-22                     |
| Status      | landed                         |
| PR          | #1461                          |

## Summary

The Jolt verifier is parameterised over two backend traits.
`FieldBackend` lifts every scalar operation, every Fiat-Shamir
transcript event, and every equality assertion off the concrete
field. `CommitmentBackend<PCS>` lifts the three PCS-shaped
operations the verifier performs (wrap, absorb, opening check) off
the concrete commitment scheme. Three implementations cover the
consumers we care about:

| Backend     | `Scalar`    | `Commitment`              | Use case                                              |
|-------------|-------------|---------------------------|-------------------------------------------------------|
| `Native`    | `F`         | `PCS::Output`             | Production: zero-cost, codegen identical to inline `F`|
| `Tracing`   | `AstNodeId` | `AstNodeId`               | Recursion / Lean export / differential testing        |
| `R1CSGen`   | `LcId`      | recursion-side group var  | Lower the verifier into an outer SNARK's R1CS         |

The verifier source code is the single source of truth: one
`verify_with_backend<B>(...)` body produces the production verifier
when monomorphized over `Native<F>` and the
recursion-/Lean-friendly trace when monomorphized over
`Tracing<PCS>`.

## Intent

### Goal

Make the `VerifierOp` interpreter generic over a
`(FieldBackend, CommitmentBackend<PCS>)` pair so that swapping
backends changes the execution semantics (concrete arithmetic vs
symbolic AST recording vs constraint emission) without changing the
verifier code. Downstream consumers (recursion, Lean export, R1CS
lowering) consume the `AstGraph<PCS>` produced by `Tracing` rather
than reimplementing the verifier.

Surface introduced by this PR (in `crates/jolt-verifier-backend`):

- `FieldBackend` (trait), `Native<F>`, `Tracing<PCS>`.
- `CommitmentBackend<PCS>` (trait, impls for `Native<F>` and
  `Tracing<PCS>`).
- `AstGraph<PCS>` (DAG of `AstOp<PCS>` nodes + `AstAssertion` list).
- `replay::<PCS>(graph, wraps, vk) -> Vec<F>` — symbolic
  re-execution; invokes `<PCS as CommitmentScheme>::verify` for each
  recorded `OpeningCheck`.
- DOT / Mermaid emitters for human inspection.

### Invariants

1. **Native parity.** For every input,
   `verify_with_backend::<Native<F>, PCS>` returns a result
   bit-identical to a hand-written verifier that calls `PCS::verify`
   directly. The
   `cargo nextest run -p jolt-equivalence modular_self_verify*`
   suite is the end-to-end witness:
   - `modular_self_verify` (`Native` backend, full proof) and
   - `modular_self_verify_via_tracing_backend` (`Tracing` backend +
     replay against the same proof)

   must both pass.
2. **Replay faithfulness.** For every `Tracing` execution that
   produces a graph `G` and a wrap-value vector `W`,
   `replay(G, W, vk)` returns a value vector of length
   `G.node_count()`, evaluates every `AstAssertion::Equality` to
   true, and discharges every `AstAssertion::OpeningHolds` by
   invoking `<PCS as CommitmentScheme>::verify` against `vk` and the
   recorded `(commitment, point, claim, proof)` tuple.
3. **Wrap accounting.** Every `Wrap` node consumes exactly one entry
   of `W`, in order. Commitment values and opening proofs are *not*
   in `W` — they are inlined on `CommitmentWrap.value` and
   `OpeningCheck.proof`.
4. **Transcript provenance.** Every `transcript.challenge()` the
   verifier makes appears in the AST as a `TranscriptChallengeValue`
   node whose `state` ancestor chain leads back to a single
   `TranscriptInit` and a sequence of `TranscriptAbsorbBytes` /
   `TranscriptAbsorbCommitment` / `OpeningCheck` nodes matching the
   bytes the `Native` path absorbs, in order.
5. **Sumcheck Fiat-Shamir.**
   `SumcheckVerifier::verify_with_backend` appends the same byte
   sequence to the transcript that `SumcheckVerifier::verify` would,
   including degree byte and label.

### Non-Goals

1. **PCS-side prover surface.** This PR is verifier-only. The
   prover continues to call `PCS::commit` / `PCS::open` directly.
2. **Performance.** Goal is correctness + a stable trait surface.
   The `Native` path is zero-cost (every method `#[inline(always)]`,
   monomorphization erases the trait). The `Tracing` path is
   acceptable for testing / recursion-prep; `AstGraph` allocations
   dominate. No benchmark targets.
3. **ZK feature coverage (`--features host,zk`).** Phases land in
   clear-mode only. ZK-mode parity is added when `refactor/crates`
   rebases onto BlindFold-bearing main.

## Design

### Architecture

```
┌──────────────────────────┐
│ jolt-verifier::verify    │ generic over <B: FieldBackend
│                          │              + CommitmentBackend<PCS>>
└────────────┬─────────────┘
             │ uses backend.{add, mul, squeeze,
             │  wrap_commitment, absorb_commitment,
             ▼  verify_opening, ...}
   ┌─────────────────────┐         ┌─────────────────────────┐
   │   FieldBackend      │         │ CommitmentBackend<PCS>  │
   │ - F, Scalar         │         │ - Commitment            │
   │ - Transcript        │         │ - wrap_commitment       │
   │ - {add,sub,mul,...} │         │ - absorb_commitment     │
   │ - {wrap_*, const_*} │         │ - verify_opening        │
   │ - new_transcript    │         └──────────┬──────────────┘
   │ - squeeze           │                    │
   │ - assert_eq         │                    │
   └─────┬───────────────┘                    │
         │ impl                               │ impl
         ├─────────────────────┐              │
         ▼                     ▼              ▼
  ┌──────────────┐     ┌──────────────────────────┐
  │  Native<F>   │     │   Tracing<PCS>           │
  │ Scalar = F   │     │ Scalar = AstNodeId       │
  │ Commitment = │     │ Commitment = AstNodeId   │
  │   PCS::Output│     │ records every op into    │
  │ everything   │     │ AstGraph<PCS>            │
  │ #[inline]    │     │ Transcript =             │
  │ Transcript = │     │ TracingTranscript<PCS>   │
  │ Blake2b<F>   │     └─────────┬────────────────┘
  └──────────────┘               │
                                 ▼
                     ┌──────────────────────┐
                     │ AstGraph<PCS>        │
                     │ nodes: Vec<AstOp<PCS>>│
                     │ assertions: Vec<…>   │
                     └─────┬────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      replay::<PCS>     to_dot         to_mermaid
      (graph, wraps,   (Graphviz)      (browser)
       vk) -> Vec<F>
```

### Backend traits

`FieldBackend` is the small surface every scalar / transcript op
goes through. `CommitmentBackend<PCS>` extends it with the three
PCS-shaped operations, statically parameterised by the
`CommitmentScheme` so the AST sees concrete `PCS::Output` /
`PCS::Proof` inhabitants.

```rust
pub trait CommitmentBackend<PCS>: FieldBackend
where
    PCS: CommitmentScheme<Field = <Self as FieldBackend>::F>,
    PCS::Output: AppendToTranscript,
    Self::Transcript: Transcript<Challenge = <Self as FieldBackend>::F>,
{
    type Commitment: Clone + Debug;

    fn wrap_commitment(
        &mut self,
        value: PCS::Output,
        origin: CommitmentOrigin,
        label: &'static str,
    ) -> Self::Commitment;

    fn absorb_commitment(
        &mut self,
        transcript: &mut Self::Transcript,
        commitment: &Self::Commitment,
        label: &'static [u8],
    );

    fn verify_opening(
        &mut self,
        vk: &PCS::VerifierSetup,
        commitment: &Self::Commitment,
        point: &[Self::Scalar],
        claim: &Self::Scalar,
        proof: &PCS::Proof,
        transcript: &mut Self::Transcript,
    ) -> Result<(), OpeningsError>;
}
```

The trait is **deliberately PCS-family agnostic**: it never names a
curve, a pairing, an MSM, or a linear combination of commitments.
Per-PCS batching (RLC for additively homomorphic schemes, FRI
folding for hash-based schemes, lattice aggregation) lives behind
`OpeningVerification::verify_batch_with_backend` (see "Opening
verification" below), not on this trait.

### AST: generic over `PCS`, values inlined

`AstGraph<PCS>`, `AstOp<PCS>`, and `Tracing<PCS>` are all
parameterised by the `CommitmentScheme`. Commitment-shaped variants
inline the inhabitant types directly:

```rust
pub enum AstOp<PCS: CommitmentScheme>
where PCS::Output: AppendToTranscript,
{
    // Field-side ops: Wrap, Constant, Neg, Add, Sub, Mul, Square,
    //                 Inverse.
    // Transcript ops: TranscriptInit, TranscriptAbsorbBytes,
    //                 TranscriptChallengeState, TranscriptChallengeValue.

    CommitmentWrap {
        value: Box<PCS::Output>,
        origin: CommitmentOrigin,
        label: &'static str,
    },
    TranscriptAbsorbCommitment {
        prev_state: AstNodeId,
        commitment: AstNodeId,    // -> CommitmentWrap node
        label: &'static [u8],
    },
    OpeningCheck {
        prev_state: AstNodeId,    // transcript state going in
        commitment: AstNodeId,    // -> CommitmentWrap node
        point: Vec<AstNodeId>,    // field nodes (challenges / consts)
        claim: AstNodeId,         // field node (claimed evaluation)
        proof: Box<PCS::Proof>,
    },
}

pub enum AstAssertion {
    Equality { lhs: AstNodeId, rhs: AstNodeId, ctx: &'static str },
    OpeningHolds { check: AstNodeId, ctx: &'static str },
}
```

Why generic-over-`PCS` with inlined values:

1. **Type-honest.** `AstGraph<DoryScheme>` literally contains
   `DoryProof`s. A Lean export reads off
   `nodes: List (AstOp DoryScheme)` directly; the soundness
   obligation on each `OpeningCheck` is "`DoryScheme.verify` accepts
   these inputs", with no marshalling or runtime tags.
2. **Exhaustively checked.** Every constructor is checked at compile
   time. Renaming a variant or changing `OpeningCheck` shape trips
   every consumer immediately.
3. **One source of truth per op.** `CommitmentWrap.value` is the
   commitment; `OpeningCheck.proof` is the proof. There is no
   parallel sidecar whose ordering can drift from the AST.
4. **Smaller blast radius for non-curve schemes.** A future
   `HachiCommitmentScheme` lights up `AstGraph<HachiCommitmentScheme>`
   with no new opcodes and no scheme-tag bookkeeping.

Cross-cutting trait obligations in `jolt-openings`:

- `CommitmentScheme::Proof: Debug` — so `AstOp<PCS>` derives
  `Debug` (the AST inlines `Self::Proof`).
- `OpeningsError: Clone` — so `replay` can stash a per-`OpeningCheck`
  rejection in its obligation map and surface it on the matching
  `OpeningHolds` assertion.

### Replay

```rust
pub fn replay<PCS: CommitmentScheme>(
    graph: &AstGraph<PCS>,
    wrap_values: &[PCS::Field],
    vk: &PCS::VerifierSetup,
) -> Result<Vec<PCS::Field>, BackendError>
where PCS::Output: AppendToTranscript;
```

Per-variant semantics:

- `Wrap` consumes the next entry of `wrap_values`; field arithmetic
  ops compute on the recorded operand ids.
- `TranscriptInit` / `TranscriptAbsorbBytes` /
  `TranscriptChallengeState` / `TranscriptChallengeValue` thread a
  live `Blake2bTranscript` through the AST so squeezed challenges
  are derived deterministically from the recorded byte stream.
- `CommitmentWrap` is a no-op for replay (value already inline).
- `TranscriptAbsorbCommitment` resolves `commitment` to its inline
  `PCS::Output`, drives the live transcript through the standard
  `LabelWithCount + AppendToTranscript` two-step, and threads the
  post-absorb state into the new node.
- `OpeningCheck` resolves the live transcript at `prev_state`,
  invokes `<PCS as CommitmentScheme>::verify(commitment, point,
  claim, proof, vk, transcript)`, threads the post-verify
  transcript into the new node, and stores the
  `Result<(), OpeningsError>` in a per-replay map.
- `AstAssertion::Equality` checks `values[lhs] == values[rhs]`,
  returning `BackendError::AssertionFailed(ctx)` on mismatch.
- `AstAssertion::OpeningHolds` looks up the stored `Result` for
  `check` and returns
  `BackendError::OpeningCheckFailed { ctx, source }` on `Err`.

### Opening verification (per-PCS batching)

`OpeningVerification` is the trait every PCS implements to verify a
**batch of opening claims as a single fused operation**. Each PCS
supplies its own implementation; there is **no blanket impl** over
`AdditivelyHomomorphic`, so a future hash-based or lattice-based
scheme (e.g. `HachiCommitmentScheme`) implements `OpeningVerification`
directly with whatever batching is natural for it.

The trait carries one associated type and two symmetric methods:

```rust
pub struct OpeningClaim<B: CommitmentBackend<Self>, Self: CommitmentScheme>
where Self::Output: AppendToTranscript,
{
    pub commitment: B::Commitment,
    pub point: Vec<B::Scalar>,
    pub eval: B::Scalar,
}

pub trait OpeningVerification: CommitmentScheme {
    /// Single proof object covering the entire batch. For additively
    /// homomorphic schemes (Mock, HyperKZG, Dory) this is just
    /// `Vec<Self::Proof>` — one inner opening per RLC group. For
    /// fused lattice schemes (Hachi) this is the scheme's native
    /// `BatchedProof` type.
    type BatchProof: ...;

    fn prove_batch<T: Transcript<Challenge = Self::Field>>(
        claims: Vec<ProverClaim<Self::Field>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut T,
    ) -> (Self::BatchProof, Vec<Self::Field>);

    fn verify_batch_with_backend<B: CommitmentBackend<Self, F = Self::Field>>(
        backend: &mut B,
        vk: &Self::VerifierSetup,
        claims: Vec<OpeningClaim<B, Self>>,
        batch_proof: &Self::BatchProof,
        transcript: &mut B::Transcript,
    ) -> Result<(), OpeningsError>
    where Self::Output: AppendToTranscript;
}
```

The two methods are **symmetric**: `prove_batch` and
`verify_batch_with_backend` consume the entire bag of claims and
emit / consume a single `BatchProof`. The earlier two-step
"reduce then verify" surface (`reduce_prover` + per-claim `open` /
`reduce_verifier` + per-claim `verify`) was collapsed into this
fused form because lattice-based schemes do not factor through that
intermediate "reduced claim" step at all — they compute one batched
proof in a single pass.

Returned `Vec<Self::Field>` from `prove_batch` is the per-group
"binding eval" — for additively homomorphic schemes, the RLC-combined
evaluation per opening point. The runtime threads these into
`Op::BindOpeningInputs` so Dory's existing transcript binding
behavior is preserved.

Per-PCS impls (`MockCommitmentScheme`, `HyperKZGScheme`,
`DoryScheme`) delegate to two helpers in `jolt-openings`:
`homomorphic_prove_batch` (group claims by point, RLC-combine
polynomials, open one per group) and
`homomorphic_verify_batch_with_backend` (mirror: group, combine
commitments + evals, verify one per group). The combine step takes a
fast path on `Native::Commitment = PCS::Output` (direct
`PCS::combine`) and an AST-emitting path on
`Tracing::Commitment = AstNodeId` (record the inputs as transcript
absorbs, push a `CommitmentWrap` for the precomputed combined
commitment). The verifier never sees the per-PCS difference.

A future `HachiCommitmentScheme::OpeningVerification` impl can set
`type BatchProof = HachiCommitmentScheme::BatchedProof` and call
the lattice scheme's fused `batched_open` / `batched_verify`
directly — no homomorphism required, no intermediate reduce step.

### Note: `ProverClaim` rename TODO

`ProverClaim<F>` (in `jolt-openings::claims`) currently bundles
`{polynomial, point, eval}` and is the prover-side analogue of
`OpeningClaim<B, PCS>`. Two follow-up refactors are likely:

1. Rename to `ProverOpeningClaim<F>` for naming symmetry with
   `OpeningClaim<B, PCS>`.
2. Optionally extend with `commitment` / slot metadata to match the
   verifier-side struct, which would let lattice schemes thread
   commitment-slot information through `prove_batch` without
   side-channel state.

Both are non-blocking; deferred to a follow-up PR. Tagged in
`jolt-openings/src/claims.rs` with a TODO.

### Wrap origins

Inputs to the AST are tagged with `ScalarOrigin` (`Public`, `Proof`,
`Challenge`) or `CommitmentOrigin` (`Public` for vk-pinned, `Proof`
for prover-supplied). This labelling is what makes the AST useful
for downstream consumers: a Lean exporter can quantify only over
`Proof` and `Challenge` inputs; a recursion verifier can fix
`Public` constants at compile time.

### Module map

- `crates/jolt-verifier-backend/src/`
  - `backend.rs` — `FieldBackend`, `ScalarOrigin`, `CommitmentOrigin`.
  - `commitment.rs` — `CommitmentBackend<PCS>`.
  - `native.rs` — `Native<F>` (zero-cost).
  - `tracing.rs` — `Tracing<PCS>`, `TracingTranscript<PCS>`,
    `AstGraph<PCS>`, `AstOp<PCS>`, `AstAssertion`, `replay`. Shared
    state is `Arc<Mutex<…>>` because `Transcript: Sync + Send + 'static`.
  - `helpers.rs` — backend-aware `eq_eval`, `lagrange_*`,
    `lt_mle`, `identity_mle`, `pow_u64`, `univariate_horner`.
  - `viz.rs` — DOT + Mermaid emitters, generic over `PCS`.
- `crates/jolt-sumcheck/src/verifier.rs` —
  `SumcheckVerifier::verify_with_backend`.
- `crates/jolt-verifier/src/verifier.rs` — `verify_with_backend`,
  `evaluate_formula_with_backend`,
  `evaluate_preprocessed_poly_with_backend`,
  `eval_io_mle_with_backend`.

## Evaluation

### Acceptance Criteria

**FieldBackend surface:**

- [x] `FieldBackend` covers every field op the verifier uses
      (`add`, `sub`, `mul`, `neg`, `square`, `inverse`, `wrap_*`,
      `const_*`, `assert_eq`, `Transcript`, `new_transcript`,
      `squeeze`).
- [x] `Native<F>` is `Scalar = F`, every method `#[inline(always)]`,
      no allocation beyond a hand-written verifier that calls
      `PCS::verify` directly.
- [x] `Tracing<PCS>` records every op as the corresponding `AstOp`.
- [x] `replay::<PCS>` re-executes every node and discharges every
      assertion or returns a structured error pinpointing the
      failing node / assertion.
- [x] `viz::to_dot` / `viz::to_mermaid` render every node kind with
      distinct styling and label every transcript-state edge.
- [x] `evaluate_formula_with_backend` covers all 13 `ClaimFactor`
      variants. 11 backend-parity tests in
      `crates/jolt-verifier/src/verifier.rs` assert
      `Native == evaluate_formula` and
      `Tracing-replay == evaluate_formula`.
- [x] `SumcheckVerifier::verify_with_backend` returns
      `(final_eval_w, challenges_w, challenges_f)` matching
      `SumcheckVerifier::verify` on the Native path.
- [x] `verify_with_backend` (top-level Jolt verifier) instantiates
      its transcript via `backend.new_transcript()` and squeezes via
      `backend.squeeze()`, with no direct `Blake2bTranscript::new` /
      `transcript.challenge()` calls remaining in the function body.

**CommitmentBackend surface:**

- [x] `OpeningVerification` is implemented per-PCS — no blanket impl
      over `AdditivelyHomomorphic`. `verify_with_backend` requires
      only `PCS: OpeningVerification`; the `HomomorphicCommitment<F>`
      bound is no longer in `jolt-verifier`.
- [x] `CommitmentBackend<PCS>` trait exists with three methods.
      `Native::Commitment = PCS::Output` (identity);
      `Tracing::Commitment = AstNodeId`.
- [x] `AstGraph<PCS>` and `AstOp<PCS>` are generic over `PCS`. The
      three commitment-shaped variants (`CommitmentWrap`,
      `TranscriptAbsorbCommitment`, `OpeningCheck`) inline
      `PCS::Output` / `PCS::Proof`. `AstAssertion::OpeningHolds`
      exists; `replay` discharges it via
      `<PCS as CommitmentScheme>::verify`.
- [x] `modular_self_verify_via_tracing_backend` exercises the full
      `Tracing<DoryScheme>` round-trip against real Dory openings:
      records ≥ 1 of each commitment-shaped variant, replays, and
      every `OpeningHolds` discharges.
- [x] No mention of `g1_msm`, `pairing`, `MSM`, or `GroupBackend`
      anywhere in `crates/jolt-verifier-backend/src/`. The trait
      surface is curve-agnostic.
- [x] `OpeningVerification::verify_batch_with_backend` and
      `OpeningVerification::prove_batch` implemented for
      `MockCommitmentScheme`, `HyperKZGScheme`, `DoryScheme`. Per-PCS
      parity tests live in
      `crates/jolt-verifier-backend/tests/verification_parity.rs`,
      `crates/jolt-hyperkzg/tests/verification_parity.rs`, and
      `crates/jolt-dory/tests/verification_parity.rs` (5 cases each:
      single, shared-point, distinct-points, mixed groups, empty),
      driving `prove_batch` then `verify_batch_with_backend` over the
      `Native<F>` backend and asserting both transcripts end in
      byte-identical state.
- [x] `verify_with_backend` calls
      `OpeningVerification::verify_batch_with_backend` exclusively.
      No direct `PCS::verify` / `PCS::reduce_verifier` /
      `reduce_verifier_with_backend` calls remain in the verifier
      crate. The `JoltProof` carries a single
      `opening_proof: PCS::BatchProof` (was previously
      `Vec<PCS::Proof>`).

### Testing Strategy

- **Per-backend unit tests.** `jolt-verifier-backend` (30+ tests):
  field-side ops, transcript ops, commitment ops (round-trip and
  tampered-claim rejection), DOT / Mermaid emitters.
- **Sumcheck.** `jolt-sumcheck` (8 verifier tests + 27 sumcheck
  tests): `verify_with_backend` matches `verify` on `Native` and
  replays correctly through `Tracing<MockCommitmentScheme<F>>`.
- **Verifier formula parity.** `jolt-verifier::verifier::tests` —
  11 backend-parity tests across all `ClaimFactor` and
  preprocessed-poly variants.
- **End-to-end.** `jolt-equivalence::muldiv`:
  - `modular_self_verify` — full prover + native verifier.
  - `modular_self_verify_via_tracing_backend` — full prover +
    `Tracing<DoryScheme>` verifier + replay against real Dory
    openings. Asserts every commitment-shaped variant appears in
    the recorded graph and every `OpeningHolds` discharges.
- **Verification parity (per PCS).** Each PCS ships a dedicated
  `verification_parity.rs` test (5 shapes: single, shared-point,
  distinct-points, mixed groups, empty) that builds a bag of opening
  claims, calls `prove_batch` against a Blake2b transcript, then
  `verify_batch_with_backend::<Native<_>>` against a fresh
  Blake2b transcript with the same label, and asserts both
  transcripts emit equal post-batch challenges. `MockCommitmentScheme`
  lives in `crates/jolt-verifier-backend/tests/`; `HyperKZGScheme`
  and `DoryScheme` live in their own crate `tests/` directories.

### Performance

- **Native path:** zero-cost. `Scalar = F`, every backend method
  is `#[inline(always)]`, and the verifier does not branch on the
  backend type. Empirical check: `modular_self_verify` runtime
  before vs after introducing the backend trait stays inside noise
  (< 5%).
- **Tracing path:** acceptable for testing / recursion-prep. Runs
  over muldiv (~4 s, of which most is proof generation).
  Allocations dominated by `Vec` growth in `AstGraph.nodes` and
  `Arc<Mutex<…>>` lock acquisition on every record. Not optimised;
  no budget enforced; not on any production hot path.
- **No prover-side work.** This PR is verifier-only.

## Alternatives Considered

1. **Two parallel verifiers (one native, one symbolic).** Rejected
   — guarantees drift. Every change to the native verifier silently
   fails to propagate to the symbolic one until a downstream
   consumer breaks. Single source of truth + per-backend impl is
   strictly safer.
2. **Macro / build-script codegen of the symbolic verifier from
   the native one.** Rejected — opaque, hard to debug, hard to
   extend with new backends (e.g. constraint synthesis for SNARK
   recursion).
3. **SNARK-composed recursion only (no AST).** Rejected — the Lean
   export need is real and cannot be served by SNARK composition
   alone. One AST that serves both keeps the surface unified.
4. **Hide the backend behind associated types of `VerifierOp`
   itself.** Rejected — couples the op enum to the backend trait,
   makes the op enum harder to serialise and harder for non-backend
   consumers (e.g. a future cost model).
5. **Treat transcript challenges as opaque inputs (no transcript
   ops in the AST).** Rejected — without transcript nodes, a Lean
   / recursion consumer would have to re-derive challenges
   externally and trust that the labelling matches the verifier,
   reintroducing the same drift hazard as alternative 1.
6. **`GroupBackend` with `g1_msm` / `pairing` / `g_combine`
   primitives.** Rejected — bakes elliptic-curve structure into
   the verifier's public surface. Any future PCS that does not
   factor through pairings or MSMs (FRI / hash-based,
   Ajtai / lattice-based) either gets shoehorned into a
   curve-shaped AST or forces a parallel backend trait. Both
   outcomes break the "one verifier source of truth" invariant.
   `CommitmentBackend` keeps the surface curve-agnostic and pushes
   per-PCS batching into `OpeningVerification::verify_batch_with_backend`.
7. **Type-erased proof sidecar (`Box<dyn Any>` per
   `OpeningCheck`).** Rejected — pushes critical typing
   information into runtime metadata, hostile to formal
   verification (Lean / Coq sees a node that merely *promises* to
   verify a `DoryProof`, not one that *is* a `DoryProof`), leaks
   the `Box<dyn Any>` indirection into every downstream consumer,
   and introduces a parallel `Vec<Box<dyn Any>>` whose ordering
   must stay in sync with the AST. Generic-over-`PCS` with inlined
   values is type-honest at every layer.
8. **`AstOp::PcsExtension { tag, payload: Vec<u8> }`.** Rejected —
   same Lean-unfriendliness as alternative 7, plus an ad-hoc
   serialisation contract per PCS. Future PCS variants get their
   own `AstGraph<NewScheme>` instead.
9. **Keep an `AdditivelyHomomorphic` blanket on
   `OpeningVerification`.** Rejected — couples every consumer of
   `OpeningVerification` to the curve-based / RLC batching strategy.
   Single trait + per-PCS impl admits any batching strategy.
10. **Two-step `OpeningReduction` (separate `reduce` + `open` /
    `verify` phases).** Rejected — does not fit fused lattice
    schemes (Hachi's `batched_open` / `batched_verify` is a single
    primitive, not two). Earlier drafts of this PR shipped this
    surface; it was collapsed into the single-method
    `OpeningVerification` trait once the Hachi mismatch surfaced.
    The fused form is strictly more general: additively
    homomorphic schemes implement it via `homomorphic_prove_batch`
    / `homomorphic_verify_batch_with_backend` helpers, lattice
    schemes implement it directly with their native batched
    primitives.

## Documentation

No `book/` changes in this PR. The user-facing surface (`verify`)
is unchanged; this PR is an internal abstraction. A "Verifier
Backends" chapter in the book is warranted as a follow-up,
covering:

- The `FieldBackend` and `CommitmentBackend` trait surfaces.
- How to write a custom backend (e.g. constraint synthesis for a
  recursive SNARK).
- Replay + viz workflows for debugging a verifier proof failure.

## References

- PR #1461 — implementation.
- Markos's `refactor/crates` base commit `c7e0869d5` — the
  `VerifierOp` interpreter the verifier is parameterised over.
- `paper-note/multi-backend-verifier-design.md` — extended design
  rationale (in `~/Documents/Notes/`).
- Closed Dory PR (`~/Documents/SNARKs/dory`) — prior art for the
  same symbolic-AST-over-native-execution pattern in a
  single-protocol setting.
