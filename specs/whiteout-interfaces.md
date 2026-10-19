# Spec appendix: Whiteout interfaces

| Field     | Value                                    |
| --------- | ---------------------------------------- |
| Author(s) | @markosg04                               |
| Created   | 2026-10-19                               |
| Status    | draft, revised after design review       |
| Parent    | `specs/shared-primitive-layer.md`        |

This appendix fixes the interfaces that later tickets build on, before any of
them is implemented. Each section names the ticket that implements it. Bodies,
helper types, and module layout are the implementer's choice; the signatures and
the stated contracts are not. A ticket that needs to change a signature here
changes this file in the same PR.

## 1. Single-instance sumcheck (M2a)

Akita runs unbatched sumchecks with grinding and sites; Jolt runs batched ones
with `challenge_small`. The batched engine (`prove_batch`, the generated
`begin_batch`/`verify_*` drivers, the uni-skip provers) keeps its current API
and its current bytes. M2a adds a single-instance driver that reproduces Akita's
bytes, and moves Akita onto it in M2b.

### Round checks have one owner

`SumcheckDomain::round_sum_coefficients` (`domain.rs`) already owns the weights
of a round check: `[2, 1, 1, …]` for `BooleanHypercube` and power sums for
`CenteredIntegerDomain`. The verifier and BlindFold's R1CS lowering consume it.
M2a adds the third format there instead of a new enum:

```rust
/// One round of an equality-factored sumcheck: s(X) = eq(τ, X) · q(X).
/// Check (1 − τ) q(0) + τ q(1) = claim, i.e. weights [1, τ, τ, …] on q.
pub struct EqFactoredRound<E> { pub tau: E }
impl<E: Field> SumcheckDomain<E> for EqFactoredRound<E> { /* weights [1, τ, τ, …] */ }
```

`SumcheckDomainSpec` gains the matching variant. Recovery of the omitted
coefficient stays with jolt-poly (`CompressedPoly`, `OmittedConstantPoly`).

Two facts the drivers and masking rely on:

- **Eq-factored claims are normalized.** The running claim after round i is
  q_i(r_i), not eq(τ_i, r_i) · q_i(r_i), exactly as Akita's
  `advance_eq_factored_claim` computes it. The product of eq factors is applied
  by the caller at the terminal check (Akita's `claim_scales`). The degree bound
  of an eq-factored instance is the degree of q.
- **Which rows masking needs.** For `Compressed` and `EqFactored` rounds the
  omitted coefficient is defined by the check, so the check row is tautological
  and masking adds none. For uni-skip rounds every coefficient is sent and
  masked, so the check is an explicit affine row.

### Context owns the transcript

Akita's grinding context (`ProverGrinding`) owns the `&mut ProverTranscript` and
is threaded through its whole prover. The driver therefore takes one context
that owns the transcript, matching Akita's existing `SumcheckProverChannel`:

```rust
#[repr(u32)]
pub enum SumcheckRole { Claim = 1, RoundBody = 3, Challenge = 4 } // Akita's discriminants

pub struct SumcheckSite { pub invocation: u32, pub round: u32, pub role: SumcheckRole }

pub trait ProverRoundContext<E> {
    type Sponge: Sponge;
    type Error;
    fn transcript_mut(&mut self) -> &mut ProverTranscript<Self::Sponge>;
    /// The transcript site for `site`, or `None` to emit no site.
    fn site(&self, site: SumcheckSite) -> Option<SiteId>;
    /// Grinding and the round challenge.
    fn challenge(&mut self, site: SumcheckSite) -> Result<E, Self::Error>;
}
pub trait VerifierRoundContext<'proof, E> { /* mirror, over VerifierTranscript<'proof, _> */ }
```

Akita implements both on its existing grinding wrappers; its grinding plan stays
in Akita. A Jolt context wrapping a bare transcript is added only when a Jolt
caller needs `prove_single`.

### Kernels and errors

```rust
pub trait ProveRounds<E: Field, K = Infallible> {
    fn num_rounds(&self) -> usize;
    /// For eq-factored instances, returns the full q. Kernels that produce
    /// [q1..qd] rebuild q0 with the jolt-poly recovery, not a local formula.
    fn prove_round(&mut self, bind: Option<E>, round: usize, previous_claim: E)
        -> Result<UnivariatePoly<E>, SumcheckError<E, K>>;
    fn finish_rounds(&mut self, bind: E) -> Result<(), SumcheckError<E, K>>;
}

pub enum SumcheckError<E, K = Infallible> { /* existing variants */ Kernel(K) }
impl<E> SumcheckError<E, Infallible> {
    pub fn widen<K>(self) -> SumcheckError<E, K>;
}
```

`K` appears only on `prove_single`/`verify_single` and the kernels they drive.
The batched engine keeps `K = Infallible`, so `MemberRound`, `MemberFinish`, and
`RoundScheduler` do not change. A blanket `From` impl is impossible
(conflicts with `impl<T> From<T> for T`), so internal helpers return
`SumcheckError<E>` and the single driver calls `widen`. Akita uses
`K = AkitaError`. Akita's fused-API adapter runs `bind_challenge(round − 1, b)`
then `round_polynomial` inside `prove_round`, and `bind_challenge` then
`finish` inside `finish_rounds`.

A sumcheck error is an invariant violation or a kernel failure. Retryable
aborts are not sumcheck errors (§5).

### Driver

```rust
pub struct SingleInstance<E> {
    pub input_claim: E,
    pub rounds: usize,
    pub degree: usize,
    pub format: SingleFormat<E>,   // Compressed | EqFactored { equality_point: Point<LOW_TO_HIGH, E> }
    pub invocation: u32,
}

/// Byte-for-byte Akita's `prove_sumcheck` / `prove_eq_factored_sumcheck` in clear
/// mode: input claim bound through the recorder at the Claim site; per round,
/// wire coefficients at the RoundBody site, then `context.challenge`. Sends no
/// output claims.
pub fn prove_single<F, E, K, C, R>(
    member: &mut dyn ProveRounds<E, K>,
    instance: &SingleInstance<E>,
    context: &mut C,
    recorder: &mut R,
) -> Result<ProvedSingle<E>, SumcheckError<E, K>>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    C: ProverRoundContext<E, Error = K>,
    R: SumcheckRecorder<E, Error = K>;

/// Replays the rounds and returns (output claim, challenges) before any oracle
/// check, as Akita's `verify_*_rounds` do; Akita exchanges late claims first.
pub fn verify_single<F, E, K, C>(/* mirror */) -> Result<SingleReplay<E>, SumcheckError<E, K>>;
```

`τ_i = equality_point[i]` matches `GruenSplitEq::current_tau` (checked against
Akita); M2b pins it with a byte-identity test.

## 2. Private messages (M3a)

Masking must cover every witness-dependent field message, and the verifier must
assign the same pad indices the prover used. One role-generic seam does both,
following Akita's existing pattern of writing each exchange once over both roles
(`GrindingReplay::exchange`).

```rust
/// Where a private message's pads sit in the pad schedule. Empty in clear mode.
pub struct PadSlots { /* start, len */ }

/// Witness-dependent field messages. Prover: sends values (clear) or
/// value + next pads (masked). Verifier: receives them and, in masked mode,
/// returns the pad slots the prover used.
pub trait PrivateChannel {
    type Error;
    /// Exchanges `values` in place (filled from the proof on the verifier).
    fn exchange_private<F, E>(&mut self, values: &mut [E]) -> Result<PadSlots, Self::Error>
    where F: Field + CanonicalEncoding, E: ExtField<F> + CanonicalDecode;
    /// Binds a claim the verifier can already derive. Clear: absorbed as a
    /// public value. Masked: nothing, since earlier masked slots bind it.
    fn bind_derived<F, E>(&mut self, claim: &E) -> Result<(), Self::Error>
    where F: Field + CanonicalEncoding, E: ExtField<F> + CanonicalDecode;
}
```

Pads cover base-field coordinates, so an extension element takes `E::DEGREE`
pads. `SumcheckRecorder` is built on this seam: input claims go through
`bind_derived`, round coefficients and output claims through `exchange_private`.
`SumcheckRecorder` and `PrivateChannel` carry an associated `Error`, so pad
exhaustion has a channel in both.

Messages that go through the seam:

- **Jolt:** every sumcheck round, claimed evaluations, uni-skip output claims.
- **Akita, folds 1..k:** stage-1 child claims and range image, L2 subclaims and
  virtual evaluations, EOR partials and final claims, stage-2/3 evaluations,
  `opening_payload`, and the `NextWitness` payload.

Messages that stay public: commitments, grinding nonces, setup-offloading
claims. Akita's fold-selection retry nonce is witness-dependent; W2 removes it,
and until then ZK mode cannot be selected.

Messages after the cut point are not masked: they are replaced by the response
proof. That covers the terminal `e_folded`, the Golomb-Rice `z` payload, and
`response_l2_sq` if it falls after fold k. W3 confirms on which side of the cut
each of these falls.

## 3. Commitments (M4, W4, W6)

M4 is cleanup only: delete `jolt-akita`'s `ZkOpeningScheme` impl (a transparent
`open_zk` and an always-erroring `verify_zk`) and its all-erroring
`ZkBatchOpeningScheme` impl. Nothing requires them, because the `akita` and `zk`
features are mutually exclusive today.

The PCS trait split lands in W6, when Akita first commits with randomness, so
both halves have a caller:

```rust
/// Commit with randomness. The blind lives in `OpeningHint`, where Dory keeps
/// it today; it is not returned separately.
pub trait HidingCommitment: CommitmentScheme {
    fn commit_hiding<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P, setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError>;
}

/// Opening that reveals nothing beyond a commitment to the evaluation.
pub trait ZkOpening: HidingCommitment {
    type EvalCommitment: /* today's HidingCommitment bounds */;
    type EvalBlind: Clone + Send + Sync;
    fn open_zk</* as today */>(/* … */) -> Result<(Self::EvalCommitment, Self::EvalBlind), OpeningsError>;
    fn verify_zk</* as today */>(/* … */) -> Result<Self::EvalCommitment, OpeningsError>;
}
```

`ZkStreamingCommitment` becomes `HidingStreamingCommitment`.
`ZkBatchOpeningScheme` keeps its shape. It cannot take `ZkOpening` as a
supertrait, because its implementors are batching markers, so it bounds its
`PCS` parameter instead.

The vector-commitment change lands in W4 with the lattice pad commitment, its
first non-scalar implementor:

- `VectorCommitment` gains `type Blind` and `sample_blind`. Its `Output: Copy`
  bound relaxes to `Clone`, because an Ajtai commitment is heap-allocated.
- `commit` takes field values and documents that any gadget decomposition
  happens inside it.
- Row openings move to `RowOpening: VectorCommitment<Blind = Self::Field>`,
  which only Pedersen implements. An Ajtai blind's norm grows under
  combination, and the lattice pad commitment is opened by the Σ-protocol
  instead.

## 4. Pads (M6b)

```rust
/// Public pad layout, derived from the protocol's public shape by one function
/// that prover and verifier both call.
pub struct PadLayout {
    pub num_pads: usize,
    /// Pads of product operands (§5), stored as constant digit polynomials in
    /// the pad commitment.
    pub operand_pads: Vec<PadIndex>,
    /// Auxiliary slots for masked intermediate products of checks of degree > 2.
    pub auxiliary_slots: Vec<AuxSlot>,
}

/// Pads committed before the first challenge. Masking recorders and
/// `PrivateChannel`s can only be built from this, so they cannot run before the
/// commitment exists.
pub struct CommittedPads<F, B> { /* pads, cursor, blind */ }
```

The back end creates `CommittedPads` (§5); the prover sends the commitment
before any challenge. At the end the cursor must equal `layout.num_pads`.
Masked intermediate products are sent through `exchange_private` at their
auxiliary slots, during the transcript, before the challenge that uses them.

## 5. ZK layer (M6b, W4)

### Residual relation

```rust
pub enum ResidualVar { Pad(PadIndex), Witness(usize) }

/// Σ_k a_k · x_k + b = 0, over the base field.
pub struct AffineRow<F> { pub terms: Vec<(ResidualVar, F)>, pub constant: F }

/// x · y = z, with x a single masked value: one constant operand per product
/// (Whiteout §3.2), so the back end can add its scalarity rows for x's pad.
pub struct MaskedOperand<F> { pub masked: F, pub pad: PadIndex }
pub struct ProductRow<F> { pub x: MaskedOperand<F>, pub y: AffineRow<F>, pub z: AffineRow<F> }

pub struct ResidualRelation<F> {
    pub affine: Vec<AffineRow<F>>,
    pub products: Vec<ProductRow<F>>,
    pub layout: PadLayout,
    pub num_witness: usize,
}

/// A verifier that contributes rows after replaying its part of the
/// transcript: Jolt's stages (through the existing verifier replay) and
/// Akita's folds up to the cut point (W3).
pub trait ResidualRows<F> {
    type Error;
    fn append_rows(&self, relation: &mut ResidualRelation<F>) -> Result<(), Self::Error>;
}
```

- **Field level.** Rows are over the base field. The masking layer owns
  expanding an extension-field affine check into `E::DEGREE` base-field rows.
- **What the back end adds.** It never sees the mask g as a row variable. g,
  the constant-term lift, the scalarity rows of operand pads, and the
  ring-linear rows N·s = t (residual and pad commitment openings) are the
  lattice back end's. N and t come from its `Setup`, and `ResidualRows` hands
  it the residual commitment.
- **Replacing BlindFold's lowering.** M6c lowers `ResidualRelation` to
  BlindFold's relaxed R1CS. That replaces the per-round lowering in
  `jolt-sumcheck/src/r1cs.rs`, which is deleted with the committed recorder.

### Back ends

```rust
pub trait ZkBackend<F> {
    type Setup;
    type Blind;
    type Witness;
    /// Samples pads, commits to them (and, for Whiteout, to g and r_p) and
    /// sends the commitment. Called before the first challenge.
    fn commit_pads<H: Sponge, R: RngCore + CryptoRng>(
        setup: &Self::Setup, layout: &PadLayout, transcript: &mut ProverTranscript<H>, rng: &mut R,
    ) -> Result<CommittedPads<F, Self::Blind>, ZkError>;
    /// Reads the pad commitment at the same transcript position, before the
    /// first challenge. The verifier-side `PrivateChannel` can only be built
    /// from the returned view.
    fn receive_pad_commitment<H: Sponge>(
        setup: &Self::Setup, layout: &PadLayout, transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<CommittedPadsView<Self::Blind>, ZkError>;
    fn prove<H: Sponge, R: RngCore + CryptoRng>(
        setup: &Self::Setup, relation: &ResidualRelation<F>, pads: CommittedPads<F, Self::Blind>,
        witness: &Self::Witness, transcript: &mut ProverTranscript<H>, rng: &mut R,
    ) -> Result<(), ZkError>;
    fn verify<H: Sponge>(
        setup: &Self::Setup, relation: &ResidualRelation<F>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError>;
}

pub enum ZkError {
    /// W2: a one-shot fold response overflowed its digit range. Retry the whole
    /// proof, resampling every pad and all randomness.
    FoldOverflow,
    // invariant violations, transcript errors, …
}
```

Σ-protocol rejection is not an error. `whiteout::prove` retries internally
with a fresh mask y and a fresh attempt nonce, as the paper requires
(Σ-protocol, §4.2, step 4). Each nonce n_i is a uniform ν-bit value from the
prover's RNG, never a counter, and the challenge oracle's input is
(transcript state, commitment to V, n_i). Only the accepted attempt's nonce and
commitment enter the transcript; a rejected attempt's nonce is never revealed.
This is what the zero-knowledge proof's hybrid H2 relies on.
`blindfold` implements `ZkBackend` in M6c, and `whiteout` implements it in W4.

### Response proof (W4, W5)

```rust
/// Whiteout §4.3: digit ranges for every coordinate of e; the compression
/// equations of the three commitments with their binary decompositions ξ, and
/// t_D = A_D G⁻¹(V); N z = f + c t; the ring-quadratic identity; ct(h) = 0.
/// Public coordinates of s get smaller digit bounds but no z = c s equations
/// (§4.2, "Public coordinates").
pub struct ResponseStatement<F> { /* … */ }
pub struct ResponseWitness<F> { /* z, V, θ, ξ_a, ξ_p, ξ_D digits */ }

/// Proves the response statement. Akita implements it (W5).
pub trait ResponseProver<F> {
    fn prove_response<H: Sponge>(
        &self, statement: &ResponseStatement<F>, witness: &ResponseWitness<F>,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError>;
    fn verify_response<H: Sponge>(
        &self, statement: &ResponseStatement<F>, transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError>;
}
```

## Open questions

- **Extension-field products.** Whether Akita's fold-k final checks contain
  products of extension-field values. A single-operand product row cannot
  express those directly; they would need `E::DEGREE` auxiliary slots or a
  sum-of-products row. W3 answers this first.
- **Cut point of `response_l2_sq`.** Whether the 16-byte L2 message falls before
  or after fold k. Before the cut, it needs a field encoding so it can be
  masked.
- **Stage-3 setup-offloading claims.** M3b routes the stage-3 setup-product
  claim and prefix evaluation through `exchange_private`. If they are derivable
  from public setup data they belong on the public path, and masking them
  wastes pad slots (I7). W3 settles this together with the cut point.
- **Remaining unrouted fold messages.** The `NextWitnessInnerState` terminal
  payload is still sent as is, and the intermediate-fold scalar openings
  (levels ≥ 1) look derivable but are sent with `public_all`. Routing the latter
  through `bind_derived` needs a slice form that keeps the transcript event log
  unchanged. Both are decided in W3.
