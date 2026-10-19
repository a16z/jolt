# Spec appendix: Whiteout interfaces

| Field     | Value                                    |
| --------- | ---------------------------------------- |
| Author(s) | @markosg04                               |
| Created   | 2026-10-19                               |
| Status    | draft                                    |
| Parent    | `specs/shared-primitive-layer.md`        |

This appendix fixes the interfaces that later tickets build on, before any of
them is implemented. Each section names the ticket that implements it. Bodies,
helper types, and module layout are the implementer's choice; the signatures and
the stated contracts are not. A ticket that needs to change a signature here
changes this file in the same PR.

## 1. Sumcheck rounds (M2a)

### Round formats

One type owns the rule for how a round polynomial travels and how the verifier
checks it. The verifier, the clear recorder, and the masking recorder (§5) all
consume it.

```rust
/// How one sumcheck instance's rounds travel.
pub enum SumcheckFormat<E> {
    /// Boolean hypercube. Check: s(0) + s(1) = claim. Sends [c0, c2, ..., cd].
    Compressed,
    /// s(X) = eq(τ_i, X) · q(X), where τ_i = equality_point[i] in binding order.
    /// Check: (1 − τ_i) q(0) + τ_i q(1) = claim. The kernel returns q.
    /// Sends [q1, ..., qd].
    EqFactored { equality_point: Vec<E> },
    /// Uni-skip round over a centered integer domain D. Check: Σ_{x∈D} s(x) = claim.
    /// Sends every coefficient.
    Full { domain: CenteredIntegerDomain },
}

/// The format of one round, derived from the instance format and the round index.
pub enum RoundFormat<E> {
    Compressed,
    EqFactored { tau: E },
    Full { domain: CenteredIntegerDomain },
}

impl<E: Field> RoundFormat<E> {
    /// The round check as weights on the full coefficient vector: Σ_j w_j c_j = claim.
    pub fn check_weights(&self, degree: usize) -> Result<Vec<E>, SumcheckError<E>>;
    /// The coefficient the wire omits, recovered from the check. `None` for `Full`.
    pub fn omitted(&self) -> Option<usize>;
    /// The coefficients sent on the wire, padded to `degree`.
    pub fn wire(&self, poly: &UnivariatePoly<E>, degree: usize) -> Result<Vec<E>, SumcheckError<E>>;
    /// Rebuilds the full polynomial from the wire coefficients and the claim.
    pub fn recover(&self, wire: &[E], claim: E) -> Result<UnivariatePoly<E>, SumcheckError<E>>;
    /// The next running claim: s(r), or eq(τ, r) · q(r) for `EqFactored`.
    pub fn next_claim(&self, poly: &UnivariatePoly<E>, r: E) -> E;
}
```

`check_weights` is the affine form Λ_i of the Whiteout paper, §4.1: masking
sends every coefficient except `omitted()` as m + p, and the residual relation
gets the row Σ_j w_j (ĉ_j − p_j) = v_{i−1}.

### Challenges and sites

Akita draws round challenges from its grinding schedule and tags every message
with a site; Jolt draws `challenge_small` and tags nothing. Both go behind one
trait, which the caller supplies per instance.

```rust
#[repr(u32)]
pub enum SumcheckRole { Claim = 1, RoundBody = 3, Challenge = 4 } // Akita's discriminants

pub struct SumcheckSite { pub invocation: u32, pub round: u32, pub role: SumcheckRole }

pub trait RoundContext<E> {
    type Error;
    /// The transcript site for `site`, or `None` to emit no site.
    fn site(&self, site: SumcheckSite) -> Option<SiteId>;
    fn prover_challenge<H: Sponge>(
        &mut self, site: SumcheckSite, transcript: &mut ProverTranscript<H>,
    ) -> Result<E, Self::Error>;
    fn verifier_challenge<H: Sponge>(
        &mut self, site: SumcheckSite, transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<E, Self::Error>;
}

/// Jolt's default: no sites, `challenge_small`.
pub struct SmallChallenges;
```

Akita implements `RoundContext` with its grinding plan, which stays in Akita
(`transcript_grinding`, `fold_draw.rs`). Its implementation must reproduce
`grinded_ext_challenge`'s bytes exactly (I3).

### Prover trait and errors

`ProveRounds` keeps its fused shape and gains a kernel error type:

```rust
pub trait ProveRounds<E: Field, K = Infallible> {
    fn num_rounds(&self) -> usize;
    /// For `EqFactored` instances, returns q, not s.
    fn prove_round(&mut self, bind: Option<E>, round: usize, previous_claim: E)
        -> Result<UnivariatePoly<E>, SumcheckError<E, K>>;
    fn finish_rounds(&mut self, bind: E) -> Result<(), SumcheckError<E, K>>;
}

pub enum SumcheckError<E, K = Infallible> {
    // existing variants unchanged
    Kernel(K),
}
```

`SumcheckError` carries invariant violations and kernel failures only. Retryable
aborts (W2's fold overflow, W4's Σ-protocol rejection) are not sumcheck
failures. They belong to the prove-level errors of `jolt-zk` and Akita (§5).

### Drivers

```rust
/// One unbatched instance. Matches Akita's `prove_sumcheck` and
/// `prove_eq_factored_sumcheck` byte for byte: claim made public at the Claim
/// site; per round, the wire coefficients at the RoundBody site, then the
/// challenge from `context`.
pub fn prove_single<F, E, K, C, R, H>(
    member: &mut dyn ProveRounds<E, K>,
    instance: &SingleInstance<E>,   // input claim, rounds, degree, format, invocation
    context: &mut C,
    recorder: &mut R,
    transcript: &mut ProverTranscript<H>,
) -> Result<ProvedSingle<E>, SumcheckError<E, K>>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    C: RoundContext<E, Error = K>,   // or map; implementer's choice
    R: SumcheckRecorder<E>,
    H: Sponge;

pub fn verify_single<F, E, C, H>(/* mirror */);
```

`prove_batch` takes a `RoundContext` too; Jolt passes `SmallChallenges` and
`SumcheckFormat::Compressed`, so Jolt proofs are unchanged. The uni-skip
provers become `Full`-format calls.

Jolt's Dory path uses `Fr`, which has no `ExtField<Fr>` impl today (only
`PseudoMersenne` fields have the reflexive impl). M2a either generalizes the
reflexive impl or gives `prove_batch` a single-field path.

## 2. Recorders (M2a, M3a)

The recorder stops drawing challenges; `RoundContext` does that.

```rust
pub trait SumcheckRecorder<E> {
    type Witness;
    fn absorb_input_claims<C: Channel>(&mut self, input_claims: &[E], channel: &mut C);
    /// Sends one round in `format`. Draws nothing.
    fn record_round<H: Sponge>(
        &mut self, poly: &UnivariatePoly<E>, format: &RoundFormat<E>, degree: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), SumcheckError<E>>;
    fn finish<H: Sponge>(self, output_claims: &[E], transcript: &mut ProverTranscript<H>)
        -> Result<Self::Witness, SumcheckError<E>>;
}
```

The committed (per-round Pedersen) recorder moves to `jolt-zk` with BlindFold
(M6a) and is deleted by M6c.

## 3. Private-message seam (M3a)

Witness-dependent field messages that are not sumcheck rounds go through one
seam: Jolt's claimed evaluations and Akita's partial-evaluation messages.

```rust
/// Sends witness-dependent field messages. Clear: unchanged. Masked: m + p,
/// with the next unused pads in schedule order.
pub trait PrivateMessages<F, H: Sponge> {
    fn send_private<E: ExtField<F> + CanonicalEncoding>(
        &mut self, values: &[E], transcript: &mut ProverTranscript<H>,
    ) -> Result<(), TranscriptError>;
}

pub struct ClearMessages; // sends values unchanged
```

Pads cover base-field coordinates, so an extension element consumes
`E::DEGREE` pads. Public messages (commitments, nonces, setup-offloading claims)
keep using the channel directly. The verifier has no counterpart: it receives
masked and clear values alike, and in ZK mode it never checks them directly.

## 4. Commitments (M4)

### PCS traits

`ZkOpeningScheme` splits into two capabilities. Its associated type
`HidingCommitment`, which commits to an evaluation, is renamed `EvalCommitment`
so the trait name is free.

```rust
pub trait HidingCommitment: CommitmentScheme {
    type Blind: Clone + Send + Sync;
    fn commit_hiding<P: MultilinearPoly<Self::Field> + ?Sized, R: RngCore + CryptoRng>(
        poly: &P, setup: &Self::ProverSetup, rng: &mut R,
    ) -> Result<(Self::Output, Self::OpeningHint, Self::Blind), OpeningsError>;
}

pub trait ZkOpening: HidingCommitment {
    type EvalCommitment: /* today's HidingCommitment bounds */;
    fn open_zk</* as today */>(/* … */) -> Result<(Self::EvalCommitment, Self::Blind), OpeningsError>;
    fn verify_zk</* as today */>(/* … */) -> Result<Self::EvalCommitment, OpeningsError>;
}
```

`ZkStreamingCommitment` becomes `HidingStreamingCommitment:
StreamingCommitment + HidingCommitment`. `ZkBatchOpeningScheme` keeps its shape
and requires `ZkOpening`. Dory implements both traits. `jolt-akita` deletes its
current `ZkOpeningScheme` impl, whose `open_zk` runs a transparent opening and
whose `verify_zk` always errors. It implements `HidingCommitment` in W6, once
Akita commits with randomness (W1).

### Vector commitments

```rust
pub trait VectorCommitment: Commitment<…> {
    type Field: JoltField;
    type Setup: Clone + Send + Sync;
    /// Scalar for Pedersen; a vector of randomness digits for Ajtai.
    type Blind: Clone + Send + Sync;
    fn capacity(setup: &Self::Setup) -> usize;
    fn sample_blind<R: RngCore + CryptoRng>(setup: &Self::Setup, rng: &mut R) -> Self::Blind;
    fn commit(setup: &Self::Setup, values: &[Self::Field], blind: &Self::Blind) -> Self::Output;
    fn verify(setup: &Self::Setup, c: &Self::Output, values: &[Self::Field], blind: &Self::Blind) -> bool;
}

/// Row openings by homomorphic combination. Pedersen only: an Ajtai blind's
/// norm grows under combination, and the lattice pad commitment is opened by
/// the Σ-protocol instead.
pub trait RowOpening: VectorCommitment<Blind = <Self as VectorCommitment>::Field> {
    fn open_committed_rows(/* as today */);
    fn verify_committed_rows(/* as today */);
}
```

## 5. ZK layer (M6b, W4)

### Residual relation

```rust
/// A variable of the residual relation: a pad, the mask g's coefficient slot,
/// or a coordinate of the residual witness (the PCS's cut-point witness and its
/// commitment openings).
pub enum ResidualVar { Pad(usize), Mask(usize), Witness(usize) }

/// Σ_k a_k · x_k + b = 0.
pub struct AffineRow<F> { pub terms: Vec<(ResidualVar, F)>, pub constant: F }

/// x · y = z. `x` is a single masked value (Whiteout §3.2: one constant operand
/// per product). `y` and `z` are arbitrary affine forms.
pub struct ProductRow<F> { pub x: AffineRow<F>, pub y: AffineRow<F>, pub z: AffineRow<F> }

pub struct ResidualRelation<F> {
    pub affine: Vec<AffineRow<F>>,
    pub products: Vec<ProductRow<F>>,
    pub num_pads: usize,
    pub num_witness: usize,
}

/// A verifier that contributes rows: Jolt's stages (through the existing
/// replay of the verifier) and Akita's folds up to the cut point (W3).
pub trait ResidualRows<F> {
    type Error;
    fn append_rows(&self, relation: &mut ResidualRelation<F>) -> Result<(), Self::Error>;
}
```

Rows are field-level for both back ends; the lattice back end applies the
constant-term lift itself.

### Pads

```rust
/// Pads in schedule order, committed before the first challenge.
pub struct PadSchedule<F> { /* pads, cursor */ }
impl<F: Field> PadSchedule<F> {
    pub fn sample<R: RngCore + CryptoRng>(count: usize, rng: &mut R) -> Self;
    /// The next `n` unused pads. Errors if the schedule is exhausted.
    pub fn take(&mut self, n: usize) -> Result<&[F], ZkError>;
}
```

The masking `PrivateMessages` and `SumcheckRecorder` implementations draw from
one `PadSchedule`. The schedule length is fixed by public parameters, so the
prover commits to it before the first challenge.

### Back ends

```rust
pub trait ZkBackend<F> {
    type Setup;
    /// Pads, their blind, and the residual witness with its openings.
    type Witness;
    fn prove<H: Sponge, R: RngCore + CryptoRng>(
        setup: &Self::Setup, relation: &ResidualRelation<F>, witness: &Self::Witness,
        transcript: &mut ProverTranscript<H>, rng: &mut R,
    ) -> Result<(), ZkError>;
    fn verify<H: Sponge>(
        setup: &Self::Setup, relation: &ResidualRelation<F>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError>;
}

pub enum ZkError {
    /// A randomized step failed; retry the whole proof with fresh randomness.
    Aborted(AbortReason),
    // invariant violations, transcript errors, …
}
```

`blindfold` implements `ZkBackend` in M6c, and `whiteout` implements it in W4.

### Response proof (W4, W5)

```rust
/// The statement of Whiteout §4.3: digit ranges, linear rows N z = f + c t,
/// the ring-quadratic identity, and ct(h) = 0. Defined in `jolt-zk::whiteout`.
pub struct ResponseStatement<F> { /* … */ }

/// Proves the response statement. Akita implements it (W5).
pub trait ResponseProver<F> {
    type Proof;
    fn prove_response<H: Sponge>(
        &self, statement: &ResponseStatement<F>, response: &ResponseWitness<F>,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError>;
    fn verify_response<H: Sponge>(
        &self, statement: &ResponseStatement<F>, transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError>;
}
```

## Open questions

- **Equality-point order.** `EqFactored`'s τ_i follows Akita's binding order
  (`GruenSplitEq::current_tau`). M2a must pin that order with a test against
  Akita's verifier.
- **Kernel error mapping.** Whether `RoundContext::Error` must equal the kernel
  error `K`, or the driver maps both into `SumcheckError<E, K>`.
- **Product-row operand.** Whether `ProductRow.x` should be a dedicated
  single-variable type instead of an `AffineRow` with one term.
