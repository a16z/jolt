//! Polynomial commitment scheme (PCS) trait hierarchy.
//!
//! Verifier-side traits live above the prover-side extensions so verifier-only
//! crates (current `jolt-verifier`, future wasm / on-chain verifiers) can bound
//! on `CommitmentSchemeVerifier` without pulling in `Polynomial`,
//! `OpeningHint`, `ProverSetup`, or `SetupParams`.
//!
//! ```text
//!                CommitmentSchemeVerifier   (verifier base)
//!                  ─ verifier_setup
//!                  ─ verify_batch
//!                  ─ bind_opening_inputs
//!                          │
//!         ┌────────────────┼─────────────────┐
//!         │                │                 │
//!         │  AdditivelyHomomorphicVerifier   ZkOpeningSchemeVerifier
//!         │  ─ combine                       ─ verify_zk
//!         │  ─ verify                        ─ HidingCommitment
//!         │                │                 │
//!     CommitmentScheme     │                 │
//!    (prover extension)    │                 │
//!     ─ setup              │                 │
//!     ─ project_verifier_  │                 │
//!         setup            │                 │
//!     ─ commit             │                 │
//!     ─ prove_batch        │                 │
//!         │                │                 │
//!         ├────────────────┤                 │
//!         │                │                 │
//!         │  AdditivelyHomomorphic           │
//!         │  ─ combine_hints                 │
//!         │  ─ open                          │
//!         │                │                 │
//!         │   ┌────────────┘                 │
//!         │   │                              │
//!         │   │           ZkOpeningScheme  ──┘
//!         │   │           ─ open_zk
//!         │   │           ─ Blind
//!         │   │
//!     StreamingCommitment
//!     (Dory only, prover-side, unchanged shape)
//!     ─ begin / feed / finish
//! ```
//!
//! The fused batched-opening protocol is exposed through [`prove_batch`] on
//! [`CommitmentScheme`] and [`verify_batch`] on [`CommitmentSchemeVerifier`].
//! Single-claim `open` / `verify` are per-group RLC primitives consumed by
//! the homomorphic helpers ([`crate::homomorphic_prove_batch`] /
//! [`crate::homomorphic_verify_batch`]); they live on
//! [`AdditivelyHomomorphic`] / [`AdditivelyHomomorphicVerifier`] so that
//! Hachi-style fused-batch schemes never have to implement them.
//!
//! [`prove_batch`]: CommitmentScheme::prove_batch
//! [`verify_batch`]: CommitmentSchemeVerifier::verify_batch

use std::fmt::Debug;

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::Field;
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{de::DeserializeOwned, Serialize};

use crate::claims::{OpeningClaim, ProverClaim};
use crate::error::OpeningsError;

/// Verifier-side surface of a polynomial commitment scheme.
///
/// Hosts only fused batched verification ([`verify_batch`]) and post-batch
/// transcript binding ([`bind_opening_inputs`]). Single-claim `verify`
/// lives on [`AdditivelyHomomorphicVerifier`].
///
/// [`verify_batch`]: Self::verify_batch
/// [`bind_opening_inputs`]: Self::bind_opening_inputs
pub trait CommitmentSchemeVerifier: Commitment + Clone + Send + Sync + 'static {
    type Field: Field;
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;
    type Proof: Clone + Send + Sync + Serialize + DeserializeOwned;
    type BatchProof: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Public inputs the verifier needs to derive its setup from scratch
    /// (URS file, max num_vars, etc.). Distinct from the prover's
    /// `SetupParams`; does NOT require toxic waste / full URS.
    type VerifierSetupParams;

    /// Build a `VerifierSetup` from public inputs alone. Used by
    /// verifier-only consumers (wasm, on-chain) that never hold a
    /// `ProverSetup`. Single-machine prove+verify roundtrips can use
    /// [`CommitmentScheme::project_verifier_setup`] instead.
    fn verifier_setup(params: Self::VerifierSetupParams) -> Self::VerifierSetup;

    /// Verify a batched opening proof produced by
    /// [`CommitmentScheme::prove_batch`]. Drives the Fiat-Shamir transcript
    /// through a sequence byte-identical to the prover's call.
    fn verify_batch(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        batch_proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;

    /// Post-batch transcript bind: absorb the (point, eval) of a verified
    /// opening so downstream Fiat-Shamir challenges depend on it.
    /// Default: no-op. Dory overrides; Mock / HyperKZG keep the default.
    fn bind_opening_inputs(
        _t: &mut impl Transcript<Challenge = Self::Field>,
        _point: &[Self::Field],
        _eval: &Self::Field,
    ) {
    }
}

/// Prover-side surface of a polynomial commitment scheme.
///
/// Strict extension of [`CommitmentSchemeVerifier`]: a prover always has the
/// verifier's surface available too (e.g., for self-test). Single-claim
/// `open` lives on [`AdditivelyHomomorphic`]; this trait carries only
/// setup, commit, and the fused [`prove_batch`].
///
/// [`prove_batch`]: Self::prove_batch
pub trait CommitmentScheme: CommitmentSchemeVerifier {
    type ProverSetup: Clone + Send + Sync;
    type Polynomial: MultilinearPoly<Self::Field> + From<Vec<Self::Field>>;

    /// Auxiliary data from `commit` reused during opening (e.g. Dory row
    /// commitments). Schemes without such data use `()`.
    type OpeningHint: Clone + Send + Sync + Default;

    type SetupParams;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup);

    /// Project the prover setup down to the verifier setup. Used by
    /// single-machine prover/verifier roundtrips that already hold a
    /// `ProverSetup` and want to skip
    /// [`CommitmentSchemeVerifier::verifier_setup`].
    fn project_verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup;

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    /// Prove a batched opening for many (polynomial, point, eval) claims.
    ///
    /// Returns the [`BatchProof`] and a vector of per-group joint
    /// evaluations (one per distinct opening point), in the order the
    /// helper produced them. The joint evals are consumed by
    /// downstream `Op::BindOpeningInputs` to seed Fiat-Shamir.
    ///
    /// [`BatchProof`]: CommitmentSchemeVerifier::BatchProof
    fn prove_batch<T: Transcript<Challenge = Self::Field>>(
        claims: Vec<ProverClaim<Self::Field>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut T,
    ) -> (Self::BatchProof, Vec<Self::Field>);
}

/// Verifier-side homomorphic extension: linear combination of commitments
/// and per-group single-claim verification.
///
/// `verify` here is the per-group RLC primitive used by
/// [`crate::homomorphic_verify_batch`]; tests/benches MAY also call it
/// directly to exercise the single-claim path.
pub trait AdditivelyHomomorphicVerifier: CommitmentSchemeVerifier
where
    Self::Output: HomomorphicCommitment<Self::Field>,
{
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output;

    /// Verify one (commitment, point, eval, proof) tuple. Single-claim
    /// primitive for homomorphic schemes.
    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;
}

/// Prover-side homomorphic extension: combine opening hints and
/// per-group single-claim opening.
///
/// `open` here is the per-group RLC primitive used by
/// [`crate::homomorphic_prove_batch`]; tests/benches MAY also call it
/// directly to exercise the single-claim path.
pub trait AdditivelyHomomorphic: AdditivelyHomomorphicVerifier + CommitmentScheme
where
    Self::Output: HomomorphicCommitment<Self::Field>,
{
    fn combine_hints(
        _hints: Vec<Self::OpeningHint>,
        _scalars: &[Self::Field],
    ) -> Self::OpeningHint {
        Self::OpeningHint::default()
    }

    /// Open one polynomial at one point. Single-claim primitive for
    /// homomorphic schemes.
    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof;
}

/// Incremental commitment without full materialization. Prover-side only.
pub trait StreamingCommitment: CommitmentScheme {
    type PartialCommitment: Clone + Send + Sync;

    fn begin(setup: &Self::ProverSetup) -> Self::PartialCommitment;

    fn feed(
        partial: &mut Self::PartialCommitment,
        chunk: &[Self::Field],
        setup: &Self::ProverSetup,
    );

    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output;
}

/// Verifier-side ZK opening surface: hiding-commitment type and the
/// verification entry point.
pub trait ZkOpeningSchemeVerifier: CommitmentSchemeVerifier {
    type HidingCommitment: Clone
        + Debug
        + Eq
        + Send
        + Sync
        + 'static
        + Serialize
        + DeserializeOwned
        + AppendToTranscript;

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval_commitment: &Self::HidingCommitment,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;
}

/// Prover-side ZK opening surface: blinding type and the proving entry
/// point. Hides the evaluation behind a [`HidingCommitment`].
///
/// [`HidingCommitment`]: ZkOpeningSchemeVerifier::HidingCommitment
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
