//! The clear/ZK recording seam for sumcheck proving.
//!
//! A batched sumcheck's round loop is identical in clear and ZK mode; only the
//! per-round recording differs (cleartext round polynomials vs. Pedersen
//! commitments) and whether input/output claims are appended to the transcript
//! in the clear. [`SumcheckRecorder`] captures exactly that difference so the
//! engine and the generated per-stage drivers are written once, generic over
//! the recorder. Whether transcript bytes are written is decided by the
//! recorder **type** — there is no runtime mode boolean to drift.
//!
//! The generated `begin_batch` drivers (`#[derive(SumcheckBatch)]` in
//! `jolt-verifier`) call [`absorb_input_claims`](SumcheckRecorder::absorb_input_claims);
//! the prove-side round loop calls `absorb_round` per round and `finish` once.
//! The clear verifier also runs `begin_batch` (with [`ClearSumcheckRecorder`])
//! so the two sides share the head's Fiat-Shamir sequence structurally.
//! Without the `prover` feature only that head remains: `absorb_round`,
//! `finish` and the committed recorder belong to the prover.

#[cfg(all(feature = "committed", feature = "prover"))]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
#[cfg(all(feature = "committed", feature = "prover"))]
use jolt_field::JoltField;
use jolt_poly::CompressedPoly;
#[cfg(feature = "prover")]
use jolt_poly::UnivariatePoly;
use jolt_transcript::{Encoding, Transcript};
#[cfg(feature = "prover")]
use jolt_transcript::{FieldTranscript, ProverTranscript};
#[cfg(all(feature = "committed", feature = "prover"))]
use rand_core::RngCore;

#[cfg(all(feature = "committed", feature = "prover"))]
use crate::committed::CommittedSumcheckBuilder;
use crate::committed::CommittedSumcheckWitness;
#[cfg(feature = "prover")]
use crate::error::SumcheckError;
#[cfg(feature = "prover")]
use crate::pow::grind_round;
use crate::proof::CompressedSumcheckProof;
#[cfg(all(feature = "committed", feature = "prover"))]
use crate::proof::SumcheckProof;
#[cfg(feature = "prover")]
use crate::round_proof::{CompressedRoundPoly, RoundMessage};

/// Records one sumcheck's proof material, abstracting over clear vs. committed
/// (ZK) recording: `absorb_input_claims` once (from `begin_batch`),
/// `absorb_round` per round (grinding the round's proof of work and returning
/// the Fiat-Shamir challenge), then
/// `finish` with the flattened output-claim values.
pub trait SumcheckRecorder<F: Field> {
    /// The proof encoding produced by this recorder.
    type Proof;

    /// Absorb the batch's per-member input claims (present members, in
    /// declaration order). Clear: each appended in that order.
    /// Committed: no-op — the claims' commitments were already absorbed by the
    /// stage that produced them, so the transcript never sees the scalars.
    fn absorb_input_claims<T: Transcript>(&mut self, input_claims: &[F], transcript: &mut T);

    /// Record one round polynomial, grind the round's proof of work at the
    /// difficulty for `degree_bound` (the batch's degree bound, which the
    /// verifier prices the round at), and squeeze the round challenge. Clear:
    /// appended compressed, after its count word. Committed: Pedersen
    /// commitment appended.
    #[cfg(feature = "prover")]
    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree_bound: usize,
        transcript: &mut T,
    ) -> Result<F, SumcheckError<F>>
    where
        T: FieldTranscript<F> + ProverTranscript;

    /// Record the flattened output-claim values (canonical absorb order) and
    /// assemble the proof. Clear: each value appended in that order.
    /// Committed: values are row-committed and only the
    /// commitments are absorbed.
    #[cfg(feature = "prover")]
    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<RecordedSumcheck<F, Self::Proof>, SumcheckError<F>>
    where
        T: FieldTranscript<F> + ProverTranscript;
}

/// A recorded sumcheck: the wire proof, plus (for a committed recorder) the
/// retained witness — round coefficients, output-claim rows, and their
/// blindings — that BlindFold later opens. `None` for a clear recorder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedSumcheck<F: Field, P> {
    pub proof: P,
    pub committed_witness: Option<CommittedSumcheckWitness<F>>,
}

/// The clear recorder: appends claims and compressed round polynomials to the
/// transcript in the clear and collects the rounds into a
/// [`CompressedSumcheckProof`]. Its transcript writes are byte-identical to
/// what the clear verifier reads back.
#[cfg_attr(
    not(feature = "prover"),
    expect(dead_code, reason = "a verifier build absorbs only the input claims")
)]
pub struct ClearSumcheckRecorder<F: Field> {
    round_polynomials: Vec<CompressedPoly<F>>,
    pow_nonces: Vec<u32>,
}

impl<F: Field> Default for ClearSumcheckRecorder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> ClearSumcheckRecorder<F> {
    pub fn new() -> Self {
        Self {
            round_polynomials: Vec::new(),
            pow_nonces: Vec::new(),
        }
    }
}

impl<F: Field + Encoding> SumcheckRecorder<F> for ClearSumcheckRecorder<F> {
    type Proof = CompressedSumcheckProof<F>;

    fn absorb_input_claims<T: Transcript>(&mut self, input_claims: &[F], transcript: &mut T) {
        for input_claim in input_claims {
            transcript.append(input_claim);
        }
    }

    #[cfg(feature = "prover")]
    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree_bound: usize,
        transcript: &mut T,
    ) -> Result<F, SumcheckError<F>>
    where
        T: FieldTranscript<F> + ProverTranscript,
    {
        CompressedRoundPoly::new(round_poly).append_to_transcript(transcript);
        grind_round(
            degree_bound,
            self.round_polynomials.len(),
            &mut self.pow_nonces,
            transcript,
        )?;
        let challenge = transcript.challenge()?;
        self.round_polynomials.push(round_poly.compress());
        Ok(challenge)
    }

    #[cfg(feature = "prover")]
    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<RecordedSumcheck<F, Self::Proof>, SumcheckError<F>>
    where
        T: FieldTranscript<F> + ProverTranscript,
    {
        for opening_claim in output_claim_values {
            transcript.append(opening_claim);
        }
        Ok(RecordedSumcheck {
            proof: CompressedSumcheckProof {
                round_polynomials: self.round_polynomials,
                pow_nonces: self.pow_nonces,
            },
            committed_witness: None,
        })
    }
}

/// The committed (ZK) recorder: Pedersen-commits each round polynomial and the
/// output-claim rows, absorbing only the commitments — the transcript never
/// sees a claim or coefficient scalar. Input-claim absorbs are no-ops: the
/// claims' commitments were already absorbed by the stage that produced them.
/// The retained witness (coefficients, rows, blindings) is returned by
/// [`finish`](SumcheckRecorder::finish) for BlindFold.
#[cfg(all(feature = "committed", feature = "prover"))]
pub struct CommittedSumcheckRecorder<'a, F, VC, R>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
    R: RngCore,
{
    builder: CommittedSumcheckBuilder<'a, F, VC, R>,
}

#[cfg(all(feature = "committed", feature = "prover"))]
impl<'a, F, VC, R> CommittedSumcheckRecorder<'a, F, VC, R>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
    R: RngCore,
{
    pub fn new(setup: &'a VC::Setup, rng: R) -> Result<Self, SumcheckError<F>> {
        Ok(Self {
            builder: CommittedSumcheckBuilder::new(setup, rng)?,
        })
    }
}

#[cfg(all(feature = "committed", feature = "prover"))]
impl<F, VC, R> SumcheckRecorder<F> for CommittedSumcheckRecorder<'_, F, VC, R>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
    R: RngCore,
{
    type Proof = SumcheckProof<F, VC::Output>;

    fn absorb_input_claims<T: Transcript>(&mut self, _input_claims: &[F], _transcript: &mut T) {}

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree_bound: usize,
        transcript: &mut T,
    ) -> Result<F, SumcheckError<F>>
    where
        T: FieldTranscript<F> + ProverTranscript,
    {
        self.builder
            .commit_round(round_poly, degree_bound, transcript)
    }

    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<RecordedSumcheck<F, Self::Proof>, SumcheckError<F>>
    where
        T: FieldTranscript<F> + ProverTranscript,
    {
        let (proof, witness) = self.builder.finish(output_claim_values, transcript)?;
        Ok(RecordedSumcheck {
            proof,
            committed_witness: Some(witness),
        })
    }
}
