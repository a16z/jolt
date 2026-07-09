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
//! the prove-side round loop calls [`absorb_round`](SumcheckRecorder::absorb_round)
//! per round and [`finish`](SumcheckRecorder::finish) once. The clear verifier
//! also runs `begin_batch` (with [`ClearSumcheckRecorder`]) so the two sides
//! share the head's Fiat-Shamir sequence structurally.

use std::marker::PhantomData;

use jolt_field::Field;
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_transcript::Transcript;

use crate::error::SumcheckError;
use crate::proof::{ClearProof, CompressedSumcheckProof, SumcheckProof};
use crate::round_proof::{CompressedLabeledRoundPoly, RoundMessage};
use crate::{append_sumcheck_claim, OPENING_CLAIM_TRANSCRIPT_LABEL};

/// Records one sumcheck's proof material, abstracting over clear vs. committed
/// (ZK) recording: `absorb_input_claims` once (from `begin_batch`),
/// `absorb_round` per round (returning the Fiat-Shamir challenge), then
/// `finish` with the flattened output-claim values.
pub trait SumcheckRecorder<F: Field> {
    /// The proof's commitment type parameter (`SumcheckProof<F, C>`). Phantom
    /// for a clear recorder; the vector-commitment output for a committed one.
    type Commitment;

    /// Absorb the batch's per-member input claims (present members, in
    /// declaration order). Clear: each appended under `b"sumcheck_claim"`.
    /// Committed: no-op — the claims' commitments were already absorbed by the
    /// stage that produced them, so the transcript never sees the scalars.
    fn absorb_input_claims<T>(&mut self, input_claims: &[F], transcript: &mut T)
    where
        T: Transcript<Challenge = F>;

    /// Record one round polynomial and squeeze the round challenge. Clear:
    /// appended compressed under `b"sumcheck_poly"`. Committed: Pedersen
    /// commitment appended.
    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, SumcheckError<F>>
    where
        T: Transcript<Challenge = F>;

    /// Record the flattened output-claim values (canonical absorb order) and
    /// assemble the proof. Clear: each value appended under
    /// `b"opening_claim"`. Committed: values are row-committed and only the
    /// commitments are absorbed.
    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<RecordedSumcheck<F, Self::Commitment>, SumcheckError<F>>
    where
        T: Transcript<Challenge = F>;
}

/// A recorded sumcheck: the wire proof. Committed recorders additionally
/// retain their blinding/witness material internally for BlindFold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedSumcheck<F: Field, C> {
    pub proof: SumcheckProof<F, C>,
}

/// The clear recorder: appends claims and compressed round polynomials to the
/// transcript in the clear and collects the rounds into a
/// [`CompressedSumcheckProof`]. Its transcript writes are byte-identical to
/// what the clear verifier reads back.
pub struct ClearSumcheckRecorder<F: Field, C> {
    round_polynomials: Vec<CompressedPoly<F>>,
    _commitment: PhantomData<C>,
}

impl<F: Field, C> Default for ClearSumcheckRecorder<F, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field, C> ClearSumcheckRecorder<F, C> {
    pub fn new() -> Self {
        Self {
            round_polynomials: Vec::new(),
            _commitment: PhantomData,
        }
    }
}

impl<F: Field, C> SumcheckRecorder<F> for ClearSumcheckRecorder<F, C> {
    type Commitment = C;

    fn absorb_input_claims<T>(&mut self, input_claims: &[F], transcript: &mut T)
    where
        T: Transcript<Challenge = F>,
    {
        for input_claim in input_claims {
            append_sumcheck_claim(transcript, input_claim);
        }
    }

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, SumcheckError<F>>
    where
        T: Transcript<Challenge = F>,
    {
        CompressedLabeledRoundPoly::sumcheck(round_poly).append_to_transcript(transcript);
        let challenge = transcript.challenge();
        self.round_polynomials.push(round_poly.compress());
        Ok(challenge)
    }

    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<RecordedSumcheck<F, Self::Commitment>, SumcheckError<F>>
    where
        T: Transcript<Challenge = F>,
    {
        for opening_claim in output_claim_values {
            transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, opening_claim);
        }
        Ok(RecordedSumcheck {
            proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
                round_polynomials: self.round_polynomials,
            })),
        })
    }
}
