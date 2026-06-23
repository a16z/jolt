//! Shared sumcheck proof recorder.
//!
//! A batched sumcheck's round loop is identical in clear and ZK mode; only the
//! per-round recording differs (cleartext round polynomials vs. Pedersen
//! commitments) and whether input/output claims are appended to the transcript
//! in the clear. [`SumcheckRecorder`] captures exactly that difference so a stage
//! can write its round loop once, generic over the recorder.
//!
//! Stages flatten their structured input/output claims to `&[F]` at the call
//! site (the per-stage `stageN_output_claim_values` helpers), keeping the
//! recorder itself stage-agnostic.

use std::marker::PhantomData;

use jolt_field::Field;
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, CompressedLabeledRoundPoly, CompressedSumcheckProof,
    RoundMessage, SumcheckProof,
};
use jolt_transcript::Transcript;

use crate::ProverError;

#[cfg(feature = "zk")]
use crate::committed::{CommittedSumcheckBuilder, CommittedSumcheckWitness};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;

/// Records the rounds of a single batched sumcheck, abstracting over clear vs.
/// committed (ZK) recording. A stage drives this from its round loop:
/// `absorb_input_claims` once, `absorb_round` per round (returning the
/// Fiat-Shamir challenge), then `finish` with the flattened output claims.
pub(crate) trait SumcheckRecorder<F: Field> {
    type Commitment;

    fn absorb_input_claims<T>(&mut self, input_claims: &[F], transcript: &mut T)
    where
        T: Transcript<Challenge = F>;

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, ProverError>
    where
        T: Transcript<Challenge = F>;

    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<RecordedSumcheck<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>;
}

/// Output of a recorded sumcheck. `committed_witness`/`output_claim_values` carry
/// the BlindFold material in ZK mode and are `None` in clear mode.
pub(crate) struct RecordedSumcheck<F: Field, C> {
    pub(crate) proof: SumcheckProof<F, C>,
    #[cfg(feature = "zk")]
    pub(crate) committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    pub(crate) output_claim_values: Option<Vec<F>>,
}

pub(crate) struct ClearSumcheckRecorder<F: Field, C> {
    round_polynomials: Vec<CompressedPoly<F>>,
    _marker: PhantomData<C>,
}

impl<F: Field, C> ClearSumcheckRecorder<F, C> {
    pub(crate) fn new(round_capacity: usize) -> Self {
        Self {
            round_polynomials: Vec::with_capacity(round_capacity),
            _marker: PhantomData,
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
    ) -> Result<F, ProverError>
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
    ) -> Result<RecordedSumcheck<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        for opening_claim in output_claim_values {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
        Ok(RecordedSumcheck {
            proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
                round_polynomials: self.round_polynomials,
            })),
            #[cfg(feature = "zk")]
            committed_witness: None,
            #[cfg(feature = "zk")]
            output_claim_values: None,
        })
    }
}

#[cfg(feature = "zk")]
pub(crate) struct CommittedSumcheckRecorder<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    builder: CommittedSumcheckBuilder<'a, F, VC>,
}

#[cfg(feature = "zk")]
impl<'a, F, VC> CommittedSumcheckRecorder<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub(crate) fn new(setup: &'a VC::Setup) -> Result<Self, ProverError> {
        Ok(Self {
            builder: CommittedSumcheckBuilder::new(setup, 0)?,
        })
    }
}

#[cfg(feature = "zk")]
impl<F, VC> SumcheckRecorder<F> for CommittedSumcheckRecorder<'_, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    type Commitment = VC::Output;

    fn absorb_input_claims<T>(&mut self, _input_claims: &[F], _transcript: &mut T)
    where
        T: Transcript<Challenge = F>,
    {
    }

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        self.builder.commit_round(round_poly, transcript)
    }

    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<RecordedSumcheck<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        let built = self.builder.finish(output_claim_values, transcript)?;
        Ok(RecordedSumcheck {
            proof: built.proof,
            committed_witness: Some(built.witness),
            output_claim_values: Some(output_claim_values.to_vec()),
        })
    }
}
