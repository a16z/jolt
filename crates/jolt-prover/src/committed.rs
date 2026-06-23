use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    CommittedOutputClaims, CommittedRound, CommittedRoundWitness, CommittedSumcheckProof,
    RoundMessage, SumcheckProof,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use rand_core::OsRng;

use crate::ProverError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CommittedSumcheckWitness<F: Field> {
    pub(crate) round_coefficients: Vec<Vec<F>>,
    pub(crate) round_blindings: Vec<F>,
    pub(crate) output_claim_rows: Vec<Vec<F>>,
    pub(crate) output_claim_blindings: Vec<F>,
}

impl<F: Field> CommittedSumcheckWitness<F> {
    fn new(round_capacity: usize) -> Self {
        Self {
            round_coefficients: Vec::with_capacity(round_capacity),
            round_blindings: Vec::with_capacity(round_capacity),
            output_claim_rows: Vec::new(),
            output_claim_blindings: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BuiltCommittedSumcheck<F: Field, C> {
    pub(crate) proof: SumcheckProof<F, C>,
    pub(crate) witness: CommittedSumcheckWitness<F>,
}

pub(crate) struct CommittedSumcheckBuilder<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    setup: &'a VC::Setup,
    rng: OsRng,
    rounds: Vec<CommittedRound<VC::Output>>,
    witness: CommittedSumcheckWitness<F>,
}

impl<'a, F, VC> CommittedSumcheckBuilder<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub(crate) fn new(setup: &'a VC::Setup, round_capacity: usize) -> Result<Self, ProverError> {
        let capacity = VC::capacity(setup);
        if capacity == 0 {
            return Err(invalid_committed_sumcheck_output(
                "vector-commitment setup has zero capacity",
            ));
        }
        Ok(Self {
            setup,
            rng: OsRng,
            rounds: Vec::with_capacity(round_capacity),
            witness: CommittedSumcheckWitness::new(round_capacity),
        })
    }

    pub(crate) fn commit_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        let coefficients = round_poly.coefficients().to_vec();
        if coefficients.len() > VC::capacity(self.setup) {
            return Err(invalid_committed_sumcheck_output(format!(
                "round polynomial has {} coefficients, but vector-commitment capacity is {}",
                coefficients.len(),
                VC::capacity(self.setup)
            )));
        }

        let blinding = F::random(&mut self.rng);
        let witness = CommittedRoundWitness {
            coefficients: coefficients.clone(),
            blinding,
        };
        let round = witness
            .commit::<VC>(self.setup)
            .map_err(|error| invalid_committed_sumcheck_output(error.to_string()))?;
        round.append_to_transcript(transcript);
        let challenge = transcript.challenge();

        self.rounds.push(round);
        self.witness.round_coefficients.push(coefficients);
        self.witness.round_blindings.push(blinding);
        Ok(challenge)
    }

    pub(crate) fn finish<T>(
        mut self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<BuiltCommittedSumcheck<F, VC::Output>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        let capacity = VC::capacity(self.setup);
        let mut commitments = Vec::with_capacity(output_claim_values.len().div_ceil(capacity));
        for row in output_claim_values.chunks(capacity) {
            let blinding = F::random(&mut self.rng);
            let commitment = VC::commit(self.setup, row, &blinding);
            commitments.push(commitment);
            self.witness.output_claim_rows.push(row.to_vec());
            self.witness.output_claim_blindings.push(blinding);
        }

        let output_claims = CommittedOutputClaims { commitments };
        output_claims.append_to_transcript(transcript);
        Ok(BuiltCommittedSumcheck {
            proof: SumcheckProof::Committed(CommittedSumcheckProof {
                rounds: self.rounds,
                output_claims,
            }),
            witness: self.witness,
        })
    }
}

fn invalid_committed_sumcheck_output(reason: impl Into<String>) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: reason.into(),
    }
}
