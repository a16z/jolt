use crate::dag::stage::{StagedSumcheck, SumcheckStages};
use crate::dag::state_manager::StateManager;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::OpeningProofReductionSumcheck;
use crate::utils::transcript::Transcript;
use std::marker::PhantomData;

/// DAG for stage 5 - opening proof reduction sumcheck
pub struct OpeningProofDAG<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    _marker: PhantomData<(F, PCS)>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> OpeningProofDAG<F, PCS> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStages<F, ProofTranscript, PCS> for OpeningProofDAG<F, PCS>
{
    fn stage5_prover_instances(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        // Get the prover accumulator which contains all the openings
        let accumulator = state_manager.get_prover_accumulator();
        let mut accumulator_borrow = accumulator.borrow_mut();

        // Get mutable references to all the OpeningProofReductionSumcheck instances
        let openings: Vec<Box<dyn StagedSumcheck<F, PCS>>> = accumulator_borrow
            .openings
            .drain(..)
            .map(|opening| Box::new(opening) as Box<dyn StagedSumcheck<F, PCS>>)
            .collect();

        openings
    }

    fn stage5_verifier_instances(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        // Get the verifier accumulator which contains all the openings
        let accumulator = state_manager.get_verifier_accumulator();
        let mut accumulator_borrow = accumulator.borrow_mut();

        // First, collect all the sumcheck claims for each instance
        let num_openings = accumulator_borrow.openings.len();
        let sumcheck_claims: Vec<F> = (0..num_openings)
            .map(|index| {
                accumulator_borrow.get_opening(
                    crate::poly::opening_proof::OpeningsKeys::OpeningsSumcheckClaim(index),
                )
            })
            .collect();

        // Now drain and update the openings with their corresponding claims
        let openings: Vec<Box<dyn StagedSumcheck<F, PCS>>> = accumulator_borrow
            .openings
            .drain(..)
            .zip(sumcheck_claims.into_iter())
            .map(|(mut opening, sumcheck_claim)| {
                // Set the sumcheck claim for this opening
                opening.sumcheck_claim = Some(sumcheck_claim);
                Box::new(opening) as Box<dyn StagedSumcheck<F, PCS>>
            })
            .collect();

        openings
    }
}

// Implement StagedSumcheck for OpeningProofReductionSumcheck
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for OpeningProofReductionSumcheck<F, PCS>
{
}
