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
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let accumulator = state_manager.get_prover_accumulator();
        let mut accumulator_borrow = accumulator.borrow_mut();

        let openings: Vec<Box<dyn StagedSumcheck<F, PCS>>> = accumulator_borrow
            .openings
            .drain(..)
            .enumerate()
            .map(|(index, mut opening)| {
                opening.instance_index = Some(index);
                Box::new(opening) as Box<dyn StagedSumcheck<F, PCS>>
            })
            .collect();

        openings
    }

    fn stage5_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let accumulator = state_manager.get_verifier_accumulator();
        let accumulator_borrow = accumulator.borrow();
        
        // Collect the claims for each index
        let num_openings = accumulator_borrow.openings.len();
        let claims: Vec<F> = (0..num_openings)
            .map(|index| {
                accumulator_borrow.get_opening(
                    crate::poly::opening_proof::OpeningsKeys::OpeningsSumcheckClaim(index)
                )
            })
            .collect();
        
        drop(accumulator_borrow);
        
        // Now drain and set both instance_index and sumcheck_claim
        let mut accumulator_borrow = accumulator.borrow_mut();
        let openings: Vec<Box<dyn StagedSumcheck<F, PCS>>> = accumulator_borrow
            .openings
            .drain(..)
            .enumerate()
            .map(|(index, mut opening)| {
                // Set the instance index
                opening.instance_index = Some(index);
                
                // Set the sumcheck claim from our pre-collected claims
                opening.sumcheck_claim = Some(claims[index]);
                
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
