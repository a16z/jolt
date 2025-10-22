use allocative::Allocative;

use super::{D, LOG_K_CHUNK};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, SumcheckId},
    },
    subprotocols::hamming_weight::{
        HammingWeightConfig, HammingWeightProverState, HammingWeightSumcheck,
    },
    transcripts::Transcript,
    zkvm::{
        dag::state_manager::StateManager,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

#[derive(Allocative)]
pub struct InstructionHammingWeightSumcheck<F: JoltField> {
    gamma: [F; D],
    prover_state: Option<HammingWeightProverState<F>>,
}

impl<F: JoltField> InstructionHammingWeightSumcheck<F> {
    #[tracing::instrument(skip_all, name = "InstructionHammingWeight::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        G: [Vec<F>; D],
    ) -> Self {
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let ra = G
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>();
        Self {
            gamma: gamma_powers,
            prover_state: Some(HammingWeightProverState { ra }),
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        Self {
            gamma: gamma_powers,
            prover_state: None,
        }
    }
}

impl<F: JoltField> HammingWeightConfig for InstructionHammingWeightSumcheck<F> {
    fn d(&self) -> usize {
        D
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK
    }

    fn polynomial_type(i: usize) -> CommittedPolynomial {
        CommittedPolynomial::InstructionRa(i)
    }

    fn sumcheck_id() -> SumcheckId {
        SumcheckId::InstructionHammingWeight
    }
}

impl<F: JoltField, T: Transcript> HammingWeightSumcheck<F, T>
    for InstructionHammingWeightSumcheck<F>
{
    fn gamma(&self) -> &[F] {
        &self.gamma
    }

    fn prover_state(&self) -> Option<&HammingWeightProverState<F>> {
        self.prover_state.as_ref()
    }

    fn prover_state_mut(&mut self) -> Option<&mut HammingWeightProverState<F>> {
        self.prover_state.as_mut()
    }

    fn get_r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge> {
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
    }
}
