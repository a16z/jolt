use crate::subprotocols::hamming_weight::Hamming;
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
    utils::math::Math,
    zkvm::{
        dag::state_manager::StateManager,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[derive(Allocative)]
pub struct BytecodeHammingWeightSumcheck<F: JoltField> {
    gamma: Vec<F>,
    log_K_chunk: usize,
    d: usize,
    prover_state: Option<HammingWeightProverState<F>>,
}

impl<F: JoltField> BytecodeHammingWeightSumcheck<F> {
    #[tracing::instrument(skip_all, name = "BytecodeHammingWeightSumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        F: Vec<Vec<F>>,
    ) -> Self {
        let d = sm.get_prover_data().0.shared.bytecode.d;
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let log_K = sm.get_bytecode().len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let ra = F
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>();
        Self {
            gamma: gamma_powers,
            log_K_chunk,
            d,
            prover_state: Some(HammingWeightProverState { ra }),
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let d = sm.get_verifier_data().0.shared.bytecode.d;
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let log_K = sm.get_bytecode().len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        Self {
            gamma: gamma_powers,
            log_K_chunk,
            d,
            prover_state: None,
        }
    }
}

impl<F: JoltField> HammingWeightConfig for BytecodeHammingWeightSumcheck<F> {
    fn d(&self) -> usize {
        self.d
    }

    fn num_rounds(&self) -> usize {
        self.log_K_chunk
    }

    fn polynomial_type(i: usize) -> CommittedPolynomial {
        CommittedPolynomial::BytecodeRa(i)
    }

    fn sumcheck_id() -> SumcheckId {
        SumcheckId::BytecodeHammingWeight
    }
}

impl<F: JoltField, T: Transcript> HammingWeightSumcheck<F, T> for BytecodeHammingWeightSumcheck<F> {
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
