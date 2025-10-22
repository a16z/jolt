use allocative::Allocative;
use common::constants::XLEN;
use rayon::prelude::*;

use super::{D, K_CHUNK, LOG_K_CHUNK};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::BindingOrder,
        opening_proof::{OpeningAccumulator, SumcheckId},
        split_eq_poly::GruenSplitEqPolynomial,
    },
    subprotocols::booleanity::{BooleanityConfig, BooleanityProverState, BooleanitySumcheck},
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        dag::state_manager::StateManager,
        instruction::LookupQuery,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

#[derive(Allocative)]
pub struct InstructionBooleanitySumcheck<F: JoltField> {
    /// Batching challenges γ_i (optimized challenges)
    gamma: [F::Challenge; D],
    prover_state: Option<BooleanityProverState<F>>,
    r_address: Vec<F::Challenge>,
    r_cycle: Vec<F::Challenge>,
    log_T: usize,
}

impl<F: JoltField> InstructionBooleanitySumcheck<F> {
    #[tracing::instrument(skip_all, name = "InstructionBooleanitySumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        G: [Vec<F>; D],
    ) -> Self {
        let gamma: [F::Challenge; D] = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(D)
            .try_into()
            .unwrap();
        let r_address: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(LOG_K_CHUNK);
        let r_cycle = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();
        let trace = sm.get_prover_data().1;

        // Build H_indices
        let H_indices: Vec<Vec<Option<u8>>> = (0..D)
            .map(|i| {
                trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(
                            ((lookup_index >> (LOG_K_CHUNK * (D - 1 - i))) % K_CHUNK as u128) as u8,
                        )
                    })
                    .collect()
            })
            .collect();

        let B = GruenSplitEqPolynomial::new(&r_address, BindingOrder::LowToHigh);
        let D_poly = GruenSplitEqPolynomial::new(&r_cycle, BindingOrder::LowToHigh);

        let mut F: Vec<F> = unsafe_allocate_zero_vec(K_CHUNK);
        F[0] = F::one();

        let prover_state = BooleanityProverState {
            B,
            D: D_poly,
            G: G.to_vec(),
            H_indices,
            H: vec![],
            F,
            eq_r_r: F::zero(),
        };

        Self {
            gamma,
            prover_state: Some(prover_state),
            r_address,
            r_cycle,
            log_T: trace.len().log_2(),
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let log_T = sm.get_verifier_data().2.log_2();
        let gamma: [F::Challenge; D] = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(D)
            .try_into()
            .unwrap();
        let r_address: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(LOG_K_CHUNK);
        let r_cycle = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();

        Self {
            gamma,
            prover_state: None,
            r_address,
            r_cycle,
            log_T,
        }
    }
}

impl<F: JoltField> BooleanityConfig for InstructionBooleanitySumcheck<F> {
    fn d(&self) -> usize {
        D
    }

    fn log_k_chunk(&self) -> usize {
        LOG_K_CHUNK
    }

    fn log_t(&self) -> usize {
        self.log_T
    }

    fn polynomial_type(i: usize) -> CommittedPolynomial {
        CommittedPolynomial::InstructionRa(i)
    }

    fn sumcheck_id() -> SumcheckId {
        SumcheckId::InstructionBooleanity
    }
}

impl<F: JoltField, T: Transcript> BooleanitySumcheck<F, T> for InstructionBooleanitySumcheck<F> {
    fn gamma(&self) -> &[F::Challenge] {
        &self.gamma
    }

    fn r_address(&self) -> &[F::Challenge] {
        &self.r_address
    }

    fn prover_state(&self) -> Option<&BooleanityProverState<F>> {
        self.prover_state.as_ref()
    }

    fn prover_state_mut(&mut self) -> Option<&mut BooleanityProverState<F>> {
        self.prover_state.as_mut()
    }

    fn get_r_cycle(&self, _accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge> {
        self.r_cycle.clone()
    }
}
