use std::{cell::RefCell, rc::Rc};

use crate::{
    dag::state_manager::StateManager,
    field::JoltField,
    jolt::witness::{CommittedPolynomial, VirtualPolynomial},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::SumcheckInstance,
    utils::{math::Math, transcript::Transcript},
};
use rayon::prelude::*;

pub struct HammingWeightProverState<F: JoltField> {
    ra: MultilinearPolynomial<F>,
}

pub struct HammingWeightSumcheck<F: JoltField> {
    log_K: usize,
    prover_state: Option<HammingWeightProverState<F>>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    pub fn new_prover(F: Vec<F>, K: usize) -> Self {
        Self {
            log_K: K.log_2(),
            prover_state: Some(HammingWeightProverState {
                ra: MultilinearPolynomial::from(F),
            }),
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let log_K = sm.get_bytecode().len().log_2();
        Self {
            log_K,
            prover_state: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        self.log_K
    }

    fn input_claim(&self) -> F {
        F::one()
    }

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let univariate_poly_eval: F = (0..prover_state.ra.len() / 2)
            .into_par_iter()
            .map(|i| prover_state.ra.get_bound_coeff(2 * i))
            .sum();

        vec![univariate_poly_eval]
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        prover_state.ra.bind_parallel(r_j, BindingOrder::LowToHigh)
    }

    fn expected_output_claim(
        &self,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F],
    ) -> F {
        opening_accumulator
            .unwrap()
            .borrow()
            .get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa,
                SumcheckId::BytecodeHammingWeight,
            )
            .1
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let ra_claim = ps.ra.final_sumcheck_claim();

        let r_cycle = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();

        accumulator.borrow_mut().append_sparse(
            vec![CommittedPolynomial::BytecodeRa],
            SumcheckId::BytecodeHammingWeight,
            opening_point.r,
            r_cycle,
            vec![ra_claim],
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_cycle = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();

        let r = opening_point
            .r
            .iter()
            .cloned()
            .chain(r_cycle.iter().cloned())
            .collect::<Vec<_>>();
        accumulator.borrow_mut().append_sparse(
            vec![CommittedPolynomial::BytecodeRa],
            SumcheckId::BytecodeHammingWeight,
            r,
        );
    }
}
