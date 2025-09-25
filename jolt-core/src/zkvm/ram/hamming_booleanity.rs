use std::cell::RefCell;
use std::rc::Rc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::witness::VirtualPolynomial;
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;

const DEGREE: usize = 3;

#[derive(Allocative)]
struct HammingBooleanityProverState<F: JoltField> {
    eq_r_cycle: MultilinearPolynomial<F>,
    H: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub struct HammingBooleanitySumcheck<F: JoltField> {
    prover_state: Option<HammingBooleanityProverState<F>>,
    log_T: usize,
}

impl<F: JoltField> HammingBooleanitySumcheck<F> {
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let (_, trace, _, _) = sm.get_prover_data();

        let T = trace.len();
        let log_T = T.log_2();

        let H = trace
            .par_iter()
            .map(|cycle| {
                if cycle.ram_access().address() == 0 {
                    F::zero()
                } else {
                    F::one()
                }
            })
            .collect::<Vec<F>>();
        let H = MultilinearPolynomial::from(H);

        let (r_cycle, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        let eq_r_cycle = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_cycle.r));

        Self {
            prover_state: Some(HammingBooleanityProverState { eq_r_cycle, H }),
            log_T,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let (_, _, T) = sm.get_verifier_data();
        let log_T = T.log_2();
        Self {
            prover_state: None,
            log_T,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for HammingBooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();

        let univariate_poly_evals: [F; 3] = (0..p.eq_r_cycle.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = p
                    .eq_r_cycle
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let H_evals =
                    p.H.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let evals = [
                    H_evals[0].square() - H_evals[0],
                    H_evals[1].square() - H_evals[1],
                    H_evals[2].square() - H_evals[2],
                ];

                [
                    eq_evals[0] * evals[0],
                    eq_evals[1] * evals[1],
                    eq_evals[2] * evals[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        univariate_poly_evals.to_vec()
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let ps = self.prover_state.as_mut().unwrap();
        rayon::join(
            || ps.eq_r_cycle.bind_parallel(r_j, BindingOrder::LowToHigh),
            || ps.H.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let H_claim = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            )
            .1;

        let (r_cycle, _) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        let eq = EqPolynomial::<F>::mle(
            r,
            &r_cycle
                .r
                .iter()
                .cloned()
                .rev()
                .collect::<Vec<F::Challenge>>(),
        );

        (H_claim.square() - H_claim) * eq
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point.reverse();
        opening_point.into()
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state.as_ref().unwrap();

        let claim = ps.H.final_sumcheck_claim();

        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            opening_point.clone(),
            claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            opening_point,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
