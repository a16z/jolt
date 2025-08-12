use std::{cell::RefCell, rc::Rc};

use crate::{
    field::{allocative_ark::MaybeAllocative, JoltField},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::SumcheckInstance,
    utils::{math::Math, thread::unsafe_allocate_zero_vec, transcript::Transcript},
    zkvm::{
        dag::state_manager::StateManager,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::REGISTER_COUNT;
use rayon::prelude::*;

#[derive(Allocative)]
pub struct ValEvaluationProverState<F: JoltField> {
    pub inc: MultilinearPolynomial<F>,
    pub wa: MultilinearPolynomial<F>,
    pub lt: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub(crate) struct ValEvaluationSumcheck<F: JoltField> {
    pub r_address: Vec<F>,
    pub input_claim: F,
    pub num_rounds: usize,
    pub r_cycle: Vec<F>,
    pub prover_state: Option<ValEvaluationProverState<F>>,
}

impl<F: JoltField + MaybeAllocative> ValEvaluationSumcheck<F> {
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, trace, _, _) = state_manager.get_prover_data();
        let accumulator = state_manager.get_prover_accumulator();

        // Get val_claim from the accumulator (from stage 2 RegistersReadWriteChecking)
        let (opening_point, val_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );

        // The opening point is r_address || r_cycle
        let r_address_len = REGISTER_COUNT.ilog2() as usize;
        let (r_address_slice, r_cycle_slice) = opening_point.split_at(r_address_len);
        let r_address: Vec<F> = r_address_slice.into();
        let r_cycle: Vec<F> = r_cycle_slice.into();

        let inc = CommittedPolynomial::RdInc.generate_witness(preprocessing, trace);

        let eq_r_address = EqPolynomial::evals(&r_address);
        let wa: Vec<F> = trace
            .par_iter()
            .map(|cycle| {
                let instr = cycle.instruction().normalize();
                eq_r_address[instr.operands.rd as usize]
            })
            .collect();
        let wa = MultilinearPolynomial::from(wa);

        let T = trace.len();
        let mut lt: Vec<F> = unsafe_allocate_zero_vec(T);
        for (i, r) in r_cycle.iter().rev().enumerate() {
            let (evals_left, evals_right) = lt.split_at_mut(1 << i);
            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r;
                    *x += *r - *y;
                });
        }
        let lt = MultilinearPolynomial::from(lt);

        let num_rounds = r_cycle.len().pow2().log_2();
        Self {
            input_claim: val_claim,
            r_address,
            num_rounds,
            r_cycle,
            prover_state: Some(ValEvaluationProverState { inc, wa, lt }),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, _, trace_length) = state_manager.get_verifier_data();

        let accumulator = state_manager.get_verifier_accumulator();
        // Get val_claim from the accumulator (from stage 2 RegistersReadWriteChecking)
        let (opening_point, val_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );

        // The opening point is r_address || r_cycle
        let r_address_len = REGISTER_COUNT.ilog2() as usize;
        let (r_address_slice, r_cycle_slice) = opening_point.split_at(r_address_len);
        let r_address: Vec<F> = r_address_slice.into();
        let r_cycle: Vec<F> = r_cycle_slice.into();

        Self {
            input_claim: val_claim,
            r_address,
            num_rounds: trace_length.log_2(),
            r_cycle,
            prover_state: None,
        }
    }
}

impl<F: JoltField + MaybeAllocative> SumcheckInstance<F> for ValEvaluationSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(
        skip_all,
        name = "RegistersValEvaluationSumcheck::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        const DEGREE: usize = 3;
        let univariate_poly_evals: [F; 3] = (0..prover_state.inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = prover_state
                    .inc
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let wa_evals = prover_state
                    .wa
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let lt_evals = prover_state
                    .lt
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                [
                    inc_evals[0] * wa_evals[0] * lt_evals[0],
                    inc_evals[1] * wa_evals[1] * lt_evals[1],
                    inc_evals[2] * wa_evals[2] * lt_evals[2],
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

    #[tracing::instrument(skip_all, name = "RegistersValEvaluationSumcheck::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            [
                &mut prover_state.inc,
                &mut prover_state.wa,
                &mut prover_state.lt,
            ]
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r.iter().zip(self.r_cycle.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        let accumulator = accumulator.as_ref().unwrap();
        let (_, inc_claim) = accumulator.borrow().get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, wa_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );

        // Return inc_claim * wa_claim * lt_eval
        inc_claim * wa_claim * lt_eval
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let inc_claim = prover_state.inc.final_sumcheck_claim();
        let wa_claim = prover_state.wa.final_sumcheck_claim();

        // Append claims to accumulator
        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::RdInc],
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
            &[inc_claim],
        );

        let r = [self.r_address.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
            wa_claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Append claims to accumulator
        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::RdInc],
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
        );

        let r = [self.r_address.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
