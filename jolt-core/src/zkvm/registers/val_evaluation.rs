use num_traits::Zero;
use std::{cell::RefCell, rc::Rc, sync::Arc};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
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

// Register value evaluation sumcheck
//
// Proves the relation:
//   Val(r) = Σ_{j=0}^{T-1} inc(r_address, j) ⋅ wa(r_address, j) ⋅ LT(r_cycle, j)
// where:
// - r = (r_address, r_cycle) is the evaluation point from the read-write checking sumcheck.
// - Val(r) is the claimed value of register r_address at time r_cycle.
// - inc is the MLE of the per-cycle increment at (r_address, j).
// - wa is the MLE of the write-indicator (1 on matching {0,1}-points).
// - LT is the MLE of strict less-than on bitstrings; evaluated at (r_cycle, j) as field points.
//
// This sumcheck ensures that the claimed final value of a register is consistent
// with all the writes that occurred to it over time (assuming initial value of 0).

#[derive(Allocative)]
pub struct ValEvaluationProverState<F: JoltField> {
    pub inc: MultilinearPolynomial<F>,
    pub wa: RaPolynomial<u8, F>,
    pub lt: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub(crate) struct ValEvaluationSumcheck<F: JoltField> {
    pub input_claim: F,
    pub num_rounds: usize,
    pub prover_state: Option<ValEvaluationProverState<F>>,
}

impl<F: JoltField> ValEvaluationSumcheck<F> {
    #[tracing::instrument(skip_all, name = "RegistersValEvaluationSumcheck::new_prover")]
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
        let r_address: Vec<F::Challenge> = r_address_slice.into();
        let r_cycle: Vec<F::Challenge> = r_cycle_slice.into();

        let inc = CommittedPolynomial::RdInc.generate_witness(preprocessing, trace);

        let eq_r_address = EqPolynomial::evals(&r_address);
        let wa: Vec<Option<u8>> = trace
            .par_iter()
            .map(|cycle| {
                let instr = cycle.instruction().normalize();
                Some(instr.operands.rd)
            })
            .collect();
        let wa = RaPolynomial::new(Arc::new(wa), eq_r_address);

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
            num_rounds,
            prover_state: Some(ValEvaluationProverState { inc, wa, lt }),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, _, trace_length) = state_manager.get_verifier_data();

        let accumulator = state_manager.get_verifier_accumulator();
        // Get val_claim from the accumulator (from stage 2 RegistersReadWriteChecking)
        let (_, val_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );

        Self {
            input_claim: val_claim,
            num_rounds: trace_length.log_2(),
            prover_state: None,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ValEvaluationSumcheck<F> {
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
        (0..prover_state.inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = prover_state
                    .inc
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let wa_evals = prover_state
                    .wa
                    .sumcheck_evals(i, DEGREE, BindingOrder::HighToLow);
                let lt_evals = prover_state
                    .lt
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                [
                    (inc_evals[0] * wa_evals[0]).mul_unreduced::<9>(lt_evals[0]),
                    (inc_evals[1] * wa_evals[1]).mul_unreduced::<9>(lt_evals[1]),
                    (inc_evals[2] * wa_evals[2]).mul_unreduced::<9>(lt_evals[2]),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); DEGREE],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            )
            .into_iter()
            .map(F::from_montgomery_reduce)
            .collect()
    }

    #[tracing::instrument(skip_all, name = "RegistersValEvaluationSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            [&mut prover_state.inc, &mut prover_state.lt]
                .par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
            prover_state.wa.bind_parallel(r_j, BindingOrder::HighToLow);
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let (opening_point, _) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle) = opening_point.split_at(REGISTER_COUNT.ilog2() as usize);

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();

        for (x, y) in r.iter().zip(r_cycle.r.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

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

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let (opening_point, _) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (r_address, _) = opening_point.split_at(REGISTER_COUNT.ilog2() as usize);

        let inc_claim = prover_state.inc.final_sumcheck_claim();
        let wa_claim = prover_state.wa.final_sumcheck_claim();

        // Append claims to accumulator
        accumulator.borrow_mut().append_dense(
            transcript,
            vec![CommittedPolynomial::RdInc],
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
            &[inc_claim],
        );

        let r = [r_address.r.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
            wa_claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let (opening_point, _) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (r_address, _) = opening_point.split_at(REGISTER_COUNT.ilog2() as usize);

        // Append claims to accumulator
        accumulator.borrow_mut().append_dense(
            transcript,
            vec![CommittedPolynomial::RdInc],
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
        );

        let r = [r_address.r.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.borrow_mut().append_virtual(
            transcript,
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
