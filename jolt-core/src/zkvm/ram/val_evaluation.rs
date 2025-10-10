use num_traits::Zero;
use std::{cell::RefCell, rc::Rc};

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
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        dag::state_manager::StateManager,
        ram::remap_address,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;

#[derive(Allocative)]
pub struct ValEvaluationProverState<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
    lt: MultilinearPolynomial<F>,
}

/// Val-evaluation sumcheck for RAM
#[derive(Allocative)]
pub struct ValEvaluationSumcheck<F: JoltField> {
    /// Initial claim value
    claimed_evaluation: F,
    /// Initial evaluation to subtract (for RAM)
    init_eval: F,
    /// log T
    num_rounds: usize,
    /// used to compute LT evaluation
    prover_state: Option<ValEvaluationProverState<F>>,
    /// ram K parameter
    K: usize,
}

impl<F: JoltField> ValEvaluationSumcheck<F> {
    #[tracing::instrument(skip_all, name = "RamValEvaluationSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        initial_ram_state: &[u64],
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, trace, program_io, _) = state_manager.get_prover_data();
        let memory_layout = &program_io.memory_layout;
        let T = trace.len();
        let K = state_manager.ram_K;

        let (r, claimed_evaluation) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, r_cycle) = r.split_at(K.log_2());

        let val_init: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());
        let init_eval = val_init.evaluate(&r_address.r);

        // Compute the size-K table storing all eq(r_address, k) evaluations for
        // k \in {0, 1}^log(K)
        let eq_r_address = EqPolynomial::evals(&r_address.r);

        let span = tracing::span!(tracing::Level::INFO, "compute wa(r_address, j)");
        let _guard = span.enter();

        // Compute the wa polynomial using the above table
        let wa: Vec<F> = trace
            .par_iter()
            .map(|cycle| {
                remap_address(cycle.ram_access().address() as u64, memory_layout)
                    .map_or(F::zero(), |k| eq_r_address[k as usize])
            })
            .collect();
        let wa = MultilinearPolynomial::from(wa);

        drop(_guard);
        drop(span);

        let inc = CommittedPolynomial::RamInc.generate_witness(preprocessing, trace);

        let span = tracing::span!(tracing::Level::INFO, "compute LT(j, r_cycle)");
        let _guard = span.enter();

        let mut lt: Vec<F> = unsafe_allocate_zero_vec(T);
        for (i, r) in r_cycle.r.iter().rev().enumerate() {
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

        drop(_guard);
        drop(span);

        ValEvaluationSumcheck {
            claimed_evaluation,
            init_eval,
            num_rounds: T.log_2(),
            prover_state: Some(ValEvaluationProverState { inc, wa, lt }),
            K,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        initial_ram_state: &[u64],
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, program_io, T) = state_manager.get_verifier_data();
        let K = state_manager.ram_K;

        let (r, claimed_evaluation) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, _) = r.split_at(K.log_2());

        let accumulator = state_manager.get_verifier_accumulator();
        let total_memory_vars = K.log_2();

        // Calculate untrusted advice contribution
        let untrusted_contribution = super::calculate_advice_memory_evaluation(
            accumulator.borrow().get_untrusted_advice_opening(),
            (program_io.memory_layout.max_untrusted_advice_size as usize / 8)
                .next_power_of_two()
                .log_2(),
            program_io.memory_layout.untrusted_advice_start,
            &program_io.memory_layout,
            &r_address.r,
            total_memory_vars,
        );

        // Calculate trusted advice contribution
        let trusted_contribution = super::calculate_advice_memory_evaluation(
            accumulator.borrow().get_trusted_advice_opening(),
            (program_io.memory_layout.max_trusted_advice_size as usize / 8)
                .next_power_of_two()
                .log_2(),
            program_io.memory_layout.trusted_advice_start,
            &program_io.memory_layout,
            &r_address.r,
            total_memory_vars,
        );

        // Compute the public part of val_init evaluation
        let val_init_public: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());

        // Combine all contributions: untrusted + trusted + public
        let init_eval =
            untrusted_contribution + trusted_contribution + val_init_public.evaluate(&r_address.r);

        ValEvaluationSumcheck {
            claimed_evaluation,
            init_eval,
            num_rounds: T.log_2(),
            prover_state: None,
            K,
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
        self.claimed_evaluation - self.init_eval
    }

    #[tracing::instrument(skip_all, name = "RamValEvaluationSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        const DEGREE: usize = 3;
        (0..ps.inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = ps
                    .inc
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let wa_evals = ps
                    .wa
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let lt_evals = ps
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

    #[tracing::instrument(skip_all, name = "RamValEvaluationSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
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
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let (r_val, _) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, r_cycle) = r_val.split_at(self.K.log_2());
        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r.iter().zip(r_cycle.r.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        let (_, inc_claim) = accumulator.borrow().get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
        );
        let (_, wa_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamValEvaluation);

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
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        if let Some(prover_state) = &self.prover_state {
            let r = accumulator
                .borrow()
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::RamVal,
                    SumcheckId::RamReadWriteChecking,
                )
                .0;
            let (r_address, _) = r.split_at(r.len() - r_cycle_prime.len());
            let wa_opening_point =
                OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

            accumulator.borrow_mut().append_virtual(
                transcript,
                VirtualPolynomial::RamRa,
                SumcheckId::RamValEvaluation,
                wa_opening_point,
                prover_state.wa.final_sumcheck_claim(),
            );

            accumulator.borrow_mut().append_dense(
                transcript,
                vec![CommittedPolynomial::RamInc],
                SumcheckId::RamValEvaluation,
                r_cycle_prime.r,
                &[prover_state.inc.final_sumcheck_claim()],
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            )
            .0;
        let (r_address, _) = r.split_at(r.len() - r_cycle_prime.len());
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValEvaluation,
            wa_opening_point,
        );

        accumulator.borrow_mut().append_dense(
            transcript,
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamValEvaluation,
            r_cycle_prime.r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
