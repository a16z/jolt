use itertools::chain;
use num_traits::Zero;
use std::{array, cell::RefCell, rc::Rc, sync::Arc};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        lt_poly::LtPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::math::Math,
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

// RAM value evaluation sumcheck
//
// Proves the relation:
//   Val(r) - Val_init(r_address) = Σ_{j=0}^{T-1} inc(r_address, j) ⋅ wa(r_address, j) ⋅ LT(r_cycle, j)
// where:
// - r = (r_address, r_cycle) is the evaluation point from the read-write checking sumcheck.
// - Val(r) is the claimed value of memory at address r_address and time r_cycle.
// - Val_init(r_address) is the initial value of memory at address r_address.
// - inc(r_address, j) is the MLE of the per-cycle increment at (r_address, j).
// - wa is the MLE of the write-indicator (1 on matching {0,1}-points).
// - LT is the MLE of strict less-than on bitstrings; evaluated at (r_cycle, j) as field points.
//
// This sumcheck ensures that the claimed final value of a memory cell is consistent
// with its initial value and all the writes that occurred to it over time.

#[derive(Allocative)]
pub struct ValEvaluationProverState<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: RaPolynomial<usize, F>,
    lt: LtPolynomial<F>,
}

/// Val-evaluation sumcheck for RAM
#[derive(Allocative)]
pub struct ValEvaluationSumcheck<F: JoltField> {
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

        let (r, _) = state_manager.get_virtual_polynomial_opening(
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
        let wa_indices: Vec<Option<usize>> = trace
            .par_iter()
            .map(|cycle| {
                remap_address(cycle.ram_access().address() as u64, memory_layout)
                    .map(|k| k as usize)
            })
            .collect();
        let wa = RaPolynomial::new(Arc::new(wa_indices), eq_r_address);

        drop(_guard);
        drop(span);

        let inc = CommittedPolynomial::RamInc.generate_witness(preprocessing, trace);
        let lt = LtPolynomial::new(&r_cycle);

        ValEvaluationSumcheck {
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

        let (r, _) = state_manager.get_virtual_polynomial_opening(
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

    fn input_claim(&self, acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        let (_, claimed_evaluation) = acc.unwrap().borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        claimed_evaluation - self.init_eval
    }

    #[tracing::instrument(skip_all, name = "RamValEvaluationSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, previous_claim: F) -> Vec<F> {
        const DEGREE: usize = 3;

        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let half_n = ps.inc.len() / 2;

        let [eval_at_1, eval_at_2, eval_at_inf] = (0..ps.inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_at_1_j = ps.inc.get_bound_coeff(j + half_n);
                let inc_at_inf_j = inc_at_1_j - ps.inc.get_bound_coeff(j);
                let inc_at_2_j = inc_at_1_j + inc_at_inf_j;

                let wa_at_1_j = ps.wa.get_bound_coeff(j + half_n);
                let wa_at_inf_j = wa_at_1_j - ps.wa.get_bound_coeff(j);
                let wa_at_2_j = wa_at_1_j + wa_at_inf_j;

                let lt_at_1_j = ps.lt.get_bound_coeff(j + half_n);
                let lt_at_inf_j = lt_at_1_j - ps.lt.get_bound_coeff(j);
                let lt_at_2_j = lt_at_1_j + lt_at_inf_j;

                // Eval inc * wa * lt.
                [
                    (inc_at_1_j * wa_at_1_j).mul_unreduced::<9>(lt_at_1_j),
                    (inc_at_2_j * wa_at_2_j).mul_unreduced::<9>(lt_at_2_j),
                    (inc_at_inf_j * wa_at_inf_j).mul_unreduced::<9>(lt_at_inf_j),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); DEGREE],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            )
            .map(F::from_montgomery_reduce);

        let eval_at_0 = previous_claim - eval_at_1;
        let poly = UniPoly::from_evals_toom(&[eval_at_0, eval_at_1, eval_at_2, eval_at_inf]);
        let domain = chain!([0], 2..).take(DEGREE).map(F::from_u64);
        domain.map(|x| poly.evaluate::<F>(&x)).collect()
    }

    #[tracing::instrument(skip_all, name = "RamValEvaluationSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            prover_state.inc.bind_parallel(r_j, BindingOrder::HighToLow);
            prover_state.wa.bind_parallel(r_j, BindingOrder::HighToLow);
            prover_state.lt.bind_high_to_low(r_j);
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
                CommittedPolynomial::RamInc,
                SumcheckId::RamValEvaluation,
                r_cycle_prime.r,
                prover_state.inc.final_sumcheck_claim(),
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
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
            r_cycle_prime.r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
