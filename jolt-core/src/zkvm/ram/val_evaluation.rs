use common::jolt_device::MemoryLayout;
use num_traits::Zero;
use std::{array, iter::zip, sync::Arc};
use tracer::{instruction::Cycle, JoltDevice};

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        lt_poly::LtPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, PolynomialId, ProverOpeningAccumulator,
            SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        ra_poly::RaPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        constraint_types::{InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
    zkvm::{
        bytecode::BytecodePreprocessing,
        config::OneHotParams,
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
//   Val(r) - Val_init(r_address) = Σ_{j=0}^{T-1} inc(j) ⋅ wa(r_address, j) ⋅ LT(j, r_cycle)
// where:
// - r = (r_address, r_cycle) is the evaluation point from the read-write checking sumcheck.
// - Val(r) is the claimed value of memory at address r_address and time r_cycle.
// - Val_init(r_address) is the initial value of memory at address r_address.
// - inc(j) is the change in value at cycle j if a write occurs, and 0 otherwise.
// - wa is the MLE of the write-indicator (1 on matching {0,1}-points).
// - LT is the MLE of strict less-than on bitstrings; LT(j, k) = 1 iff j < k for bitstrings j, k.
//
// This sumcheck ensures that the claimed final value of a memory cell is consistent
// with its initial value and all the writes that occurred to it over time.

/// Degree bound of the sumcheck round polynomials in [`ValEvaluationSumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct ValEvaluationSumcheckParams<F: JoltField> {
    /// Full init_eval for the prover's `input_claim()` computation.
    pub init_eval: F,
    /// Public-only portion of init_eval (bytecode + inputs), used by BlindFold constraint.
    pub init_eval_public: F,
    /// Advice contributions decomposed for BlindFold: each is (-selector, opening_id).
    pub advice_contributions: Vec<(F, OpeningId)>,
    pub T: usize,
    pub K: usize,
    pub r_address: OpeningPoint<BIG_ENDIAN, F>,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> ValEvaluationSumcheckParams<F> {
    pub fn new_from_prover(
        one_hot_params: &OneHotParams,
        opening_accumulator: &ProverOpeningAccumulator<F>,
        initial_ram_state: &[u64],
        trace_len: usize,
        ram_preprocessing: &super::RAMPreprocessing,
        program_io: &JoltDevice,
    ) -> Self {
        let K = one_hot_params.ram_k;
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, r_cycle) = r.split_at(K.log_2());
        let val_init: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());
        let init_eval = val_init.evaluate(&r_address.r);

        let init_eval_public =
            super::eval_initial_ram_mle::<F>(ram_preprocessing, program_io, &r_address.r);

        let n_memory_vars = K.log_2();
        let advice_contributions = super::compute_advice_init_contributions(
            opening_accumulator,
            &program_io.memory_layout,
            &r_address.r,
            n_memory_vars,
            SumcheckId::RamValEvaluation,
        );

        Self {
            init_eval,
            init_eval_public,
            advice_contributions,
            T: trace_len,
            K,
            r_address,
            r_cycle,
        }
    }

    pub fn new_from_verifier(
        ram_preprocessing: &super::RAMPreprocessing,
        program_io: &JoltDevice,
        trace_len: usize,
        ram_K: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, r_cycle) = r.split_at(ram_K.log_2());

        let n_memory_vars = ram_K.log_2();

        let init_eval_public =
            super::eval_initial_ram_mle::<F>(ram_preprocessing, program_io, &r_address.r);

        let advice_contributions = super::compute_advice_init_contributions(
            opening_accumulator,
            &program_io.memory_layout,
            &r_address.r,
            n_memory_vars,
            SumcheckId::RamValEvaluation,
        );

        // Reconstruct full init_eval from public portion + advice contributions.
        // In ZK mode advice evals are zero (not pre-populated), so this is a no-op.
        let init_eval = super::reconstruct_full_eval(
            init_eval_public,
            &advice_contributions,
            opening_accumulator,
        );

        ValEvaluationSumcheckParams {
            init_eval,
            init_eval_public,
            advice_contributions,
            T: trace_len,
            K: ram_K,
            r_address,
            r_cycle,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ValEvaluationSumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, claimed_evaluation) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        claimed_evaluation - self.init_eval
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn input_claim_constraint(&self) -> InputClaimConstraint {
        let opening = OpeningId::Polynomial(
            PolynomialId::Virtual(VirtualPolynomial::RamVal),
            SumcheckId::RamReadWriteChecking,
        );
        // input_claim = val_opening - eval_public - Σ(selector_i * advice_opening_i)
        let mut terms = vec![
            ProductTerm::single(ValueSource::Opening(opening)),
            ProductTerm::single(ValueSource::Challenge(0)), // -eval_public
        ];
        for (i, (_, advice_opening_id)) in self.advice_contributions.iter().enumerate() {
            terms.push(ProductTerm::product(vec![
                ValueSource::Challenge(i + 1),
                ValueSource::Opening(*advice_opening_id),
            ]));
        }
        InputClaimConstraint::sum_of_products(terms)
    }

    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        let mut values = vec![-self.init_eval_public];
        for (neg_selector, _) in &self.advice_contributions {
            values.push(*neg_selector);
        }
        values
    }

    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let inc = OpeningId::Polynomial(
            PolynomialId::Committed(CommittedPolynomial::RamInc),
            SumcheckId::RamValEvaluation,
        );
        let wa = OpeningId::Polynomial(
            PolynomialId::Virtual(VirtualPolynomial::RamRa),
            SumcheckId::RamValEvaluation,
        );

        let lt_eval = ValueSource::Challenge(0);

        let terms = vec![ProductTerm::product(vec![
            ValueSource::Opening(inc),
            ValueSource::Opening(wa),
            lt_eval,
        ])];

        Some(OutputClaimConstraint::sum_of_products(terms))
    }

    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let r = self.normalize_opening_point(sumcheck_challenges);

        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in zip(&r.r, &self.r_cycle.r) {
            lt_eval += (F::one() - *x) * *y * eq_term;
            eq_term *= F::one() - *x - *y + *x * *y + *x * *y;
        }

        vec![lt_eval]
    }
}

#[derive(Allocative)]
pub struct ValEvaluationSumcheckProver<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: RaPolynomial<usize, F>,
    lt: LtPolynomial<F>,
    pub params: ValEvaluationSumcheckParams<F>,
}

impl<F: JoltField> ValEvaluationSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RamValEvaluationSumcheckProver::initialize")]
    pub fn initialize(
        params: ValEvaluationSumcheckParams<F>,
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
    ) -> Self {
        // Compute the size-K table storing all eq(r_address, k) evaluations for
        // k \in {0, 1}^log(K)
        let eq_r_address = EqPolynomial::evals(&params.r_address.r);

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

        let inc = CommittedPolynomial::RamInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            trace,
            None,
        );
        let lt = LtPolynomial::new(&params.r_cycle);

        Self {
            inc,
            wa,
            lt,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ValEvaluationSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RamValEvaluationSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let [eval_at_1, eval_at_2, eval_at_inf] = (0..self.inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_at_j_1 = self.inc.get_bound_coeff(j * 2 + 1);
                let inc_at_j_inf = inc_at_j_1 - self.inc.get_bound_coeff(j * 2);
                let inc_at_j_2 = inc_at_j_1 + inc_at_j_inf;

                let wa_at_j_1 = self.wa.get_bound_coeff(j * 2 + 1);
                let wa_at_j_inf = wa_at_j_1 - self.wa.get_bound_coeff(j * 2);
                let wa_at_j_2 = wa_at_j_1 + wa_at_j_inf;

                let lt_at_j_1 = self.lt.get_bound_coeff(j * 2 + 1);
                let lt_at_j_inf = lt_at_j_1 - self.lt.get_bound_coeff(j * 2);
                let lt_at_j_2 = lt_at_j_1 + lt_at_j_inf;

                // Eval inc * wa * lt.
                [
                    (inc_at_j_1 * wa_at_j_1).mul_unreduced::<9>(lt_at_j_1),
                    (inc_at_j_2 * wa_at_j_2).mul_unreduced::<9>(lt_at_j_2),
                    (inc_at_j_inf * wa_at_j_inf).mul_unreduced::<9>(lt_at_j_inf),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            )
            .map(F::from_montgomery_reduce);

        let eval_at_0 = previous_claim - eval_at_1;
        UniPoly::from_evals_toom(&[eval_at_0, eval_at_1, eval_at_2, eval_at_inf])
    }

    #[tracing::instrument(skip_all, name = "RamValEvaluationSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.wa.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.lt.bind(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle_prime = self.params.normalize_opening_point(sumcheck_challenges);
        let r = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            )
            .0;
        let (r_address, _) = r.split_at(r.len() - r_cycle_prime.len());
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValEvaluation,
            wa_opening_point,
            self.wa.final_sumcheck_claim(),
        );

        accumulator.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
            r_cycle_prime.r,
            self.inc.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Val-evaluation sumcheck for RAM.
pub struct ValEvaluationSumcheckVerifier<F: JoltField> {
    params: ValEvaluationSumcheckParams<F>,
}

impl<F: JoltField> ValEvaluationSumcheckVerifier<F> {
    pub fn new(
        ram_preprocessing: &super::RAMPreprocessing,
        program_io: &JoltDevice,
        trace_len: usize,
        ram_K: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = ValEvaluationSumcheckParams::new_from_verifier(
            ram_preprocessing,
            program_io,
            trace_len,
            ram_K,
            opening_accumulator,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for ValEvaluationSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let (r_val, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, r_cycle) = r_val.split_at(self.params.K.log_2());
        let r = self.params.normalize_opening_point(sumcheck_challenges);

        // Compute LT(r, r_cycle) using the MLE formula:
        //   LT(x, y) = Σ_i (1 - x_i) · y_i · eq(x[i+1:], y[i+1:])
        //
        // The prover constructs LtPolynomial with r_cycle, giving LT(j, r_cycle) for all j.
        // After binding j to r (the sumcheck challenges), the prover gets LT(r, r_cycle).
        // The verifier computes the same value directly using the formula above.
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in zip(&r.r, &r_cycle.r) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
        );
        let (_, wa_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamValEvaluation);

        // Return inc_claim * wa_claim * lt_eval
        inc_claim * wa_claim * lt_eval
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle_prime = self.params.normalize_opening_point(sumcheck_challenges);
        let r = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            )
            .0;
        let (r_address, _) = r.split_at(r.len() - r_cycle_prime.len());
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValEvaluation,
            wa_opening_point,
        );
        accumulator.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
            r_cycle_prime.r,
        );
    }
}
