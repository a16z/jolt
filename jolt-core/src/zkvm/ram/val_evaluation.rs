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
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        ra_poly::RaPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
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

pub struct ValEvaluationSumcheckParams<F: JoltField> {
    /// Initial evaluation to subtract (for RAM).
    pub init_eval: F,
    /// Trace length.
    pub T: usize,
    /// Ram K parameter.
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

        Self {
            init_eval,
            T: trace_len,
            K,
            r_address,
            r_cycle,
        }
    }

    pub fn new_from_verifier(
        initial_ram_state: &[u64],
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

        // Calculate untrusted advice contribution
        let untrusted_contribution = super::calculate_advice_memory_evaluation(
            opening_accumulator.get_untrusted_advice_opening(SumcheckId::RamValEvaluation),
            (program_io.memory_layout.max_untrusted_advice_size as usize / 8)
                .next_power_of_two()
                .log_2(),
            program_io.memory_layout.untrusted_advice_start,
            &program_io.memory_layout,
            &r_address.r,
            n_memory_vars,
        );

        // Calculate trusted advice contribution
        let trusted_contribution = super::calculate_advice_memory_evaluation(
            opening_accumulator.get_trusted_advice_opening(SumcheckId::RamValEvaluation),
            (program_io.memory_layout.max_trusted_advice_size as usize / 8)
                .next_power_of_two()
                .log_2(),
            program_io.memory_layout.trusted_advice_start,
            &program_io.memory_layout,
            &r_address.r,
            n_memory_vars,
        );

        // Compute the public part of val_init evaluation
        let val_init_public: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());

        // Combine all contributions: untrusted + trusted + public
        let init_eval =
            untrusted_contribution + trusted_contribution + val_init_public.evaluate(&r_address.r);

        ValEvaluationSumcheckParams {
            init_eval,
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
}

/// Sumcheck prover for [`ValEvaluationSumcheckVerifier`].
#[derive(Allocative)]
pub struct ValEvaluationSumcheckProver<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: RaPolynomial<usize, F>,
    lt: LtPolynomial<F>,
    #[allocative(skip)]
    params: ValEvaluationSumcheckParams<F>,
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
        transcript: &mut T,
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
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValEvaluation,
            wa_opening_point,
            self.wa.final_sumcheck_claim(),
        );

        accumulator.append_dense(
            transcript,
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
        initial_ram_state: &[u64],
        program_io: &JoltDevice,
        trace_len: usize,
        ram_K: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = ValEvaluationSumcheckParams::new_from_verifier(
            initial_ram_state,
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
        transcript: &mut T,
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
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValEvaluation,
            wa_opening_point,
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
            r_cycle_prime.r,
        );
    }
}
