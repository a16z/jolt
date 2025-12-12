use num_traits::Zero;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
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
        ram::remap_address,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::{instruction::Cycle, JoltDevice};

/// Degree bound of the sumcheck round polynomials in [`ValFinalSumcheckVerifier`].
const VAL_FINAL_SUMCHECK_DEGREE_BOUND: usize = 2;

pub struct ValFinalSumcheckParams<F: JoltField> {
    pub T: usize,
    pub r_address: Vec<F::Challenge>,
    pub val_init_eval: F,
}

impl<F: JoltField> ValFinalSumcheckParams<F> {
    pub fn new_from_prover(
        trace_len: usize,
        opening_accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
        let (r_address, val_init_eval) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
        );

        Self {
            T: trace_len,
            val_init_eval,
            r_address: r_address.r.clone(),
        }
    }

    pub fn new_from_verifier(
        initial_ram_state: &[u64],
        program_io: &JoltDevice,
        trace_len: usize,
        ram_K: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let r_address = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0
            .r;

        let n_memory_vars = ram_K.log_2();

        // When needs_single_advice_opening(T) is true, advice is only opened at RamValEvaluation
        // (the two points are identical). Otherwise, we use RamValFinalEvaluation.
        let advice_sumcheck_id =
            if super::read_write_checking::needs_single_advice_opening(trace_len) {
                SumcheckId::RamValEvaluation
            } else {
                SumcheckId::RamValFinalEvaluation
            };

        let untrusted_advice_contribution = super::calculate_advice_memory_evaluation(
            opening_accumulator.get_untrusted_advice_opening(advice_sumcheck_id),
            (program_io.memory_layout.max_untrusted_advice_size as usize / 8)
                .next_power_of_two()
                .log_2(),
            program_io.memory_layout.untrusted_advice_start,
            &program_io.memory_layout,
            &r_address,
            n_memory_vars,
        );

        let trusted_advice_contribution = super::calculate_advice_memory_evaluation(
            opening_accumulator.get_trusted_advice_opening(advice_sumcheck_id),
            (program_io.memory_layout.max_trusted_advice_size as usize / 8)
                .next_power_of_two()
                .log_2(),
            program_io.memory_layout.trusted_advice_start,
            &program_io.memory_layout,
            &r_address,
            n_memory_vars,
        );

        // Compute the public part of val_init evaluation
        let val_init_public: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());

        // Combine all contributions: untrusted + trusted + public
        let val_init_eval = untrusted_advice_contribution
            + trusted_advice_contribution
            + val_init_public.evaluate(&r_address);

        ValFinalSumcheckParams {
            T: trace_len,
            val_init_eval,
            r_address,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ValFinalSumcheckParams<F> {
    fn degree(&self) -> usize {
        VAL_FINAL_SUMCHECK_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, val_final_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
        );
        val_final_claim - self.val_init_eval
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct ValFinalSumcheckProver<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
    #[allocative(skip)]
    params: ValFinalSumcheckParams<F>,
}

impl<F: JoltField> ValFinalSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "ValFinalSumcheckProver::initialize")]
    pub fn initialize(
        params: ValFinalSumcheckParams<F>,
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
    ) -> Self {
        // Compute the size-K table storing all eq(r_address, k) evaluations for
        // k \in {0, 1}^log(K)
        // TODO(moodlezoup): Can reuse from OutputSumcheck
        let eq_r_address = EqPolynomial::evals(&params.r_address);

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

        let inc = CommittedPolynomial::RamInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            trace,
            None,
        );

        // #[cfg(test)]
        // {
        //     let OutputSumcheckProverState {
        //         val_init,
        //         val_final,
        //         ..
        //     } = &output_sumcheck_prover_state;
        //     // Check that Val_init(r), wa(r, j), and Inc(j) are consistent with
        //     // the claim Val_final(r)
        //     let expected = val_final.final_sumcheck_claim();
        //     let actual = val_init.final_sumcheck_claim()
        //         + wa_r_address
        //             .par_iter()
        //             .enumerate()
        //             .map(|(j, wa)| inc.get_coeff(j) * wa)
        //             .sum::<F>();
        //     assert_eq!(
        //         expected, actual,
        //         "Val_final(r_address) ≠ Val_init(r_address) + \\sum_j wa(r_address, j) * Inc(j)"
        //     );
        // }

        Self { wa, inc, params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ValFinalSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "ValFinalSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let evals = (0..self.inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_evals = self
                    .inc
                    .sumcheck_evals_array::<VAL_FINAL_SUMCHECK_DEGREE_BOUND>(
                        j,
                        BindingOrder::LowToHigh,
                    );
                let wa_evals = self
                    .wa
                    .sumcheck_evals_array::<VAL_FINAL_SUMCHECK_DEGREE_BOUND>(
                        j,
                        BindingOrder::LowToHigh,
                    );
                [
                    inc_evals[0].mul_unreduced::<9>(wa_evals[0]),
                    inc_evals[1].mul_unreduced::<9>(wa_evals[1]),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); VAL_FINAL_SUMCHECK_DEGREE_BOUND],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            )
            .map(F::from_montgomery_reduce);

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "ValFinalSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _: usize) {
        self.inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.wa.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle_prime = self.params.normalize_opening_point(sumcheck_challenges);
        let r_address = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0;
        let wa_opening_point = OpeningPoint::new([&*r_address.r, &*r_cycle_prime.r].concat());

        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamValFinalEvaluation,
            r_cycle_prime.r,
            self.inc.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
            wa_opening_point,
            self.wa.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ValFinalSumcheckVerifier<F: JoltField> {
    params: ValFinalSumcheckParams<F>,
}

impl<F: JoltField> ValFinalSumcheckVerifier<F> {
    pub fn new(
        initial_ram_state: &[u64],
        program_io: &JoltDevice,
        trace_len: usize,
        ram_K: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = ValFinalSumcheckParams::new_from_verifier(
            initial_ram_state,
            program_io,
            trace_len,
            ram_K,
            opening_accumulator,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ValFinalSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let inc_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamValFinalEvaluation,
            )
            .1;
        let wa_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamRa,
                SumcheckId::RamValFinalEvaluation,
            )
            .1;
        inc_claim * wa_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let r_cycle_prime = self.params.normalize_opening_point(sumcheck_challenges);
        let r_address = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0;
        let wa_opening_point = OpeningPoint::new([&*r_address.r, &*r_cycle_prime.r].concat());

        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamValFinalEvaluation,
            r_cycle_prime.r,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
            wa_opening_point,
        );
    }
}
