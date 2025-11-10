use num_traits::Zero;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        program_io_polynomial::ProgramIOPolynomial,
        range_mask_polynomial::RangeMaskPolynomial,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
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
use common::constants::RAM_START_ADDRESS;
use rayon::prelude::*;
use tracer::JoltDevice;

// RAM output sumchecks
//
// OutputSumcheck:
//   Proves the zero-check
//     Σ_k eq(r_address, k) ⋅ io_mask(k) ⋅ (Val_final(k) − Val_io(k)) = 0,
//   where:
//   - r_address is a random address challenge vector.
//   - io_mask is the MLE of the I/O-region indicator (1 on matching {0,1}-points).
//   - Val_final(k) is the final memory value at address k.
//   - Val_io(k) is the publicly claimed output value at address k.
//
// ValFinalSumcheck:
//   Proves the relation
//     Val_final(r_address) − Val_init(r_address) = Σ_j inc(r_address, j) ⋅ wa(r_address, j),
//   where:
//   - Val_init(r_address) is the initial memory value at r_address.
//   - inc is the MLE of the per-cycle increment; wa is the MLE of the write indicator.

/// Degree bonud of the sumcheck round polynomials in [`OutputSumcheckVerifier`].
const OUTPUT_SUMCHECK_DEGREE_BOUND: usize = 3;

/// Degree bound of the sumcheck round polynomials in [`ValFinalSumcheckVerifier`].
const VAL_FINAL_SUMCHECK_DEGREE_BOUND: usize = 2;

/// Sumcheck prover for [`OutputSumcheckVerifier`].
#[derive(Allocative)]
pub struct OutputSumcheckProver<F: JoltField> {
    /// Val(k, 0)
    val_init: MultilinearPolynomial<F>,
    /// The MLE of the final RAM state
    val_final: MultilinearPolynomial<F>,
    /// Val_io(k) = Val_final(k) if k is in the "IO" region of memory,
    /// and 0 otherwise.
    /// Equivalently, Val_io(k) = Val(k, T) * io_mask(k) for
    /// k \in {0, 1}^log(K)
    val_io: MultilinearPolynomial<F>,
    /// EQ(k, r_address)
    eq_poly: MultilinearPolynomial<F>,
    /// io_mask(k) serves as a "mask" for the IO region of memory,
    /// i.e. io_mask(k) = 1 if k is in the "IO" region of memory,
    /// and 0 otherwise.
    io_mask: MultilinearPolynomial<F>,
    #[allocative(skip)]
    params: OutputSumcheckParams<F>,
}

impl<F: JoltField> OutputSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "OutputSumcheckProver::gen")]
    pub fn gen(
        initial_ram_state: &[u64],
        final_ram_state: &[u64],
        state_manager: &mut StateManager<'_, F, impl CommitmentScheme<Field = F>>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let (_, _, _, program_io, _) = state_manager.get_prover_data();
        let params = OutputSumcheckParams::new(state_manager.ram_K, program_io, transcript);

        let K = final_ram_state.len();
        debug_assert_eq!(initial_ram_state.len(), final_ram_state.len());
        debug_assert!(K.is_power_of_two());

        // Compute the witness indices corresponding to the start and end of the IO
        // region of memory
        let io_start = remap_address(
            params.program_io.memory_layout.input_start,
            &params.program_io.memory_layout,
        )
        .unwrap() as usize;
        let io_end =
            remap_address(RAM_START_ADDRESS, &params.program_io.memory_layout).unwrap() as usize;

        // Compute Val_io by copying the relevant slice of Val_final
        let mut val_io = vec![0; K];
        val_io[io_start..io_end]
            .par_iter_mut()
            .zip(final_ram_state[io_start..io_end].par_iter())
            .for_each(|(dest, src)| *dest = *src);

        // Compute io_mask by setting the relevant coefficients to true
        let mut io_mask = vec![false; K];
        io_mask[io_start..io_end]
            .par_iter_mut()
            .for_each(|k| *k = true);

        Self {
            val_init: initial_ram_state.to_vec().into(),
            val_final: final_ram_state.to_vec().into(),
            val_io: val_io.into(),
            eq_poly: EqPolynomial::<F>::evals(&params.r_address).into(),
            io_mask: io_mask.into(),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for OutputSumcheckProver<F> {
    fn degree(&self) -> usize {
        OUTPUT_SUMCHECK_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "OutputSumcheckProver::compute_prover_message")]
    fn compute_prover_message(&mut self, _: usize, _previous_claim: F) -> Vec<F> {
        let Self {
            eq_poly,
            io_mask,
            val_final,
            val_io,
            ..
        } = self;

        (0..eq_poly.len() / 2)
            .into_par_iter()
            .map(|k| {
                let eq_evals = eq_poly.sumcheck_evals_array::<OUTPUT_SUMCHECK_DEGREE_BOUND>(
                    k,
                    BindingOrder::LowToHigh,
                );
                let io_mask_evals = io_mask.sumcheck_evals_array::<OUTPUT_SUMCHECK_DEGREE_BOUND>(
                    k,
                    BindingOrder::LowToHigh,
                );
                let val_final_evals = val_final
                    .sumcheck_evals_array::<OUTPUT_SUMCHECK_DEGREE_BOUND>(
                        k,
                        BindingOrder::LowToHigh,
                    );
                let val_io_evals = val_io.sumcheck_evals_array::<OUTPUT_SUMCHECK_DEGREE_BOUND>(
                    k,
                    BindingOrder::LowToHigh,
                );
                [
                    (eq_evals[0] * io_mask_evals[0])
                        .mul_unreduced::<9>(val_final_evals[0] - val_io_evals[0]),
                    (eq_evals[1] * io_mask_evals[1])
                        .mul_unreduced::<9>(val_final_evals[1] - val_io_evals[1]),
                    (eq_evals[2] * io_mask_evals[2])
                        .mul_unreduced::<9>(val_final_evals[2] - val_io_evals[2]),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); OUTPUT_SUMCHECK_DEGREE_BOUND],
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

    #[tracing::instrument(skip_all, name = "OutputSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _: usize) {
        // Bind address variable
        let Self {
            val_init,
            val_final,
            val_io,
            eq_poly,
            io_mask,
            ..
        } = self;

        // We bind Val_init here despite the fact that it is not used in `compute_prover_message`
        // because we'll need Val_init(r) in `ValFinalSumcheck`
        val_init.bind_parallel(r_j, BindingOrder::LowToHigh);
        val_final.bind_parallel(r_j, BindingOrder::LowToHigh);
        val_io.bind_parallel(r_j, BindingOrder::LowToHigh);
        eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        io_mask.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let Self {
            val_final,
            val_init,
            ..
        } = self;
        let opening_point = get_output_sumcheck_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
            val_final.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
            opening_point,
            val_init.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct OutputSumcheckVerifier<F: JoltField> {
    params: OutputSumcheckParams<F>,
}

impl<F: JoltField> OutputSumcheckVerifier<F> {
    pub fn new(ram_K: usize, program_io: &JoltDevice, transcript: &mut impl Transcript) -> Self {
        let params = OutputSumcheckParams::new(ram_K, program_io, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for OutputSumcheckVerifier<F> {
    fn degree(&self) -> usize {
        OUTPUT_SUMCHECK_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let val_final_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .1;

        let r_address = &self.params.r_address;
        let r_address_prime = &sumcheck_challenges[..self.params.r_address.len()]
            .iter()
            .cloned()
            .rev()
            .collect::<Vec<_>>();
        let program_io = &self.params.program_io;

        // let io_mask = RangeMaskPolynomial::new(
        //     remap_address(
        //         program_io.memory_layout.input_start,
        //         &program_io.memory_layout,
        //     )
        //     .unwrap()
        //     .into(),
        //     remap_address(RAM_START_ADDRESS, &program_io.memory_layout)
        //         .unwrap()
        //         .into(),
        // );
        let io_mask = RangeMaskPolynomial::<F>::new(
            remap_address(
                program_io.memory_layout.input_start,
                &program_io.memory_layout,
            )
            .unwrap() as u128,
            remap_address(RAM_START_ADDRESS, &program_io.memory_layout).unwrap() as u128,
        );
        let val_io = ProgramIOPolynomial::new(program_io);

        let eq_eval: F = EqPolynomial::<F>::mle(r_address, r_address_prime);
        let io_mask_eval = io_mask.evaluate_mle(r_address_prime);
        let val_io_eval: F = val_io.evaluate(r_address_prime);

        // Recall that the sumcheck expression is:
        //   0 = \sum_k eq(r_address, k) * io_range(k) * (Val_final(k) - Val_io(k))
        eq_eval * io_mask_eval * (val_final_claim - val_io_eval)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = get_output_sumcheck_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
            opening_point,
        );
    }
}

struct OutputSumcheckParams<F: JoltField> {
    K: usize,
    r_address: Vec<F::Challenge>,
    program_io: JoltDevice,
}

impl<F: JoltField> OutputSumcheckParams<F> {
    pub fn new(ram_K: usize, program_io: &JoltDevice, transcript: &mut impl Transcript) -> Self {
        let r_address = transcript.challenge_vector_optimized::<F>(ram_K.log_2());
        Self {
            K: ram_K,
            r_address,
            program_io: program_io.clone(),
        }
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }
}

fn get_output_sumcheck_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness()
}

#[derive(Allocative)]
pub struct ValFinalSumcheckProver<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
    #[allocative(skip)]
    params: ValFinalSumcheckParams<F>,
}

impl<F: JoltField> ValFinalSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "ValFinalSumcheckProver::gen")]
    pub fn gen(
        state_manager: &mut StateManager<'_, F, impl CommitmentScheme<Field = F>>,
        opening_accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
        let (preprocessing, _, trace, program_io, _) = state_manager.get_prover_data();
        let memory_layout = &program_io.memory_layout;
        let T = trace.len();

        let r_address = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0
            .r;

        // Compute the size-K table storing all eq(r_address, k) evaluations for
        // k \in {0, 1}^log(K)
        // TODO(moodlezoup): Can reuse from OutputSumcheck
        let eq_r_address = EqPolynomial::evals(&r_address);

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

        let inc =
            CommittedPolynomial::RamInc.generate_witness(preprocessing, trace, state_manager.ram_d);

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

        let val_init_eval = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValInit,
                SumcheckId::RamOutputCheck,
            )
            .1;

        let params = ValFinalSumcheckParams { T, val_init_eval };

        Self { wa, inc, params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ValFinalSumcheckProver<F> {
    fn degree(&self) -> usize {
        VAL_FINAL_SUMCHECK_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(skip_all, name = "ValFinalSumcheckProver::compute_prover_message")]
    fn compute_prover_message(&mut self, _: usize, _previous_claim: F) -> Vec<F> {
        (0..self.inc.len() / 2)
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
            .into_iter()
            .map(F::from_montgomery_reduce)
            .collect()
    }

    #[tracing::instrument(skip_all, name = "ValFinalSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _: usize) {
        self.inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.wa.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle_prime = get_val_final_sumcheck_opening_point::<F>(sumcheck_challenges);
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
        let r_address = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0
            .r;

        let n_memory_vars = ram_K.log_2();

        {
            // Verify that val_evaluation and output_check use the same opening point for initial_ram_state.
            // This allows us to reuse a single untrusted_advice opening instead of providing two.
            let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            );
            let (r_address_val_evaluation, _) = r.split_at(n_memory_vars);
            assert_eq!(r_address_val_evaluation.r, r_address);
        }

        let untrusted_advice_contribution = super::calculate_advice_memory_evaluation(
            opening_accumulator.get_untrusted_advice_opening(),
            (program_io.memory_layout.max_untrusted_advice_size as usize / 8)
                .next_power_of_two()
                .log_2(),
            program_io.memory_layout.untrusted_advice_start,
            &program_io.memory_layout,
            &r_address,
            n_memory_vars,
        );

        let trusted_advice_contribution = super::calculate_advice_memory_evaluation(
            opening_accumulator.get_trusted_advice_opening(),
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

        let params = ValFinalSumcheckParams {
            T: trace_len,
            val_init_eval,
        };

        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ValFinalSumcheckVerifier<F> {
    fn degree(&self) -> usize {
        VAL_FINAL_SUMCHECK_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
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
        let r_cycle_prime = get_val_final_sumcheck_opening_point::<F>(sumcheck_challenges);
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

struct ValFinalSumcheckParams<F: JoltField> {
    T: usize,
    val_init_eval: F,
}

impl<F: JoltField> ValFinalSumcheckParams<F> {
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
}

fn get_val_final_sumcheck_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness()
}
