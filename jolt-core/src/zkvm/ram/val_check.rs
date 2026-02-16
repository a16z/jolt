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
        claim_reductions::AdviceKind,
        config::{OneHotParams, ReadWriteConfig},
        ram::remap_address,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;

// RAM batched value sumcheck (ValEvaluation + ValFinal)
//
// This is a *single* log(T)-round sumcheck that checks RAM value consistency at one
// unified RAM address point `r_address` (after Stage 2 address-round alignment):
//
//   (1) Val(r_address, r_cycle) - Val_init(r_address)
//         = Σ_j inc(j) · wa(r_address, j) · LT(j, r_cycle)
//
//   (2) Val_final(r_address) - Val_init(r_address)
//         = Σ_j inc(j) · wa(r_address, j)
//
// using an explicit batching challenge γ sampled from the transcript *before* the sumcheck:
//
//   (LHS1) + γ·(LHS2) = Σ_j inc(j) · wa(r_address, j) · ( LT(j, r_cycle) + γ )
//
// Integration note:
// - Stage 4 must replace {ValEvaluation, ValFinal} with this instance and must sample γ in
//   the transcript at the same point on both prover and verifier.
// - This instance caches the RAM write-address (RA) opening only under
//   `SumcheckId::RamValCheck`.

/// Degree bound of the sumcheck round polynomials in [`RamValCheckSumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct RamValCheckSumcheckParams<F: JoltField> {
    pub T: usize,
    pub K: usize,

    /// Batching challenge γ.
    pub gamma: F,

    /// r = (r_address, r_cycle) from `RamVal`/`RamReadWriteChecking`.
    pub r_address: OpeningPoint<BIG_ENDIAN, F>,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,

    /// Val_init(r_address) evaluation to subtract on both LHS terms.
    pub init_eval: F,
}

impl<F: JoltField> RamValCheckSumcheckParams<F> {
    pub fn new_from_prover(
        one_hot_params: &OneHotParams,
        opening_accumulator: &ProverOpeningAccumulator<F>,
        initial_ram_state: &[u64],
        trace_len: usize,
        gamma: F,
    ) -> Self {
        let K = one_hot_params.ram_k;

        // (r_address, r_cycle) comes from RamVal/RamReadWriteChecking.
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, r_cycle) = r.split_at(K.log_2());

        // After Stage 2 alignment, OutputCheck's opening point should use the same address rounds
        // as RW-check, so these addresses coincide.
        #[cfg(debug_assertions)]
        {
            let (r_out, _) = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamValInit,
                SumcheckId::RamOutputCheck,
            );
            debug_assert_eq!(r_out.r, r_address.r);
        }

        let val_init: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());
        let init_eval = val_init.evaluate(&r_address.r);

        Self {
            T: trace_len,
            K,
            gamma,
            r_address,
            r_cycle,
            init_eval,
        }
    }

    pub fn new_from_verifier(
        initial_ram_state: &[u64],
        program_io: &JoltDevice,
        trace_len: usize,
        ram_K: usize,
        _rw_config: &ReadWriteConfig,
        gamma: F,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        // (r_address, r_cycle) from RamVal/RamReadWriteChecking.
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, r_cycle) = r.split_at(ram_K.log_2());

        #[cfg(debug_assertions)]
        {
            let r_out = opening_accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::RamValFinal,
                    SumcheckId::RamOutputCheck,
                )
                .0;
            debug_assert_eq!(r_out.r, r_address.r);
        }

        let n_memory_vars = ram_K.log_2();

        // Advice openings are cached under `SumcheckId::RamValCheck`. After Stage 2
        // alignment, `r_address` is the unique RAM address point used in Stage 4.
        let init_eval = {
            let untrusted_contribution = super::calculate_advice_memory_evaluation(
                opening_accumulator
                    .get_advice_opening(AdviceKind::Untrusted, SumcheckId::RamValCheck),
                (program_io.memory_layout.max_untrusted_advice_size as usize / 8)
                    .next_power_of_two()
                    .log_2(),
                program_io.memory_layout.untrusted_advice_start,
                &program_io.memory_layout,
                &r_address.r,
                n_memory_vars,
            );
            let trusted_contribution = super::calculate_advice_memory_evaluation(
                opening_accumulator
                    .get_advice_opening(AdviceKind::Trusted, SumcheckId::RamValCheck),
                (program_io.memory_layout.max_trusted_advice_size as usize / 8)
                    .next_power_of_two()
                    .log_2(),
                program_io.memory_layout.trusted_advice_start,
                &program_io.memory_layout,
                &r_address.r,
                n_memory_vars,
            );
            let val_init_public: MultilinearPolynomial<F> =
                MultilinearPolynomial::from(initial_ram_state.to_vec());
            untrusted_contribution + trusted_contribution + val_init_public.evaluate(&r_address.r)
        };

        Self {
            T: trace_len,
            K: ram_K,
            gamma,
            r_address,
            r_cycle,
            init_eval,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RamValCheckSumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, val_rw_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, val_final_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
        );
        (val_rw_claim - self.init_eval) + self.gamma * (val_final_claim - self.init_eval)
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct RamValCheckSumcheckProver<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: RaPolynomial<usize, F>,
    lt: LtPolynomial<F>,
    pub params: RamValCheckSumcheckParams<F>,
}

impl<F: JoltField> RamValCheckSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RamValCheckSumcheckProver::initialize")]
    pub fn initialize(
        params: RamValCheckSumcheckParams<F>,
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
    ) -> Self {
        // Shared witness indices for the write address at each cycle.
        let wa_indices: Vec<Option<usize>> = trace
            .par_iter()
            .map(|cycle| {
                remap_address(cycle.ram_access().address() as u64, memory_layout)
                    .map(|k| k as usize)
            })
            .collect();
        let wa_indices = Arc::new(wa_indices);

        // After Stage 2 alignment, both identities use the same address point.
        let eq = EqPolynomial::evals(&params.r_address.r);
        let wa = RaPolynomial::new(Arc::clone(&wa_indices), eq);

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

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for RamValCheckSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RamValCheckSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let gamma = self.params.gamma;

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

                // Term (1): inc * wa * lt (degree 3).
                let t1_at_1 = (inc_at_j_1 * wa_at_j_1).mul_unreduced::<9>(lt_at_j_1);
                let t1_at_2 = (inc_at_j_2 * wa_at_j_2).mul_unreduced::<9>(lt_at_j_2);
                let t1_at_inf = (inc_at_j_inf * wa_at_j_inf).mul_unreduced::<9>(lt_at_j_inf);

                // Term (2): γ * inc * wa (degree 2).
                //
                // IMPORTANT: In the cubic Toom reconstruction, the "∞ evaluation" corresponds to
                // the coefficient of X^3. The quadratic term contributes nothing to that
                // coefficient, so it must NOT be included in eval_at_inf.
                let t2_at_1 = (inc_at_j_1 * wa_at_j_1).mul_unreduced::<9>(gamma);
                let t2_at_2 = (inc_at_j_2 * wa_at_j_2).mul_unreduced::<9>(gamma);

                [t1_at_1 + t2_at_1, t1_at_2 + t2_at_2, t1_at_inf]
            })
            .reduce(
                || [F::Unreduced::zero(); DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            )
            .map(F::from_montgomery_reduce);

        let eval_at_0 = previous_claim - eval_at_1;
        UniPoly::from_evals_toom(&[eval_at_0, eval_at_1, eval_at_2, eval_at_inf])
    }

    #[tracing::instrument(skip_all, name = "RamValCheckSumcheckProver::ingest_challenge")]
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

        // r_address from RamVal/RamReadWriteChecking
        let r_rw = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            )
            .0;
        let (r_address, _) = r_rw.split_at(r_rw.len() - r_cycle_prime.len());
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        // Cache a single RAM RA opening under RamValCheck.
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValCheck,
            wa_opening_point,
            self.wa.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamValCheck,
            r_cycle_prime.r,
            self.inc.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct RamValCheckSumcheckVerifier<F: JoltField> {
    params: RamValCheckSumcheckParams<F>,
}

impl<F: JoltField> RamValCheckSumcheckVerifier<F> {
    pub fn new(
        initial_ram_state: &[u64],
        program_io: &JoltDevice,
        trace_len: usize,
        ram_K: usize,
        rw_config: &ReadWriteConfig,
        gamma: F,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = RamValCheckSumcheckParams::new_from_verifier(
            initial_ram_state,
            program_io,
            trace_len,
            ram_K,
            rw_config,
            gamma,
            opening_accumulator,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RamValCheckSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // LT(r_cycle′, r_cycle) term for (1), computed the same way as ValEvaluation verifier.
        let (r_val, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, r_cycle) = r_val.split_at(self.params.K.log_2());
        let r_cycle_prime = self.params.normalize_opening_point(sumcheck_challenges);

        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in zip(&r_cycle_prime.r, &r_cycle.r) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        let inc_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamValCheck,
            )
            .1;
        let wa_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamRa,
                SumcheckId::RamValCheck,
            )
            .1;

        inc_claim * wa_claim * (lt_eval + self.params.gamma)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let r_cycle_prime = self.params.normalize_opening_point(sumcheck_challenges);

        // r_address from RamVal/RamReadWriteChecking.
        let r_rw = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            )
            .0;
        let (r_address, _) = r_rw.split_at(r_rw.len() - r_cycle_prime.len());
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValCheck,
            wa_opening_point,
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamValCheck,
            r_cycle_prime.r,
        );
    }
}
