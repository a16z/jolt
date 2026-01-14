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

// RAM fused value sumcheck (ValEvaluation + ValFinal)
//
// This is a *single* log(T)-round sumcheck that batches the two RAM value identities:
//
//   (1) Val(r_address_rw, r_cycle) - Val_init(r_address_rw)
//         = Σ_j inc(j) · wa(r_address_rw, j) · LT(j, r_cycle)
//
//   (2) Val_final(r_address_raf) - Val_init(r_address_raf)
//         = Σ_j inc(j) · wa(r_address_raf, j)
//
// using an explicit fusion challenge γ sampled from the transcript *before* the sumcheck:
//
//   LHS1 + γ·LHS2 = Σ_j inc(j) · ( wa_rw(j)·LT(j,r_cycle) + γ·wa_raf(j) )
//
// Integration note:
// - Stage 4 must replace {ValEvaluation, ValFinal} with this instance and must sample γ in
//   the transcript at the same point on both prover and verifier.
// - This instance still caches openings under `SumcheckId::RamValEvaluation` and
//   `SumcheckId::RamValFinalEvaluation`, so downstream protocols (e.g. RA reduction) are unchanged.

/// Degree bound of the sumcheck round polynomials in [`RamValFusedSumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct RamValFusedSumcheckParams<F: JoltField> {
    pub T: usize,
    pub K: usize,

    /// Fusion challenge γ.
    pub gamma: F,

    /// r = (r_address_rw, r_cycle) from `RamVal`/`RamReadWriteChecking`.
    pub r_address_rw: OpeningPoint<BIG_ENDIAN, F>,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,

    /// r_address_raf from `RamValFinal`/`RamOutputCheck`.
    pub r_address_raf: OpeningPoint<BIG_ENDIAN, F>,

    /// Val_init(r_address_rw) evaluation to subtract on the LHS of (1).
    pub init_eval_rw: F,
    /// Val_init(r_address_raf) evaluation to subtract on the LHS of (2).
    pub init_eval_raf: F,
}

impl<F: JoltField> RamValFusedSumcheckParams<F> {
    pub fn new_from_prover(
        one_hot_params: &OneHotParams,
        opening_accumulator: &ProverOpeningAccumulator<F>,
        initial_ram_state: &[u64],
        trace_len: usize,
        gamma: F,
    ) -> Self {
        let K = one_hot_params.ram_k;

        // (r_address_rw, r_cycle) comes from RamVal/RamReadWriteChecking.
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address_rw, r_cycle) = r.split_at(K.log_2());

        // r_address_raf comes from RamValInit/RamOutputCheck (same point as RamValFinal).
        let (r_address_raf, init_eval_raf) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
        );

        let val_init: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());
        let init_eval_rw = val_init.evaluate(&r_address_rw.r);

        Self {
            T: trace_len,
            K,
            gamma,
            r_address_rw,
            r_cycle,
            r_address_raf,
            init_eval_rw,
            init_eval_raf,
        }
    }

    pub fn new_from_verifier(
        initial_ram_state: &[u64],
        program_io: &JoltDevice,
        trace_len: usize,
        ram_K: usize,
        rw_config: &ReadWriteConfig,
        gamma: F,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        // (r_address_rw, r_cycle) from RamVal/RamReadWriteChecking.
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address_rw, r_cycle) = r.split_at(ram_K.log_2());

        // r_address_raf from RamValFinal/RamOutputCheck.
        let r_address_raf = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0;

        let n_memory_vars = ram_K.log_2();
        let log_T = trace_len.log_2();

        // Advice openings are always available under `RamValEvaluation`.
        let init_eval_rw = {
            let untrusted_contribution = super::calculate_advice_memory_evaluation(
                opening_accumulator
                    .get_advice_opening(AdviceKind::Untrusted, SumcheckId::RamValEvaluation),
                (program_io.memory_layout.max_untrusted_advice_size as usize / 8)
                    .next_power_of_two()
                    .log_2(),
                program_io.memory_layout.untrusted_advice_start,
                &program_io.memory_layout,
                &r_address_rw.r,
                n_memory_vars,
            );
            let trusted_contribution = super::calculate_advice_memory_evaluation(
                opening_accumulator
                    .get_advice_opening(AdviceKind::Trusted, SumcheckId::RamValEvaluation),
                (program_io.memory_layout.max_trusted_advice_size as usize / 8)
                    .next_power_of_two()
                    .log_2(),
                program_io.memory_layout.trusted_advice_start,
                &program_io.memory_layout,
                &r_address_rw.r,
                n_memory_vars,
            );
            let val_init_public: MultilinearPolynomial<F> =
                MultilinearPolynomial::from(initial_ram_state.to_vec());
            untrusted_contribution
                + trusted_contribution
                + val_init_public.evaluate(&r_address_rw.r)
        };

        // For r_address_raf, advice may be opened only once (at RamValEvaluation) or twice
        // (also at RamValFinalEvaluation), depending on rw_config.
        let advice_sumcheck_id = if rw_config.needs_single_advice_opening(log_T) {
            SumcheckId::RamValEvaluation
        } else {
            SumcheckId::RamValFinalEvaluation
        };

        let init_eval_raf = {
            let untrusted_contribution = super::calculate_advice_memory_evaluation(
                opening_accumulator.get_advice_opening(AdviceKind::Untrusted, advice_sumcheck_id),
                (program_io.memory_layout.max_untrusted_advice_size as usize / 8)
                    .next_power_of_two()
                    .log_2(),
                program_io.memory_layout.untrusted_advice_start,
                &program_io.memory_layout,
                &r_address_raf.r,
                n_memory_vars,
            );
            let trusted_contribution = super::calculate_advice_memory_evaluation(
                opening_accumulator.get_advice_opening(AdviceKind::Trusted, advice_sumcheck_id),
                (program_io.memory_layout.max_trusted_advice_size as usize / 8)
                    .next_power_of_two()
                    .log_2(),
                program_io.memory_layout.trusted_advice_start,
                &program_io.memory_layout,
                &r_address_raf.r,
                n_memory_vars,
            );
            let val_init_public: MultilinearPolynomial<F> =
                MultilinearPolynomial::from(initial_ram_state.to_vec());
            untrusted_contribution
                + trusted_contribution
                + val_init_public.evaluate(&r_address_raf.r)
        };

        Self {
            T: trace_len,
            K: ram_K,
            gamma,
            r_address_rw,
            r_cycle,
            r_address_raf,
            init_eval_rw,
            init_eval_raf,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RamValFusedSumcheckParams<F> {
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
        (val_rw_claim - self.init_eval_rw) + self.gamma * (val_final_claim - self.init_eval_raf)
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct RamValFusedSumcheckProver<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa_rw: RaPolynomial<usize, F>,
    wa_raf: RaPolynomial<usize, F>,
    lt: LtPolynomial<F>,
    pub params: RamValFusedSumcheckParams<F>,
}

impl<F: JoltField> RamValFusedSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RamValFusedSumcheckProver::initialize")]
    pub fn initialize(
        params: RamValFusedSumcheckParams<F>,
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

        // Build two RA polynomials with different eq(r_address, ·) tables but shared indices.
        let eq_rw = EqPolynomial::evals(&params.r_address_rw.r);
        let eq_raf = EqPolynomial::evals(&params.r_address_raf.r);
        let wa_rw = RaPolynomial::new(Arc::clone(&wa_indices), eq_rw);
        let wa_raf = RaPolynomial::new(Arc::clone(&wa_indices), eq_raf);

        let inc = CommittedPolynomial::RamInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            trace,
            None,
        );
        let lt = LtPolynomial::new(&params.r_cycle);

        Self {
            inc,
            wa_rw,
            wa_raf,
            lt,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RamValFusedSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RamValFusedSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let gamma = self.params.gamma;

        let [eval_at_1, eval_at_2, eval_at_inf] = (0..self.inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_at_j_1 = self.inc.get_bound_coeff(j * 2 + 1);
                let inc_at_j_inf = inc_at_j_1 - self.inc.get_bound_coeff(j * 2);
                let inc_at_j_2 = inc_at_j_1 + inc_at_j_inf;

                let wa_rw_at_j_1 = self.wa_rw.get_bound_coeff(j * 2 + 1);
                let wa_rw_at_j_inf = wa_rw_at_j_1 - self.wa_rw.get_bound_coeff(j * 2);
                let wa_rw_at_j_2 = wa_rw_at_j_1 + wa_rw_at_j_inf;

                let wa_raf_at_j_1 = self.wa_raf.get_bound_coeff(j * 2 + 1);
                let wa_raf_at_j_inf = wa_raf_at_j_1 - self.wa_raf.get_bound_coeff(j * 2);
                let wa_raf_at_j_2 = wa_raf_at_j_1 + wa_raf_at_j_inf;

                let lt_at_j_1 = self.lt.get_bound_coeff(j * 2 + 1);
                let lt_at_j_inf = lt_at_j_1 - self.lt.get_bound_coeff(j * 2);
                let lt_at_j_2 = lt_at_j_1 + lt_at_j_inf;

                // Term (1): inc * wa_rw * lt (degree 3).
                let t1_at_1 = (inc_at_j_1 * wa_rw_at_j_1).mul_unreduced::<9>(lt_at_j_1);
                let t1_at_2 = (inc_at_j_2 * wa_rw_at_j_2).mul_unreduced::<9>(lt_at_j_2);
                let t1_at_inf = (inc_at_j_inf * wa_rw_at_j_inf).mul_unreduced::<9>(lt_at_j_inf);

                // Term (2): γ * inc * wa_raf (degree 2).
                //
                // IMPORTANT: In the cubic Toom reconstruction, the "∞ evaluation" corresponds to
                // the coefficient of X^3. The quadratic term contributes nothing to that
                // coefficient, so it must NOT be included in eval_at_inf.
                let t2_at_1 = (inc_at_j_1 * wa_raf_at_j_1).mul_unreduced::<9>(gamma);
                let t2_at_2 = (inc_at_j_2 * wa_raf_at_j_2).mul_unreduced::<9>(gamma);

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

    #[tracing::instrument(skip_all, name = "RamValFusedSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.wa_rw.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.wa_raf.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.lt.bind(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle_prime = self.params.normalize_opening_point(sumcheck_challenges);

        // r_address_rw from RamVal/RamReadWriteChecking
        let r_rw = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            )
            .0;
        let (r_address_rw, _) = r_rw.split_at(r_rw.len() - r_cycle_prime.len());
        let wa_rw_opening_point =
            OpeningPoint::new([r_address_rw.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        // r_address_raf from RamValFinal/RamOutputCheck
        let r_address_raf = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0;
        let wa_raf_opening_point =
            OpeningPoint::new([&*r_address_raf.r, &*r_cycle_prime.r].concat());

        // Cache openings exactly as the two separate sumchecks did.
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValEvaluation,
            wa_rw_opening_point,
            self.wa_rw.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
            r_cycle_prime.r.clone(),
            self.inc.final_sumcheck_claim(),
        );

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
            wa_raf_opening_point,
            self.wa_raf.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct RamValFusedSumcheckVerifier<F: JoltField> {
    params: RamValFusedSumcheckParams<F>,
}

impl<F: JoltField> RamValFusedSumcheckVerifier<F> {
    pub fn new(
        initial_ram_state: &[u64],
        program_io: &JoltDevice,
        trace_len: usize,
        ram_K: usize,
        rw_config: &ReadWriteConfig,
        gamma: F,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = RamValFusedSumcheckParams::new_from_verifier(
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
    for RamValFusedSumcheckVerifier<F>
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
                SumcheckId::RamValEvaluation,
            )
            .1;
        let wa_rw_claim = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamValEvaluation)
            .1;
        let wa_raf_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamRa,
                SumcheckId::RamValFinalEvaluation,
            )
            .1;

        inc_claim * wa_rw_claim * lt_eval + self.params.gamma * inc_claim * wa_raf_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let r_cycle_prime = self.params.normalize_opening_point(sumcheck_challenges);

        // r_address_rw from RamVal/RamReadWriteChecking.
        let r_rw = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            )
            .0;
        let (r_address_rw, _) = r_rw.split_at(r_rw.len() - r_cycle_prime.len());
        let wa_rw_opening_point =
            OpeningPoint::new([r_address_rw.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        // r_address_raf from RamValFinal/RamOutputCheck.
        let r_address_raf = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0;
        let wa_raf_opening_point =
            OpeningPoint::new([&*r_address_raf.r, &*r_cycle_prime.r].concat());

        // Mirror the two separate sumchecks' cached openings.
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValEvaluation,
            wa_rw_opening_point,
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
            r_cycle_prime.r.clone(),
        );

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
            wa_raf_opening_point,
        );
    }
}
