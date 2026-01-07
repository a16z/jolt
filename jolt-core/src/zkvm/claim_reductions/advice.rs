//! Two-phase advice claim reduction (Stage 6 cycle → Stage 7 address)
//!
//! This module generalizes the previous single-phase `AdviceClaimReduction` so that trusted and
//! untrusted advice can be committed as an arbitrary Dory matrix `2^{nu_a} x 2^{sigma_a}` (balanced
//! by default), while still keeping a **single Stage 8 Dory opening** at the unified Dory point.
//!
//! ## Variable order (Dory order)
//! Dory evaluates at:
//! - `point_dory = reverse(opening_point_be) = [cycle6_le || addr7_le]`
//! - `col_coords = point_dory[0..sigma_main]`
//! - `row_coords = point_dory[sigma_main..]`
//!
//! For an advice matrix embedded as the **top-left block** `2^{nu_a} x 2^{sigma_a}`, the *native*
//! advice evaluation point (in Dory order, LSB-first) is:
//! - `advice_cols = col_coords[0..sigma_a]`
//! - `advice_rows = row_coords[0..nu_a]`
//! - `advice_point = [advice_cols || advice_rows]`
//!
//! In our current pipeline, `cycle` coordinates come from Stage 6 and `addr` coordinates come from
//! Stage 7. When `row_coords[0..nu_a]` crosses into the address segment, we must consume
//! coordinates from *both* stages; hence the 2-phase reduction:
//! - **Phase 1 (Stage 6)**: bind the cycle-derived advice coordinates and output an intermediate
//!   scalar claim `C_mid`.
//! - **Phase 2 (Stage 7)**: resume from `C_mid`, bind the address-derived advice coordinates, and
//!   cache the final advice opening `AdviceMLE(advice_point)` for batching into Stage 8.
//!
//! ## Dummy-gap scaling (within Stage 6)
//! When `nu_a > 0`, the advice row bits that come from `cycle6_le` start at `cycle6_le[sigma_main]`,
//! while the advice column bits are `cycle6_le[0..sigma_a]`. If `sigma_a < sigma_main`, Phase 1
//! must *pass through* the intervening cycle coordinates `cycle6_le[sigma_a .. sigma_main)` that
//! are **not** advice variables (they are handled by the Stage 7 embedding selector).
//!
//! We handle this without modifying the generic batched sumcheck by treating those intervening
//! rounds as **dummy internal rounds** (constant univariates), and maintaining a running scaling
//! factor `2^{-dummy_done}` so the per-round univariates remain consistent.
//!
//! Trusted and untrusted advice run as **separate** sumcheck instances (each may have different
//! dimensions).
//!

use crate::field::JoltField;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::config::OneHotConfig;
use crate::zkvm::witness::CommittedPolynomial;
use allocative::Allocative;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;

const DEGREE_BOUND: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Allocative)]
pub enum AdviceKind {
    Trusted,
    Untrusted,
}

#[derive(Clone, Allocative)]
pub struct AdviceClaimReductionPhase1Params<F: JoltField> {
    pub kind: AdviceKind,
    pub gamma: F,
    pub advice_vars: usize,
    pub sigma_a: usize, // number of advice columns
    pub nu_a: usize,    // number of advice rows
    pub single_opening: bool,
    pub log_k_chunk: usize,
    pub log_t: usize,
    pub sigma_main: usize, // number of main columns
    pub nu_a_cycle: usize, // number of advice rows that come from the cycle variables
    pub nu_a_addr: usize,  // number of advice rows that come from the address variables
    /// Dummy rounds are `[dummy_start, dummy_end)` within the Phase 1 local round index space.
    pub dummy_start: usize,
    pub dummy_end: usize,
    pub r_val_eval: OpeningPoint<BIG_ENDIAN, F>,
    pub r_val_final: Option<OpeningPoint<BIG_ENDIAN, F>>,
}

impl<F: JoltField> AdviceClaimReductionPhase1Params<F> {
    pub fn new(
        kind: AdviceKind,
        memory_layout: &MemoryLayout,
        trace_len: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        single_opening: bool,
    ) -> Option<Self> {
        let max_advice_size_bytes = match kind {
            AdviceKind::Trusted => memory_layout.max_trusted_advice_size as usize,
            AdviceKind::Untrusted => memory_layout.max_untrusted_advice_size as usize,
        };

        let log_t = trace_len.log_2();
        let log_k_chunk = OneHotConfig::new(log_t).log_k_chunk as usize;
        let (sigma_main, _nu_main) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);

        let r_val_eval = accumulator
            .get_advice_opening(kind, SumcheckId::RamValEvaluation)
            .map(|(p, _)| p)?;
        let r_val_final = if single_opening {
            None
        } else {
            accumulator
                .get_advice_opening(kind, SumcheckId::RamValFinalEvaluation)
                .map(|(p, _)| p)
        };
        let (r_val_eval, r_val_final) = (r_val_eval, r_val_final);

        let gamma: F = transcript.challenge_scalar();

        let (sigma_a, nu_a) = DoryGlobals::advice_sigma_nu_from_max_bytes(max_advice_size_bytes);
        let advice_vars = sigma_a + nu_a;

        let row_cycle = DoryGlobals::cycle_row_len(log_t, sigma_main);
        let nu_a_cycle = std::cmp::min(nu_a, row_cycle);
        let nu_a_addr = nu_a - nu_a_cycle;

        // Phase 1 only needs to traverse the cycle gap if we actually need cycle-derived row bits.
        let (dummy_start, dummy_end) = if nu_a_cycle == 0 {
            (0, 0)
        } else {
            (sigma_a, sigma_main)
        };

        Some(Self {
            kind,
            gamma,
            advice_vars,
            sigma_a,
            nu_a,
            single_opening,
            log_k_chunk,
            log_t,
            sigma_main,
            nu_a_cycle,
            nu_a_addr,
            dummy_start,
            dummy_end,
            r_val_eval,
            r_val_final,
        })
    }

    #[inline]
    fn is_dummy_round(&self, local_round: usize) -> bool {
        self.dummy_start < self.dummy_end
            && local_round >= self.dummy_start
            && local_round < self.dummy_end
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for AdviceClaimReductionPhase1Params<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let mut claim = F::zero();
        if let Some((_, eval)) =
            accumulator.get_advice_opening(self.kind, SumcheckId::RamValEvaluation)
        {
            claim += eval;
        }
        if !self.single_opening {
            if let Some((_, final_eval)) =
                accumulator.get_advice_opening(self.kind, SumcheckId::RamValFinalEvaluation)
            {
                claim += self.gamma * final_eval;
            }
        }
        claim
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        if self.nu_a_cycle == 0 {
            // Only need the low column bits; no need to traverse the cycle gap.
            self.sigma_a
        } else {
            // Need to reach cycle row bits at index sigma_main.
            self.sigma_main + self.nu_a_cycle
        }
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // Instance-local rounds are interpreted as little-endian in time order.
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct AdviceClaimReductionPhase1Prover<F: JoltField> {
    params: AdviceClaimReductionPhase1Params<F>,
    advice_poly: MultilinearPolynomial<F>,
    eq_poly: MultilinearPolynomial<F>,
    /// Maintains the running internal scaling factor 2^{-dummy_done}.
    scale: F,
    inv_scale: F,
    /// Constant scaling for trailing dummy rounds *after* the active window in the batched Stage 6
    /// sumcheck (i.e., dummy cycle variables beyond this instance's `num_rounds()`).
    after_scale: F,
    after_inv_scale: F,
    two_inv: F,
}

impl<F: JoltField> AdviceClaimReductionPhase1Prover<F> {
    pub fn initialize(
        params: AdviceClaimReductionPhase1Params<F>,
        advice_poly: MultilinearPolynomial<F>,
    ) -> Self {
        let gamma = params.gamma;
        let eq_eval = EqPolynomial::evals(&params.r_val_eval.r);
        let eq_poly = if params.single_opening {
            MultilinearPolynomial::from(eq_eval)
        } else {
            let r_final = params
                .r_val_final
                .as_ref()
                .expect("r_val_final must exist when !single_opening");
            let eq_final = EqPolynomial::evals_with_scaling(&r_final.r, Some(gamma));
            let combined: Vec<F> = eq_eval
                .par_iter()
                .zip(eq_final.par_iter())
                .map(|(e1, e2)| *e1 + e2)
                .collect();
            MultilinearPolynomial::from(combined)
        };

        let two_inv = F::from_u64(2).inverse().unwrap();
        let dummy_after = params.log_t.saturating_sub(params.num_rounds());
        let after_scale = F::one().mul_pow_2(dummy_after);
        let after_inv_scale = after_scale.inverse().unwrap();
        Self {
            params,
            advice_poly,
            eq_poly,
            scale: F::one(),
            inv_scale: F::one(),
            after_scale,
            after_inv_scale,
            two_inv,
        }
    }

    fn compute_message_unscaled(&mut self, previous_claim_unscaled: F) -> UniPoly<F> {
        let half = self.advice_poly.len() / 2;
        let evals: [F; DEGREE_BOUND] = (0..half)
            .into_par_iter()
            .map(|j| {
                let a_evals = self
                    .advice_poly
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_evals = self
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                let mut out = [F::zero(); DEGREE_BOUND];
                for i in 0..DEGREE_BOUND {
                    out[i] = a_evals[i] * eq_evals[i];
                }
                out
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, arr| {
                    acc.par_iter_mut()
                        .zip(arr.par_iter())
                        .for_each(|(a, b)| *a += *b);
                    acc
                },
            );
        UniPoly::from_evals_and_hint(previous_claim_unscaled, &evals)
    }

    fn bind_real_var(&mut self, r_j: F::Challenge) {
        self.advice_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn step_dummy(&mut self) {
        // Each dummy internal round halves the running claim; equivalently, we multiply the
        // scaling factor by 1/2.
        self.scale *= self.two_inv;
        self.inv_scale *= F::from_u64(2);
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for AdviceClaimReductionPhase1Prover<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if self.params.is_dummy_round(round) {
            // Dummy internal variable: constant univariate with H(0)=H(1)=previous_claim/2.
            UniPoly::from_coeff(vec![previous_claim * self.two_inv])
        } else {
            // Account for (1) internal dummy rounds already traversed (scale/inv_scale) and
            // (2) trailing dummy rounds after this instance's active window in the batched sumcheck.
            let prev_unscaled = previous_claim * self.inv_scale * self.after_inv_scale;
            let poly_unscaled = self.compute_message_unscaled(prev_unscaled);
            poly_unscaled * self.scale * self.after_scale
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if self.params.is_dummy_round(round) {
            self.step_dummy();
        } else {
            self.bind_real_var(r_j);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Compute the intermediate claim C_mid = (2^{-gap}) * Σ_y advice(y) * eq(y),
        // where y are the remaining (address-derived) advice row variables.
        let len = self.advice_poly.len();
        debug_assert_eq!(len, self.eq_poly.len());

        let mut sum = F::zero();
        for i in 0..len {
            sum += self.advice_poly.get_bound_coeff(i) * self.eq_poly.get_bound_coeff(i);
        }
        let c_mid = sum * self.scale;

        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        // Append C_mid to transcript for explicit Fiat-Shamir binding (defensive).
        // While C_mid is already implicitly determined by the sumcheck univariates,
        // explicit transcript binding makes the security argument clearer.
        match self.params.kind {
            AdviceKind::Trusted => accumulator.append_trusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase1,
                opening_point,
                c_mid,
            ),
            AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase1,
                opening_point,
                c_mid,
            ),
        }

        // If there is no Phase 2 (all advice row bits come from cycle), cache the final advice opening
        // directly here under `SumcheckId::AdviceClaimReduction` so Stage 7 can scale/embed it.
        if self.params.nu_a_addr == 0 {
            let mut advice_le: Vec<F::Challenge> = Vec::with_capacity(self.params.advice_vars);
            // col bits are the first sigma_a cycle coords
            advice_le.extend_from_slice(&sumcheck_challenges[0..self.params.sigma_a]);
            // row bits (from cycle) start at sigma_main
            if self.params.nu_a_cycle > 0 {
                advice_le.extend_from_slice(
                    &sumcheck_challenges
                        [self.params.sigma_main..self.params.sigma_main + self.params.nu_a_cycle],
                );
            }
            let advice_point: OpeningPoint<BIG_ENDIAN, F> =
                OpeningPoint::<LITTLE_ENDIAN, F>::new(advice_le).match_endianness();
            let advice_claim = self.advice_poly.final_sumcheck_claim();

            match self.params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReductionPhase2,
                    advice_point,
                    advice_claim,
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReductionPhase2,
                    advice_point,
                    advice_claim,
                ),
            }
        }
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        // Align Phase 1 to the *start* of Booleanity's cycle segment, so local rounds correspond
        // to low Dory column bits in the unified point ordering.
        let booleanity_rounds = self.params.log_k_chunk + self.params.log_t;
        let booleanity_offset = max_num_rounds - booleanity_rounds;
        booleanity_offset + self.params.log_k_chunk
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct AdviceClaimReductionPhase1Verifier<F: JoltField> {
    params: AdviceClaimReductionPhase1Params<F>,
}

impl<F: JoltField> AdviceClaimReductionPhase1Verifier<F> {
    pub fn new(
        kind: AdviceKind,
        memory_layout: &MemoryLayout,
        trace_len: usize,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        single_opening: bool,
    ) -> Option<Self> {
        let params = AdviceClaimReductionPhase1Params::new(
            kind,
            memory_layout,
            trace_len,
            accumulator,
            transcript,
            single_opening,
        )?;
        Some(Self { params })
    }

    pub fn gamma(&self) -> F {
        self.params.gamma
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for AdviceClaimReductionPhase1Verifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let (key, label) = match self.params.kind {
            AdviceKind::Trusted => (
                SumcheckId::AdviceClaimReductionPhase1,
                "Trusted Phase1 intermediate claim",
            ),
            AdviceKind::Untrusted => (
                SumcheckId::AdviceClaimReductionPhase1,
                "Untrusted Phase1 intermediate claim",
            ),
        };

        accumulator
            .get_advice_opening(self.params.kind, key)
            .unwrap_or_else(|| panic!("{label} not found"))
            .1
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = SumcheckInstanceVerifier::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        // Append C_mid to transcript for explicit Fiat-Shamir binding (defensive).
        match self.params.kind {
            AdviceKind::Trusted => accumulator.append_trusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase1,
                opening_point,
            ),
            AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase1,
                opening_point,
            ),
        }

        // If Phase 2 is absent (nu_a_addr == 0), Phase 1 cached the final advice opening in Stage 6.
        // Populate its opening point now so downstream stages can extract it.
        if self.params.nu_a_addr == 0 {
            let mut advice_le: Vec<F::Challenge> = Vec::with_capacity(self.params.advice_vars);
            advice_le.extend_from_slice(&sumcheck_challenges[0..self.params.sigma_a]);
            if self.params.nu_a_cycle > 0 {
                advice_le.extend_from_slice(
                    &sumcheck_challenges
                        [self.params.sigma_main..self.params.sigma_main + self.params.nu_a_cycle],
                );
            }
            let advice_point: OpeningPoint<BIG_ENDIAN, F> =
                OpeningPoint::<LITTLE_ENDIAN, F>::new(advice_le).match_endianness();
            match self.params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReductionPhase2,
                    advice_point,
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReductionPhase2,
                    advice_point,
                ),
            }
        }
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        let booleanity_rounds = self.params.log_k_chunk + self.params.log_t;
        let booleanity_offset = max_num_rounds - booleanity_rounds;
        booleanity_offset + self.params.log_k_chunk
    }
}

#[derive(Clone, Allocative)]
pub struct AdviceClaimReductionPhase2Params<F: JoltField> {
    pub kind: AdviceKind,
    pub gamma: F,
    pub advice_vars: usize,
    pub sigma_a: usize, // number of advice columns
    pub nu_a: usize,    // number of advice rows
    pub single_opening: bool,
    pub log_k_chunk: usize,
    pub log_t: usize,
    pub sigma_main: usize, // number of main columns
    pub nu_a_cycle: usize, // number of advice rows that come from the cycle variables
    pub nu_a_addr: usize,  // number of advice rows that come from the address variables
    /// Constant scaling factor carried from Phase 1 (equals 2^{-gap_len}).
    pub scale: F,
    pub inv_scale: F,
    /// Cycle-derived prefix (Dory order) used to pre-bind advice vars in Phase 2:
    /// `[col_bits (sigma_a) || row_cycle_bits (nu_a_cycle)]` (little-endian).
    pub cycle_bind_le: Vec<F::Challenge>,
    pub r_val_eval: OpeningPoint<BIG_ENDIAN, F>,
    pub r_val_final: Option<OpeningPoint<BIG_ENDIAN, F>>,
}

impl<F: JoltField> AdviceClaimReductionPhase2Params<F> {
    pub fn new(
        kind: AdviceKind,
        memory_layout: &MemoryLayout,
        trace_len: usize,
        gamma: F,
        accumulator: &dyn OpeningAccumulator<F>,
        single_opening: bool,
    ) -> Option<Self> {
        let max_advice_size_bytes = match kind {
            AdviceKind::Trusted => memory_layout.max_trusted_advice_size as usize,
            AdviceKind::Untrusted => memory_layout.max_untrusted_advice_size as usize,
        };
        let log_t = trace_len.log_2();
        let log_k_chunk = OneHotConfig::new(log_t).log_k_chunk as usize;
        let (sigma_main, _nu_main) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);

        let r_val_eval = accumulator
            .get_advice_opening(kind, SumcheckId::RamValEvaluation)
            .map(|(p, _)| p)?;
        let r_val_final = if single_opening {
            None
        } else {
            accumulator
                .get_advice_opening(kind, SumcheckId::RamValFinalEvaluation)
                .map(|(p, _)| p)
        };
        let (r_val_eval, r_val_final) = (r_val_eval, r_val_final);

        let (sigma_a, nu_a) = DoryGlobals::advice_sigma_nu_from_max_bytes(max_advice_size_bytes);
        let advice_vars = sigma_a + nu_a;

        let row_cycle = DoryGlobals::cycle_row_len(log_t, sigma_main);
        let nu_a_cycle = std::cmp::min(nu_a, row_cycle);
        let nu_a_addr = nu_a - nu_a_cycle;

        // If no address-derived advice vars, Phase 2 is not needed.
        if nu_a_addr == 0 {
            return None;
        }

        // Gap dummy rounds exist only if Phase 1 had to traverse from sigma_a to sigma_main.
        let gap_len = if nu_a_cycle == 0 {
            0
        } else {
            sigma_main.saturating_sub(sigma_a)
        };

        let two_inv = F::from_u64(2).inverse().unwrap();
        let scale = (0..gap_len).fold(F::one(), |acc, _| acc * two_inv);
        let inv_scale = (0..gap_len).fold(F::one(), |acc, _| acc * F::from_u64(2));

        // Extract cycle6_le from Booleanity's unified opening point in the accumulator.
        let (unified_point, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::Booleanity,
        );
        let r_cycle_be = &unified_point.r[log_k_chunk..];
        let mut cycle_le = r_cycle_be.to_vec();
        cycle_le.reverse();

        let mut cycle_bind_le = Vec::with_capacity(sigma_a + nu_a_cycle);
        // col bits: cycle_le[0..sigma_a]
        cycle_bind_le.extend_from_slice(&cycle_le[0..sigma_a]);
        // row cycle bits: cycle_le[sigma_main .. sigma_main+nu_a_cycle]
        if nu_a_cycle > 0 {
            cycle_bind_le.extend_from_slice(&cycle_le[sigma_main..sigma_main + nu_a_cycle]);
        }

        Some(Self {
            kind,
            gamma,
            advice_vars,
            sigma_a,
            nu_a,
            single_opening,
            log_k_chunk,
            log_t,
            sigma_main,
            nu_a_cycle,
            nu_a_addr,
            scale,
            inv_scale,
            cycle_bind_le,
            r_val_eval,
            r_val_final,
        })
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for AdviceClaimReductionPhase2Params<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        // Phase 2 starts from the Phase 1 intermediate claim.
        accumulator
            .get_advice_opening(self.kind, SumcheckId::AdviceClaimReductionPhase1)
            .expect("Phase1 intermediate claim not found")
            .1
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.nu_a_addr
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct AdviceClaimReductionPhase2Prover<F: JoltField> {
    params: AdviceClaimReductionPhase2Params<F>,
    advice_poly: MultilinearPolynomial<F>,
    eq_poly: MultilinearPolynomial<F>,
    /// Constant scaling for trailing dummy rounds after the active window in Stage 7 batching
    /// (remaining address bits not consumed by advice).
    after_scale: F,
    after_inv_scale: F,
    two_inv: F,
}

impl<F: JoltField> AdviceClaimReductionPhase2Prover<F> {
    pub fn initialize(
        params: AdviceClaimReductionPhase2Params<F>,
        mut advice_poly: MultilinearPolynomial<F>,
    ) -> Self {
        let gamma = params.gamma;
        let eq_eval = EqPolynomial::evals(&params.r_val_eval.r);
        let mut eq_poly = if params.single_opening {
            MultilinearPolynomial::from(eq_eval)
        } else {
            let r_final = params
                .r_val_final
                .as_ref()
                .expect("r_val_final must exist when !single_opening");
            let eq_final = EqPolynomial::evals_with_scaling(&r_final.r, Some(gamma));
            let combined: Vec<F> = eq_eval
                .par_iter()
                .zip(eq_final.par_iter())
                .map(|(e1, e2)| *e1 + e2)
                .collect();
            MultilinearPolynomial::from(combined)
        };

        // Pre-bind cycle-derived advice variables in Dory order:
        // [col_bits || row_cycle_bits].
        for &r in params.cycle_bind_le.iter() {
            advice_poly.bind_parallel(r, BindingOrder::LowToHigh);
            eq_poly.bind_parallel(r, BindingOrder::LowToHigh);
        }

        let two_inv = F::from_u64(2).inverse().unwrap();
        let dummy_after = params.log_k_chunk.saturating_sub(params.nu_a_addr);
        let after_scale = F::one().mul_pow_2(dummy_after);
        let after_inv_scale = after_scale.inverse().unwrap();
        Self {
            params,
            advice_poly,
            eq_poly,
            after_scale,
            after_inv_scale,
            two_inv,
        }
    }

    fn compute_message_unscaled(&mut self, previous_claim_unscaled: F) -> UniPoly<F> {
        let half = self.advice_poly.len() / 2;

        let evals: [F; DEGREE_BOUND] = (0..half)
            .into_par_iter()
            .map(|j| {
                let a_evals = self
                    .advice_poly
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_evals = self
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                let mut out = [F::zero(); DEGREE_BOUND];
                for i in 0..DEGREE_BOUND {
                    out[i] = a_evals[i] * eq_evals[i];
                }
                out
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, arr| {
                    acc.par_iter_mut()
                        .zip(arr.par_iter())
                        .for_each(|(a, b)| *a += *b);
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim_unscaled, &evals)
    }

    fn bind_real_var(&mut self, r_j: F::Challenge) {
        self.advice_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for AdviceClaimReductionPhase2Prover<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        // Account for (1) constant scale from Phase 1's internal dummy-gap traversal and
        // (2) trailing dummy rounds after this instance's active window in Stage 7 batching.
        let prev_unscaled = previous_claim * self.params.inv_scale * self.after_inv_scale;
        let poly_unscaled = self.compute_message_unscaled(prev_unscaled);
        poly_unscaled * self.params.scale * self.after_scale
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.bind_real_var(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Build full advice opening point (little-endian Dory order):
        // [col_bits (sigma_a) || row_cycle_bits (nu_a_cycle) || row_addr_bits (nu_a_addr)]
        debug_assert_eq!(sumcheck_challenges.len(), self.params.nu_a_addr);
        debug_assert_eq!(
            self.params.cycle_bind_le.len(),
            self.params.sigma_a + self.params.nu_a_cycle
        );

        let mut advice_le: Vec<F::Challenge> = Vec::with_capacity(self.params.advice_vars);
        advice_le.extend_from_slice(&self.params.cycle_bind_le[0..self.params.sigma_a]);
        advice_le.extend_from_slice(&self.params.cycle_bind_le[self.params.sigma_a..]);
        advice_le.extend_from_slice(sumcheck_challenges);

        let opening_point: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(advice_le).match_endianness();

        let claim = self.advice_poly.final_sumcheck_claim();

        match self.params.kind {
            AdviceKind::Trusted => accumulator.append_trusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase2,
                opening_point,
                claim,
            ),
            AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase2,
                opening_point,
                claim,
            ),
        }
    }

    fn round_offset(&self, _max_num_rounds: usize) -> usize {
        // Stage 7 rounds are the address bits `addr7_le = [b0..b_{log_k_chunk-1}]`.
        // We need the prefix `b0..b_{nu_a_addr-1}`, so we start at offset 0.
        0
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct AdviceClaimReductionPhase2Verifier<F: JoltField> {
    params: AdviceClaimReductionPhase2Params<F>,
}

impl<F: JoltField> AdviceClaimReductionPhase2Verifier<F> {
    pub fn new(
        kind: AdviceKind,
        memory_layout: &MemoryLayout,
        trace_len: usize,
        gamma: F,
        accumulator: &VerifierOpeningAccumulator<F>,
        single_opening: bool,
    ) -> Option<Self> {
        let params = AdviceClaimReductionPhase2Params::new(
            kind,
            memory_layout,
            trace_len,
            gamma,
            accumulator,
            single_opening,
        )?;
        Some(Self { params })
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for AdviceClaimReductionPhase2Verifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Build the full advice opening point in BIG_ENDIAN for eq computations.
        debug_assert_eq!(sumcheck_challenges.len(), self.params.nu_a_addr);
        let mut advice_le: Vec<F::Challenge> = Vec::with_capacity(self.params.advice_vars);
        advice_le.extend_from_slice(&self.params.cycle_bind_le[0..self.params.sigma_a]);
        advice_le.extend_from_slice(&self.params.cycle_bind_le[self.params.sigma_a..]);
        advice_le.extend_from_slice(sumcheck_challenges);
        let opening_point: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(advice_le).match_endianness();

        let advice_claim = accumulator
            .get_advice_opening(self.params.kind, SumcheckId::AdviceClaimReductionPhase2)
            .expect("Final advice claim not found")
            .1;

        let eq_eval = EqPolynomial::mle(&opening_point.r, &self.params.r_val_eval.r);
        let eq_combined = if self.params.single_opening {
            eq_eval
        } else {
            let r_final = self
                .params
                .r_val_final
                .as_ref()
                .expect("r_val_final must exist when !single_opening");
            let eq_final = EqPolynomial::mle(&opening_point.r, &r_final.r);
            eq_eval + self.params.gamma * eq_final
        };

        // Account for Phase 1's internal dummy-gap traversal via constant scaling.
        advice_claim * eq_combined * self.params.scale
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        debug_assert_eq!(sumcheck_challenges.len(), self.params.nu_a_addr);
        let mut advice_le: Vec<F::Challenge> = Vec::with_capacity(self.params.advice_vars);
        advice_le.extend_from_slice(&self.params.cycle_bind_le[0..self.params.sigma_a]);
        advice_le.extend_from_slice(&self.params.cycle_bind_le[self.params.sigma_a..]);
        advice_le.extend_from_slice(sumcheck_challenges);
        let opening_point: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(advice_le).match_endianness();

        match self.params.kind {
            AdviceKind::Trusted => accumulator.append_trusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase2,
                opening_point,
            ),
            AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase2,
                opening_point,
            ),
        }
    }

    fn round_offset(&self, _max_num_rounds: usize) -> usize {
        0
    }
}
