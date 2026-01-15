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

use std::cmp::{min, Ordering};
use std::ops::Range;

use crate::field::JoltField;
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
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
    pub single_opening: bool,
    pub log_k_chunk: usize,
    pub log_t: usize,
    pub advice_col_vars: usize,
    pub advice_row_vars: usize,
    /// Number of column variables in the main Dory matrix
    pub main_col_vars: usize,
    pub main_row_vars: usize,
    #[allocative(skip)]
    pub row_binding_rounds: Range<usize>,
    #[allocative(skip)]
    pub col_binding_rounds: Range<usize>,
    pub r_val_eval: OpeningPoint<BIG_ENDIAN, F>,
    pub r_val_final: Option<OpeningPoint<BIG_ENDIAN, F>>,
}

fn phase1_round_schedule(
    log_T: usize,
    log_k_chunk: usize,
    main_col_vars: usize,
    advice_row_vars: usize,
    advice_col_vars: usize,
) -> (Range<usize>, Range<usize>) {
    match DoryGlobals::get_layout() {
        DoryLayout::CycleMajor => {
            let col_binding_rounds = 0..min(log_T, advice_col_vars);
            let row_binding_rounds =
                min(log_T, main_col_vars)..min(log_T, main_col_vars + advice_row_vars);
            (col_binding_rounds, row_binding_rounds)
        }
        DoryLayout::AddressMajor => {
            let col_binding_rounds = 0..advice_col_vars.saturating_sub(log_k_chunk);
            let row_binding_rounds =
                col_binding_rounds.end..min(log_T, col_binding_rounds.end + advice_row_vars);
            (col_binding_rounds, row_binding_rounds)
        }
    }
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
        let (main_col_vars, main_row_vars) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);

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

        let gamma: F = transcript.challenge_scalar();

        let (advice_col_vars, advice_row_vars) =
            DoryGlobals::advice_sigma_nu_from_max_bytes(max_advice_size_bytes);
        let (col_binding_rounds, row_binding_rounds) = phase1_round_schedule(
            log_t,
            log_k_chunk,
            main_col_vars,
            advice_row_vars,
            advice_col_vars,
        );

        Some(Self {
            kind,
            gamma,
            advice_col_vars,
            advice_row_vars,
            single_opening,
            log_k_chunk,
            log_t,
            main_col_vars,
            main_row_vars,
            row_binding_rounds,
            col_binding_rounds,
            r_val_eval,
            r_val_final,
        })
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
        if !self.row_binding_rounds.is_empty() {
            self.row_binding_rounds.end - self.col_binding_rounds.start
        } else {
            self.col_binding_rounds.len()
        }
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let advice_vars = self.advice_col_vars + self.advice_row_vars;
        let mut advice_var_challenges: Vec<F::Challenge> = Vec::with_capacity(advice_vars);
        advice_var_challenges.extend_from_slice(&challenges[self.col_binding_rounds.clone()]);
        advice_var_challenges.extend_from_slice(&challenges[self.row_binding_rounds.clone()]);
        OpeningPoint::<LITTLE_ENDIAN, F>::new(advice_var_challenges).match_endianness()
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
        let eq_evals = if params.single_opening {
            EqPolynomial::evals(&params.r_val_eval.r)
        } else {
            let evals = EqPolynomial::evals(&params.r_val_eval.r);
            let r_final = params
                .r_val_final
                .as_ref()
                .expect("r_val_final must exist when !single_opening");
            let eq_final = EqPolynomial::evals_with_scaling(&r_final.r, Some(params.gamma));
            evals
                .par_iter()
                .zip(eq_final.par_iter())
                .map(|(e1, e2)| *e1 + e2)
                .collect()
        };

        let main_cols = 1 << params.main_col_vars;
        let row_col_to_address_cycle = |row: usize, col: usize| -> (usize, usize) {
            match DoryGlobals::get_layout() {
                DoryLayout::CycleMajor => {
                    let global_index = row as u128 * main_cols + col as u128;
                    let address = global_index / (1 << params.log_t);
                    let cycle = global_index % (1 << params.log_t);
                    (address as usize, cycle as usize)
                }
                DoryLayout::AddressMajor => {
                    let global_index = row as u128 * main_cols + col as u128;
                    let address = global_index % (1 << params.log_k_chunk);
                    let cycle = global_index / (1 << params.log_k_chunk);
                    (address as usize, cycle as usize)
                }
            }
        };

        let advice_cols = 1 << params.advice_col_vars;

        let advice_index_to_address_cycle = |index: usize| -> (usize, usize) {
            let row = index / advice_cols;
            let col = index % advice_cols;
            row_col_to_address_cycle(row, col)
        };

        let mut permuted_coeffs: Vec<(usize, (u64, F))> = match advice_poly {
            MultilinearPolynomial::U64Scalars(poly) => poly
                .coeffs
                .into_par_iter()
                .zip(eq_evals.into_par_iter())
                .enumerate()
                .collect(),
            _ => panic!("Advice should have u64 coefficients"),
        };
        permuted_coeffs.par_sort_by(|&(index_a, _), &(index_b, _)| {
            let (address_a, cycle_a) = advice_index_to_address_cycle(index_a);
            let (address_b, cycle_b) = advice_index_to_address_cycle(index_b);
            match address_a.cmp(&address_b) {
                Ordering::Less => Ordering::Less,
                Ordering::Greater => Ordering::Greater,
                Ordering::Equal => cycle_a.cmp(&cycle_b),
            }
        });

        let (advice_coeffs, eq_coeffs): (Vec<_>, Vec<_>) = permuted_coeffs
            .into_par_iter()
            .map(|(_, coeffs)| coeffs)
            .unzip();
        let advice_poly = advice_coeffs.into();
        let eq_poly = eq_coeffs.into();

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
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for AdviceClaimReductionPhase1Prover<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if !self.params.col_binding_rounds.contains(&round)
            && !self.params.row_binding_rounds.contains(&round)
        {
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
        if !self.params.col_binding_rounds.contains(&round)
            && !self.params.row_binding_rounds.contains(&round)
        {
            // Each dummy internal round halves the running claim; equivalently, we multiply the
            // scaling factor by 1/2.
            self.scale *= self.two_inv;
            self.inv_scale *= F::from_u64(2);
        } else {
            self.advice_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
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

        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        match self.params.kind {
            AdviceKind::Trusted => accumulator.append_trusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase1,
                opening_point.clone(),
                c_mid,
            ),
            AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase1,
                opening_point.clone(),
                c_mid,
            ),
        }

        // If there is no Phase 2 (all advice row bits come from cycle), cache the final advice opening
        // directly here under `SumcheckId::AdviceClaimReduction` so Stage 7 can scale/embed it.
        if self.advice_poly.len() == 1 {
            let advice_claim = self.advice_poly.final_sumcheck_claim();
            match self.params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReductionPhase2,
                    opening_point,
                    advice_claim,
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReductionPhase2,
                    opening_point,
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
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        match self.params.kind {
            AdviceKind::Trusted => accumulator.append_trusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase1,
                opening_point.clone(),
            ),
            AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                transcript,
                SumcheckId::AdviceClaimReductionPhase1,
                opening_point.clone(),
            ),
        }

        // If Phase 2 is absent (nu_a_addr == 0), Phase 1 cached the final advice opening in Stage 6.
        // Populate its opening point now so downstream stages can extract it.
        if self.params.row_binding_rounds.len() + self.params.col_binding_rounds.len()
            == self.params.advice_row_vars + self.params.advice_col_vars
        {
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
    pub advice_col_vars: usize,
    pub advice_row_vars: usize,
    pub single_opening: bool,
    pub log_k_chunk: usize,
    pub log_t: usize,
    pub main_col_vars: usize,
    pub main_row_vars: usize,
    pub num_rounds: usize,
    /// Constant scaling factor carried from Phase 1 (equals 2^{-gap_len}).
    pub scale: F,
    pub inv_scale: F,
    /// Cycle-derived prefix (Dory order) used to pre-bind advice vars in Phase 2
    pub cycle_challenges: Vec<F::Challenge>,
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
        let (main_col_vars, main_row_vars) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);

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

        let (advice_col_vars, advice_row_vars) =
            DoryGlobals::advice_sigma_nu_from_max_bytes(max_advice_size_bytes);

        let (col_binding_rounds, row_binding_rounds) = phase1_round_schedule(
            log_t,
            log_k_chunk,
            main_col_vars,
            advice_row_vars,
            advice_col_vars,
        );
        let remaining_rounds = (advice_col_vars + advice_row_vars)
            - (row_binding_rounds.len() + col_binding_rounds.len());
        if remaining_rounds == 0 {
            return None;
        }

        // Gap dummy rounds exist only if Phase 1 had to traverse from sigma_a to sigma_main.
        let gap_len = if row_binding_rounds.is_empty() || col_binding_rounds.is_empty() {
            0
        } else {
            row_binding_rounds.start - col_binding_rounds.end
        };

        let two_inv = F::from_u64(2).inverse().unwrap();
        let scale = (0..gap_len).fold(F::one(), |acc, _| acc * two_inv);
        let inv_scale = (0..gap_len).fold(F::one(), |acc, _| acc * F::from_u64(2));

        let r_cycle_le: OpeningPoint<LITTLE_ENDIAN, F> = accumulator
            .get_advice_opening(kind, SumcheckId::AdviceClaimReductionPhase1)
            .unwrap()
            .0
            .match_endianness();
        let cycle_challenges = r_cycle_le.r;

        Some(Self {
            kind,
            gamma,
            advice_col_vars,
            advice_row_vars,
            num_rounds: remaining_rounds,
            single_opening,
            log_k_chunk,
            log_t,
            main_col_vars,
            main_row_vars,
            scale,
            inv_scale,
            cycle_challenges,
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
        self.num_rounds
    }

    fn normalize_opening_point(
        &self,
        address_challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        match DoryGlobals::get_layout() {
            DoryLayout::CycleMajor => OpeningPoint::<LITTLE_ENDIAN, F>::new(
                [self.cycle_challenges.as_slice(), address_challenges].concat(),
            )
            .match_endianness(),
            DoryLayout::AddressMajor => {
                let (col_binding_rounds, row_binding_rounds) = phase1_round_schedule(
                    self.log_t,
                    self.log_k_chunk,
                    self.main_col_vars,
                    self.advice_row_vars,
                    self.advice_col_vars,
                );
                OpeningPoint::<LITTLE_ENDIAN, F>::new(
                    [
                        address_challenges,
                        &self.cycle_challenges[col_binding_rounds],
                        &self.cycle_challenges[row_binding_rounds],
                    ]
                    .concat(),
                )
                .match_endianness()
            }
        }
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
        advice_poly: MultilinearPolynomial<F>,
    ) -> Self {
        let gamma = params.gamma;
        let eq_evals = if params.single_opening {
            EqPolynomial::evals(&params.r_val_eval.r)
        } else {
            let evals = EqPolynomial::evals(&params.r_val_eval.r);
            let r_final = params
                .r_val_final
                .as_ref()
                .expect("r_val_final must exist when !single_opening");
            let eq_final = EqPolynomial::evals_with_scaling(&r_final.r, Some(gamma));
            evals
                .par_iter()
                .zip(eq_final.par_iter())
                .map(|(e1, e2)| *e1 + e2)
                .collect()
        };

        let main_cols = 1 << params.main_col_vars;
        let row_col_to_address_cycle = |row: usize, col: usize| -> (usize, usize) {
            match DoryGlobals::get_layout() {
                DoryLayout::CycleMajor => {
                    let global_index = row as u128 * main_cols + col as u128;
                    let address = global_index / (1 << params.log_t);
                    let cycle = global_index % (1 << params.log_t);
                    (address as usize, cycle as usize)
                }
                DoryLayout::AddressMajor => {
                    let global_index = row as u128 * main_cols + col as u128;
                    let address = global_index % (1 << params.log_k_chunk);
                    let cycle = global_index / (1 << params.log_k_chunk);
                    (address as usize, cycle as usize)
                }
            }
        };

        let advice_cols = 1 << params.advice_col_vars;
        let advice_index_to_address_cycle = |index: usize| -> (usize, usize) {
            let row = index / advice_cols;
            let col = index % advice_cols;
            row_col_to_address_cycle(row, col)
        };

        let mut permuted_coeffs: Vec<(usize, (u64, F))> = match advice_poly {
            MultilinearPolynomial::U64Scalars(poly) => poly
                .coeffs
                .into_par_iter()
                .zip(eq_evals.into_par_iter())
                .enumerate()
                .collect(),
            _ => panic!("Advice should have u64 coefficients"),
        };
        permuted_coeffs.par_sort_by(|&(index_a, _), &(index_b, _)| {
            let (address_a, cycle_a) = advice_index_to_address_cycle(index_a);
            let (address_b, cycle_b) = advice_index_to_address_cycle(index_b);
            match address_a.cmp(&address_b) {
                Ordering::Less => Ordering::Less,
                Ordering::Greater => Ordering::Greater,
                Ordering::Equal => cycle_a.cmp(&cycle_b),
            }
        });

        let (advice_coeffs, eq_coeffs): (Vec<_>, Vec<_>) = permuted_coeffs
            .into_par_iter()
            .map(|(_, coeffs)| coeffs)
            .unzip();
        let mut advice_poly: MultilinearPolynomial<F> = advice_coeffs.into();
        let mut eq_poly: MultilinearPolynomial<F> = eq_coeffs.into();

        // Pre-bind cycle-derived advice variables in Dory order
        for &r in params.cycle_challenges.iter() {
            advice_poly.bind_parallel(r, BindingOrder::LowToHigh);
            eq_poly.bind_parallel(r, BindingOrder::LowToHigh);
        }

        let two_inv = F::from_u64(2).inverse().unwrap();
        let dummy_after = params.log_k_chunk.saturating_sub(params.num_rounds);
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
        self.advice_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Build full advice opening point (little-endian Dory order):
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
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
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
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
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

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
