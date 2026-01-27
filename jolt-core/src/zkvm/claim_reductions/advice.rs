//! Two-phase advice claim reduction (Stage 6 cycle → Stage 7 address)
//!
//! This module generalizes the previous single-phase `AdviceClaimReduction` so that trusted and
//! untrusted advice can be committed as an arbitrary Dory matrix `2^{nu_a} x 2^{sigma_a}` (balanced
//! by default), while still keeping a **single Stage 8 Dory opening** at the unified Dory point.
//!
//! For an advice matrix embedded as the **top-left block** `2^{nu_a} x 2^{sigma_a}`, the *native*
//! advice evaluation point (in Dory order, LSB-first) is:
//! - `advice_cols = col_coords[0..sigma_a]`
//! - `advice_rows = row_coords[0..nu_a]`
//! - `advice_point = [advice_cols || advice_rows]`
//!
//! In our current pipeline, `cycle` coordinates come from Stage 6 and `addr` coordinates come from
//! Stage 7.
//! - **Phase 1 (Stage 6)**: bind the cycle-derived advice coordinates and output an intermediate
//!   scalar claim `C_mid`.
//! - **Phase 2 (Stage 7)**: resume from `C_mid`, bind the address-derived advice coordinates, and
//!   cache the final advice opening `AdviceMLE(advice_point)` for batching into Stage 8.
//!
//! ## Dummy-gap scaling (within Stage 6)
//! With cycle-major order, there may be a gap during the cycle phase where the cycle variables
//! being bound in the batched sumcheck do not appear in the advice polynommial.
//!
//! We handle this without modifying the generic batched sumcheck by treating those intervening
//! rounds as **dummy internal rounds** (constant univariates), and maintaining a running scaling
//! factor `2^{-dummy_done}` so the per-round univariates remain consistent.
//!
//! Trusted and untrusted advice run as **separate** sumcheck instances (each may have different
//! dimensions).
//!

use std::cell::RefCell;
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

#[derive(Debug, Clone, Allocative, PartialEq, Eq)]
pub enum ReductionPhase {
    CycleVariables,
    AddressVariables,
}

#[derive(Clone, Allocative)]
pub struct AdviceClaimReductionParams<F: JoltField> {
    pub kind: AdviceKind,
    pub phase: ReductionPhase,
    pub gamma: F,
    pub single_opening: bool,
    pub log_k_chunk: usize,
    pub log_t: usize,
    pub advice_col_vars: usize,
    pub advice_row_vars: usize,
    /// Number of column variables in the main Dory matrix
    pub main_col_vars: usize,
    /// Number of row variables in the main Dory matrix
    pub main_row_vars: usize,
    #[allocative(skip)]
    pub cycle_phase_row_rounds: Range<usize>,
    #[allocative(skip)]
    pub cycle_phase_col_rounds: Range<usize>,
    pub r_val_eval: OpeningPoint<BIG_ENDIAN, F>,
    pub r_val_final: Option<OpeningPoint<BIG_ENDIAN, F>>,
    /// (little-endian) challenges for the cycle phase variables
    pub cycle_var_challenges: Vec<F::Challenge>,
}

fn cycle_phase_round_schedule(
    log_T: usize,
    log_k_chunk: usize,
    main_col_vars: usize,
    advice_row_vars: usize,
    advice_col_vars: usize,
) -> (Range<usize>, Range<usize>) {
    match DoryGlobals::get_layout() {
        DoryLayout::CycleMajor => {
            // Low-order cycle variables correspond to the low-order bits of the
            // column index
            let col_binding_rounds = 0..min(log_T, advice_col_vars);
            // High-order cycle variables correspond to the low-order bits of the
            // rows index
            let row_binding_rounds =
                min(log_T, main_col_vars)..min(log_T, main_col_vars + advice_row_vars);
            (col_binding_rounds, row_binding_rounds)
        }
        DoryLayout::AddressMajor => {
            // Low-order cycle variables correspond to the high-order bits of the
            // column index
            let col_binding_rounds = 0..advice_col_vars.saturating_sub(log_k_chunk);
            // High-order cycle variables correspond to the bits of the row index
            let row_binding_rounds = main_col_vars.saturating_sub(log_k_chunk)
                ..min(
                    log_T,
                    main_col_vars.saturating_sub(log_k_chunk) + advice_row_vars,
                );
            (col_binding_rounds, row_binding_rounds)
        }
    }
}

impl<F: JoltField> AdviceClaimReductionParams<F> {
    pub fn new(
        kind: AdviceKind,
        memory_layout: &MemoryLayout,
        trace_len: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        single_opening: bool,
    ) -> Self {
        let max_advice_size_bytes = match kind {
            AdviceKind::Trusted => memory_layout.max_trusted_advice_size as usize,
            AdviceKind::Untrusted => memory_layout.max_untrusted_advice_size as usize,
        };

        let log_t = trace_len.log_2();
        let log_k_chunk = OneHotConfig::new(log_t).log_k_chunk as usize;
        let (main_col_vars, main_row_vars) = DoryGlobals::try_get_main_sigma_nu()
            .unwrap_or_else(|| DoryGlobals::main_sigma_nu(log_k_chunk, log_t));

        let r_val_eval = accumulator
            .get_advice_opening(kind, SumcheckId::RamValEvaluation)
            .map(|(p, _)| p)
            .unwrap();
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
        let (col_binding_rounds, row_binding_rounds) = cycle_phase_round_schedule(
            log_t,
            log_k_chunk,
            main_col_vars,
            advice_row_vars,
            advice_col_vars,
        );

        Self {
            kind,
            phase: ReductionPhase::CycleVariables,
            gamma,
            advice_col_vars,
            advice_row_vars,
            single_opening,
            log_k_chunk,
            log_t,
            main_col_vars,
            main_row_vars,
            cycle_phase_row_rounds: row_binding_rounds,
            cycle_phase_col_rounds: col_binding_rounds,
            r_val_eval,
            r_val_final,
            cycle_var_challenges: vec![],
        }
    }

    /// (Total # advice variables) - (# variables bound during cycle phase)
    pub fn num_address_phase_rounds(&self) -> usize {
        (self.advice_col_vars + self.advice_row_vars)
            - (self.cycle_phase_col_rounds.len() + self.cycle_phase_row_rounds.len())
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for AdviceClaimReductionParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        match self.phase {
            ReductionPhase::CycleVariables => {
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
            ReductionPhase::AddressVariables => {
                // Address phase starts from the cycle phase intermediate claim.
                accumulator
                    .get_advice_opening(self.kind, SumcheckId::AdviceClaimReductionCyclePhase)
                    .expect("Cycle phase intermediate claim not found")
                    .1
            }
        }
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        match self.phase {
            ReductionPhase::CycleVariables => {
                if !self.cycle_phase_row_rounds.is_empty() {
                    self.cycle_phase_row_rounds.end - self.cycle_phase_col_rounds.start
                } else {
                    self.cycle_phase_col_rounds.len()
                }
            }
            ReductionPhase::AddressVariables => {
                let first_phase_rounds =
                    self.cycle_phase_row_rounds.len() + self.cycle_phase_col_rounds.len();
                // Total advice variables, minus the variables bound during the cycle phase
                (self.advice_col_vars + self.advice_row_vars) - first_phase_rounds
            }
        }
    }

    /// Rearrange the opening point so that it is big-endian with respect to the original,
    /// unpermuted advice/EQ polynomials.
    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        if self.phase == ReductionPhase::CycleVariables {
            let advice_vars = self.advice_col_vars + self.advice_row_vars;
            let mut advice_var_challenges: Vec<F::Challenge> = Vec::with_capacity(advice_vars);
            advice_var_challenges
                .extend_from_slice(&challenges[self.cycle_phase_col_rounds.clone()]);
            advice_var_challenges
                .extend_from_slice(&challenges[self.cycle_phase_row_rounds.clone()]);
            return OpeningPoint::<LITTLE_ENDIAN, F>::new(advice_var_challenges).match_endianness();
        }

        match DoryGlobals::get_layout() {
            DoryLayout::CycleMajor => OpeningPoint::<LITTLE_ENDIAN, F>::new(
                [self.cycle_var_challenges.as_slice(), challenges].concat(),
            )
            .match_endianness(),
            DoryLayout::AddressMajor => OpeningPoint::<LITTLE_ENDIAN, F>::new(
                [challenges, self.cycle_var_challenges.as_slice()].concat(),
            )
            .match_endianness(),
        }
    }
}

#[derive(Allocative)]
pub struct AdviceClaimReductionProver<F: JoltField> {
    pub params: AdviceClaimReductionParams<F>,
    advice_poly: MultilinearPolynomial<F>,
    eq_poly: MultilinearPolynomial<F>,
    /// Maintains the running internal scaling factor 2^{-dummy_done}.
    scale: F,
}

impl<F: JoltField> AdviceClaimReductionProver<F> {
    pub fn initialize(
        params: AdviceClaimReductionParams<F>,
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
        // Maps a (row, col) position in the Dory matrix layout to its
        // implied (address, cycle).
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
        // Maps an index in the advice vector to its implied (address, cycle), based
        // on the position the index maps to in the Dory matrix layout.
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
        // Sort the advice and EQ polynomial coefficients by (address, cycle).
        // By sorting this way, binding the resulting polynomials in low-to-high
        // order is equivalent to binding the original polynomials' "cycle" variables
        // low-to-high, then their "address" variables low-to-high.
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

        Self {
            params,
            advice_poly,
            eq_poly,
            scale: F::one(),
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

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for AdviceClaimReductionProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if self.params.phase == ReductionPhase::CycleVariables
            && !self.params.cycle_phase_col_rounds.contains(&round)
            && !self.params.cycle_phase_row_rounds.contains(&round)
        {
            // Current sumcheck variable does not appear in advice polynomial, so we
            // can simply send a constant polynomial equal to the previous claim divided by 2
            UniPoly::from_coeff(vec![previous_claim * F::from_u64(2).inverse().unwrap()])
        } else {
            // Account for (1) internal dummy rounds already traversed and
            // (2) trailing dummy rounds after this instance's active window in the batched sumcheck.
            let num_trailing_variables = match self.params.phase {
                ReductionPhase::CycleVariables => {
                    self.params.log_t.saturating_sub(self.params.num_rounds())
                }
                ReductionPhase::AddressVariables => self
                    .params
                    .log_k_chunk
                    .saturating_sub(self.params.num_rounds()),
            };
            let scaling_factor = self.scale * F::one().mul_pow_2(num_trailing_variables);
            let prev_unscaled = previous_claim * scaling_factor.inverse().unwrap();
            let poly_unscaled = self.compute_message_unscaled(prev_unscaled);
            poly_unscaled * scaling_factor
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        match self.params.phase {
            ReductionPhase::CycleVariables => {
                if !self.params.cycle_phase_col_rounds.contains(&round)
                    && !self.params.cycle_phase_row_rounds.contains(&round)
                {
                    // Each dummy internal round halves the running claim; equivalently, we multiply the
                    // scaling factor by 1/2.
                    self.scale *= F::from_u64(2).inverse().unwrap();
                } else {
                    self.advice_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
                    self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
                    self.params.cycle_var_challenges.push(r_j);
                }
            }
            ReductionPhase::AddressVariables => {
                self.advice_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
                self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        if self.params.phase == ReductionPhase::CycleVariables {
            // Compute the intermediate claim C_mid = (2^{-gap}) * Σ_y advice(y) * eq(y),
            // where y are the remaining (address-derived) advice row variables.
            let len = self.advice_poly.len();
            debug_assert_eq!(len, self.eq_poly.len());

            let mut sum = F::zero();
            for i in 0..len {
                sum += self.advice_poly.get_bound_coeff(i) * self.eq_poly.get_bound_coeff(i);
            }
            let c_mid = sum * self.scale;

            match self.params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    opening_point.clone(),
                    c_mid,
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    opening_point.clone(),
                    c_mid,
                ),
            }
        }

        // If we're done binding advice variables, cache the final advice opening
        if self.advice_poly.len() == 1 {
            let advice_claim = self.advice_poly.final_sumcheck_claim();
            match self.params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReduction,
                    opening_point,
                    advice_claim,
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReduction,
                    opening_point,
                    advice_claim,
                ),
            }
        }
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        match self.params.phase {
            ReductionPhase::CycleVariables => {
                // Stage 6b only spans cycle variables; align to the start of the cycle segment.
                max_num_rounds.saturating_sub(self.params.log_t)
            }
            ReductionPhase::AddressVariables => 0,
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct AdviceClaimReductionVerifier<F: JoltField> {
    pub params: RefCell<AdviceClaimReductionParams<F>>,
}

impl<F: JoltField> AdviceClaimReductionVerifier<F> {
    pub fn new(
        kind: AdviceKind,
        memory_layout: &MemoryLayout,
        trace_len: usize,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        single_opening: bool,
    ) -> Self {
        let params = AdviceClaimReductionParams::new(
            kind,
            memory_layout,
            trace_len,
            accumulator,
            transcript,
            single_opening,
        );

        Self {
            params: RefCell::new(params),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for AdviceClaimReductionVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        unsafe { &*self.params.as_ptr() }
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let params = self.params.borrow();
        match params.phase {
            ReductionPhase::CycleVariables => {
                accumulator
                    .get_advice_opening(params.kind, SumcheckId::AdviceClaimReductionCyclePhase)
                    .unwrap_or_else(|| panic!("Cycle phase intermediate claim not found",))
                    .1
            }
            ReductionPhase::AddressVariables => {
                let opening_point = params.normalize_opening_point(sumcheck_challenges);
                let advice_claim = accumulator
                    .get_advice_opening(params.kind, SumcheckId::AdviceClaimReduction)
                    .expect("Final advice claim not found")
                    .1;

                let eq_eval = EqPolynomial::mle(&opening_point.r, &params.r_val_eval.r);
                let eq_combined = if params.single_opening {
                    eq_eval
                } else {
                    let r_final = params
                        .r_val_final
                        .as_ref()
                        .expect("r_val_final must exist when !single_opening");
                    let eq_final = EqPolynomial::mle(&opening_point.r, &r_final.r);
                    eq_eval + params.gamma * eq_final
                };

                let gap_len = if params.cycle_phase_row_rounds.is_empty()
                    || params.cycle_phase_col_rounds.is_empty()
                {
                    0
                } else {
                    params.cycle_phase_row_rounds.start - params.cycle_phase_col_rounds.end
                };
                let two_inv = F::from_u64(2).inverse().unwrap();
                let scale = (0..gap_len).fold(F::one(), |acc, _| acc * two_inv);

                // Account for Phase 1's internal dummy-gap traversal via constant scaling.
                advice_claim * eq_combined * scale
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let mut params = self.params.borrow_mut();
        if params.phase == ReductionPhase::CycleVariables {
            let opening_point = params.normalize_opening_point(sumcheck_challenges);
            match params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    opening_point.clone(),
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    opening_point.clone(),
                ),
            }
            let opening_point_le: OpeningPoint<LITTLE_ENDIAN, F> = opening_point.match_endianness();
            params.cycle_var_challenges = opening_point_le.r;
        }

        if params.num_address_phase_rounds() == 0
            || params.phase == ReductionPhase::AddressVariables
        {
            let opening_point = params.normalize_opening_point(sumcheck_challenges);
            match params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReduction,
                    opening_point,
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    transcript,
                    SumcheckId::AdviceClaimReduction,
                    opening_point,
                ),
            }
        }
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        let params = self.params.borrow();
        match params.phase {
            ReductionPhase::CycleVariables => max_num_rounds.saturating_sub(params.log_t),
            ReductionPhase::AddressVariables => 0,
        }
    }
}
