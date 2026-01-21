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
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialBinding};
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
            // In AddressMajor layout, the unified opening point is (r_address || r_cycle) where:
            // - `r_address` binds the **low** `log_k_chunk` bits of the (row,col) global index (i.e. low column bits)
            // - `r_cycle` binds the remaining `log_T` bits, in **low-to-high** (LSB-first) order during sumcheck
            //
            // The main Dory matrix has `main_col_vars` column bits total; after the `log_k_chunk` address bits,
            // the next `(main_col_vars - log_k_chunk)` cycle bits are still **column** bits. Any additional cycle
            // bits land in **row** bits.
            //
            // For an advice matrix with `advice_col_vars` column bits, its cycle-column bits are exactly
            // `advice_col_vars - log_k_chunk` (the cycle bits immediately above the address bits).
            //
            // If `advice_col_vars < main_col_vars`, then there is an internal "gap" of cycle-column bits present
            // in the main matrix but not in the advice matrix. We must treat those as dummy internal rounds.
            let col_cycle_bits = advice_col_vars.saturating_sub(log_k_chunk);
            let main_col_cycle_bits = main_col_vars.saturating_sub(log_k_chunk);

            // Column-cycle rounds are the lowest cycle bits that still belong to the (advice) column index.
            let col_binding_rounds = 0..min(log_T, col_cycle_bits);

            // Row-cycle rounds begin once we move past all cycle bits that belong to the *main* column index.
            // This correctly introduces dummy internal rounds when `main_col_cycle_bits > col_cycle_bits`.
            let row_start = min(log_T, main_col_cycle_bits);
            let row_binding_rounds = row_start..min(log_T, row_start + advice_row_vars);
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
        let (main_col_vars, main_row_vars) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);

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
        tracing::info!("col_binding_rounds: {:?}", col_binding_rounds);
        tracing::info!("row_binding_rounds: {:?}", row_binding_rounds);
        tracing::info!("advice_col_vars: {}", advice_col_vars);
        tracing::info!("advice_row_vars: {}", advice_row_vars);
        tracing::info!("main_col_vars: {}", main_col_vars);
        tracing::info!("main_row_vars: {}", main_row_vars);
        tracing::info!("log_t: {}", log_t);
        tracing::info!("log_k_chunk: {}", log_k_chunk);

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
        // For debugging: print the exact "permutation" (source ordering) used to construct the
        // normalized opening point. This describes where each output position pulls from:
        // - `challenges[i]` refers to the input slice passed to this function
        // - `cycle_var[i]` refers to `self.cycle_var_challenges[i]`
        //
        // Note: We intentionally print symbolic indices (not field elements) to avoid huge logs.
        let mut permutation: Vec<String> = Vec::new();

        if self.phase == ReductionPhase::CycleVariables {
            let advice_vars = self.advice_col_vars + self.advice_row_vars;
            let mut advice_var_challenges: Vec<F::Challenge> = Vec::with_capacity(advice_vars);
            // Output = challenges[cycle_phase_col_rounds] || challenges[cycle_phase_row_rounds]
            for i in self.cycle_phase_col_rounds.clone() {
                permutation.push(format!("challenges[{i}]"));
            }
            for i in self.cycle_phase_row_rounds.clone() {
                permutation.push(format!("challenges[{i}]"));
            }

            tracing::info!(
                target: "jolt_core::zkvm::claim_reductions::advice",
                phase = ?self.phase,
                layout = ?DoryGlobals::get_layout(),
                out_len = permutation.len(),
                "normalize_opening_point permutation: [{}]",
                permutation.join(", ")
            );
            advice_var_challenges
                .extend_from_slice(&challenges[self.cycle_phase_col_rounds.clone()]);
            advice_var_challenges
                .extend_from_slice(&challenges[self.cycle_phase_row_rounds.clone()]);
            return OpeningPoint::<LITTLE_ENDIAN, F>::new(advice_var_challenges).match_endianness();
        }

        match DoryGlobals::get_layout() {
            DoryLayout::CycleMajor => {
                // Output = cycle_var[0..] || challenges[0..]
                for i in 0..self.cycle_var_challenges.len() {
                    permutation.push(format!("cycle_var[{i}]"));
                }
                for i in 0..challenges.len() {
                    permutation.push(format!("challenges[{i}]"));
                }

                tracing::info!(
                    target: "jolt_core::zkvm::claim_reductions::advice",
                    phase = ?self.phase,
                    layout = ?DoryGlobals::get_layout(),
                    out_len = permutation.len(),
                    "normalize_opening_point permutation: [{}]",
                    permutation.join(", ")
                );

                OpeningPoint::<LITTLE_ENDIAN, F>::new(
                    [self.cycle_var_challenges.as_slice(), challenges].concat(),
                )
                .match_endianness()
            }
            DoryLayout::AddressMajor => {
                // IMPORTANT:
                // `self.cycle_var_challenges` is stored as a *compacted* vector of only the cycle-phase
                // advice variables that were actually bound (col-cycle vars followed by row-cycle vars),
                // NOT as a length-`log_t` array indexed by the global cycle round number.
                //
                // Therefore we must NOT slice it using `cycle_phase_round_schedule`'s round ranges
                // (which are in the `0..log_t` round-index space and may include gaps).
                //
                // Output = address-phase challenges || (cycle-phase bound challenges, in the same order
                // they were normalized/cached in phase 1).
                for i in 0..challenges.len() {
                    permutation.push(format!("challenges[{i}]"));
                }
                for i in 0..self.cycle_var_challenges.len() {
                    permutation.push(format!("cycle_var_compact[{i}]"));
                }

                tracing::info!(
                    target: "jolt_core::zkvm::claim_reductions::advice",
                    phase = ?self.phase,
                    layout = ?DoryGlobals::get_layout(),
                    out_len = permutation.len(),
                    cycle_var_len = self.cycle_var_challenges.len(),
                    "normalize_opening_point permutation: [{}]",
                    permutation.join(", ")
                );

                OpeningPoint::<LITTLE_ENDIAN, F>::new(
                    [challenges, self.cycle_var_challenges.as_slice()].concat(),
                )
                .match_endianness()
            }
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
    /// Maps the *current* polynomial variable positions (0 = low/LSB) to the corresponding
    /// "logical" advice variable index in the original (unpermuted) advice polynomial indexing:
    /// [advice_cols (LSB-first) || advice_rows (LSB-first)].
    ///
    /// As we bind variables (possibly out of the naive sequential order), we remove the bound
    /// position so we can continue to resolve logical variables to current positions.
    var_pos_to_logical_var: Vec<usize>,
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

        // Derive how the coefficient permutation reorders the *variables* (bit positions).
        //
        // After sorting, we have a permutation P such that:
        //   old_index = P[new_index]
        // If P is a pure bit-permutation of the Boolean hypercube indexing, then flipping a single
        // bit of `new_index` flips exactly one bit of `old_index`. We use this to map the current
        // variable positions (low/LSB-first) to the corresponding logical variable indices in the
        // original advice polynomial indexing ([cols || rows], both LSB-first).
        let num_vars = params.advice_col_vars + params.advice_row_vars;
        let permuted_indices: Vec<usize> = permuted_coeffs.iter().map(|(idx, _)| *idx).collect();
        let base = permuted_indices[0];
        let mut var_pos_to_logical_var: Vec<usize> = Vec::with_capacity(num_vars);
        for new_bit in 0..num_vars {
            let diff = base ^ permuted_indices[1usize << new_bit];
            debug_assert!(
                diff.is_power_of_two(),
                "advice permutation is not a pure bit-permutation (new_bit={new_bit}, diff={diff})"
            );
            var_pos_to_logical_var.push(diff.trailing_zeros() as usize);
        }

                // Dump the permuted advice coefficient *indices* (original positions) in their post-sort
        // order as fixed-width binary strings, inserting a newline after each advice-row
        // (i.e., every `advice_cols` entries).
        //
        // Output is always written (hardcoded path) to make debugging deterministic.
        {
            use std::io::Write;

            // Hardcoded dump location (relative to current working directory).
            let dump_path = "advice_permutation_dump.txt";
            // Default width: total number of advice variables.
            let width = params.advice_col_vars + params.advice_row_vars;

            let file = std::fs::File::create(dump_path)
                .unwrap_or_else(|e| panic!("failed to create {dump_path}: {e}"));
            let mut w = std::io::BufWriter::new(file);

            for (pos, (idx, _)) in permuted_coeffs.iter().enumerate() {
                write!(w, "{:0width$b}", *idx, width = width)
                    .expect("failed to write permutation entry");

                if (pos + 1) % advice_cols == 0 {
                    writeln!(w).expect("failed to write newline");
                } else {
                    write!(w, ", ").expect("failed to write separator");
                }
            }
        }


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
            var_pos_to_logical_var,
        }
    }

    fn compute_message_unscaled(
        &mut self,
        previous_claim_unscaled: F,
        var_pos_from_low: usize,
    ) -> UniPoly<F> {
        let half = self.advice_poly.len() / 2;
        let evals: [F; DEGREE_BOUND] = (0..half)
            .into_par_iter()
            .map(|j| {
                let a_evals = self
                    .advice_poly
                    .sumcheck_evals_array_at_var::<DEGREE_BOUND>(j, var_pos_from_low);
                let eq_evals = self
                    .eq_poly
                    .sumcheck_evals_array_at_var::<DEGREE_BOUND>(j, var_pos_from_low);

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

    fn current_var_pos_for_logical_var(&self, logical_var: usize) -> usize {
        self.var_pos_to_logical_var
            .iter()
            .position(|&v| v == logical_var)
            .unwrap_or_else(|| panic!("logical var {logical_var} not present in remaining vars"))
    }

    fn logical_var_for_round(&self, round: usize) -> Option<usize> {
        match self.params.phase {
            ReductionPhase::CycleVariables => {
                if !self.params.cycle_phase_col_rounds.contains(&round)
                    && !self.params.cycle_phase_row_rounds.contains(&round)
                {
                    return None;
                }
                match DoryGlobals::get_layout() {
                    DoryLayout::CycleMajor => {
                        if self.params.cycle_phase_col_rounds.contains(&round) {
                            Some(round)
                        } else {
                            Some(
                                self.params.advice_col_vars
                                    + (round - self.params.cycle_phase_row_rounds.start),
                            )
                        }
                    }
                    DoryLayout::AddressMajor => {
                        if self.params.cycle_phase_col_rounds.contains(&round) {
                            Some(self.params.log_k_chunk + round)
                        } else {
                            Some(
                                self.params.advice_col_vars
                                    + (round - self.params.cycle_phase_row_rounds.start),
                            )
                        }
                    }
                }
            }
            ReductionPhase::AddressVariables => self.var_pos_to_logical_var.iter().copied().min(),
        }
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
            let logical_var = self
                .logical_var_for_round(round)
                .expect("expected logical var for non-dummy round");
            let var_pos_from_low = self.current_var_pos_for_logical_var(logical_var);
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
            let poly_unscaled = self.compute_message_unscaled(prev_unscaled, var_pos_from_low);
            poly_unscaled * scaling_factor
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        tracing::info!("ingesting challenge: round = {round}, r_j = {r_j}");

        match self.params.phase {
            ReductionPhase::CycleVariables => {
                if !self.params.cycle_phase_col_rounds.contains(&round)
                    && !self.params.cycle_phase_row_rounds.contains(&round)
                {
                    // Each dummy internal round halves the running claim; equivalently, we multiply the
                    // scaling factor by 1/2.
                    self.scale *= F::from_u64(2).inverse().unwrap();
                } else {
                    tracing::info!("binding cycle variables: round = {round}, r_j = {r_j}");
                    let logical_var = self.logical_var_for_round(round).expect("logical var missing");
                    let var_pos_from_low = self.current_var_pos_for_logical_var(logical_var);
                    self.advice_poly.bind_var_at_parallel(r_j, var_pos_from_low);
                    self.eq_poly.bind_var_at_parallel(r_j, var_pos_from_low);
                    self.var_pos_to_logical_var.remove(var_pos_from_low);
                    self.params.cycle_var_challenges.push(r_j);
                }
            }
            ReductionPhase::AddressVariables => {
                tracing::info!("binding address variables: round = {round}, r_j = {r_j}");
                let logical_var = self.logical_var_for_round(round).expect("logical var missing");
                let var_pos_from_low = self.current_var_pos_for_logical_var(logical_var);
                self.advice_poly.bind_var_at_parallel(r_j, var_pos_from_low);
                self.eq_poly.bind_var_at_parallel(r_j, var_pos_from_low);
                self.var_pos_to_logical_var.remove(var_pos_from_low);
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
                // Align to the *start* of Booleanity's cycle segment, so local rounds correspond
                // to low Dory column bits in the unified point ordering.
                let booleanity_rounds = self.params.log_k_chunk + self.params.log_t;
                let booleanity_offset = max_num_rounds - booleanity_rounds;
                booleanity_offset + self.params.log_k_chunk
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
            ReductionPhase::CycleVariables => {
                let booleanity_rounds = params.log_k_chunk + params.log_t;
                let booleanity_offset = max_num_rounds - booleanity_rounds;
                booleanity_offset + params.log_k_chunk
            }
            ReductionPhase::AddressVariables => 0,
        }
    }
}
