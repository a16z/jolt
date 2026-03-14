use allocative::Allocative;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::utils::math::Math;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Allocative)]
pub enum PrecommittedEmbeddingMode {
    DominantPrecommitted,
    EmbeddedPrecommitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Allocative)]
pub enum PrecommittedPhase {
    CycleVariables,
    AddressVariables,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Allocative)]
pub struct PrecommittedSchedulingReference {
    pub main_total_vars: usize,
    pub reference_total_vars: usize,
    pub cycle_alignment_rounds: usize,
    pub address_rounds: usize,
    pub joint_col_vars: usize,
}

#[derive(Debug, Clone, Allocative)]
pub struct PrecommittedClaimReduction<F: JoltField> {
    pub scheduling_reference: PrecommittedSchedulingReference,
    pub embedding_mode: PrecommittedEmbeddingMode,
    pub cycle_var_challenges: Vec<F::Challenge>,
    dory_opening_round_permutation_be: Vec<usize>,
    poly_opening_round_permutation_be: Vec<usize>,
    cycle_phase_rounds: Vec<usize>,
    cycle_phase_total_rounds: usize,
    address_phase_rounds: Vec<usize>,
    address_phase_total_rounds: usize,
}

impl<F: JoltField> PrecommittedClaimReduction<F> {
    /// Compute shared scheduling dimensions from Main and precommitted candidates.
    ///
    /// `reference_total_vars` is the largest total var count across Main and candidates.
    pub fn scheduling_reference(
        main_total_vars: usize,
        candidates: &[usize],
    ) -> PrecommittedSchedulingReference {
        let address_rounds = DoryGlobals::main_k().log_2();
        let max_precommitted = candidates.iter().copied().max().unwrap_or(0);
        let reference_total_vars = std::cmp::max(main_total_vars, max_precommitted);
        let cycle_alignment_rounds = reference_total_vars.saturating_sub(address_rounds);
        let (reference_sigma, _) = DoryGlobals::balanced_sigma_nu(reference_total_vars);
        let joint_col_vars = std::cmp::max(
            DoryGlobals::configured_main_num_columns().log_2(),
            reference_sigma,
        );
        PrecommittedSchedulingReference {
            main_total_vars,
            reference_total_vars,
            cycle_alignment_rounds,
            address_rounds,
            joint_col_vars,
        }
    }

    #[inline]
    pub fn new(
        poly_total_vars: usize,
        poly_row_vars: usize,
        poly_col_vars: usize,
        scheduling_reference: PrecommittedSchedulingReference,
    ) -> Self {
        let has_precommitted_dominance =
            scheduling_reference.reference_total_vars > scheduling_reference.main_total_vars;
        let embedding_mode = Self::embedding_mode_for_poly(poly_total_vars, &scheduling_reference);
        let dory_opening_round_permutation_be = Self::reference_dory_opening_round_permutation_be(
            &scheduling_reference,
            has_precommitted_dominance,
            DoryGlobals::main_t().log_2(),
        );
        let poly_opening_round_permutation_be = Self::project_dory_round_permutation_for_poly(
            &dory_opening_round_permutation_be,
            &scheduling_reference,
            poly_row_vars,
            poly_col_vars,
        );
        let (cycle_phase_rounds, address_phase_rounds) = Self::active_rounds_from_poly_permutation(
            &poly_opening_round_permutation_be,
            scheduling_reference.cycle_alignment_rounds,
        );
        Self {
            scheduling_reference,
            embedding_mode,
            cycle_var_challenges: vec![],
            dory_opening_round_permutation_be,
            poly_opening_round_permutation_be,
            cycle_phase_rounds,
            cycle_phase_total_rounds: scheduling_reference.cycle_alignment_rounds,
            address_phase_rounds,
            address_phase_total_rounds: scheduling_reference.address_rounds,
        }
    }

    #[inline]
    fn embedding_mode_for_poly(
        poly_total_vars: usize,
        reference: &PrecommittedSchedulingReference,
    ) -> PrecommittedEmbeddingMode {
        let has_precommitted_dominance = reference.reference_total_vars > reference.main_total_vars;
        let embedding_mode =
            if has_precommitted_dominance && poly_total_vars == reference.reference_total_vars {
                PrecommittedEmbeddingMode::DominantPrecommitted
            } else {
                PrecommittedEmbeddingMode::EmbeddedPrecommitted
            };
        if embedding_mode == PrecommittedEmbeddingMode::DominantPrecommitted {
            assert_eq!(poly_total_vars, reference.reference_total_vars);
        }
        embedding_mode
    }

    fn reference_dory_opening_round_permutation_be(
        reference: &PrecommittedSchedulingReference,
        has_precommitted_dominance: bool,
        dense_cycle_prefix_rounds: usize,
    ) -> Vec<usize> {
        let cycle_rounds = reference.cycle_alignment_rounds;
        let address_rounds = reference.address_rounds;
        let total_rounds = cycle_rounds + address_rounds;
        if has_precommitted_dominance {
            let address_rev = (cycle_rounds..total_rounds).rev();
            match DoryGlobals::get_layout() {
                DoryLayout::CycleMajor => {
                    let t = dense_cycle_prefix_rounds.min(cycle_rounds);
                    let prefix_rev = (0..cycle_rounds.saturating_sub(t)).rev();
                    let dense_rev = (cycle_rounds.saturating_sub(t)..cycle_rounds).rev();
                    return prefix_rev.chain(address_rev).chain(dense_rev).collect();
                }
                DoryLayout::AddressMajor => {
                    let t = dense_cycle_prefix_rounds.min(cycle_rounds);
                    let prefix_rev = (0..cycle_rounds.saturating_sub(t)).rev();
                    let dense_rev = (cycle_rounds.saturating_sub(t)..cycle_rounds).rev();
                    return dense_rev.chain(address_rev).chain(prefix_rev).collect();
                }
            }
        }

        match DoryGlobals::get_layout() {
            DoryLayout::CycleMajor => (0..total_rounds).rev().collect(),
            DoryLayout::AddressMajor => {
                let cycle_rev = (0..cycle_rounds).rev();
                let address_rev = (cycle_rounds..total_rounds).rev();
                cycle_rev.chain(address_rev).collect()
            }
        }
    }

    fn project_dory_round_permutation_for_poly(
        dory_opening_round_permutation_be: &[usize],
        reference: &PrecommittedSchedulingReference,
        poly_row_vars: usize,
        poly_col_vars: usize,
    ) -> Vec<usize> {
        let total_full = reference.reference_total_vars;
        let sigma_full = reference.joint_col_vars;
        let nu_full = total_full.saturating_sub(sigma_full);
        assert_eq!(
            dory_opening_round_permutation_be.len(),
            total_full,
            "reference dory round permutation length mismatch",
        );
        assert!(
            poly_row_vars <= nu_full && poly_col_vars <= sigma_full,
            "top-left projection requires poly dims <= full dims (poly row/col vars={poly_row_vars}/{poly_col_vars}, full row/col vars={nu_full}/{sigma_full})"
        );
        let row_be = &dory_opening_round_permutation_be[..nu_full];
        let col_be = &dory_opening_round_permutation_be[nu_full..nu_full + sigma_full];
        let row_tail = &row_be[nu_full - poly_row_vars..];
        let col_tail = &col_be[sigma_full - poly_col_vars..];
        [row_tail, col_tail].concat()
    }

    fn active_rounds_from_poly_permutation(
        poly_opening_round_permutation_be: &[usize],
        cycle_alignment_rounds: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut cycle_phase_rounds = Vec::new();
        let mut address_phase_rounds = Vec::new();
        for &global_round in poly_opening_round_permutation_be.iter() {
            if global_round < cycle_alignment_rounds {
                cycle_phase_rounds.push(global_round);
            } else {
                address_phase_rounds.push(global_round - cycle_alignment_rounds);
            }
        }
        cycle_phase_rounds.sort_unstable();
        cycle_phase_rounds.dedup();
        address_phase_rounds.sort_unstable();
        address_phase_rounds.dedup();
        (cycle_phase_rounds, address_phase_rounds)
    }

    #[inline]
    pub fn num_address_phase_rounds(&self) -> usize {
        self.address_phase_rounds.len()
    }

    #[inline]
    pub fn is_cycle_phase_round(&self, round: usize) -> bool {
        self.cycle_phase_rounds
            .iter()
            .any(|&scheduled| scheduled == round)
    }

    #[inline]
    pub fn is_address_phase_round(&self, round: usize) -> bool {
        self.address_phase_rounds
            .iter()
            .any(|&scheduled| scheduled == round)
    }

    #[inline]
    pub fn cycle_alignment_rounds(&self) -> usize {
        self.scheduling_reference.cycle_alignment_rounds
    }

    #[inline]
    pub fn address_alignment_rounds(&self) -> usize {
        self.scheduling_reference.address_rounds
    }

    #[inline]
    pub fn num_rounds_for_phase(&self, is_cycle_phase: bool) -> usize {
        if is_cycle_phase {
            self.cycle_phase_total_rounds
        } else {
            self.address_phase_total_rounds
        }
    }

    pub fn round_offset(&self, is_cycle_phase: bool, max_num_rounds: usize) -> usize {
        let _ = (is_cycle_phase, max_num_rounds);
        0
    }

    fn cycle_challenge_for_round(&self, round: usize) -> F::Challenge {
        let idx = self
            .cycle_phase_rounds
            .iter()
            .position(|&scheduled_round| scheduled_round == round)
            .unwrap_or_else(|| {
                panic!(
                    "missing recorded cycle challenge for round={} (active rounds={:?})",
                    round, self.cycle_phase_rounds
                )
            });
        assert!(
            idx < self.cycle_var_challenges.len(),
            "cycle challenge vector too short: idx={} len={}",
            idx,
            self.cycle_var_challenges.len()
        );
        self.cycle_var_challenges[idx]
    }

    pub fn normalize_opening_point(
        &self,
        is_cycle_phase: bool,
        challenges: &[F::Challenge],
        dense_cycle_prefix_rounds: usize,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let _ = dense_cycle_prefix_rounds;
        if is_cycle_phase {
            let local_cycle_challenges: Vec<F::Challenge> = self
                .cycle_phase_rounds
                .iter()
                .map(|&round| {
                    assert!(
                        round < challenges.len(),
                        "cycle round index out of local bounds: round={} local_len={}",
                        round,
                        challenges.len()
                    );
                    challenges[round]
                })
                .collect();
            return OpeningPoint::<LITTLE_ENDIAN, F>::new(local_cycle_challenges)
                .match_endianness();
        }

        debug_assert_eq!(
            self.dory_opening_round_permutation_be.len(),
            self.scheduling_reference.reference_total_vars
        );
        let cycle_round_limit = self.cycle_alignment_rounds();
        let opening_rounds = &self.poly_opening_round_permutation_be;
        let mut opening_point_be = Vec::with_capacity(opening_rounds.len());
        for &global_round in opening_rounds.iter() {
            if global_round < cycle_round_limit {
                opening_point_be.push(self.cycle_challenge_for_round(global_round));
            } else {
                let address_round = global_round - cycle_round_limit;
                assert!(
                    address_round < challenges.len(),
                    "address round index out of local bounds: round={} local_len={}",
                    address_round,
                    challenges.len()
                );
                opening_point_be.push(challenges[address_round]);
            }
        }
        OpeningPoint::<BIG_ENDIAN, F>::new(opening_point_be)
    }

    #[inline]
    pub fn record_cycle_challenge(&mut self, challenge: F::Challenge) {
        self.cycle_var_challenges.push(challenge);
    }

    #[inline]
    pub fn set_cycle_var_challenges(&mut self, challenges: Vec<F::Challenge>) {
        self.cycle_var_challenges = challenges;
    }
}

pub fn permute_precommitted_polys<V: Copy + Send + Sync, F: JoltField>(
    coeffs_by_poly: Vec<Vec<V>>,
    precommitted: &PrecommittedClaimReduction<F>,
) -> Vec<MultilinearPolynomial<F>>
where
    MultilinearPolynomial<F>: From<Vec<V>>,
{
    if coeffs_by_poly.is_empty() {
        return Vec::new();
    }
    let coeffs_len = coeffs_by_poly[0].len();
    assert!(
        coeffs_by_poly
            .iter()
            .all(|coeffs| coeffs.len() == coeffs_len),
        "all precommitted polynomials must have equal coefficient lengths",
    );
    let inverse_permutation = precommitted_sumcheck_inverse_index_permutation(
        coeffs_len,
        &precommitted.poly_opening_round_permutation_be,
    );
    let permuted_coeffs_by_poly: Vec<Vec<V>> =
        if let Some(inverse_permutation) = inverse_permutation {
            coeffs_by_poly
                .into_iter()
                .map(|coeffs| {
                    (0..coeffs_len)
                        .into_par_iter()
                        .map(|new_idx| {
                            let old_idx = inverse_permutation[new_idx];
                            coeffs[old_idx]
                        })
                        .collect()
                })
                .collect()
        } else {
            coeffs_by_poly
        };
    permuted_coeffs_by_poly
        .into_iter()
        .map(Into::into)
        .collect()
}

pub fn precommitted_eq_evals_with_scaling<F: JoltField>(
    challenges_be: &[F::Challenge],
    scaling_factor: Option<F>,
    precommitted: &PrecommittedClaimReduction<F>,
) -> Vec<F>
where
    F: std::ops::Mul<F::Challenge, Output = F> + std::ops::SubAssign<F>,
{
    let permuted_challenges = precommitted_permute_eq_challenges(
        challenges_be,
        &precommitted.poly_opening_round_permutation_be,
    );
    if let Some(permuted_challenges) = permuted_challenges {
        EqPolynomial::evals_with_scaling(&permuted_challenges, scaling_factor)
    } else {
        EqPolynomial::evals_with_scaling(challenges_be, scaling_factor)
    }
}

fn precommitted_permute_eq_challenges<C: Copy>(
    challenges_be: &[C],
    poly_opening_round_permutation_be: &[usize],
) -> Option<Vec<C>> {
    let old_lsb_to_new_lsb =
        precommitted_sumcheck_lsb_permutation(poly_opening_round_permutation_be)?;
    assert_eq!(
        challenges_be.len(),
        old_lsb_to_new_lsb.len(),
        "challenge vector length mismatch for precommitted eq permutation",
    );
    let num_vars = challenges_be.len();
    let mut permuted_challenges = challenges_be.to_vec();
    for old_be in 0..num_vars {
        let old_lsb = num_vars - 1 - old_be;
        let new_lsb = old_lsb_to_new_lsb[old_lsb];
        let new_be = num_vars - 1 - new_lsb;
        permuted_challenges[new_be] = challenges_be[old_be];
    }
    Some(permuted_challenges)
}

fn precommitted_sumcheck_lsb_permutation(
    poly_opening_round_permutation_be: &[usize],
) -> Option<Vec<usize>> {
    let num_vars = poly_opening_round_permutation_be.len();
    let mut be_var_by_round: Vec<usize> = (0..num_vars).collect();
    be_var_by_round.sort_unstable_by_key(|&be_idx| poly_opening_round_permutation_be[be_idx]);

    let mut old_lsb_to_new_lsb = vec![0usize; num_vars];
    for (new_lsb, be_var_idx) in be_var_by_round.into_iter().enumerate() {
        let old_lsb = num_vars - 1 - be_var_idx;
        old_lsb_to_new_lsb[old_lsb] = new_lsb;
    }

    if old_lsb_to_new_lsb
        .iter()
        .enumerate()
        .all(|(old_lsb, &new_lsb)| old_lsb == new_lsb)
    {
        return None;
    }
    Some(old_lsb_to_new_lsb)
}

fn precommitted_sumcheck_inverse_index_permutation(
    coeffs_len: usize,
    poly_opening_round_permutation_be: &[usize],
) -> Option<Vec<usize>> {
    let num_vars = poly_opening_round_permutation_be.len();
    assert_eq!(
        coeffs_len,
        1usize << num_vars,
        "precommitted coeff vector length mismatch: len={} expected=2^{}",
        coeffs_len,
        num_vars
    );
    let old_lsb_to_new_lsb =
        precommitted_sumcheck_lsb_permutation(poly_opening_round_permutation_be)?;

    let mut new_lsb_to_old_lsb = vec![0usize; num_vars];
    for (old_lsb, &new_lsb) in old_lsb_to_new_lsb.iter().enumerate() {
        new_lsb_to_old_lsb[new_lsb] = old_lsb;
    }

    let inverse_permutation: Vec<usize> = (0..coeffs_len)
        .into_par_iter()
        .map(|new_idx| {
            let mut old_idx = 0usize;
            for new_lsb in 0..num_vars {
                let bit = (new_idx >> new_lsb) & 1usize;
                let old_lsb = new_lsb_to_old_lsb[new_lsb];
                old_idx |= bit << old_lsb;
            }
            old_idx
        })
        .collect();
    Some(inverse_permutation)
}

pub const TWO_PHASE_DEGREE_BOUND: usize = 2;

pub trait PrecomittedParams<F: JoltField>: SumcheckInstanceParams<F> {
    fn is_cycle_phase(&self) -> bool;
    fn is_cycle_phase_round(&self, round: usize) -> bool;
    fn is_address_phase_round(&self, round: usize) -> bool;
    fn cycle_alignment_rounds(&self) -> usize;
    fn address_alignment_rounds(&self) -> usize;
    fn record_cycle_challenge(&mut self, challenge: F::Challenge);
}

#[derive(Allocative)]
pub struct PrecomittedProver<F: JoltField, P: PrecomittedParams<F>> {
    params: P,
    value_poly: MultilinearPolynomial<F>,
    eq_poly: MultilinearPolynomial<F>,
    scale: F,
}

impl<F: JoltField, P: PrecomittedParams<F>> PrecomittedProver<F, P> {
    pub fn new(
        params: P,
        value_poly: MultilinearPolynomial<F>,
        eq_poly: MultilinearPolynomial<F>,
    ) -> Self {
        Self {
            params,
            value_poly,
            eq_poly,
            scale: F::one(),
        }
    }

    pub fn params(&self) -> &P {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut P {
        &mut self.params
    }

    fn compute_message_unscaled(&self, previous_claim_unscaled: F) -> UniPoly<F> {
        let half = self.value_poly.len() / 2;
        let value_poly = &self.value_poly;
        let eq_poly = &self.eq_poly;
        let evals: [F; TWO_PHASE_DEGREE_BOUND] = (0..half)
            .into_par_iter()
            .map(|j| {
                let value_evals = value_poly
                    .sumcheck_evals_array::<TWO_PHASE_DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_evals = eq_poly
                    .sumcheck_evals_array::<TWO_PHASE_DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                let mut out = [F::zero(); TWO_PHASE_DEGREE_BOUND];
                for i in 0..TWO_PHASE_DEGREE_BOUND {
                    out[i] = value_evals[i] * eq_evals[i];
                }
                out
            })
            .reduce(
                || [F::zero(); TWO_PHASE_DEGREE_BOUND],
                |mut acc, arr| {
                    acc.iter_mut().zip(arr.iter()).for_each(|(a, b)| *a += *b);
                    acc
                },
            );
        UniPoly::from_evals_and_hint(previous_claim_unscaled, &evals)
    }

    pub fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let is_active_round = if self.params.is_cycle_phase() {
            self.params.is_cycle_phase_round(round)
        } else {
            self.params.is_address_phase_round(round)
        };
        if !is_active_round {
            return UniPoly::from_coeff(vec![previous_claim * F::from_u64(2).inverse().unwrap()]);
        }

        let trailing_cap = if self.params.is_cycle_phase() {
            self.params.cycle_alignment_rounds()
        } else {
            self.params.address_alignment_rounds()
        };
        let num_trailing_variables = trailing_cap.saturating_sub(self.params.num_rounds());
        let scaling_factor = self.scale * F::one().mul_pow_2(num_trailing_variables);
        let prev_unscaled = previous_claim * scaling_factor.inverse().unwrap();
        let poly_unscaled = self.compute_message_unscaled(prev_unscaled);
        poly_unscaled * scaling_factor
    }

    pub fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        let is_active_round = if self.params.is_cycle_phase() {
            self.params.is_cycle_phase_round(round)
        } else {
            self.params.is_address_phase_round(round)
        };
        if !is_active_round {
            self.scale *= F::from_u64(2).inverse().unwrap();
            return;
        }

        self.value_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        if self.params.is_cycle_phase() {
            self.params.record_cycle_challenge(r_j);
        }
    }

    pub fn cycle_intermediate_claim(&self) -> F {
        let len = self.value_poly.len();
        assert_eq!(len, self.eq_poly.len());

        let mut sum = F::zero();
        for i in 0..len {
            sum += self.value_poly.get_bound_coeff(i) * self.eq_poly.get_bound_coeff(i);
        }
        sum * self.scale
    }

    pub fn final_claim_if_ready(&self) -> Option<F> {
        if self.value_poly.len() == 1 {
            Some(self.value_poly.get_bound_coeff(0))
        } else {
            None
        }
    }
}

pub fn precommitted_skip_round_scale<F: JoltField>(
    precommitted: &PrecommittedClaimReduction<F>,
) -> F {
    let cycle_gap_len =
        precommitted.cycle_phase_total_rounds - precommitted.cycle_phase_rounds.len();
    let address_gap_len =
        precommitted.address_phase_total_rounds - precommitted.address_phase_rounds.len();
    let gap_len = cycle_gap_len + address_gap_len;
    let two_inv = F::from_u64(2).inverse().unwrap();
    (0..gap_len).fold(F::one(), |acc, _| acc * two_inv)
}
