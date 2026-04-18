use allocative::Allocative;
use rayon::prelude::*;
use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::utils::math::Math;
use crate::zkvm::bytecode::chunks::committed_lanes;

#[derive(Clone, Debug)]
pub enum PrecommittedPolynomial<F: JoltField> {
    Dense(MultilinearPolynomial<F>),
    BytecodeChunk {
        chunk_index: usize,
        chunk_cycle_len: usize,
    },
    ProgramImage {
        words: Arc<Vec<u64>>,
        padded_len: usize,
    },
}

impl<F: JoltField> PrecommittedPolynomial<F> {
    pub(crate) fn original_len(&self) -> usize {
        match self {
            Self::Dense(poly) => poly.original_len(),
            Self::BytecodeChunk {
                chunk_cycle_len, ..
            } => committed_lanes() * *chunk_cycle_len,
            Self::ProgramImage { padded_len, .. } => *padded_len,
        }
    }
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
    pub cycle_var_challenges: Vec<F::Challenge>,
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
        poly_row_vars: usize,
        poly_col_vars: usize,
        scheduling_reference: PrecommittedSchedulingReference,
    ) -> Self {
        let has_precommitted_dominance =
            scheduling_reference.reference_total_vars > scheduling_reference.main_total_vars;
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
            cycle_var_challenges: vec![],
            poly_opening_round_permutation_be,
            cycle_phase_rounds,
            cycle_phase_total_rounds: scheduling_reference.cycle_alignment_rounds,
            address_phase_rounds,
            address_phase_total_rounds: scheduling_reference.address_rounds,
        }
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
        self.cycle_phase_rounds.contains(&round)
    }

    /// Indices of the cycle-phase rounds that this poly actively participates
    /// in (i.e. rounds where the verifier evaluates the poly rather than
    /// scaling by 1/2). The vector is sorted ascending and deduplicated.
    pub fn cycle_phase_rounds(&self) -> &[usize] {
        &self.cycle_phase_rounds
    }

    /// Indices of the address-phase rounds that this poly actively
    /// participates in. Same conventions as [`Self::cycle_phase_rounds`].
    pub fn address_phase_rounds(&self) -> &[usize] {
        &self.address_phase_rounds
    }

    /// Big-endian round-permutation projected onto this poly's
    /// `(poly_row_vars, poly_col_vars)` rectangle.
    ///
    /// The slice is `poly_row_vars + poly_col_vars` long: the first
    /// `poly_row_vars` entries describe the row-side rounds, the rest the
    /// column-side rounds. Pair this with
    /// [`precommitted_sumcheck_inverse_index_permutation`] to permute a
    /// length-`2^len` coefficient vector into opening order, instead of
    /// re-deriving it from `scheduling_reference`.
    pub fn poly_opening_round_permutation_be(&self) -> &[usize] {
        &self.poly_opening_round_permutation_be
    }

    /// The `(1/2)^cycle_gap` factor that "non-active" cycle-phase rounds
    /// contribute to the running scale. Returns `F::one()` when there are
    /// no inactive cycle-phase rounds.
    ///
    /// This is the cycle-only counterpart of
    /// [`precommitted_skip_round_scale`], intended for callers that need
    /// the scale strictly at the cycle-to-address handoff (e.g. when
    /// constructing the address-phase prover).
    #[inline]
    pub fn cycle_phase_skip_scale(&self) -> F {
        let cycle_gap_len = self.cycle_phase_total_rounds - self.cycle_phase_rounds.len();
        if cycle_gap_len == 0 {
            return F::one();
        }
        let two_inv = F::from_u64(2).inverse().unwrap();
        (0..cycle_gap_len).fold(F::one(), |acc, _| acc * two_inv)
    }

    pub fn is_address_phase_active_round(&self, round: usize) -> bool {
        self.address_phase_rounds.contains(&round)
    }

    #[inline]
    pub fn is_address_phase_round(&self, round: usize) -> bool {
        self.address_phase_rounds.contains(&round)
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
    ) -> OpeningPoint<BIG_ENDIAN, F> {
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

pub fn precommitted_eq_evals_with_scaling<F, C>(
    challenges_be: &[C],
    scaling_factor: Option<F>,
    precommitted: &PrecommittedClaimReduction<F>,
) -> Vec<F>
where
    C: Copy + Send + Sync + Into<F>,
    F: JoltField + std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
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

/// Inverse index permutation for permuting a precommitted polynomial's
/// coefficient vector into the order implied by `poly_opening_round_permutation_be`.
///
/// Returns `Some(perm)` such that `perm[new_idx] = old_idx`, suitable for
/// driving an out-of-place permute of a length-`coeffs_len` vector. Returns
/// `None` when the requested permutation is the identity, so callers can
/// short-circuit and skip the permute entirely.
///
/// `coeffs_len` must equal `1 << poly_opening_round_permutation_be.len()`;
/// asserts otherwise.
pub fn precommitted_sumcheck_inverse_index_permutation(
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

    pub fn set_scale(&mut self, scale: F) {
        self.scale = scale;
    }

    pub fn scale(&self) -> F {
        self.scale
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::JoltField;
    use ark_bn254::Fr;
    use num_traits::One;

    fn make_reduction(
        poly_opening_round_permutation_be: Vec<usize>,
        cycle_phase_rounds: Vec<usize>,
        cycle_phase_total_rounds: usize,
        address_phase_rounds: Vec<usize>,
        address_phase_total_rounds: usize,
    ) -> PrecommittedClaimReduction<Fr> {
        let scheduling_reference = PrecommittedSchedulingReference {
            main_total_vars: cycle_phase_total_rounds + address_phase_total_rounds,
            reference_total_vars: cycle_phase_total_rounds + address_phase_total_rounds,
            cycle_alignment_rounds: cycle_phase_total_rounds,
            address_rounds: address_phase_total_rounds,
            joint_col_vars: 0,
        };
        PrecommittedClaimReduction {
            scheduling_reference,
            cycle_var_challenges: vec![],
            poly_opening_round_permutation_be,
            cycle_phase_rounds,
            cycle_phase_total_rounds,
            address_phase_rounds,
            address_phase_total_rounds,
        }
    }

    #[test]
    fn poly_opening_round_permutation_be_returns_stored_field() {
        let perm = vec![3usize, 0, 1, 2];
        let r = make_reduction(perm.clone(), vec![0, 1], 2, vec![0, 1], 2);
        assert_eq!(r.poly_opening_round_permutation_be(), perm.as_slice());
    }

    #[test]
    fn cycle_and_address_phase_rounds_accessors_match_internal_storage() {
        let cycle = vec![0usize, 2, 3];
        let address = vec![1usize];
        let r = make_reduction(vec![0, 1, 2, 3], cycle.clone(), 4, address.clone(), 2);
        assert_eq!(r.cycle_phase_rounds(), cycle.as_slice());
        assert_eq!(r.address_phase_rounds(), address.as_slice());
    }

    #[test]
    fn cycle_phase_skip_scale_is_one_when_no_gap() {
        let r = make_reduction(vec![0, 1], vec![0, 1], 2, vec![], 0);
        assert_eq!(r.cycle_phase_skip_scale(), Fr::one());
    }

    #[test]
    fn cycle_phase_skip_scale_is_two_inverse_per_inactive_round() {
        let two_inv = Fr::from_u64(2).inverse().unwrap();
        // 1 inactive cycle round
        let r1 = make_reduction(vec![0], vec![0], 2, vec![], 0);
        assert_eq!(r1.cycle_phase_skip_scale(), two_inv);
        // 3 inactive cycle rounds
        let r3 = make_reduction(vec![0], vec![0], 4, vec![], 0);
        assert_eq!(r3.cycle_phase_skip_scale(), two_inv * two_inv * two_inv);
        // address-phase gap must NOT contribute (this is the cycle-only flavour)
        let r_ignore = make_reduction(vec![0], vec![0], 1, vec![], 5);
        assert_eq!(r_ignore.cycle_phase_skip_scale(), Fr::one());
    }

    #[test]
    fn cycle_phase_skip_scale_agrees_with_full_skip_when_address_gap_is_zero() {
        let r = make_reduction(vec![0, 1], vec![0], 3, vec![0, 1], 2);
        // cycle_gap = 3 - 1 = 2, address_gap = 2 - 2 = 0
        // full = (1/2)^2; cycle_only = (1/2)^2; they should agree.
        let full = precommitted_skip_round_scale(&r);
        assert_eq!(r.cycle_phase_skip_scale(), full);
    }

    #[test]
    fn inverse_index_permutation_returns_none_for_identity() {
        // BE descending = identity LSB permutation (no reordering).
        let identity_be: Vec<usize> = (0..4).rev().collect();
        let perm = precommitted_sumcheck_inverse_index_permutation(1 << 4, &identity_be);
        assert!(
            perm.is_none(),
            "identity permutation should be reported as None, got Some(len={})",
            perm.map(|p| p.len()).unwrap_or(0),
        );
    }

    #[test]
    fn inverse_index_permutation_is_a_genuine_permutation_when_nontrivial() {
        // Swap the two LSBs by reversing the BE round order partially.
        let poly_perm_be: Vec<usize> = vec![0, 1, 3, 2];
        let coeffs_len = 1usize << poly_perm_be.len();
        let perm = precommitted_sumcheck_inverse_index_permutation(coeffs_len, &poly_perm_be)
            .expect("non-identity permutation expected for this input");
        assert_eq!(perm.len(), coeffs_len);
        let mut seen = vec![false; coeffs_len];
        for (new_idx, &old_idx) in perm.iter().enumerate() {
            assert!(
                old_idx < coeffs_len,
                "perm[{new_idx}] = {old_idx} is out of bounds for coeffs_len={coeffs_len}",
            );
            assert!(
                !seen[old_idx],
                "perm contains duplicate old_idx={old_idx} (at new_idx={new_idx})",
            );
            seen[old_idx] = true;
        }
    }
}
