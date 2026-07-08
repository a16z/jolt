//! Shared two-phase scheduling for precommitted polynomial claim reductions.
//!
//! Precommitted polynomials (trusted/untrusted advice today; committed bytecode
//! chunks and the program image in committed program mode) are smaller or larger
//! than the main trace-domain polynomials. All of them are reduced over a shared
//! reference domain so their final openings embed consistently into the stage 8
//! batched PCS opening. Mirrors `jolt-prover-legacy`'s
//! `zkvm/claim_reductions/precommitted.rs`, with the Dory globals
//! (`main_k`, `main_t`, layout, configured column count) parameter-passed.

use jolt_field::Field;

use super::super::dimensions::{CommitmentMatrixShape, TracePolynomialOrder};
use super::super::error::JoltFormulaPointError;

/// Degree bound shared by all two-phase precommitted reduction sumchecks.
pub const TWO_PHASE_DEGREE_BOUND: usize = 2;

/// Round counts of a two-phase precommitted claim reduction, shared by the
/// advice, committed-bytecode, and program-image reductions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PrecommittedReductionDimensions {
    cycle_phase_total_rounds: usize,
    address_phase_total_rounds: usize,
    has_address_phase: bool,
}

impl PrecommittedReductionDimensions {
    pub const fn new(
        cycle_phase_total_rounds: usize,
        address_phase_total_rounds: usize,
        has_address_phase: bool,
    ) -> Self {
        Self {
            cycle_phase_total_rounds,
            address_phase_total_rounds,
            has_address_phase,
        }
    }

    /// Full cycle-phase sumcheck round count, including rounds this
    /// polynomial skips.
    pub const fn cycle_phase_total_rounds(self) -> usize {
        self.cycle_phase_total_rounds
    }

    /// Full address-phase sumcheck round count, including rounds this
    /// polynomial skips.
    pub const fn address_phase_total_rounds(self) -> usize {
        self.address_phase_total_rounds
    }

    /// The address phase only runs for this polynomial when it has active
    /// address-phase rounds; otherwise the reduction finalizes at the
    /// cycle-phase handoff.
    pub const fn has_address_phase(self) -> bool {
        self.has_address_phase
    }
}

/// Common two-phase schedule surface of the per-reduction layout types
/// (advice, committed bytecode, program image), forwarding to the shared
/// [`PrecommittedClaimReduction`].
pub trait PrecommittedReductionLayout {
    fn precommitted(&self) -> &PrecommittedClaimReduction;

    fn dimensions(&self) -> PrecommittedReductionDimensions {
        self.precommitted().reduction_dimensions()
    }

    fn cycle_phase_opening_point<F: Field>(
        &self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        self.precommitted().cycle_phase_opening_point(challenges)
    }

    fn cycle_phase_variable_challenges<F: Field>(
        &self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        self.precommitted()
            .cycle_phase_variable_challenges(challenges)
    }

    fn address_phase_opening_point<F: Field>(
        &self,
        cycle_var_challenges: &[F],
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        self.precommitted()
            .address_phase_opening_point(cycle_var_challenges, challenges)
    }
}

/// Shared scheduling dimensions derived from the main trace domain and all
/// precommitted candidate domains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrecommittedSchedulingReference {
    pub main_total_vars: usize,
    pub reference_total_vars: usize,
    pub cycle_alignment_rounds: usize,
    pub address_rounds: usize,
    pub joint_col_vars: usize,
}

/// Per-polynomial two-phase round schedule projected from the shared reference
/// domain.
///
/// Unlike core's stateful counterpart, this layout is pure: the verifier passes
/// the recorded cycle-phase challenges explicitly when completing the address
/// phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecommittedClaimReduction {
    scheduling_reference: PrecommittedSchedulingReference,
    poly_opening_round_permutation_be: Vec<usize>,
    cycle_phase_rounds: Vec<usize>,
    cycle_phase_total_rounds: usize,
    address_phase_rounds: Vec<usize>,
    address_phase_total_rounds: usize,
}

impl PrecommittedClaimReduction {
    /// Compute shared scheduling dimensions from the main trace domain and
    /// precommitted candidate total-var counts.
    ///
    /// `joint_col_vars` mirrors core's
    /// `max(configured_main_num_columns().log_2(), reference_sigma)`: after the
    /// stage 6 Dory re-embedding the main matrix is balanced over
    /// `reference_total_vars`, so both operands equal
    /// `ceil(reference_total_vars / 2)`.
    pub fn scheduling_reference(
        main_total_vars: usize,
        candidates: &[usize],
        log_k_chunk: usize,
    ) -> PrecommittedSchedulingReference {
        let address_rounds = log_k_chunk;
        let max_precommitted = candidates.iter().copied().max().unwrap_or(0);
        let reference_total_vars = main_total_vars.max(max_precommitted);
        let cycle_alignment_rounds = reference_total_vars.saturating_sub(address_rounds);
        let joint_col_vars = CommitmentMatrixShape::balanced(reference_total_vars).column_vars();
        PrecommittedSchedulingReference {
            main_total_vars,
            reference_total_vars,
            cycle_alignment_rounds,
            address_rounds,
            joint_col_vars,
        }
    }

    pub fn new(
        poly_row_vars: usize,
        poly_col_vars: usize,
        scheduling_reference: PrecommittedSchedulingReference,
        trace_order: TracePolynomialOrder,
        log_t: usize,
    ) -> Result<Self, JoltFormulaPointError> {
        let has_precommitted_dominance =
            scheduling_reference.reference_total_vars > scheduling_reference.main_total_vars;
        let dense_cycle_prefix_rounds = if has_precommitted_dominance { log_t } else { 0 };
        let dory_opening_round_permutation_be = Self::reference_dory_opening_round_permutation_be(
            &scheduling_reference,
            trace_order,
            has_precommitted_dominance,
            dense_cycle_prefix_rounds,
        );
        let poly_opening_round_permutation_be = Self::project_dory_round_permutation_for_poly(
            &dory_opening_round_permutation_be,
            &scheduling_reference,
            poly_row_vars,
            poly_col_vars,
        )?;
        let (cycle_phase_rounds, address_phase_rounds) = Self::active_rounds_from_poly_permutation(
            &poly_opening_round_permutation_be,
            scheduling_reference.cycle_alignment_rounds,
        );
        Ok(Self {
            scheduling_reference,
            poly_opening_round_permutation_be,
            cycle_phase_rounds,
            cycle_phase_total_rounds: scheduling_reference.cycle_alignment_rounds,
            address_phase_rounds,
            address_phase_total_rounds: scheduling_reference.address_rounds,
        })
    }

    fn reference_dory_opening_round_permutation_be(
        reference: &PrecommittedSchedulingReference,
        trace_order: TracePolynomialOrder,
        has_precommitted_dominance: bool,
        dense_cycle_prefix_rounds: usize,
    ) -> Vec<usize> {
        let cycle_rounds = reference.cycle_alignment_rounds;
        let address_rounds = reference.address_rounds;
        let total_rounds = cycle_rounds + address_rounds;
        if has_precommitted_dominance {
            let address_rev = (cycle_rounds..total_rounds).rev();
            let t = dense_cycle_prefix_rounds.min(cycle_rounds);
            let prefix_rev = (0..cycle_rounds.saturating_sub(t)).rev();
            let dense_rev = (cycle_rounds.saturating_sub(t)..cycle_rounds).rev();
            return match trace_order {
                TracePolynomialOrder::CycleMajor => {
                    prefix_rev.chain(address_rev).chain(dense_rev).collect()
                }
                TracePolynomialOrder::AddressMajor => {
                    dense_rev.chain(address_rev).chain(prefix_rev).collect()
                }
            };
        }

        match trace_order {
            TracePolynomialOrder::CycleMajor => (0..total_rounds).rev().collect(),
            TracePolynomialOrder::AddressMajor => {
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
    ) -> Result<Vec<usize>, JoltFormulaPointError> {
        let total_full = reference.reference_total_vars;
        let sigma_full = reference.joint_col_vars;
        let nu_full = total_full.saturating_sub(sigma_full);
        if dory_opening_round_permutation_be.len() != total_full {
            return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
                expected: total_full,
                got: dory_opening_round_permutation_be.len(),
            });
        }
        if poly_row_vars > nu_full || poly_col_vars > sigma_full {
            return Err(JoltFormulaPointError::PolyDimsExceedReference {
                poly_row_vars,
                poly_col_vars,
                reference_row_vars: nu_full,
                reference_col_vars: sigma_full,
            });
        }
        let row_be = &dory_opening_round_permutation_be[..nu_full];
        let col_be = &dory_opening_round_permutation_be[nu_full..nu_full + sigma_full];
        let row_tail = &row_be[nu_full - poly_row_vars..];
        let col_tail = &col_be[sigma_full - poly_col_vars..];
        Ok([row_tail, col_tail].concat())
    }

    fn active_rounds_from_poly_permutation(
        poly_opening_round_permutation_be: &[usize],
        cycle_alignment_rounds: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut cycle_phase_rounds = Vec::new();
        let mut address_phase_rounds = Vec::new();
        for &global_round in poly_opening_round_permutation_be {
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

    pub fn reduction_dimensions(&self) -> PrecommittedReductionDimensions {
        PrecommittedReductionDimensions::new(
            self.cycle_phase_total_rounds,
            self.address_phase_total_rounds,
            self.num_address_phase_rounds() > 0,
        )
    }

    pub fn num_address_phase_rounds(&self) -> usize {
        self.address_phase_rounds.len()
    }

    /// Indices of the cycle-phase rounds this polynomial actively participates
    /// in (sorted ascending, deduplicated). Inactive rounds contribute a `1/2`
    /// factor instead.
    pub fn cycle_phase_rounds(&self) -> &[usize] {
        &self.cycle_phase_rounds
    }

    /// Indices of the address-phase rounds this polynomial actively
    /// participates in. Same conventions as [`Self::cycle_phase_rounds`].
    pub fn address_phase_rounds(&self) -> &[usize] {
        &self.address_phase_rounds
    }

    /// Big-endian Dory opening-round permutation projected onto this
    /// polynomial's `(row_vars, col_vars)` rectangle: the first `row_vars`
    /// entries describe the row-side rounds, the rest the column-side rounds.
    pub fn poly_opening_round_permutation_be(&self) -> &[usize] {
        &self.poly_opening_round_permutation_be
    }

    pub const fn cycle_alignment_rounds(&self) -> usize {
        self.scheduling_reference.cycle_alignment_rounds
    }

    /// Total cycle-phase sumcheck rounds, including rounds this polynomial
    /// skips.
    pub const fn cycle_phase_total_rounds(&self) -> usize {
        self.cycle_phase_total_rounds
    }

    /// Total address-phase sumcheck rounds, including rounds this polynomial
    /// skips.
    pub const fn address_phase_total_rounds(&self) -> usize {
        self.address_phase_total_rounds
    }

    /// The `(1/2)^cycle_gap` factor contributed by inactive cycle-phase
    /// rounds. This is the cycle-only counterpart of
    /// [`precommitted_skip_round_scale`], used when the reduction finishes in
    /// the cycle phase.
    pub fn cycle_phase_skip_scale<F: Field>(&self) -> F {
        skip_round_scale(self.cycle_phase_total_rounds - self.cycle_phase_rounds.len())
    }

    fn cycle_challenge_for_round<F: Field>(
        &self,
        cycle_var_challenges: &[F],
        round: usize,
    ) -> Result<F, JoltFormulaPointError> {
        let idx = self
            .cycle_phase_rounds
            .binary_search(&round)
            .map_err(|_| JoltFormulaPointError::InactiveCycleRound { round })?;
        cycle_var_challenges.get(idx).copied().ok_or(
            JoltFormulaPointError::ChallengeLengthMismatch {
                expected: self.cycle_phase_rounds.len(),
                got: cycle_var_challenges.len(),
            },
        )
    }

    /// Cycle-phase challenges this polynomial actively binds, in ascending
    /// round order. The verifier carries these into the address phase.
    pub fn cycle_phase_variable_challenges<F: Field>(
        &self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        if challenges.len() != self.cycle_phase_total_rounds {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: self.cycle_phase_total_rounds,
                got: challenges.len(),
            });
        }
        Ok(self
            .cycle_phase_rounds
            .iter()
            .map(|&round| challenges[round])
            .collect())
    }

    /// Big-endian opening point cached at the cycle-phase handoff.
    pub fn cycle_phase_opening_point<F: Field>(
        &self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let mut point = self.cycle_phase_variable_challenges(challenges)?;
        point.reverse();
        Ok(point)
    }

    /// Big-endian permutation-ordered point for a reduction that completes in
    /// the cycle phase. Errors if the polynomial still has active
    /// address-phase rounds.
    pub fn cycle_phase_permuted_opening_point<F: Field>(
        &self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        if challenges.len() != self.cycle_phase_total_rounds {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: self.cycle_phase_total_rounds,
                got: challenges.len(),
            });
        }
        let cycle_round_limit = self.cycle_alignment_rounds();
        self.poly_opening_round_permutation_be
            .iter()
            .map(|&global_round| {
                if global_round < cycle_round_limit {
                    Ok(challenges[global_round])
                } else {
                    Err(JoltFormulaPointError::CyclePhaseNotFinal {
                        active_address_rounds: self.num_address_phase_rounds(),
                    })
                }
            })
            .collect()
    }

    /// [`cycle_phase_permuted_opening_point`] recovered from the produced
    /// `cycle_phase_opening_point` (the reverse-ordered active cycle challenges)
    /// instead of the full sumcheck challenges. The cycle-phase opening point
    /// holds exactly this polynomial's active cycle challenges, and the Dory
    /// permutation only references active cycle rounds (see
    /// `active_rounds_from_poly_permutation`), so the permuted point is fully
    /// recoverable without the rounds this polynomial skips. Lets a relation
    /// object's `resolve_public` derive the cycle-phase `FinalScale` from the
    /// opening point it produced, mirroring the address-phase
    /// `*_at_opening_point` helpers.
    ///
    /// [`cycle_phase_permuted_opening_point`]: Self::cycle_phase_permuted_opening_point
    pub fn cycle_phase_permuted_from_opening_point<F: Field>(
        &self,
        opening_point: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        if opening_point.len() != self.cycle_phase_rounds.len() {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: self.cycle_phase_rounds.len(),
                got: opening_point.len(),
            });
        }
        let mut cycle_var_challenges = opening_point.to_vec();
        cycle_var_challenges.reverse();
        let cycle_round_limit = self.cycle_alignment_rounds();
        self.poly_opening_round_permutation_be
            .iter()
            .map(|&global_round| {
                if global_round < cycle_round_limit {
                    self.cycle_challenge_for_round(&cycle_var_challenges, global_round)
                } else {
                    Err(JoltFormulaPointError::CyclePhaseNotFinal {
                        active_address_rounds: self.num_address_phase_rounds(),
                    })
                }
            })
            .collect()
    }

    /// Big-endian final opening point in Dory opening-round order, assembled
    /// from the recorded cycle-phase challenges and the address-phase
    /// sumcheck challenges.
    pub fn address_phase_opening_point<F: Field>(
        &self,
        cycle_var_challenges: &[F],
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        if cycle_var_challenges.len() != self.cycle_phase_rounds.len() {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: self.cycle_phase_rounds.len(),
                got: cycle_var_challenges.len(),
            });
        }
        if challenges.len() != self.address_phase_total_rounds {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: self.address_phase_total_rounds,
                got: challenges.len(),
            });
        }
        let cycle_round_limit = self.cycle_alignment_rounds();
        self.poly_opening_round_permutation_be
            .iter()
            .map(|&global_round| {
                if global_round < cycle_round_limit {
                    self.cycle_challenge_for_round(cycle_var_challenges, global_round)
                } else {
                    Ok(challenges[global_round - cycle_round_limit])
                }
            })
            .collect()
    }
}

/// The `(1/2)^gap` factor contributed by all rounds (cycle and address) this
/// polynomial skips across both phases. Used for the final output claim when
/// the reduction completes in the address phase.
pub fn precommitted_skip_round_scale<F: Field>(precommitted: &PrecommittedClaimReduction) -> F {
    let gap = (precommitted.cycle_phase_total_rounds - precommitted.cycle_phase_rounds.len())
        + (precommitted.address_phase_total_rounds - precommitted.address_phase_rounds.len());
    skip_round_scale(gap)
}

fn skip_round_scale<F: Field>(gap: usize) -> F {
    if gap == 0 {
        return F::one();
    }
    let two_inv = F::from_u64(2).inv_or_zero();
    (0..gap).fold(F::one(), |scale, _| scale * two_inv)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on unexpected errors")]

    use super::*;
    use crate::protocols::jolt::geometry::dimensions::TracePolynomialOrder;
    use jolt_field::{Fr, FromPrimitiveInt, Invertible};

    #[test]
    fn cycle_skip_scale_counts_inactive_cycle_rounds() {
        let scheduling_reference = PrecommittedSchedulingReference {
            main_total_vars: 3,
            reference_total_vars: 3,
            cycle_alignment_rounds: 3,
            address_rounds: 0,
            joint_col_vars: 0,
        };
        let precommitted = PrecommittedClaimReduction::new(
            1,
            0,
            scheduling_reference,
            TracePolynomialOrder::CycleMajor,
            3,
        )
        .unwrap_or_else(|error| panic!("schedule should build: {error}"));

        let two_inv = Fr::from_u64(2).inv_or_zero();
        assert_eq!(precommitted.cycle_phase_rounds(), &[0]);
        assert_eq!(precommitted.num_address_phase_rounds(), 0);
        assert_eq!(
            precommitted.cycle_phase_skip_scale::<Fr>(),
            two_inv * two_inv
        );
    }

    #[test]
    fn cycle_skip_scale_ignores_unrun_address_dummy_rounds() {
        let scheduling_reference = PrecommittedSchedulingReference {
            main_total_vars: 4,
            reference_total_vars: 4,
            cycle_alignment_rounds: 2,
            address_rounds: 2,
            joint_col_vars: 0,
        };
        let precommitted = PrecommittedClaimReduction::new(
            1,
            0,
            scheduling_reference,
            TracePolynomialOrder::CycleMajor,
            4,
        )
        .unwrap_or_else(|error| panic!("schedule should build: {error}"));

        let two_inv = Fr::from_u64(2).inv_or_zero();
        assert_eq!(precommitted.cycle_phase_rounds(), &[0]);
        assert_eq!(precommitted.num_address_phase_rounds(), 0);
        // The cycle-phase handoff scale excludes the address rounds this
        // polynomial never participates in...
        assert_eq!(precommitted.cycle_phase_skip_scale::<Fr>(), two_inv);
        // ...while the full final scale counts both phase gaps.
        assert_eq!(
            precommitted_skip_round_scale::<Fr>(&precommitted),
            two_inv * two_inv * two_inv
        );
    }

    #[test]
    fn dominant_precommitted_permutation_reorders_dense_cycle_prefix() {
        let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(4, &[6, 3], 2);
        assert_eq!(
            scheduling_reference,
            PrecommittedSchedulingReference {
                main_total_vars: 4,
                reference_total_vars: 6,
                cycle_alignment_rounds: 4,
                address_rounds: 2,
                joint_col_vars: 3,
            }
        );

        let precommitted = PrecommittedClaimReduction::new(
            3,
            3,
            scheduling_reference,
            TracePolynomialOrder::CycleMajor,
            2,
        )
        .unwrap_or_else(|error| panic!("schedule should build: {error}"));
        assert_eq!(
            precommitted.poly_opening_round_permutation_be(),
            &[1, 0, 5, 4, 3, 2]
        );
        assert_eq!(precommitted.cycle_phase_rounds(), &[0, 1, 2, 3]);
        assert_eq!(precommitted.address_phase_rounds(), &[0, 1]);
    }

    #[test]
    fn cycle_phase_permuted_recovers_from_opening_point() {
        // A precommitted-dominant, cycle-completed schedule: the Dory permutation
        // reorders the cycle rounds, so the permuted opening point differs from
        // the produced (reverse-ordered) cycle opening point — exercising the
        // recovery rather than a trivial reverse.
        let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(2, &[5], 0);
        let precommitted = PrecommittedClaimReduction::new(
            2,
            3,
            scheduling_reference,
            TracePolynomialOrder::CycleMajor,
            2,
        )
        .unwrap_or_else(|error| panic!("schedule should build: {error}"));
        assert_eq!(precommitted.num_address_phase_rounds(), 0);

        let challenges: Vec<Fr> = (0..precommitted.cycle_phase_total_rounds())
            .map(|index| Fr::from_u64(10 + index as u64))
            .collect();
        let opening_point = precommitted
            .cycle_phase_opening_point(&challenges)
            .unwrap_or_else(|error| panic!("cycle opening point: {error}"));
        let from_challenges = precommitted
            .cycle_phase_permuted_opening_point(&challenges)
            .unwrap_or_else(|error| panic!("permuted from challenges: {error}"));
        // Guard against a vacuous test: the permutation is not just the reverse.
        assert_ne!(from_challenges, opening_point);
        assert_eq!(
            precommitted
                .cycle_phase_permuted_from_opening_point(&opening_point)
                .unwrap_or_else(|error| panic!("permuted from opening point: {error}")),
            from_challenges,
        );
    }
}
