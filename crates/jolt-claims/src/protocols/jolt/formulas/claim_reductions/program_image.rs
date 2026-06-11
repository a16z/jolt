//! Two-phase program-image (initial RAM) claim reduction.
//!
//! In committed program mode, stage 4 stages the scalar program-image
//! contribution to `Val_init(r_address)` as the virtual opening
//! `ProgramImageInitContributionRw` instead of having the verifier materialize
//! the initial RAM words. This reduction binds that scalar to the trusted
//! `ProgramImageInit` commitment over the shared precommitted schedule.
//! Mirrors `jolt-core`'s `zkvm/claim_reductions/program_image.rs`.

use jolt_field::{Field, RingCore};

use crate::{opening, public};

use super::super::super::{
    JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationClaims,
    JoltRelationId, JoltVirtualPolynomial, ProgramImageClaimReductionPublic,
};
use super::super::dimensions::{log2_power_of_two, CommitmentMatrixShape, TracePolynomialOrder};
use super::super::error::JoltFormulaPointError;
use super::precommitted::{
    precommitted_skip_round_scale, PrecommittedClaimReduction, PrecommittedReductionDimensions,
    PrecommittedReductionLayout, PrecommittedSchedulingReference,
};

/// Committed length of the program-image polynomial: the initial RAM
/// bytecode-word slice padded to a power of two (at least two words).
pub fn padded_program_image_len_words(program_image_len_words: usize) -> usize {
    program_image_len_words.next_power_of_two().max(2)
}

/// Total-var count of the committed program-image polynomial, used as this
/// reduction's candidate in the shared precommitted scheduling reference.
pub fn precommitted_candidate(program_image_len_words: usize) -> usize {
    log2_power_of_two(padded_program_image_len_words(program_image_len_words))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramImageClaimReductionLayout {
    image_shape: CommitmentMatrixShape,
    precommitted: PrecommittedClaimReduction,
    start_index: usize,
    padded_len_words: usize,
}

impl ProgramImageClaimReductionLayout {
    /// `start_index` is the remapped RAM address of the first program-image
    /// word (`remap_address(min_bytecode_address)`).
    pub fn balanced(
        trace_order: TracePolynomialOrder,
        log_t: usize,
        scheduling_reference: PrecommittedSchedulingReference,
        program_image_len_words: usize,
        start_index: usize,
    ) -> Self {
        let padded_len_words = padded_program_image_len_words(program_image_len_words);
        let image_shape = CommitmentMatrixShape::balanced(log2_power_of_two(padded_len_words));
        let precommitted = PrecommittedClaimReduction::new(
            image_shape.row_vars(),
            image_shape.column_vars(),
            scheduling_reference,
            trace_order,
            log_t,
        );
        Self {
            image_shape,
            precommitted,
            start_index,
            padded_len_words,
        }
    }

    pub const fn image_shape(&self) -> CommitmentMatrixShape {
        self.image_shape
    }

    pub const fn start_index(&self) -> usize {
        self.start_index
    }

    pub const fn padded_len_words(&self) -> usize {
        self.padded_len_words
    }

    /// `FinalScale` value when the reduction completes in the cycle phase
    /// (i.e. no active address-phase rounds remain). `r_addr_rw` is the RAM
    /// address component of the `RamVal` opening from RAM read-write checking.
    pub fn cycle_phase_final_output_scale<F: Field>(
        &self,
        r_addr_rw: &[F],
        challenges: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        let opening_point = self
            .precommitted
            .cycle_phase_permuted_opening_point(challenges)?;
        let eq_eval =
            eval_shifted_eq_poly_at_opening_point(r_addr_rw, self.start_index, &opening_point)?;
        Ok(eq_eval * self.precommitted.cycle_phase_skip_scale::<F>())
    }

    /// `FinalScale` value when the reduction completes in the address phase.
    pub fn address_phase_final_output_scale<F: Field>(
        &self,
        r_addr_rw: &[F],
        cycle_var_challenges: &[F],
        challenges: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        let opening_point = self
            .precommitted
            .address_phase_opening_point(cycle_var_challenges, challenges)?;
        let eq_eval =
            eval_shifted_eq_poly_at_opening_point(r_addr_rw, self.start_index, &opening_point)?;
        Ok(eq_eval * precommitted_skip_round_scale::<F>(&self.precommitted))
    }
}

impl PrecommittedReductionLayout for ProgramImageClaimReductionLayout {
    fn precommitted(&self) -> &PrecommittedClaimReduction {
        &self.precommitted
    }
}

pub fn cycle_phase<F>(dimensions: PrecommittedReductionDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let output = if dimensions.has_address_phase() {
        opening(cycle_phase_program_image_opening())
    } else {
        final_output_expr()
    };

    JoltRelationClaims::new(
        JoltRelationId::ProgramImageClaimReductionCyclePhase,
        dimensions.cycle_sumcheck(),
        opening(ram_val_check_contribution_opening()),
        output,
    )
}

pub fn address_phase<F>(dimensions: PrecommittedReductionDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    JoltRelationClaims::new(
        JoltRelationId::ProgramImageClaimReduction,
        dimensions.address_sumcheck(),
        opening(cycle_phase_program_image_opening()),
        final_output_expr(),
    )
}

fn final_output_expr<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(
        ProgramImageClaimReductionPublic::FinalScale,
    )) * opening(final_program_image_opening())
}

pub fn cycle_phase_output_openings(
    dimensions: PrecommittedReductionDimensions,
) -> Vec<JoltOpeningId> {
    if dimensions.has_address_phase() {
        vec![cycle_phase_program_image_opening()]
    } else {
        vec![final_program_image_opening()]
    }
}

pub fn ram_val_check_contribution_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::ProgramImageInitContributionRw,
        JoltRelationId::RamValCheck,
    )
}

pub fn cycle_phase_program_image_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::ProgramImageInit,
        JoltRelationId::ProgramImageClaimReductionCyclePhase,
    )
}

pub fn final_program_image_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::ProgramImageInit,
        JoltRelationId::ProgramImageClaimReduction,
    )
}

/// Evaluate the shifted program-image eq slice at the reduction's opening
/// point without materializing the slice.
///
/// The slice is `eq_slice[j] = eq(r_addr, start_index + j)` for `j` in
/// `0..2^m` where `m = opening_point_be.len()`; this computes its multilinear
/// extension at `opening_point_be` with a carry-propagation DP over the
/// address bits (low to high), tracking whether the running sum
/// `start_index + y` has produced a carry into the current bit.
fn eval_shifted_eq_poly_at_opening_point<F: Field>(
    r_addr_be: &[F],
    start_index: usize,
    opening_point_be: &[F],
) -> Result<F, JoltFormulaPointError> {
    let ell = r_addr_be.len();
    let m = opening_point_be.len();
    if m > ell {
        return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
            expected: ell,
            got: m,
        });
    }

    let mut dp0 = F::one();
    let mut dp1 = F::zero();

    for old_lsb in 0..ell {
        let start_bit = ((start_index >> old_lsb) & 1) as u8;
        let r_addr_bit = r_addr_be[ell - 1 - old_lsb];
        let k0 = F::one() - r_addr_bit;
        let k1 = r_addr_bit;
        let y_var = old_lsb < m;
        let r_y = if y_var {
            opening_point_be[m - 1 - old_lsb]
        } else {
            F::zero()
        };

        let mut next_dp0 = F::zero();
        let mut next_dp1 = F::zero();

        let update_state = |weight: F, carry: u8, next_dp0: &mut F, next_dp1: &mut F| {
            if weight.is_zero() {
                return;
            }

            if y_var {
                let sum0 = start_bit + carry;
                let k_bit0 = sum0 & 1;
                let carry0 = (sum0 >> 1) & 1;
                let addr_factor0 = if k_bit0 == 1 { k1 } else { k0 };
                let y_factor0 = F::one() - r_y;
                if carry0 == 0 {
                    *next_dp0 += weight * addr_factor0 * y_factor0;
                } else {
                    *next_dp1 += weight * addr_factor0 * y_factor0;
                }

                let sum1 = start_bit + carry + 1;
                let k_bit1 = sum1 & 1;
                let carry1 = (sum1 >> 1) & 1;
                let addr_factor1 = if k_bit1 == 1 { k1 } else { k0 };
                if carry1 == 0 {
                    *next_dp0 += weight * addr_factor1 * r_y;
                } else {
                    *next_dp1 += weight * addr_factor1 * r_y;
                }
            } else {
                let sum0 = start_bit + carry;
                let k_bit0 = sum0 & 1;
                let carry0 = (sum0 >> 1) & 1;
                let addr_factor0 = if k_bit0 == 1 { k1 } else { k0 };
                if carry0 == 0 {
                    *next_dp0 += weight * addr_factor0;
                } else {
                    *next_dp1 += weight * addr_factor0;
                }
            }
        };

        update_state(dp0, 0, &mut next_dp0, &mut next_dp1);
        update_state(dp1, 1, &mut next_dp0, &mut next_dp1);
        dp0 = next_dp0;
        dp1 = next_dp1;
    }

    Ok(dp0 + dp1)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on unexpected errors")]

    use super::*;
    use crate::protocols::jolt::JoltPublicId;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn naive_shifted_eq_eval(r_addr_be: &[Fr], start_index: usize, opening_point_be: &[Fr]) -> Fr {
        let padded_len = 1usize << opening_point_be.len();
        let address_eq = EqPolynomial::<Fr>::evals(r_addr_be, None);
        let opening_eq = EqPolynomial::<Fr>::evals(opening_point_be, None);
        (0..padded_len)
            .map(|offset| {
                address_eq[(start_index + offset) % address_eq.len()] * opening_eq[offset]
            })
            .sum()
    }

    #[test]
    fn padded_length_is_a_power_of_two_with_a_two_word_floor() {
        assert_eq!(padded_program_image_len_words(0), 2);
        assert_eq!(padded_program_image_len_words(1), 2);
        assert_eq!(padded_program_image_len_words(3), 4);
        assert_eq!(padded_program_image_len_words(8), 8);
        assert_eq!(precommitted_candidate(3), 2);
        assert_eq!(precommitted_candidate(1024), 10);
    }

    #[test]
    fn shifted_eq_dp_matches_naive_slice_evaluation() {
        let r_addr: Vec<Fr> = [3, 5, 7, 11].into_iter().map(fr).collect();
        let opening_point: Vec<Fr> = [13, 17].into_iter().map(fr).collect();

        // 14 and 15 wrap past 2^ell, exercising the DP's carry-out path.
        for start_index in [0usize, 3, 4, 9, 12, 14, 15] {
            let dp = eval_shifted_eq_poly_at_opening_point(&r_addr, start_index, &opening_point)
                .unwrap_or_else(|error| panic!("shifted eq should evaluate: {error}"));
            assert_eq!(
                dp,
                naive_shifted_eq_eval(&r_addr, start_index, &opening_point),
                "start_index={start_index}"
            );
        }
    }

    #[test]
    fn shifted_eq_dp_handles_full_width_opening_points() {
        let r_addr: Vec<Fr> = [3, 5, 7].into_iter().map(fr).collect();
        let opening_point: Vec<Fr> = [13, 17, 19].into_iter().map(fr).collect();

        let dp = eval_shifted_eq_poly_at_opening_point(&r_addr, 2, &opening_point)
            .unwrap_or_else(|error| panic!("shifted eq should evaluate: {error}"));
        assert_eq!(dp, naive_shifted_eq_eval(&r_addr, 2, &opening_point));
    }

    #[test]
    fn shifted_eq_dp_rejects_oversized_opening_points() {
        let r_addr = [fr(3)];
        let opening_point = [fr(5), fr(7)];

        assert_eq!(
            eval_shifted_eq_poly_at_opening_point(&r_addr, 0, &opening_point),
            Err(JoltFormulaPointError::OpeningPointLengthMismatch {
                expected: 1,
                got: 2,
            })
        );
    }

    #[test]
    fn cycle_phase_with_address_phase_exposes_expected_dependencies() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        let claims = cycle_phase::<Fr>(dimensions);

        assert_eq!(
            claims.id,
            JoltRelationId::ProgramImageClaimReductionCyclePhase
        );
        assert_eq!(claims.sumcheck, dimensions.cycle_sumcheck());
        assert_eq!(
            claims.input.required_openings,
            vec![ram_val_check_contribution_opening()]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![cycle_phase_program_image_opening()]
        );
        assert!(claims.required_challenges().is_empty());
        assert!(claims.required_publics().is_empty());
        assert_eq!(
            cycle_phase_output_openings(dimensions),
            vec![cycle_phase_program_image_opening()]
        );
    }

    #[test]
    fn cycle_phase_without_address_phase_exposes_final_scale() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, false);
        let claims = cycle_phase::<Fr>(dimensions);

        assert_eq!(
            claims.output.required_openings,
            vec![final_program_image_opening()]
        );
        assert_eq!(
            claims.required_publics(),
            vec![JoltPublicId::from(
                ProgramImageClaimReductionPublic::FinalScale
            )]
        );
        assert_eq!(
            cycle_phase_output_openings(dimensions),
            vec![final_program_image_opening()]
        );
    }

    #[test]
    fn address_phase_evaluates_like_core_formula() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        let claims = address_phase::<Fr>(dimensions);

        let intermediate = fr(11);
        let final_claim = fr(13);
        let final_scale = fr(17);
        let zero = fr(0);

        assert_eq!(claims.id, JoltRelationId::ProgramImageClaimReduction);
        assert_eq!(claims.sumcheck, dimensions.address_sumcheck());

        let input = claims.input.expression().evaluate(
            |id| {
                if *id == cycle_phase_program_image_opening() {
                    intermediate
                } else {
                    zero
                }
            },
            |_| zero,
            |_| zero,
        );
        let output = claims.output.expression().evaluate(
            |id| {
                if *id == final_program_image_opening() {
                    final_claim
                } else {
                    zero
                }
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::ProgramImageClaimReduction(
                    ProgramImageClaimReductionPublic::FinalScale,
                ) => final_scale,
                _ => zero,
            },
        );

        assert_eq!(input, intermediate);
        assert_eq!(output, final_scale * final_claim);
    }

    #[test]
    fn final_output_scale_combines_shifted_eq_and_skip_scale() {
        let log_t = 8;
        let log_k_chunk = 4;
        let program_image_len_words = 4;
        let start_index = 4;
        let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(
            log_t + log_k_chunk,
            &[precommitted_candidate(program_image_len_words)],
            log_k_chunk,
        );
        let layout = ProgramImageClaimReductionLayout::balanced(
            TracePolynomialOrder::CycleMajor,
            log_t,
            scheduling_reference,
            program_image_len_words,
            start_index,
        );
        assert_eq!(layout.padded_len_words(), 4);
        assert_eq!(layout.image_shape(), CommitmentMatrixShape::balanced(2));

        let precommitted = layout.precommitted();
        assert_eq!(precommitted.num_address_phase_rounds(), 0);

        let challenges: Vec<Fr> = (0..precommitted.cycle_phase_total_rounds())
            .map(|index| fr(40 + index as u64))
            .collect();
        let r_addr_rw: Vec<Fr> = (1..=6).map(fr).collect();

        let opening_point = precommitted
            .cycle_phase_permuted_opening_point(&challenges)
            .unwrap_or_else(|error| panic!("cycle phase point should normalize: {error}"));
        let expected =
            eval_shifted_eq_poly_at_opening_point::<Fr>(&r_addr_rw, start_index, &opening_point)
                .unwrap_or_else(|error| panic!("shifted eq should evaluate: {error}"))
                * precommitted.cycle_phase_skip_scale::<Fr>();

        let scale = layout
            .cycle_phase_final_output_scale(&r_addr_rw, &challenges)
            .unwrap_or_else(|error| panic!("final output scale should compute: {error}"));
        assert_eq!(scale, expected);
    }
}
