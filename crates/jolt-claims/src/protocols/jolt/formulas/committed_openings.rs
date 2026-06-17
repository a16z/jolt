//! Jolt committed-polynomial proof and final-opening orders.

use jolt_field::Field;

use super::super::{JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};
use super::dimensions::TracePolynomialOrder;
use super::error::JoltFormulaPointError;
use super::ra::JoltRaPolynomialLayout;

pub fn proof_commitment_order(layout: JoltRaPolynomialLayout) -> Vec<JoltCommittedPolynomial> {
    let mut polynomials = Vec::with_capacity(2 + layout.total());
    polynomials.push(JoltCommittedPolynomial::RdInc);
    polynomials.push(JoltCommittedPolynomial::RamInc);
    polynomials.extend((0..layout.instruction()).map(JoltCommittedPolynomial::InstructionRa));
    polynomials.extend((0..layout.ram()).map(JoltCommittedPolynomial::RamRa));
    polynomials.extend((0..layout.bytecode()).map(JoltCommittedPolynomial::BytecodeRa));
    polynomials
}

/// `committed_program_chunks` is `Some(bytecode_chunk_count)` in committed
/// program mode, which appends the trusted bytecode chunk and program-image
/// commitments to the batch.
pub fn final_opening_polynomial_order(
    layout: JoltRaPolynomialLayout,
    include_trusted_advice: bool,
    include_untrusted_advice: bool,
    committed_program_chunks: Option<usize>,
) -> Vec<JoltCommittedPolynomial> {
    let mut polynomials = Vec::with_capacity(
        2 + layout.total()
            + usize::from(include_trusted_advice)
            + usize::from(include_untrusted_advice)
            + committed_program_chunks.map_or(0, |chunk_count| chunk_count + 1),
    );
    polynomials.push(JoltCommittedPolynomial::RamInc);
    polynomials.push(JoltCommittedPolynomial::RdInc);
    polynomials.extend((0..layout.instruction()).map(JoltCommittedPolynomial::InstructionRa));
    polynomials.extend((0..layout.bytecode()).map(JoltCommittedPolynomial::BytecodeRa));
    polynomials.extend((0..layout.ram()).map(JoltCommittedPolynomial::RamRa));
    if include_trusted_advice {
        polynomials.push(JoltCommittedPolynomial::TrustedAdvice);
    }
    if include_untrusted_advice {
        polynomials.push(JoltCommittedPolynomial::UntrustedAdvice);
    }
    if let Some(chunk_count) = committed_program_chunks {
        polynomials.extend((0..chunk_count).map(JoltCommittedPolynomial::BytecodeChunk));
        polynomials.push(JoltCommittedPolynomial::ProgramImageInit);
    }
    polynomials
}

pub fn final_opening_ids(
    layout: JoltRaPolynomialLayout,
    include_trusted_advice: bool,
    include_untrusted_advice: bool,
    committed_program_chunks: Option<usize>,
) -> Vec<JoltOpeningId> {
    final_opening_polynomial_order(
        layout,
        include_trusted_advice,
        include_untrusted_advice,
        committed_program_chunks,
    )
    .into_iter()
    .map(final_opening_id)
    .collect()
}

pub fn final_opening_id(polynomial: JoltCommittedPolynomial) -> JoltOpeningId {
    match polynomial {
        JoltCommittedPolynomial::TrustedAdvice => {
            JoltOpeningId::trusted_advice(JoltRelationId::AdviceClaimReduction)
        }
        JoltCommittedPolynomial::UntrustedAdvice => {
            JoltOpeningId::untrusted_advice(JoltRelationId::AdviceClaimReduction)
        }
        polynomial => JoltOpeningId::committed(polynomial, final_opening_relation(polynomial)),
    }
}

pub fn final_opening_relation(polynomial: JoltCommittedPolynomial) -> JoltRelationId {
    match polynomial {
        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc => {
            JoltRelationId::IncClaimReduction
        }
        JoltCommittedPolynomial::InstructionRa(_)
        | JoltCommittedPolynomial::BytecodeRa(_)
        | JoltCommittedPolynomial::RamRa(_) => JoltRelationId::HammingWeightClaimReduction,
        JoltCommittedPolynomial::BytecodeChunk(_) => JoltRelationId::BytecodeClaimReduction,
        JoltCommittedPolynomial::ProgramImageInit => JoltRelationId::ProgramImageClaimReduction,
        JoltCommittedPolynomial::TrustedAdvice | JoltCommittedPolynomial::UntrustedAdvice => {
            JoltRelationId::AdviceClaimReduction
        }
    }
}

/// Lagrange factor for embedding a smaller polynomial's opening into the
/// top-left block of the unified final opening point: `1` on variables the
/// embedded point binds, `1 - r` on the rest.
pub fn advice_commitment_embedding_scale<F: Field>(
    opening_point: &[F],
    embedded_opening_point: &[F],
) -> F {
    commitment_embedding_scale(opening_point, embedded_opening_point)
}

pub fn commitment_embedding_scale<F: Field>(
    opening_point: &[F],
    embedded_opening_point: &[F],
) -> F {
    debug_assert!(
        embedded_opening_point
            .iter()
            .all(|challenge| opening_point.contains(challenge)),
        "embedded opening point must be a subset of the unified opening point"
    );
    opening_point
        .iter()
        .map(|challenge| {
            if embedded_opening_point.contains(challenge) {
                F::one()
            } else {
                F::one() - challenge
            }
        })
        .product()
}

/// Inputs to [`final_opening_point`], gathered from earlier verification
/// stages.
pub struct FinalOpeningPointInputs<'a, F: Field> {
    pub log_t: usize,
    pub log_k_chunk: usize,
    pub trace_order: TracePolynomialOrder,
    /// Stage 7 hamming-weight claim-reduction opening point.
    pub hamming_weight_opening_point: &'a [F],
    /// Stage 6 increment claim-reduction opening point (the stage 6 cycle
    /// challenges).
    pub inc_claim_reduction_opening_point: &'a [F],
    /// Final opening points of present precommitted polynomials, in stage 8
    /// batch order (trusted advice, untrusted advice).
    pub precommitted_anchor_points: &'a [&'a [F]],
}

/// Unified big-endian opening point for the stage 8 batched PCS opening.
///
/// When a precommitted polynomial spans more variables than the native trace
/// domain, its opening point anchors the batch (all dominant anchors must
/// agree). Otherwise the point is assembled from the stage 6 cycle challenges
/// and the stage 7 address challenges in the order the active trace layout
/// expects.
pub fn final_opening_point<F: Field>(
    inputs: FinalOpeningPointInputs<'_, F>,
) -> Result<Vec<F>, JoltFormulaPointError> {
    let native_main_vars = inputs.log_t + inputs.log_k_chunk;
    let mut dominant: Option<(usize, &[F])> = None;
    for (index, point) in inputs.precommitted_anchor_points.iter().enumerate() {
        if dominant.is_none_or(|(_, dominant_point)| point.len() > dominant_point.len()) {
            dominant = Some((index, point));
        }
    }
    if let Some((first, dominant_point)) = dominant {
        if dominant_point.len() > native_main_vars {
            for (index, point) in inputs.precommitted_anchor_points.iter().enumerate() {
                if point.len() == dominant_point.len() && *point != dominant_point {
                    return Err(JoltFormulaPointError::IncompatibleDominantAnchors {
                        first,
                        second: index,
                    });
                }
            }
            return Ok(dominant_point.to_vec());
        }
    }

    if inputs.hamming_weight_opening_point.len() < inputs.log_k_chunk {
        return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
            expected: inputs.log_k_chunk,
            got: inputs.hamming_weight_opening_point.len(),
        });
    }
    let r_address_stage7 = &inputs.hamming_weight_opening_point[..inputs.log_k_chunk];
    let r_cycle_stage6 = inputs.inc_claim_reduction_opening_point;
    match inputs.trace_order {
        TracePolynomialOrder::AddressMajor => Ok([r_cycle_stage6, r_address_stage7].concat()),
        TracePolynomialOrder::CycleMajor => {
            let native_cycle = &inputs.hamming_weight_opening_point[inputs.log_k_chunk..];
            if r_cycle_stage6.len() < native_cycle.len() {
                return Err(
                    JoltFormulaPointError::CycleChallengesShorterThanNativeCycle {
                        expected: native_cycle.len(),
                        got: r_cycle_stage6.len(),
                    },
                );
            }
            if &r_cycle_stage6[..native_cycle.len()] != native_cycle {
                return Err(JoltFormulaPointError::CycleMajorCyclePrefixMismatch);
            }
            let cycle_extra = &r_cycle_stage6[native_cycle.len()..];
            Ok([cycle_extra, r_address_stage7, native_cycle].concat())
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on unexpected errors")]

    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn layout() -> JoltRaPolynomialLayout {
        JoltRaPolynomialLayout::new(2, 1, 2).unwrap_or_else(|error| {
            panic!("test layout should be valid: {error}");
        })
    }

    #[test]
    fn proof_commitment_order_matches_proof_payload_order() {
        assert_eq!(
            proof_commitment_order(layout()),
            vec![
                JoltCommittedPolynomial::RdInc,
                JoltCommittedPolynomial::RamInc,
                JoltCommittedPolynomial::InstructionRa(0),
                JoltCommittedPolynomial::InstructionRa(1),
                JoltCommittedPolynomial::RamRa(0),
                JoltCommittedPolynomial::RamRa(1),
                JoltCommittedPolynomial::BytecodeRa(0),
            ]
        );
    }

    #[test]
    fn final_opening_order_matches_stage8_rlc_order() {
        assert_eq!(
            final_opening_polynomial_order(layout(), true, true, None),
            vec![
                JoltCommittedPolynomial::RamInc,
                JoltCommittedPolynomial::RdInc,
                JoltCommittedPolynomial::InstructionRa(0),
                JoltCommittedPolynomial::InstructionRa(1),
                JoltCommittedPolynomial::BytecodeRa(0),
                JoltCommittedPolynomial::RamRa(0),
                JoltCommittedPolynomial::RamRa(1),
                JoltCommittedPolynomial::TrustedAdvice,
                JoltCommittedPolynomial::UntrustedAdvice,
            ]
        );
    }

    #[test]
    fn final_opening_ids_use_sumcheck_sources() {
        assert_eq!(
            final_opening_ids(layout(), true, false, None),
            vec![
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RamInc,
                    JoltRelationId::IncClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RdInc,
                    JoltRelationId::IncClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::InstructionRa(0),
                    JoltRelationId::HammingWeightClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::InstructionRa(1),
                    JoltRelationId::HammingWeightClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::BytecodeRa(0),
                    JoltRelationId::HammingWeightClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RamRa(0),
                    JoltRelationId::HammingWeightClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RamRa(1),
                    JoltRelationId::HammingWeightClaimReduction,
                ),
                JoltOpeningId::trusted_advice(JoltRelationId::AdviceClaimReduction),
            ]
        );
    }

    #[test]
    fn embedding_scale_selects_variables_outside_embedded_point() {
        let opening_point = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let embedded_point = [Fr::from_u64(3)];

        assert_eq!(
            commitment_embedding_scale(&opening_point, &embedded_point),
            (Fr::from_u64(1) - Fr::from_u64(2)) * (Fr::from_u64(1) - Fr::from_u64(5))
        );
    }

    #[test]
    fn final_opening_point_passes_through_hamming_point_for_cycle_major() {
        let hamming_point: Vec<Fr> = (1..=6).map(Fr::from_u64).collect();
        let inc_point: Vec<Fr> = hamming_point[2..].to_vec();

        let point = final_opening_point(FinalOpeningPointInputs {
            log_t: 4,
            log_k_chunk: 2,
            trace_order: TracePolynomialOrder::CycleMajor,
            hamming_weight_opening_point: &hamming_point,
            inc_claim_reduction_opening_point: &inc_point,
            precommitted_anchor_points: &[],
        })
        .unwrap_or_else(|error| panic!("final opening point should assemble: {error}"));

        assert_eq!(point, hamming_point);
    }

    #[test]
    fn final_opening_point_orders_cycle_before_address_for_address_major() {
        let hamming_point: Vec<Fr> = (1..=6).map(Fr::from_u64).collect();
        let inc_point: Vec<Fr> = (11..=14).map(Fr::from_u64).collect();

        let point = final_opening_point(FinalOpeningPointInputs {
            log_t: 4,
            log_k_chunk: 2,
            trace_order: TracePolynomialOrder::AddressMajor,
            hamming_weight_opening_point: &hamming_point,
            inc_claim_reduction_opening_point: &inc_point,
            precommitted_anchor_points: &[],
        })
        .unwrap_or_else(|error| panic!("final opening point should assemble: {error}"));

        let expected: Vec<Fr> = inc_point
            .iter()
            .chain(&hamming_point[..2])
            .copied()
            .collect();
        assert_eq!(point, expected);
    }

    #[test]
    fn final_opening_point_anchors_on_dominant_precommitted_opening() {
        let hamming_point: Vec<Fr> = (1..=6).map(Fr::from_u64).collect();
        let inc_point: Vec<Fr> = hamming_point[2..].to_vec();
        let dominant: Vec<Fr> = (21..=28).map(Fr::from_u64).collect();
        let conflicting: Vec<Fr> = (31..=38).map(Fr::from_u64).collect();

        let point = final_opening_point(FinalOpeningPointInputs {
            log_t: 4,
            log_k_chunk: 2,
            trace_order: TracePolynomialOrder::CycleMajor,
            hamming_weight_opening_point: &hamming_point,
            inc_claim_reduction_opening_point: &inc_point,
            precommitted_anchor_points: &[&dominant, &dominant],
        })
        .unwrap_or_else(|error| panic!("final opening point should assemble: {error}"));
        assert_eq!(point, dominant);

        assert_eq!(
            final_opening_point(FinalOpeningPointInputs {
                log_t: 4,
                log_k_chunk: 2,
                trace_order: TracePolynomialOrder::CycleMajor,
                hamming_weight_opening_point: &hamming_point,
                inc_claim_reduction_opening_point: &inc_point,
                precommitted_anchor_points: &[&dominant, &conflicting],
            }),
            Err(JoltFormulaPointError::IncompatibleDominantAnchors {
                first: 0,
                second: 1
            })
        );
    }
}
