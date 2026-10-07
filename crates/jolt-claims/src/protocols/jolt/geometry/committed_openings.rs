//! Jolt committed-polynomial proof and final-opening orders.

use jolt_field::JoltField;

use super::super::{JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};
use super::dimensions::{CommitmentMatrixShape, TracePolynomialOrder};
use super::error::PointGeometryError;
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

fn final_opening_relation(polynomial: JoltCommittedPolynomial) -> JoltRelationId {
    match polynomial {
        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc => {
            JoltRelationId::IncClaimReduction
        }
        JoltCommittedPolynomial::InstructionRa(_)
        | JoltCommittedPolynomial::BytecodeRa(_)
        | JoltCommittedPolynomial::RamRa(_) => JoltRelationId::HammingWeightClaimReduction,
        JoltCommittedPolynomial::TrustedAdvice | JoltCommittedPolynomial::UntrustedAdvice => {
            JoltRelationId::AdviceClaimReduction
        }
        JoltCommittedPolynomial::BytecodeChunk(_) => JoltRelationId::BytecodeClaimReduction,
        JoltCommittedPolynomial::ProgramImageInit => JoltRelationId::ProgramImageClaimReduction,

        JoltCommittedPolynomial::BalancedIncDigit(_)
        | JoltCommittedPolynomial::BalancedIncCarry => JoltRelationId::HammingWeightClaimReduction,
    }
}

/// Coefficient placement in the unified commitment grid.
#[derive(Clone, Copy, Debug)]
pub enum CommitmentEmbedding {
    /// Native points are `[address, cycle]`; dense increments have no address variables.
    Trace {
        order: TracePolynomialOrder,
        log_t: usize,
    },
    /// A balanced matrix placed in the top-left corner of the unified balanced matrix.
    Precommitted,
}

/// Check the embedded point at its layout-selected coordinates and multiply
/// `(1-r)` over the zero-padding coordinates. Returns `None` for an oversized
/// domain, invalid trace width, or mismatched coordinate. Equal challenge values
/// at different coordinates remain distinct variables.
pub fn commitment_embedding_scale<F: JoltField>(
    opening_point: &[F],
    embedded_opening_point: &[F],
    embedding: CommitmentEmbedding,
) -> Option<F> {
    let total = opening_point.len();
    let native = embedded_opening_point.len();
    let padding = total.checked_sub(native)?;
    let positions = match embedding {
        CommitmentEmbedding::Trace { order, log_t } => {
            let _ = native.checked_sub(log_t)?;
            match order {
                TracePolynomialOrder::CycleMajor => [padding..total, 0..0],
                TracePolynomialOrder::AddressMajor => [log_t..native, 0..log_t],
            }
        }
        CommitmentEmbedding::Precommitted => {
            let grid = CommitmentMatrixShape::balanced(total);
            let poly = CommitmentMatrixShape::balanced(native);
            [
                grid.row_vars().checked_sub(poly.row_vars())?..grid.row_vars(),
                total.checked_sub(poly.column_vars())?..total,
            ]
        }
    };
    if !positions
        .iter()
        .cloned()
        .flatten()
        .zip(embedded_opening_point)
        .all(|(index, value)| opening_point.get(index) == Some(value))
    {
        return None;
    }
    Some(
        opening_point
            .iter()
            .enumerate()
            .filter(|(index, _)| !positions.iter().any(|range| range.contains(index)))
            .map(|(_, value)| F::one() - value)
            .product(),
    )
}

pub struct FinalOpeningPointInputs<'a, F: JoltField> {
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
pub fn final_opening_point<F: JoltField>(
    inputs: FinalOpeningPointInputs<'_, F>,
) -> Result<Vec<F>, PointGeometryError> {
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
                    return Err(PointGeometryError::IncompatibleDominantAnchors {
                        first,
                        second: index,
                    });
                }
            }
            return Ok(dominant_point.to_vec());
        }
    }

    if inputs.hamming_weight_opening_point.len() < inputs.log_k_chunk {
        return Err(PointGeometryError::OpeningPointLengthMismatch {
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
                return Err(PointGeometryError::CycleChallengesShorterThanNativeCycle {
                    expected: native_cycle.len(),
                    got: r_cycle_stage6.len(),
                });
            }
            if &r_cycle_stage6[..native_cycle.len()] != native_cycle {
                return Err(PointGeometryError::CycleMajorCyclePrefixMismatch);
            }
            let cycle_extra = &r_cycle_stage6[native_cycle.len()..];
            Ok([cycle_extra, r_address_stage7, native_cycle].concat())
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::panic,
        clippy::unwrap_used,
        reason = "tests fail loudly on unexpected errors"
    )]

    use super::*;
    use jolt_field::{Fr, Ring, Zero};

    #[test]
    fn embedding_scale_checks_positions_and_repeated_coordinates() {
        use jolt_poly::Polynomial;
        let r = Fr::from_u64(3);
        let embedding = CommitmentEmbedding::Trace {
            order: TracePolynomialOrder::CycleMajor,
            log_t: 1,
        };
        let value = Polynomial::new(vec![Fr::from_u64(5), Fr::from_u64(7)]).evaluate(&[r]);
        let padded = Polynomial::new(vec![
            Fr::from_u64(5),
            Fr::from_u64(7),
            Fr::zero(),
            Fr::zero(),
        ]);
        let scale = commitment_embedding_scale(&[r, r], &[r], embedding).unwrap();
        assert_eq!(scale * value, padded.evaluate(&[r, r]));
        let point = [Fr::from_u64(2), r];
        assert!(commitment_embedding_scale(&point, &[r, point[0]], embedding).is_none());
        assert!(commitment_embedding_scale(&point, &[point[0], r, r], embedding).is_none());
        assert!(commitment_embedding_scale(&point, &[point[0]], embedding).is_none());
    }

    #[test]
    fn embedding_scales_match_independently_placed_tables() {
        use jolt_poly::Polynomial;
        let native = [1, 2, 3, 4].map(Fr::from_u64).to_vec();
        let cases = [
            (
                CommitmentEmbedding::Trace {
                    order: TracePolynomialOrder::CycleMajor,
                    log_t: 1,
                },
                [0, 1, 2, 3],
                [2, 3],
            ),
            (
                CommitmentEmbedding::Trace {
                    order: TracePolynomialOrder::AddressMajor,
                    log_t: 1,
                },
                [0, 8, 4, 12],
                [1, 0],
            ),
            (CommitmentEmbedding::Precommitted, [0, 1, 4, 5], [1, 3]),
        ];
        for point in [[2, 3, 5, 7], [3, 3, 3, 3]] {
            let point = point.map(Fr::from_u64);
            for (embedding, indices, coordinates) in cases {
                let mut padded = vec![Fr::zero(); 16];
                for (index, value) in indices.into_iter().zip(&native) {
                    padded[index] = *value;
                }
                let own = coordinates.map(|index| point[index]);
                let value = Polynomial::new(native.clone()).evaluate(&own);
                let scale = commitment_embedding_scale(&point, &own, embedding).unwrap();
                assert_eq!(scale * value, Polynomial::new(padded).evaluate(&point));
            }
        }
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
            Err(PointGeometryError::IncompatibleDominantAnchors {
                first: 0,
                second: 1
            })
        );
    }
}
