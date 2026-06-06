use jolt_field::{FromPrimitiveInt, RingCore};

use super::super::{DoryAssistProtocolClaims, DoryAssistRelationId};
use super::dimensions::DoryAssistDimensions;
use super::{composition, dory_reduce, g1, g2, gt, miller_loop, wiring};

pub const CANONICAL_RELATION_ORDER: [DoryAssistRelationId; 30] = [
    DoryAssistRelationId::GtExponentiation,
    DoryAssistRelationId::GtExponentiationDigitSelector,
    DoryAssistRelationId::GtExponentiationBasePower,
    DoryAssistRelationId::GtExponentiationDigitBitness,
    DoryAssistRelationId::GtExponentiationShift,
    DoryAssistRelationId::GtExponentiationBoundary,
    DoryAssistRelationId::GtMultiplication,
    DoryAssistRelationId::G1ScalarMultiplication,
    DoryAssistRelationId::G1ScalarMultiplicationShift,
    DoryAssistRelationId::G1ScalarMultiplicationBoundary,
    DoryAssistRelationId::G1Addition,
    DoryAssistRelationId::G2ScalarMultiplication,
    DoryAssistRelationId::G2ScalarMultiplicationShift,
    DoryAssistRelationId::G2ScalarMultiplicationBoundary,
    DoryAssistRelationId::G2Addition,
    DoryAssistRelationId::MillerLoopLineStep,
    DoryAssistRelationId::MillerLoopLineEvaluation,
    DoryAssistRelationId::MillerLoopPairProduct,
    DoryAssistRelationId::MillerLoopAccumulator,
    DoryAssistRelationId::MillerLoopBoundary,
    DoryAssistRelationId::DoryReduceGtTransition,
    DoryAssistRelationId::DoryReduceG1Transition,
    DoryAssistRelationId::DoryReduceG2Transition,
    DoryAssistRelationId::DoryReduceScalarFold,
    DoryAssistRelationId::DoryReduceStateChain,
    DoryAssistRelationId::DoryReduceBoundary,
    DoryAssistRelationId::WiringGt,
    DoryAssistRelationId::WiringG1,
    DoryAssistRelationId::WiringG2,
    DoryAssistRelationId::PrefixPacking,
];

pub fn protocol_claims<F>(dimensions: DoryAssistDimensions) -> DoryAssistProtocolClaims<F>
where
    F: RingCore + FromPrimitiveInt,
{
    let packing_catalog = composition::prefix_packing_catalog(dimensions);

    DoryAssistProtocolClaims::new(vec![
        gt::exponentiation(dimensions.gt),
        gt::exponentiation_digit_selector(dimensions.gt),
        gt::exponentiation_base_power(dimensions.gt),
        gt::exponentiation_digit_bitness(dimensions.gt),
        gt::exponentiation_shift(dimensions.gt),
        gt::exponentiation_boundary(dimensions.gt),
        gt::multiplication(dimensions.gt),
        g1::scalar_multiplication(dimensions.g1),
        g1::scalar_multiplication_shift(dimensions.g1),
        g1::scalar_multiplication_boundary(dimensions.g1),
        g1::addition(dimensions.g1),
        g2::scalar_multiplication(dimensions.g2),
        g2::scalar_multiplication_shift(dimensions.g2),
        g2::scalar_multiplication_boundary(dimensions.g2),
        g2::addition(dimensions.g2),
        miller_loop::line_step(dimensions.miller_loop),
        miller_loop::line_evaluation(dimensions.miller_loop),
        miller_loop::pair_product(dimensions.miller_loop),
        miller_loop::accumulator(dimensions.miller_loop),
        miller_loop::boundary(dimensions.miller_loop),
        dory_reduce::gt_transition(dimensions.dory_reduce),
        dory_reduce::g1_transition(dimensions.dory_reduce),
        dory_reduce::g2_transition(dimensions.dory_reduce),
        dory_reduce::scalar_fold(dimensions.dory_reduce),
        dory_reduce::state_chain(dimensions.dory_reduce),
        dory_reduce::boundary(dimensions.dory_reduce),
        wiring::gt_wiring(dimensions.wiring),
        wiring::g1_wiring(dimensions.wiring),
        wiring::g2_wiring(dimensions.wiring),
        composition::prefix_packing_claims(dimensions.packing, &packing_catalog),
    ])
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        clippy::unwrap_used,
        reason = "tests fail loudly on invalid fixture dimensions"
    )]

    use std::collections::BTreeSet;

    use jolt_field::Fr;

    use super::super::super::{DoryAssistOpeningId, DoryAssistPublicId, DoryAssistValueRef};
    use super::super::composition;
    use super::super::dimensions::{
        DoryReduceDimensions, G1Dimensions, G2Dimensions, GtDimensions, MillerLoopDimensions,
        PrefixPackingDimensions, WiringDimensions,
    };
    use super::super::packing;
    use super::*;

    fn dimensions() -> DoryAssistDimensions {
        let unpacked = DoryAssistDimensions::new(
            GtDimensions::new(7, 2, 3),
            G1Dimensions::new(8, 2, 3),
            G2Dimensions::new(8, 2, 3),
            MillerLoopDimensions::new(7, 2, 8),
            DoryReduceDimensions::new(2, 1),
            WiringDimensions::new(6),
            PrefixPackingDimensions::new(0, 0, 0).unwrap(),
        );
        let packing = composition::prefix_packing_catalog(unpacked)
            .minimal_dimensions()
            .unwrap();

        DoryAssistDimensions::new(
            unpacked.gt,
            unpacked.g1,
            unpacked.g2,
            unpacked.miller_loop,
            unpacked.dory_reduce,
            unpacked.wiring,
            packing,
        )
    }

    #[test]
    fn protocol_claims_follow_canonical_relation_order() {
        let claims = protocol_claims::<Fr>(dimensions());
        let actual_order = claims
            .relations
            .iter()
            .map(|relation| relation.id)
            .collect::<Vec<_>>();

        assert_eq!(actual_order, CANONICAL_RELATION_ORDER);
    }

    #[test]
    fn protocol_claims_have_no_duplicate_relation_ids() {
        let claims = protocol_claims::<Fr>(dimensions());
        let relation_ids = claims
            .relations
            .iter()
            .map(|relation| relation.id)
            .collect::<Vec<_>>();
        let unique_relation_ids = relation_ids.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(relation_ids.len(), unique_relation_ids.len());
        assert_eq!(relation_ids.len(), CANONICAL_RELATION_ORDER.len());
    }

    #[test]
    fn protocol_claims_surface_all_relation_dependencies() {
        let claims = protocol_claims::<Fr>(dimensions());
        let protocol_openings = claims
            .required_openings()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let protocol_publics = claims
            .required_publics()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let protocol_challenges = claims
            .required_challenges()
            .into_iter()
            .collect::<BTreeSet<_>>();

        for relation in &claims {
            for opening in relation.required_openings() {
                assert!(
                    protocol_openings.contains(&opening),
                    "missing opening {opening:?} from relation {:?}",
                    relation.id
                );
            }
            for public in relation.required_publics() {
                assert!(
                    protocol_publics.contains(&public),
                    "missing public {public:?} from relation {:?}",
                    relation.id
                );
            }
            for challenge in relation.required_challenges() {
                assert!(
                    protocol_challenges.contains(&challenge),
                    "missing challenge {challenge:?} from relation {:?}",
                    relation.id
                );
            }
        }
    }

    #[test]
    fn prefix_packing_relation_matches_composition_catalog() {
        let dimensions = dimensions();
        let claims = protocol_claims::<Fr>(dimensions);
        let catalog = composition::prefix_packing_catalog(dimensions);
        let packing_relation = claims
            .relation(DoryAssistRelationId::PrefixPacking)
            .expect("prefix packing relation exists");

        assert_eq!(
            packing_relation.input.required_openings,
            vec![packing::dense_witness_opening()]
        );
        assert_eq!(
            packing_relation.output.required_openings,
            catalog.openings()
        );
        assert_eq!(
            packing_relation.output.required_publics,
            (0..catalog.num_claims())
                .map(DoryAssistPublicId::PrefixPackingWeight)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            packing_relation.sumcheck.rounds,
            dimensions.packing.prefix_vars()
        );
    }

    #[test]
    fn miller_loop_copy_edge_witness_endpoints_are_packable_in_protocol_bundle() {
        let dimensions = dimensions();
        let catalog_openings = composition::prefix_packing_catalog(dimensions)
            .openings()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let protocol_openings = protocol_claims::<Fr>(dimensions)
            .required_openings()
            .into_iter()
            .collect::<BTreeSet<_>>();

        for edge in composition::miller_loop_copy_constraints() {
            for endpoint in [edge.source, edge.target] {
                if let DoryAssistValueRef::Witness { .. } = endpoint {
                    let opening = endpoint
                        .witness_opening()
                        .expect("witness endpoint has an opening");
                    assert!(
                        catalog_openings.contains(&opening),
                        "copy endpoint {opening:?} missing from packing catalog"
                    );
                    assert!(
                        protocol_openings.contains(&opening),
                        "copy endpoint {opening:?} missing from protocol openings"
                    );
                }
            }
        }
    }

    #[test]
    fn protocol_exposes_single_dense_witness_opening() {
        let dense_opening_count = protocol_claims::<Fr>(dimensions())
            .required_openings()
            .into_iter()
            .filter(|opening| {
                *opening == DoryAssistOpeningId::dense_witness(DoryAssistRelationId::PrefixPacking)
            })
            .count();

        assert_eq!(dense_opening_count, 1);
    }
}
