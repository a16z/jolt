use jolt_field::RingCore;

use crate::{challenge, constant, opening};

use super::super::{
    DoryAssistChallengeId, DoryAssistOpeningId, DoryAssistRelationClaims, DoryAssistRelationId,
    DoryAssistVirtualPolynomial, WiringChallenge, WiringPolynomial,
};
use super::dimensions::{DoryAssistSumcheckSpec, WiringDimensions};

pub const fn copy_zero_check_sumcheck(dimensions: WiringDimensions) -> DoryAssistSumcheckSpec {
    dimensions.sumcheck()
}

pub fn gt_wiring<F>(dimensions: WiringDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    copy_zero_check(dimensions, DoryAssistRelationId::WiringGt)
}

pub fn g1_wiring<F>(dimensions: WiringDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    copy_zero_check(dimensions, DoryAssistRelationId::WiringG1)
}

pub fn g2_wiring<F>(dimensions: WiringDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    copy_zero_check(dimensions, DoryAssistRelationId::WiringG2)
}

pub fn copy_zero_check<F>(
    dimensions: WiringDimensions,
    relation: DoryAssistRelationId,
) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let edge_batch = challenge(DoryAssistChallengeId::from(WiringChallenge::EdgeBatch));
    let output = edge_batch
        * opening(enabled_opening(relation))
        * (opening(source_opening(relation)) - opening(destination_opening(relation)));

    DoryAssistRelationClaims::new(
        relation,
        copy_zero_check_sumcheck(dimensions),
        constant(F::zero()),
        output,
    )
    .with_input_challenges([
        DoryAssistChallengeId::from(WiringChallenge::CopyPoint),
        DoryAssistChallengeId::from(WiringChallenge::TupleCompression),
    ])
}

pub fn copy_zero_check_output_openings(relation: DoryAssistRelationId) -> [DoryAssistOpeningId; 3] {
    [
        enabled_opening(relation),
        source_opening(relation),
        destination_opening(relation),
    ]
}

pub fn source_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    wiring_opening(WiringPolynomial::Source, relation)
}

pub fn destination_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    wiring_opening(WiringPolynomial::Destination, relation)
}

pub fn enabled_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    wiring_opening(WiringPolynomial::Enabled, relation)
}

pub fn difference_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    wiring_opening(WiringPolynomial::Difference, relation)
}

fn wiring_opening(
    polynomial: WiringPolynomial,
    relation: DoryAssistRelationId,
) -> DoryAssistOpeningId {
    DoryAssistOpeningId::virtual_polynomial(
        DoryAssistVirtualPolynomial::Wiring(polynomial),
        relation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn copy_zero_check_claims_expose_expected_dependencies() {
        let dimensions = WiringDimensions::new(6);
        let claims = gt_wiring::<Fr>(dimensions);

        assert_eq!(claims.id, DoryAssistRelationId::WiringGt);
        assert_eq!(claims.sumcheck, copy_zero_check_sumcheck(dimensions));
        assert!(claims.input.required_openings.is_empty());
        assert_eq!(
            claims.output.required_openings,
            copy_zero_check_output_openings(DoryAssistRelationId::WiringGt).to_vec()
        );
        assert_eq!(
            claims.required_challenges(),
            vec![
                DoryAssistChallengeId::from(WiringChallenge::CopyPoint),
                DoryAssistChallengeId::from(WiringChallenge::TupleCompression),
                DoryAssistChallengeId::from(WiringChallenge::EdgeBatch),
            ]
        );
        assert!(claims.required_publics().is_empty());
    }

    #[test]
    fn copy_zero_check_evaluates_enabled_source_destination_difference() {
        let claims = g1_wiring::<Fr>(WiringDimensions::new(3));
        let zero = Fr::from_u64(0);

        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                id if id == enabled_opening(DoryAssistRelationId::WiringG1) => Fr::from_u64(1),
                id if id == source_opening(DoryAssistRelationId::WiringG1) => Fr::from_u64(13),
                id if id == destination_opening(DoryAssistRelationId::WiringG1) => Fr::from_u64(5),
                _ => zero,
            },
            |_| Fr::from_u64(3),
            |_| zero,
        );

        assert_eq!(output, Fr::from_u64(24));
    }
}
