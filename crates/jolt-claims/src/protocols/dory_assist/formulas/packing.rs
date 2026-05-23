use jolt_field::RingCore;

use crate::{challenge, opening, public};

use super::super::{
    DoryAssistChallengeId, DoryAssistExpr, DoryAssistOpeningId, DoryAssistPublicId,
    DoryAssistRelationClaims, DoryAssistRelationId, PackingChallenge,
};
use super::dimensions::{DoryAssistSumcheckSpec, PrefixPackingDimensions};

pub const fn prefix_packing_sumcheck(
    dimensions: PrefixPackingDimensions,
) -> DoryAssistSumcheckSpec {
    DoryAssistSumcheckSpec::boolean(dimensions.prefix_vars(), 1)
}

pub fn prefix_packing<F, I>(
    dimensions: PrefixPackingDimensions,
    reduced_claims: I,
) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
    I: IntoIterator<Item = DoryAssistOpeningId>,
{
    let reduced_claims = reduced_claims.into_iter().collect::<Vec<_>>();
    let output = reduced_claims
        .iter()
        .enumerate()
        .fold(DoryAssistExpr::zero(), |acc, (index, opening_id)| {
            acc + prefix_weight(index) * opening(*opening_id)
        });

    DoryAssistRelationClaims::new(
        DoryAssistRelationId::PrefixPacking,
        prefix_packing_sumcheck(dimensions),
        opening(dense_witness_opening()),
        output,
    )
    .with_input_challenges([DoryAssistChallengeId::from(PackingChallenge::PrefixPoint)])
}

pub fn dense_witness_opening() -> DoryAssistOpeningId {
    DoryAssistOpeningId::dense_witness(DoryAssistRelationId::PrefixPacking)
}

pub fn prefix_weight<F>(index: usize) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    public(DoryAssistPublicId::PrefixPackingWeight(index))
}

pub fn prefix_point_challenge<F>() -> DoryAssistExpr<F>
where
    F: RingCore,
{
    challenge(DoryAssistChallengeId::from(PackingChallenge::PrefixPoint))
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "tests fail loudly on invalid fixture dimensions"
    )]

    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::super::super::{
        DoryAssistVirtualPolynomial, GtPolynomial, PackingPolynomial, WiringPolynomial,
    };

    fn dimensions() -> PrefixPackingDimensions {
        PrefixPackingDimensions::new(6, 4, 2).unwrap()
    }

    #[test]
    fn prefix_packing_claims_expose_expected_dependencies() {
        let gt_claim = DoryAssistOpeningId::virtual_polynomial(
            DoryAssistVirtualPolynomial::Gt(GtPolynomial::MulOutput),
            DoryAssistRelationId::GtMultiplication,
        );
        let wiring_claim = DoryAssistOpeningId::virtual_polynomial(
            DoryAssistVirtualPolynomial::Wiring(WiringPolynomial::Source),
            DoryAssistRelationId::WiringGt,
        );
        let claims = prefix_packing::<Fr, _>(dimensions(), [gt_claim, wiring_claim]);

        assert_eq!(claims.id, DoryAssistRelationId::PrefixPacking);
        assert_eq!(
            claims.input.required_openings,
            vec![dense_witness_opening()]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![gt_claim, wiring_claim]
        );
        assert_eq!(
            claims.output.required_publics,
            vec![
                DoryAssistPublicId::PrefixPackingWeight(0),
                DoryAssistPublicId::PrefixPackingWeight(1),
            ]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![DoryAssistChallengeId::from(PackingChallenge::PrefixPoint)]
        );
    }

    #[test]
    fn prefix_packing_evaluates_weighted_claim_fold() {
        let first = DoryAssistOpeningId::virtual_polynomial(
            DoryAssistVirtualPolynomial::Packing(PackingPolynomial::PrefixSelector),
            DoryAssistRelationId::PrefixPacking,
        );
        let second = DoryAssistOpeningId::virtual_polynomial(
            DoryAssistVirtualPolynomial::Packing(PackingPolynomial::DenseWitness),
            DoryAssistRelationId::PrefixPacking,
        );
        let claims = prefix_packing::<Fr, _>(dimensions(), [first, second]);
        let zero = Fr::from_u64(0);

        let output = claims.output.expression().evaluate(
            |opening| match *opening {
                id if id == first => Fr::from_u64(5),
                id if id == second => Fr::from_u64(7),
                _ => zero,
            },
            |_| zero,
            |public| match *public {
                DoryAssistPublicId::PrefixPackingWeight(0) => Fr::from_u64(11),
                DoryAssistPublicId::PrefixPackingWeight(1) => Fr::from_u64(13),
                _ => zero,
            },
        );

        assert_eq!(output, Fr::from_u64(146));
    }
}
