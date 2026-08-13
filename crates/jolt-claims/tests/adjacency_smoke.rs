use jolt_claims::protocols::jolt::relations::spartan::OuterRemainderInputClaims;
use jolt_claims::protocols::jolt::{JoltOpeningId, JoltRelationId, JoltVirtualPolynomial};
use jolt_claims::{ClaimAdjacency, ClaimArity, ClaimEdge};

/// Pins the derive-emitted adjacency of one real claim struct against its
/// `#[opening(UnivariateSkip, from = SpartanOuter)]` declaration.
#[test]
fn outer_remainder_input_adjacency_matches_declaration() {
    let edges = <OuterRemainderInputClaims<()> as ClaimAdjacency>::EDGES;
    assert_eq!(
        edges,
        &[ClaimEdge {
            id: JoltOpeningId::virtual_polynomial(
                JoltVirtualPolynomial::UnivariateSkip,
                JoltRelationId::SpartanOuter,
            ),
            arity: ClaimArity::Scalar,
        }]
    );
}
