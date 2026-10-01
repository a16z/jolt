//! Symbolic relations composing ordinary Jolt with field-inline constraints.

use super::claims::{
    ComposedClaims, ComposedExpr, FieldInlineBytecodeReadRafInputs, OuterInputs, OuterOutputs,
    ProductInputs, ProductOutputs, UniskipInputs, UniskipOutputs,
};
use super::geometry::{
    SPARTAN_PRODUCT_BASE_LANES, SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
    SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use super::ComposedOpeningId;
use crate::protocols::field_inline::geometry::{
    product as field_product, spartan as field_spartan,
};
use crate::protocols::field_inline::{
    FieldInlineOpeningId, FieldInlineRelationId, FieldInlineVirtualPolynomial,
};
use crate::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
use crate::protocols::jolt::geometry::spartan::{
    self, SpartanOuterDimensions, SpartanProductDimensions,
};
use crate::protocols::jolt::relations::spartan as base;
use crate::protocols::jolt::{
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltOpeningId, JoltRelationId,
    SpartanOuterPublic, SpartanProductVirtualizationPublic,
};
use crate::{derived, opening, Expr, NoChallenges, Source, SumcheckDomain, SymbolicSumcheck, Term};
use jolt_field::Ring;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};

fn lift<F>(expr: Expr<F, JoltOpeningId, JoltDerivedId, JoltChallengeId>) -> ComposedExpr<F> {
    Expr {
        terms: expr
            .terms
            .into_iter()
            .map(|term| Term {
                coefficient: term.coefficient,
                factors: term
                    .factors
                    .into_iter()
                    .map(|source| match source {
                        Source::Opening(id) => Source::Opening(id.into()),
                        Source::Derived(id) => Source::Derived(id),
                        Source::Challenge(id) => Source::Challenge(id),
                    })
                    .collect(),
            })
            .collect(),
    }
}

#[derive(Clone)]
pub struct OuterRemainder {
    shape: SpartanOuterDimensions,
}

impl SymbolicSumcheck for OuterRemainder {
    type RelationId = JoltRelationId;
    type OpeningId = ComposedOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanOuterDimensions;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = OuterInputs<C>;
    type Outputs<C> = OuterOutputs<C>;
    fn new(shape: Self::Shape) -> Self {
        Self { shape }
    }
    fn id() -> JoltRelationId {
        JoltRelationId::SpartanOuter
    }
    fn rounds(&self) -> usize {
        self.shape.remainder_rounds()
    }
    fn degree(&self) -> usize {
        base::OuterRemainder::new(self.shape.clone()).degree()
    }
    fn input_expression<F: Ring>(&self) -> ComposedExpr<F> {
        opening(spartan::outer_uniskip_opening())
    }
    fn output_expression<F: Ring>(&self) -> ComposedExpr<F> {
        let (az, bz) = base::OuterRemainder::new(self.shape.clone()).output_factor_expressions();
        let mut az = lift(az);
        let mut bz = lift(bz);
        let ids = (self.shape.variables().len()..).zip(field_spartan::outer_output_openings());
        for (index, id) in ids {
            az = az
                + derived(JoltDerivedId::from(SpartanOuterPublic::AzWeight(index))) * opening(id);
            bz = bz
                + derived(JoltDerivedId::from(SpartanOuterPublic::BzWeight(index))) * opening(id);
        }
        derived(JoltDerivedId::from(SpartanOuterPublic::TauKernel)) * az * bz
    }
}

#[derive(Clone)]
pub struct ProductUniskip {
    base: base::ProductUniskip,
}
impl SymbolicSumcheck for ProductUniskip {
    type RelationId = JoltRelationId;
    type OpeningId = ComposedOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanProductDimensions;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = UniskipInputs<C>;
    type Outputs<C> = UniskipOutputs<C>;
    fn new(shape: Self::Shape) -> Self {
        Self {
            base: base::ProductUniskip::new(shape),
        }
    }
    fn id() -> JoltRelationId {
        JoltRelationId::SpartanProductVirtualization
    }
    fn rounds(&self) -> usize {
        self.base.rounds()
    }
    fn degree(&self) -> usize {
        SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE
    }
    fn domain(&self) -> SumcheckDomain {
        SumcheckDomain::centered_integer(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE)
    }
    fn input_expression<F: Ring>(&self) -> ComposedExpr<F> {
        let mut expr = lift(self.base.input_expression());
        for (index, lane) in
            (SPARTAN_PRODUCT_BASE_LANES..).zip(field_product::selected_product_lanes())
        {
            let FieldInlineOpeningId::Polynomial { polynomial, .. } = lane.input_opening();
            let id = FieldInlineOpeningId::Polynomial {
                polynomial,
                relation: FieldInlineRelationId::FieldRegistersSpartanOuter,
            };
            expr = expr
                + derived(JoltDerivedId::from(
                    SpartanProductVirtualizationPublic::UniskipLagrangeWeight(index),
                )) * opening(id);
        }
        expr
    }
    fn output_expression<F: Ring>(&self) -> ComposedExpr<F> {
        lift(self.base.output_expression())
    }
}

#[derive(Clone)]
pub struct ProductRemainder {
    base: base::ProductRemainder,
}
impl SymbolicSumcheck for ProductRemainder {
    type RelationId = JoltRelationId;
    type OpeningId = ComposedOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanProductDimensions;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = ProductInputs<C>;
    type Outputs<C> = ProductOutputs<C>;
    fn new(shape: Self::Shape) -> Self {
        Self {
            base: base::ProductRemainder::new(shape),
        }
    }
    fn id() -> JoltRelationId {
        JoltRelationId::SpartanProductVirtualization
    }
    fn rounds(&self) -> usize {
        self.base.rounds()
    }
    fn degree(&self) -> usize {
        self.base.degree()
    }
    fn input_expression<F: Ring>(&self) -> ComposedExpr<F> {
        lift(self.base.input_expression())
    }
    fn output_expression<F: Ring>(&self) -> ComposedExpr<F> {
        let mut left = lift(self.base.left_factor_expression());
        let mut right = lift(self.base.right_factor_expression());
        for (index, lane) in
            (SPARTAN_PRODUCT_BASE_LANES..).zip(field_product::selected_product_lanes())
        {
            let [l, r] = lane.factor_openings();
            let weight = derived(JoltDerivedId::from(
                SpartanProductVirtualizationPublic::LagrangeWeight(index),
            ));
            left = left + weight.clone() * opening(l);
            right = right + weight * opening(r);
        }
        derived(JoltDerivedId::from(
            SpartanProductVirtualizationPublic::TauKernel,
        )) * left
            * right
    }
}

#[derive(Clone)]
pub struct ReadRafAddressPhase<B> {
    base: B,
}
impl<B> SymbolicSumcheck for ReadRafAddressPhase<B>
where
    B: SymbolicSumcheck<
        OpeningId = JoltOpeningId,
        DerivedId = JoltDerivedId,
        ChallengeId = JoltChallengeId,
        RelationId = JoltRelationId,
        Shape = BytecodeReadRafDimensions,
    >,
{
    type RelationId = JoltRelationId;
    type OpeningId = ComposedOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafDimensions;
    type Challenges<F> = B::Challenges<F>;
    type Inputs<C> = ComposedClaims<B::Inputs<C>, FieldInlineBytecodeReadRafInputs<C>>;
    type Outputs<C> = ComposedClaims<B::Outputs<C>>;
    fn new(shape: Self::Shape) -> Self {
        Self {
            base: B::new(shape),
        }
    }
    fn id() -> JoltRelationId {
        B::id()
    }
    fn rounds(&self) -> usize {
        self.base.rounds()
    }
    fn degree(&self) -> usize {
        self.base.degree()
    }
    fn input_expression<F: Ring>(&self) -> ComposedExpr<F> {
        lift(self.base.input_expression()) + Self::bytecode_input_extension_expr()
    }
    fn output_expression<F: Ring>(&self) -> ComposedExpr<F> {
        lift(self.base.output_expression())
    }
}

impl<B> ReadRafAddressPhase<B> {
    fn bytecode_input_extension_expr<F: Ring>() -> ComposedExpr<F> {
        let gamma_public = |challenge: BytecodeReadRafChallenge| -> ComposedExpr<F> {
            crate::challenge(JoltChallengeId::from(challenge))
        };
        let gamma = gamma_public(BytecodeReadRafChallenge::Gamma);
        let stage4_gamma = gamma_public(BytecodeReadRafChallenge::Stage4Gamma);
        let stage5_gamma = gamma_public(BytecodeReadRafChallenge::Stage5Gamma);

        // Stage-4 extension: FieldRdWa/FieldRs1Ra/FieldRs2Ra at powers
        // `stage4_gamma^(3 + j)` (the ordinary stage-4 power count is 3), riding
        // the outer γ³.
        let stage4_rows = [
            FieldInlineVirtualPolynomial::FieldRdWa,
            FieldInlineVirtualPolynomial::FieldRs1Ra,
            FieldInlineVirtualPolynomial::FieldRs2Ra,
        ];
        let mut stage4_extension = ComposedExpr::zero();
        for (index, polynomial) in stage4_rows.into_iter().enumerate() {
            #[expect(
                clippy::arithmetic_side_effects,
                reason = "3 + index is a small constant sum over the three field-register access rows"
            )]
            let power = 3 + index;
            stage4_extension = stage4_extension
                + stage4_gamma.clone().pow(power)
                    * opening(FieldInlineOpeningId::virtual_polynomial(
                        polynomial,
                        FieldInlineRelationId::FieldRegistersReadWriteChecking,
                    ));
        }
        let extension = gamma.clone().pow(3) * stage4_extension;

        // Stage-5 extension: the val-evaluation FieldRdWa at the power following
        // the ordinary stage-5 count (`2 + lookup-table count`), riding the outer
        // γ⁴.
        let stage5_power = 2 + LookupTableKind::<RISCV_XLEN>::COUNT;
        extension
            + gamma.pow(4)
                * stage5_gamma.pow(stage5_power)
                * opening(FieldInlineOpeningId::virtual_polynomial(
                    FieldInlineVirtualPolynomial::FieldRdWa,
                    FieldInlineRelationId::FieldRegistersValEvaluation,
                ))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test fixtures")]
mod tests {
    use super::super::claims::FieldProductUniskipInputs;
    use super::*;
    use crate::{InputClaims, OutputClaims};
    use jolt_field::Fr;
    use std::collections::BTreeSet;

    #[test]
    fn symbolic_claims_include_every_extension_opening() {
        let outer = OuterRemainder::new(SpartanOuterDimensions::rv64(3));
        let outer_ids = outer.expected_output_openings::<Fr>();
        assert_eq!(outer_ids.len(), 50);
        let outputs = OuterOutputs::<Fr>::from_opening_values(|id| {
            outer_ids.contains(id).then_some(Fr::from_u64(1))
        })
        .unwrap();
        assert_eq!(
            outputs
                .canonical_order()
                .into_iter()
                .collect::<BTreeSet<_>>(),
            outer_ids
        );

        let product = ProductRemainder::new(SpartanProductDimensions::new(3));
        let product_ids = product.expected_output_openings::<Fr>();
        for id in field_product::selected_product_remainder_output_openings() {
            assert!(product_ids.contains(&id.into()));
        }
        let uniskip = ProductUniskip::new(SpartanProductDimensions::new(3));
        let input_ids: BTreeSet<_> = uniskip
            .input_expression::<Fr>()
            .terms
            .into_iter()
            .flat_map(|term| term.factors)
            .filter_map(|source| {
                if let Source::Opening(id) = source {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(input_ids.len(), 5);
        assert_eq!(
            UniskipInputs::<Fr> {
                base: base::ProductUniskipInputClaims {
                    product: Fr::from_u64(0),
                    should_branch: Fr::from_u64(0),
                    should_jump: Fr::from_u64(0)
                },
                field_inline: FieldProductUniskipInputs::default()
            }
            .canonical_order()
            .into_iter()
            .collect::<BTreeSet<_>>(),
            input_ids
        );

        let bytecode_inputs = FieldInlineBytecodeReadRafInputs::<Fr>::default();
        let extension_ids: BTreeSet<_> =
            ReadRafAddressPhase::<()>::bytecode_input_extension_expr::<Fr>()
                .terms
                .into_iter()
                .flat_map(|term| term.factors)
                .filter_map(|source| {
                    if let Source::Opening(id) = source {
                        Some(id)
                    } else {
                        None
                    }
                })
                .collect();
        assert_eq!(extension_ids.len(), 4);
        assert_eq!(
            bytecode_inputs
                .canonical_order()
                .into_iter()
                .map(ComposedOpeningId::from)
                .collect::<BTreeSet<_>>(),
            extension_ids
        );
    }
}
