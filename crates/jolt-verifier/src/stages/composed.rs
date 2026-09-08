//! Composition of protocol-local claims and expressions at the verifier boundary.

use super::ids::VerifierOpeningId;
use super::stage6a::field_inline::FieldInlineBytecodeReadRafInputs;
use jolt_claims::protocols::field_inline::geometry::{
    product as field_product, spartan as field_spartan,
};
use jolt_claims::protocols::field_inline::relations::{
    product::FieldRegistersProductOutputClaims, spartan::FieldRegistersSpartanOuterOutputClaims,
};
use jolt_claims::protocols::field_inline::FieldInlineOpeningId;
use jolt_claims::protocols::field_inline::FieldInlineRelationId;
use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
use jolt_claims::protocols::jolt::geometry::spartan::{
    self, SpartanOuterDimensions, SpartanProductDimensions,
};
use jolt_claims::protocols::jolt::relations::spartan as base;
use jolt_claims::protocols::jolt::BytecodeReadRafChallenge;
use jolt_claims::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltOpeningId};
use jolt_claims::protocols::jolt::{
    JoltRelationId, SpartanOuterPublic, SpartanProductVirtualizationPublic,
};
use jolt_claims::{derived, opening, NoChallenges, SumcheckDomain, SymbolicSumcheck};
use jolt_claims::{Expr, InputClaims, MissingOpeningValue, OutputClaims, Source, Term};
use jolt_field::JoltField;
use jolt_field::Ring;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_r1cs::constraints::jolt::{
    SPARTAN_PRODUCT_BASE_LANES, SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
    SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_riscv::NUM_CIRCUIT_FLAGS;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

/// Claims from the base protocol followed by the extension's claims. The two
/// namespaces stay disjoint; only this verifier-owned carrier resolves both.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct ComposedClaims<B, E = EmptyClaims> {
    pub base: B,
    pub field_inline: E,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct EmptyClaims;

impl<B, E: Default> From<B> for ComposedClaims<B, E> {
    fn from(base: B) -> Self {
        Self {
            base,
            field_inline: E::default(),
        }
    }
}

impl<B, E> Deref for ComposedClaims<B, E> {
    type Target = B;
    fn deref(&self) -> &B {
        &self.base
    }
}

impl<B, E> DerefMut for ComposedClaims<B, E> {
    fn deref_mut(&mut self) -> &mut B {
        &mut self.base
    }
}

impl<F: JoltField> InputClaims<F, FieldInlineOpeningId> for EmptyClaims {
    fn canonical_order(&self) -> Vec<FieldInlineOpeningId> {
        Vec::new()
    }
    fn resolve_input(&self, _: &FieldInlineOpeningId) -> Option<F> {
        None
    }
}
impl<F: JoltField> OutputClaims<F, FieldInlineOpeningId> for EmptyClaims {
    fn canonical_order(&self) -> Vec<FieldInlineOpeningId> {
        Vec::new()
    }
    fn resolve_output(&self, _: &FieldInlineOpeningId) -> Option<F> {
        None
    }
    fn from_opening_values(
        _: impl FnMut(&FieldInlineOpeningId) -> Option<F>,
    ) -> Result<Self, MissingOpeningValue<FieldInlineOpeningId>> {
        Ok(Self)
    }
}

impl<F: JoltField, B: InputClaims<F>, E: InputClaims<F, FieldInlineOpeningId>>
    InputClaims<F, VerifierOpeningId> for ComposedClaims<B, E>
{
    fn canonical_order(&self) -> Vec<VerifierOpeningId> {
        self.base
            .canonical_order()
            .into_iter()
            .map(VerifierOpeningId::from)
            .chain(
                self.field_inline
                    .canonical_order()
                    .into_iter()
                    .map(VerifierOpeningId::from),
            )
            .collect()
    }
    fn resolve_input(&self, id: &VerifierOpeningId) -> Option<F> {
        match id {
            VerifierOpeningId::Jolt(id) => self.base.resolve_input(id),
            VerifierOpeningId::FieldInline(id) => self.field_inline.resolve_input(id),
        }
    }
}
impl<F: JoltField, B: OutputClaims<F>, E: OutputClaims<F, FieldInlineOpeningId>>
    OutputClaims<F, VerifierOpeningId> for ComposedClaims<B, E>
{
    fn canonical_order(&self) -> Vec<VerifierOpeningId> {
        self.base
            .canonical_order()
            .into_iter()
            .map(VerifierOpeningId::from)
            .chain(
                self.field_inline
                    .canonical_order()
                    .into_iter()
                    .map(VerifierOpeningId::from),
            )
            .collect()
    }
    fn resolve_output(&self, id: &VerifierOpeningId) -> Option<F> {
        match id {
            VerifierOpeningId::Jolt(id) => self.base.resolve_output(id),
            VerifierOpeningId::FieldInline(id) => self.field_inline.resolve_output(id),
        }
    }
    fn from_opening_values(
        mut resolve: impl FnMut(&VerifierOpeningId) -> Option<F>,
    ) -> Result<Self, MissingOpeningValue<VerifierOpeningId>> {
        Ok(Self {
            base: B::from_opening_values(|id| resolve(&(*id).into()))
                .map_err(|e| MissingOpeningValue { id: e.id.into() })?,
            field_inline: E::from_opening_values(|id| resolve(&(*id).into()))
                .map_err(|e| MissingOpeningValue { id: e.id.into() })?,
        })
    }
}

pub type ComposedExpr<F> = Expr<F, VerifierOpeningId, JoltDerivedId, JoltChallengeId>;

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

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(field_inline)]
pub struct FieldProductUniskipInputs<C> {
    #[opening(FieldProduct, from = FieldRegistersSpartanOuter)]
    pub product: C,
    #[opening(FieldInvProduct, from = FieldRegistersSpartanOuter)]
    pub inv_product: C,
}

pub type OuterInputs<C> = ComposedClaims<base::OuterRemainderInputClaims<C>>;
pub type OuterOutputs<C> =
    ComposedClaims<base::OuterRemainderOutputClaims<C>, FieldRegistersSpartanOuterOutputClaims<C>>;
pub type ProductInputs<C> = ComposedClaims<base::ProductRemainderInputClaims<C>>;
pub type ProductOutputs<C> =
    ComposedClaims<base::ProductRemainderOutputClaims<C>, FieldRegistersProductOutputClaims<C>>;
pub type UniskipInputs<C> =
    ComposedClaims<base::ProductUniskipInputClaims<C>, FieldProductUniskipInputs<C>>;
pub type UniskipOutputs<C> = ComposedClaims<base::ProductUniskipOutputClaims<C>>;

#[derive(Clone)]
pub struct OuterRemainder {
    shape: SpartanOuterDimensions,
}

impl SymbolicSumcheck for OuterRemainder {
    type RelationId = JoltRelationId;
    type OpeningId = VerifierOpeningId;
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
    type OpeningId = VerifierOpeningId;
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
    type OpeningId = VerifierOpeningId;
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
    type OpeningId = VerifierOpeningId;
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
        use jolt_claims::protocols::field_inline::geometry::bytecode::FIELD_INLINE_BYTECODE_STAGE1_FLAGS;
        use jolt_claims::protocols::field_inline::FieldInlineVirtualPolynomial;

        let gamma_public = |challenge: BytecodeReadRafChallenge| -> ComposedExpr<F> {
            jolt_claims::challenge(JoltChallengeId::from(challenge))
        };
        let gamma = gamma_public(BytecodeReadRafChallenge::Gamma);
        let stage1_gamma = gamma_public(BytecodeReadRafChallenge::Stage1Gamma);
        let stage4_gamma = gamma_public(BytecodeReadRafChallenge::Stage4Gamma);
        let stage5_gamma = gamma_public(BytecodeReadRafChallenge::Stage5Gamma);

        // Stage-1 extension: the eight FieldOpFlag rows at powers
        // `stage1_gamma^(2 + NUM_CIRCUIT_FLAGS + i)` (the ordinary stage-1 power
        // count is `2 + NUM_CIRCUIT_FLAGS`), riding the outer γ⁰.
        let mut extension = ComposedExpr::zero();
        for (index, flag) in FIELD_INLINE_BYTECODE_STAGE1_FLAGS.into_iter().enumerate() {
            #[expect(
                clippy::arithmetic_side_effects,
                reason = "2 + NUM_CIRCUIT_FLAGS + index is a small constant sum over the eight FR flags"
            )]
            let power = 2 + NUM_CIRCUIT_FLAGS + index;
            extension = extension
                + stage1_gamma.clone().pow(power)
                    * opening(field_spartan::outer_opening(
                        FieldInlineVirtualPolynomial::FieldOpFlag(flag),
                    ));
        }

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
                reason = "3 + index is a small constant sum over the three FR access rows"
            )]
            let power = 3 + index;
            stage4_extension = stage4_extension
                + stage4_gamma.clone().pow(power)
                    * opening(FieldInlineOpeningId::virtual_polynomial(
                        polynomial,
                        FieldInlineRelationId::FieldRegistersReadWriteChecking,
                    ));
        }
        extension = extension + gamma.clone().pow(3) * stage4_extension;

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
    use super::*;
    use jolt_field::Fr;
    use std::collections::BTreeSet;

    #[test]
    fn symbolic_claims_include_every_extension_opening() {
        let outer = OuterRemainder::new(SpartanOuterDimensions::rv64(3));
        let outer_ids = outer.expected_output_openings::<Fr>();
        assert_eq!(outer_ids.len(), 48);
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
        assert_eq!(extension_ids.len(), 12);
        assert_eq!(
            bytecode_inputs
                .canonical_order()
                .into_iter()
                .map(VerifierOpeningId::from)
                .collect::<BTreeSet<_>>(),
            extension_ids
        );
    }
}
