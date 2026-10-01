//! Claim carriers for relations combining ordinary Jolt and field-inline openings.

use super::ComposedOpeningId;
use crate::protocols::field_inline::relations::{
    product::FieldRegistersProductOutputClaims, spartan::FieldRegistersSpartanOuterOutputClaims,
};
use crate::protocols::field_inline::FieldInlineOpeningId;
use crate::protocols::jolt::relations::spartan as base;
use crate::protocols::jolt::{JoltChallengeId, JoltDerivedId};
use crate::{Expr, InputClaims, MissingOpeningValue, OutputClaims};
use jolt_field::JoltField;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

/// Claims from the base protocol followed by the extension's claims. The two
/// namespaces stay disjoint; the composed carrier resolves both.
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
    InputClaims<F, ComposedOpeningId> for ComposedClaims<B, E>
{
    fn canonical_order(&self) -> Vec<ComposedOpeningId> {
        self.base
            .canonical_order()
            .into_iter()
            .map(ComposedOpeningId::from)
            .chain(
                self.field_inline
                    .canonical_order()
                    .into_iter()
                    .map(ComposedOpeningId::from),
            )
            .collect()
    }
    fn resolve_input(&self, id: &ComposedOpeningId) -> Option<F> {
        match id {
            ComposedOpeningId::Jolt(id) => self.base.resolve_input(id),
            ComposedOpeningId::FieldInline(id) => self.field_inline.resolve_input(id),
        }
    }
}
impl<F: JoltField, B: OutputClaims<F>, E: OutputClaims<F, FieldInlineOpeningId>>
    OutputClaims<F, ComposedOpeningId> for ComposedClaims<B, E>
{
    fn canonical_order(&self) -> Vec<ComposedOpeningId> {
        self.base
            .canonical_order()
            .into_iter()
            .map(ComposedOpeningId::from)
            .chain(
                self.field_inline
                    .canonical_order()
                    .into_iter()
                    .map(ComposedOpeningId::from),
            )
            .collect()
    }
    fn resolve_output(&self, id: &ComposedOpeningId) -> Option<F> {
        match id {
            ComposedOpeningId::Jolt(id) => self.base.resolve_output(id),
            ComposedOpeningId::FieldInline(id) => self.field_inline.resolve_output(id),
        }
    }
    fn from_opening_values(
        mut resolve: impl FnMut(&ComposedOpeningId) -> Option<F>,
    ) -> Result<Self, MissingOpeningValue<ComposedOpeningId>> {
        Ok(Self {
            base: B::from_opening_values(|id| resolve(&(*id).into()))
                .map_err(|e| MissingOpeningValue { id: e.id.into() })?,
            field_inline: E::from_opening_values(|id| resolve(&(*id).into()))
                .map_err(|e| MissingOpeningValue { id: e.id.into() })?,
        })
    }
}

pub type ComposedExpr<F> = Expr<F, ComposedOpeningId, JoltDerivedId, JoltChallengeId>;

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

/// Field-register access claims folded into bytecode read-RAF, in canonical
/// input order: stage-4 destination and sources, then the stage-5 destination.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(field_inline)]
pub struct FieldInlineBytecodeReadRafInputs<C> {
    #[opening(FieldRdWa, from = FieldRegistersReadWriteChecking)]
    pub rd_wa_read_write: C,
    #[opening(FieldRs1Ra, from = FieldRegistersReadWriteChecking)]
    pub rs1_ra: C,
    #[opening(FieldRs2Ra, from = FieldRegistersReadWriteChecking)]
    pub rs2_ra: C,
    #[opening(FieldRdWa, from = FieldRegistersValEvaluation)]
    pub rd_wa_val_evaluation: C,
}
