//! RAM output-check symbolic sumcheck relation.

use core::marker::PhantomData;

use jolt_field::{Field, RingCore};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::ram_val_final;
use crate::protocols::jolt::{
    JoltExpr, JoltOpeningId, JoltRelationId, RamOutputCheckPublic, ReadWriteDimensions,
};
use crate::SymbolicSumcheck;
use crate::{derived, opening, InputClaims, OutputClaims};

/// The produced RAM `val_final` opening, sharing the single output-check opening
/// point. Generic over the opening cell (`F` for the serialized wire value,
/// `Vec<F>` for the derived opening point).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamOutputCheck)]
pub struct RamOutputCheckOutputClaims<C> {
    #[opening(RamValFinal)]
    pub val_final: C,
}

/// The RAM output check consumes no openings (its input claim is the constant
/// zero), so this carries only the cell marker. Hand-implements [`InputClaims`]
/// since the derive requires at least one `#[opening]` field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamOutputCheckInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for RamOutputCheckInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: Field> InputClaims<F> for RamOutputCheckInputClaims<F> {
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        Vec::new()
    }

    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

/// The RAM output-check sumcheck: a degree-one output that pins `Val_final` on
/// the I/O region (via `EqIoMask`) and offsets by the masked I/O value; no
/// input claim.
pub struct OutputCheck {
    shape: ReadWriteDimensions,
}

impl SymbolicSumcheck for OutputCheck {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = ReadWriteDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = RamOutputCheckInputClaims<C>;
    type Outputs<C> = RamOutputCheckOutputClaims<C>;

    fn new(shape: ReadWriteDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamOutputCheck
    }

    fn rounds(&self) -> usize {
        self.shape.output_check_rounds()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        derived(RamOutputCheckPublic::EqIoMask) * opening(ram_val_final())
            + derived(RamOutputCheckPublic::NegEqIoMaskValIo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::JoltDerivedId;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn read_write_dimensions() -> ReadWriteDimensions {
        ReadWriteDimensions::new(5, 4, 2, 1)
    }

    #[test]
    fn output_check_evaluates_like_core_formula() {
        let relation = OutputCheck::new(read_write_dimensions());

        let val_final = Fr::from_u64(7);
        let eq_io_mask = Fr::from_u64(11);
        let neg_eq_io_mask_val_io = -Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = relation
            .input_expression::<Fr>()
            .evaluate(|_| zero, |_| zero, |_| zero);
        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_val_final() => val_final,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::RamOutputCheck(RamOutputCheckPublic::EqIoMask) => eq_io_mask,
                JoltDerivedId::RamOutputCheck(RamOutputCheckPublic::NegEqIoMaskValIo) => {
                    neg_eq_io_mask_val_io
                }
                _ => zero,
            },
        );

        assert_eq!(input, zero);
        assert_eq!(output, eq_io_mask * val_final + neg_eq_io_mask_val_io);
    }

    #[test]
    fn output_check_symbolic_matches_dependencies() {
        let relation = OutputCheck::new(read_write_dimensions());

        assert_eq!(OutputCheck::id(), JoltRelationId::RamOutputCheck);
        assert_eq!(
            relation.rounds(),
            read_write_dimensions().output_check_rounds()
        );
        assert_eq!(relation.degree(), 3);
        assert_eq!(relation.required_openings::<Fr>(), vec![ram_val_final()]);
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![
                JoltDerivedId::from(RamOutputCheckPublic::EqIoMask),
                JoltDerivedId::from(RamOutputCheckPublic::NegEqIoMaskValIo),
            ]
        );
    }
}
