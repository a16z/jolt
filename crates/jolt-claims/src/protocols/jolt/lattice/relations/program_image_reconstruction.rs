//! Program-image word virtualization: in lattice mode the program image is
//! committed as a byte one-hot, so the `ProgramImageInit` word claim (the
//! base `ProgramImageClaimReduction` terminus) is settled against
//! `ProgramImageBytes` by a decode sumcheck — the word polynomial is never
//! PCS-opened.
//!
//! Same single-leg shape as the trusted advice reconstruction: the
//! polynomial is precommitted public data whose one-hot structure is checked
//! offline, so only the decode leg is spent. Binds the `(byte ‖ place)`
//! variables; the word point is fixed by the incoming claim.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::program_image::final_program_image_opening;
use crate::protocols::jolt::{
    JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltRelationId,
    ProgramImageReconstructionPublic,
};
use crate::{derived, opening, InputClaims, OutputClaims, SymbolicSumcheck};

use super::super::geometry::byte_place_vars;

/// The program-image byte one-hot opening at `(bound (byte ‖ place) ‖
/// r_word)` — the final claim the packed opening consumes for the slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(ProgramImageReconstruction)]
pub struct ProgramImageReconstructionOutputClaims<C> {
    #[opening(committed = ProgramImageBytes)]
    pub bytes: C,
}

/// The consumed word claim: the base program-image reduction's terminus.
#[derive(Clone, Debug, InputClaims)]
pub struct ProgramImageReconstructionInputClaims<C> {
    #[opening(committed = ProgramImageInit, from = ProgramImageClaimReduction)]
    pub word: C,
}

pub struct ProgramImageReconstruction;

impl SymbolicSumcheck for ProgramImageReconstruction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = ();
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = ProgramImageReconstructionInputClaims<C>;
    type Outputs<C> = ProgramImageReconstructionOutputClaims<C>;

    fn new(_shape: ()) -> Self {
        Self
    }

    fn id() -> JoltRelationId {
        JoltRelationId::ProgramImageReconstruction
    }

    fn rounds(&self) -> usize {
        byte_place_vars()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(final_program_image_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        derived(ProgramImageReconstructionPublic::ByteDecode)
            * opening(program_image_bytes_opening())
    }
}

pub fn program_image_bytes_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::ProgramImageBytes,
        JoltRelationId::ProgramImageReconstruction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::JoltDerivedId;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn program_image_reconstruction_evaluates_like_core_formula() {
        let relation = ProgramImageReconstruction::new(());

        let bytes = Fr::from_u64(3);
        let byte_decode = Fr::from_u64(7);
        let word_claim = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == final_program_image_opening() => word_claim,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        assert_eq!(input, word_claim);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == program_image_bytes_opening() => bytes,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::ProgramImageReconstruction(
                    ProgramImageReconstructionPublic::ByteDecode,
                ) => byte_decode,
                _ => zero,
            },
        );
        assert_eq!(output, byte_decode * bytes);
    }

    #[test]
    fn program_image_reconstruction_exposes_expected_dependencies() {
        let relation = ProgramImageReconstruction::new(());

        assert_eq!(
            ProgramImageReconstruction::id(),
            JoltRelationId::ProgramImageReconstruction
        );
        assert_eq!(relation.rounds(), 8 + 3);
        assert_eq!(relation.degree(), 2);
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![final_program_image_opening()]
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![program_image_bytes_opening()]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(
                ProgramImageReconstructionPublic::ByteDecode
            )]
        );
    }
}
