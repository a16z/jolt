pub mod formulas;
pub mod relations;

mod config;
mod ids;

pub use config::{FieldInlineConfig, FieldInlineRepresentation, FIELD_REGISTERS_LOG_K};
pub use formulas::dimensions::{
    FieldInlineSumcheckSpec, FieldRegistersReadWriteDimensions,
    FieldRegistersReadWriteOpeningPoint, FieldRegistersTraceDimensions,
};
pub use ids::{
    FieldInlineChallengeId, FieldInlineCommittedPolynomial, FieldInlineExpr, FieldInlineOpFlag,
    FieldInlineOpeningId, FieldInlinePolynomialId, FieldInlinePublicId, FieldInlineRelationId,
    FieldInlineVirtualPolynomial, FieldRegistersClaimReductionChallenge,
    FieldRegistersClaimReductionPublic, FieldRegistersIncClaimReductionChallenge,
    FieldRegistersIncClaimReductionPublic, FieldRegistersReadWriteChallenge,
    FieldRegistersReadWritePublic, FieldRegistersValEvaluationPublic,
};
