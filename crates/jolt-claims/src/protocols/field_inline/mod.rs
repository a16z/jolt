pub mod formulas;

mod config;
mod ids;
mod relation;

pub use config::{FieldInlineConfig, FieldInlineRepresentation, FIELD_REGISTERS_LOG_K};
pub use formulas::dimensions::{
    FieldInlineSumcheckSpec, FieldRegistersReadWriteDimensions,
    FieldRegistersReadWriteOpeningPoint, FieldRegistersTraceDimensions,
};
pub use ids::{
    FieldInlineChallengeId, FieldInlineCommittedPolynomial, FieldInlineOpFlag,
    FieldInlineOpeningId, FieldInlinePolynomialId, FieldInlinePublicId, FieldInlineRelationId,
    FieldInlineVirtualPolynomial, FieldRegistersClaimReductionChallenge,
    FieldRegistersClaimReductionPublic, FieldRegistersIncClaimReductionChallenge,
    FieldRegistersIncClaimReductionPublic, FieldRegistersReadWriteChallenge,
    FieldRegistersReadWritePublic, FieldRegistersValEvaluationPublic,
};
pub use relation::{
    FieldInlineConsistencyClaim, FieldInlineExpr, FieldInlineInputClaimExpression,
    FieldInlineOutputClaimExpression, FieldInlineProtocolClaims, FieldInlineRelationClaims,
};
