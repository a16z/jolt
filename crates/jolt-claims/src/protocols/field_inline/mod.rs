pub mod formulas;

mod config;
mod ids;
mod relation;

pub use config::{FieldInlineConfig, FieldInlineRepresentation, FIELD_REGISTERS_LOG_K};
pub use formulas::dimensions::{
    FieldInlineSumcheckSpec, FieldRegistersReadWriteDimensions, FieldRegistersTraceDimensions,
};
pub use ids::{
    FieldInlineChallengeId, FieldInlineCommittedPolynomial, FieldInlineOpeningId,
    FieldInlinePolynomialId, FieldInlinePublicId, FieldInlineRelationId,
    FieldInlineVirtualPolynomial, FieldRegistersClaimReductionChallenge,
    FieldRegistersIncClaimReductionChallenge, FieldRegistersIncClaimReductionPublic,
    FieldRegistersReadWriteChallenge, FieldRegistersValEvaluationChallenge,
};
pub use relation::{
    FieldInlineConsistencyClaim, FieldInlineExpr, FieldInlineInputClaimExpression,
    FieldInlineOutputClaimExpression, FieldInlineProtocolClaims, FieldInlineRelationClaims,
};
