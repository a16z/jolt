pub mod geometry;
pub mod relations;

mod config;
mod ids;

pub use config::{FieldInlineConfig, FieldInlineRepresentation, FIELD_REGISTERS_LOG_K};
pub use geometry::dimensions::{
    FieldRegistersReadWriteDimensions, FieldRegistersReadWriteOpeningPoint,
    FieldRegistersTraceDimensions,
};
pub use ids::{
    FieldInlineChallengeId, FieldInlineCommittedPolynomial, FieldInlineDerivedId, FieldInlineExpr,
    FieldInlineOpFlag, FieldInlineOpeningId, FieldInlinePolynomialId, FieldInlineRelationId,
    FieldInlineVirtualPolynomial, FieldRegistersClaimReductionChallenge,
    FieldRegistersClaimReductionPublic, FieldRegistersIncClaimReductionChallenge,
    FieldRegistersIncClaimReductionPublic, FieldRegistersReadWriteChallenge,
    FieldRegistersReadWritePublic, FieldRegistersValEvaluationPublic,
};
