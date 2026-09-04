//! The field-inline (FR) protocol family: its id families, geometry, and
//! relation instantiations.
//!
//! Ownership rule: this module owns the field-inline ids and instantiates the
//! shared Twist identities (`crate::twist`) with them; that module owns
//! the algebra. `protocols::jolt` is a completely separate protocol family —
//! the two protocol modules never import each other (FR ids never enter the
//! jolt namespace), and their composition happens only in `jolt-verifier`
//! (pinned by the `protocol_modules_are_import_disjoint` boundary test).

pub mod geometry;
pub mod lattice;
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
