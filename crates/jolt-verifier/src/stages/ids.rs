//! Verifier-local relation, derived-value, and challenge ids for diagnostics.
//!
//! The verifier composes the ordinary Jolt protocol with optional protocol
//! extensions (today: field-inline). Diagnostics and reporting surfaces that
//! outlive a single relation carry these composites so they can name ids from
//! either family without collapsing the namespaces in jolt-claims.

use jolt_claims::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineRelationId,
};
use jolt_claims::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltRelationId};

/// A relation id from either protocol family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VerifierRelationId {
    Jolt(JoltRelationId),
    FieldInline(FieldInlineRelationId),
}

impl From<JoltRelationId> for VerifierRelationId {
    fn from(id: JoltRelationId) -> Self {
        Self::Jolt(id)
    }
}

impl From<FieldInlineRelationId> for VerifierRelationId {
    fn from(id: FieldInlineRelationId) -> Self {
        Self::FieldInline(id)
    }
}

/// A derived-value id from either protocol family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VerifierDerivedId {
    Jolt(JoltDerivedId),
    FieldInline(FieldInlineDerivedId),
}

impl From<JoltDerivedId> for VerifierDerivedId {
    fn from(id: JoltDerivedId) -> Self {
        Self::Jolt(id)
    }
}

impl From<FieldInlineDerivedId> for VerifierDerivedId {
    fn from(id: FieldInlineDerivedId) -> Self {
        Self::FieldInline(id)
    }
}

/// A challenge id from either protocol family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VerifierChallengeId {
    Jolt(JoltChallengeId),
    FieldInline(FieldInlineChallengeId),
}

impl From<JoltChallengeId> for VerifierChallengeId {
    fn from(id: JoltChallengeId) -> Self {
        Self::Jolt(id)
    }
}

impl From<FieldInlineChallengeId> for VerifierChallengeId {
    fn from(id: FieldInlineChallengeId) -> Self {
        Self::FieldInline(id)
    }
}
