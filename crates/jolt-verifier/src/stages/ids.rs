//! Verifier-local relation, derived-value, and challenge ids for diagnostics.
//!
//! The verifier composes the ordinary Jolt protocol with optional protocol
//! extensions (today: field-inline). Diagnostics and reporting surfaces that
//! outlive a single relation carry these composites so they can name ids from
//! either family without collapsing the namespaces in jolt-claims.

use jolt_claims::protocols::composed::ExternalId;
use jolt_claims::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineRelationId,
};
use jolt_claims::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltRelationId};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VerifierDerivedId {
    Jolt(JoltDerivedId),
    FieldInline(FieldInlineDerivedId),
    External(ExternalId),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VerifierChallengeId {
    Jolt(JoltChallengeId),
    FieldInline(FieldInlineChallengeId),
    External(ExternalId),
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

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_claims::protocols::field_inline::{
        FieldRegistersReadWriteChallenge, FieldRegistersReadWritePublic,
    };
    use jolt_claims::protocols::jolt::{RamReadWriteChallenge, RamReadWritePublic};

    #[test]
    fn external_diagnostic_ids_sort_after_workspace_ids() {
        let external = ExternalId {
            family: "external",
            index: 0,
        };
        let jolt_derived =
            VerifierDerivedId::Jolt(JoltDerivedId::RamReadWrite(RamReadWritePublic::EqCycle));
        let field_derived = VerifierDerivedId::FieldInline(
            FieldInlineDerivedId::FieldRegistersReadWrite(FieldRegistersReadWritePublic::EqCycle),
        );
        assert!(jolt_derived < field_derived);
        assert!(field_derived < VerifierDerivedId::External(external));
        let jolt_challenge =
            VerifierChallengeId::Jolt(JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma));
        let field_challenge =
            VerifierChallengeId::FieldInline(FieldInlineChallengeId::FieldRegistersReadWrite(
                FieldRegistersReadWriteChallenge::Gamma,
            ));
        assert!(jolt_challenge < field_challenge);
        assert!(field_challenge < VerifierChallengeId::External(external));
    }
}
