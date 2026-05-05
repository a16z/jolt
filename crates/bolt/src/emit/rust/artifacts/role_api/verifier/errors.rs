use super::super::super::types::ProtocolArtifactExtension;
use super::super::declarations::push_error_enum;
use super::super::items::push_verifier_error_conversions;
use super::super::role::RoleApiRole;
use super::super::{CommitmentRustApi, StageRustApi};

const ROLE: RoleApiRole = RoleApiRole::Verifier;

pub(super) fn push_errors(
    source: &mut String,
    protocol_snake: &str,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    verify_error_type: &str,
    extension: Option<&ProtocolArtifactExtension>,
) {
    push_error_enum(
        source,
        verify_error_type,
        commitment,
        stages,
        ROLE,
        extension.map(|extension| ROLE.extension_error_variants(extension)),
    );

    if let Some(extension) = extension {
        source.push_str(ROLE.extension_error_items(extension));
    }

    push_verifier_error_conversions(
        source,
        protocol_snake,
        commitment,
        stages,
        verify_error_type,
    );
    if let Some(extension) = extension {
        source.push_str(ROLE.extension_error_conversions(extension));
    }
}
