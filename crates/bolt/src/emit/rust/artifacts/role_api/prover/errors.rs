use super::super::super::types::ProtocolArtifactExtension;
use super::super::declarations::push_error_enum;
use super::super::items::push_prover_error_conversions;
use super::super::role::RoleApiRole;
use super::super::{CommitmentRustApi, StageRustApi};

const ROLE: RoleApiRole = RoleApiRole::Prover;

pub(super) fn push_errors(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    prove_error_type: &str,
    extension: Option<&ProtocolArtifactExtension>,
) {
    push_error_enum(
        source,
        prove_error_type,
        commitment,
        stages,
        ROLE,
        extension.map(|extension| ROLE.extension_error_variants(extension)),
    );

    if let Some(extension) = extension {
        source.push_str(ROLE.extension_error_items(extension));
    }

    push_prover_error_conversions(source, commitment, stages, prove_error_type);
    if let Some(extension) = extension {
        source.push_str(ROLE.extension_error_conversions(extension));
    }
}
