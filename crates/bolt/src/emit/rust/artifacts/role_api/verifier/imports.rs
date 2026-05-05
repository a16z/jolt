use super::super::super::types::{ProtocolArtifactConfig, ProtocolArtifactExtension};
use super::super::imports::push_stage_imports;
use super::super::role::RoleApiRole;
use super::super::CommitmentRustApi;

const ROLE: RoleApiRole = RoleApiRole::Verifier;

pub(super) fn push_imports(
    source: &mut String,
    config: &ProtocolArtifactConfig,
    modules: &[String],
    commitment: Option<&CommitmentRustApi>,
    extension: Option<&ProtocolArtifactExtension>,
) {
    if let Some(extension) = extension {
        source.push_str(ROLE.extension_imports(extension));
    } else {
        if commitment.is_some() {
            source.push_str(&config.commitment_type.use_line());
        }
        source.push_str(&config.field_type.use_line());
        source.push_str(&config.transcript_trait.use_line());
    }
    source.push('\n');
    push_stage_imports(source, modules);
}
