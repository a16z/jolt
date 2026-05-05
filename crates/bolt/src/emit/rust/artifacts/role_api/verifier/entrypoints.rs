mod bounds;
mod signatures;
mod types;

use super::super::super::types::ProtocolArtifactExtension;
use super::super::{CommitmentRustApi, StageRustApi};
use super::execution::push_with_programs_body;
use signatures::{push_default_signature, push_with_programs_signature};
pub(super) use types::VerifierEntryTypes;

pub(super) fn push_entrypoints(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    types: VerifierEntryTypes<'_>,
    extension: Option<&ProtocolArtifactExtension>,
) {
    push_default_signature(source, &types);
    source.push_str(&format!(
        "    verify_{protocol}_with_programs(proof, inputs, default_verifier_programs(), transcript)\n}}\n\n",
        protocol = types.protocol_snake
    ));
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.after_default_verify);
    }
    push_with_programs_signature(source, &types);
    push_with_programs_body(
        source,
        commitment,
        stages,
        types.verification_artifacts_type,
        extension,
        types.instrumentation_prefix,
    );
}
