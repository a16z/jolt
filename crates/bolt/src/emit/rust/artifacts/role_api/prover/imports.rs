use super::super::super::types::{ProtocolArtifactConfig, ProtocolArtifactExtension};
use super::super::imports::push_stage_imports;
use super::super::names::RoleApiNames;
use super::super::role::RoleApiRole;
use super::super::CommitmentRustApi;

const ROLE: RoleApiRole = RoleApiRole::Prover;

pub(super) fn push_imports(
    source: &mut String,
    config: &ProtocolArtifactConfig,
    modules: &[String],
    kernel_modules: &[String],
    commitment: Option<&CommitmentRustApi>,
    names: &RoleApiNames,
    extension: Option<&ProtocolArtifactExtension>,
) {
    if let Some(extension) = extension {
        source.push_str(ROLE.extension_imports(extension));
    } else {
        if commitment.is_some() {
            source.push_str(&config.prover_setup_type.use_line());
        }
        source.push_str(&config.field_type.use_line());
        if !kernel_modules.is_empty() {
            let kernel_crate = config
                .kernel_crate
                .as_ref()
                .map(|kernel_crate| kernel_crate.import.as_str())
                .unwrap_or("missing_kernel_crate");
            source.push_str(&format!(
                "use {kernel_crate}::{{{}}};\n",
                kernel_modules.join(", ")
            ));
        }
        source.push_str(&config.default_transcript_type.use_line());
        source.push_str(&config.transcript_trait.use_line());
        source.push_str(&format!(
            "use {}::{{{}, {}, {}, {}}};\n\n",
            config.verifier_crate_import(),
            names.named_eval,
            names.proof,
            names.stage_proof,
            names.sumcheck_output,
        ));
    }
    push_stage_imports(source, modules);
}
