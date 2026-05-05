mod aliases;
mod entrypoints;
mod errors;
mod execution;
mod imports;
mod proof_inputs;

use super::super::types::{ProtocolArtifactConfig, ProtocolRustArtifact};
use super::context::RoleApiSourceContext;
use super::declarations::push_role_program_artifact_declarations;
use super::role::RoleApiRole;
use aliases::push_type_aliases;
use entrypoints::{push_entrypoints, VerifierEntryTypes};
use errors::push_errors;
use imports::push_imports;
use proof_inputs::{push_proof_type, push_verifier_inputs};

const ROLE: RoleApiRole = RoleApiRole::Verifier;

pub(super) fn generated_verifier_api(
    config: &ProtocolArtifactConfig,
    artifacts: &[ProtocolRustArtifact],
) -> String {
    let context = RoleApiSourceContext::new(config, artifacts);
    let role_names = context.names.role(ROLE);
    let commitment_type = config.commitment_type.ident();
    let runtime_named_eval_type = &config.verifier_named_eval_type.path;
    let runtime_sumcheck_output_type = &config.verifier_sumcheck_output_type.path;
    let runtime_stage_proof_type = &config.verifier_stage_proof_type.path;

    let mut source = String::new();
    push_imports(
        &mut source,
        config,
        &context.modules,
        context.commitment.as_ref(),
        context.extension,
    );

    push_type_aliases(
        &mut source,
        &context.names,
        &context.field_type,
        runtime_named_eval_type,
        runtime_sumcheck_output_type,
        runtime_stage_proof_type,
    );
    push_proof_type(
        &mut source,
        context.commitment.as_ref(),
        &context.stages,
        &commitment_type,
        &context.names.proof,
        &context.names.stage_proof,
        context.extension,
    );
    push_verifier_inputs(
        &mut source,
        &context.stages,
        &context.field_type,
        role_names.inputs,
        context.extension,
    );

    push_role_program_artifact_declarations(
        &mut source,
        context.commitment.as_ref(),
        &context.stages,
        ROLE,
        context.declaration_types(ROLE),
        context.extension,
    );

    push_errors(
        &mut source,
        &context.protocol_snake,
        context.commitment.as_ref(),
        &context.stages,
        role_names.error,
        context.extension,
    );

    push_entrypoints(
        &mut source,
        context.commitment.as_ref(),
        &context.stages,
        VerifierEntryTypes {
            protocol_snake: &context.protocol_snake,
            field_type: &context.field_type,
            transcript_trait: &context.transcript_trait,
            proof_type: &context.names.proof,
            verifier_inputs_type: role_names.inputs,
            verifier_programs_type: role_names.programs,
            verification_artifacts_type: role_names.artifacts,
            verify_error_type: role_names.error,
            instrumentation_prefix: config.instrumentation_prefix.as_deref(),
        },
        context.extension,
    );

    if let Some(extension) = context.extension {
        source.push_str(ROLE.extension_helper_items(extension));
    }

    source
}
