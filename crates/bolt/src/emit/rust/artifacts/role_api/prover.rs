mod aliases;
mod entrypoints;
mod errors;
mod execution;
mod imports;
mod inputs;
mod proof_helpers;

use super::super::types::{ProtocolArtifactConfig, ProtocolRustArtifact};
use super::context::RoleApiSourceContext;
use super::declarations::push_role_program_artifact_declarations;
use super::discovery::{prover_generic_params, unique_kernel_modules};
use super::role::RoleApiRole;
use aliases::push_type_aliases;
use entrypoints::{push_entrypoints, ProverEntryTypes};
use errors::push_errors;
use imports::push_imports;
use inputs::push_prover_inputs;
use proof_helpers::push_proof_helpers;

const ROLE: RoleApiRole = RoleApiRole::Prover;

pub(super) fn generated_prover_api(
    config: &ProtocolArtifactConfig,
    artifacts: &[ProtocolRustArtifact],
) -> String {
    let context = RoleApiSourceContext::new(config, artifacts);
    let role_names = context.names.role(ROLE);
    let kernel_modules = unique_kernel_modules(&context.stages);
    let generic_params = prover_generic_params(&context.stages, context.commitment.as_ref());
    let default_transcript_type = config.default_transcript_type.ident();
    let prover_setup_type = config.prover_setup_type.ident();

    let mut source = String::new();
    push_imports(
        &mut source,
        config,
        &context.modules,
        &kernel_modules,
        context.commitment.as_ref(),
        &context.names,
        context.extension,
    );
    push_type_aliases(
        &mut source,
        &context.names,
        &default_transcript_type,
        &context.field_type,
    );

    push_prover_inputs(
        &mut source,
        &context.stages,
        context.commitment.as_ref(),
        &prover_setup_type,
        role_names.inputs,
        &generic_params,
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
        context.commitment.as_ref(),
        &context.stages,
        role_names.error,
        context.extension,
    );

    push_entrypoints(
        &mut source,
        context.commitment.as_ref(),
        &context.stages,
        ProverEntryTypes {
            protocol_snake: &context.protocol_snake,
            generic_params: &generic_params,
            field_type: &context.field_type,
            transcript_trait: &context.transcript_trait,
            proof_type: &context.names.proof,
            prover_inputs_type: role_names.inputs,
            prover_programs_type: role_names.programs,
            prover_artifacts_type: role_names.artifacts,
            prove_error_type: role_names.error,
            instrumentation_prefix: config.instrumentation_prefix.as_deref(),
        },
        context.extension,
    );

    if let Some(extension) = context.extension {
        source.push_str(ROLE.extension_helper_items(extension));
    }

    push_proof_helpers(
        &mut source,
        &context.stages,
        &context.field_type,
        &context.names.stage_proof,
        &context.names.sumcheck_output,
        &context.names.named_eval,
    );
    source
}
