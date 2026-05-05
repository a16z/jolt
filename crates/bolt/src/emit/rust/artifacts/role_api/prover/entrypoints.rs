mod bounds;
mod signatures;
mod types;

use super::super::super::types::ProtocolArtifactExtension;
use super::super::{CommitmentRustApi, StageRustApi};
use super::execution::push_with_programs_body;
use bounds::push_where_bounds;
use signatures::{push_default_signature, push_with_programs_signature};
pub(super) use types::ProverEntryTypes;

pub(super) fn push_entrypoints(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    types: ProverEntryTypes<'_>,
    extension: Option<&ProtocolArtifactExtension>,
) {
    let generic_params = types.generic_params.join(", ");
    push_default_signature(source, &types, &generic_params);
    push_where_bounds(
        source,
        commitment,
        stages,
        types.field_type,
        types.transcript_trait,
    );
    source.push_str("{\n");
    source.push_str(&format!(
        "    prove_{protocol}_with_programs(inputs, default_prover_programs(), transcript)\n}}\n\n",
        protocol = types.protocol_snake
    ));

    push_with_programs_signature(source, &types, &generic_params);
    push_where_bounds(
        source,
        commitment,
        stages,
        types.field_type,
        types.transcript_trait,
    );
    push_with_programs_body(
        source,
        commitment,
        stages,
        types.proof_type,
        types.prover_artifacts_type,
        extension,
    );
}
