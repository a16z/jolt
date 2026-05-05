use super::ProverEntryTypes;

pub(super) fn push_default_signature(
    source: &mut String,
    types: &ProverEntryTypes<'_>,
    generic_params: &str,
) {
    source.push_str(&format!(
        "pub fn prove_{protocol}<{generic_params}, T>(\n    inputs: {inputs}<'_, {generic_params}>,\n    transcript: &mut T,\n) -> Result<({proof}, {artifacts}), {error}>\nwhere\n",
        protocol = types.protocol_snake,
        inputs = types.prover_inputs_type,
        proof = types.proof_type,
        artifacts = types.prover_artifacts_type,
        error = types.prove_error_type
    ));
}

pub(super) fn push_with_programs_signature(
    source: &mut String,
    types: &ProverEntryTypes<'_>,
    generic_params: &str,
) {
    source.push_str(&format!(
        "pub fn prove_{protocol}_with_programs<{generic_params}, T>(\n    inputs: {inputs}<'_, {generic_params}>,\n    programs: {programs},\n    transcript: &mut T,\n) -> Result<({proof}, {artifacts}), {error}>\nwhere\n",
        protocol = types.protocol_snake,
        inputs = types.prover_inputs_type,
        programs = types.prover_programs_type,
        proof = types.proof_type,
        artifacts = types.prover_artifacts_type,
        error = types.prove_error_type
    ));
}
