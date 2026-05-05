use super::bounds::transcript_generic_bound;
use super::VerifierEntryTypes;

pub(super) fn push_default_signature(source: &mut String, types: &VerifierEntryTypes<'_>) {
    source.push_str(&format!(
        "pub fn verify_{protocol}<T: {transcript_bound}>(proof: &{proof}, inputs: {inputs}<'_>, transcript: &mut T) -> Result<{artifacts}, {error}> {{\n",
        transcript_bound = transcript_generic_bound(types.transcript_trait, types.field_type),
        protocol = types.protocol_snake,
        proof = types.proof_type,
        inputs = types.verifier_inputs_type,
        artifacts = types.verification_artifacts_type,
        error = types.verify_error_type,
    ));
}

pub(super) fn push_with_programs_signature(source: &mut String, types: &VerifierEntryTypes<'_>) {
    source.push_str(&format!(
        "pub fn verify_{protocol}_with_programs<T: {transcript_bound}>(proof: &{proof}, inputs: {inputs}<'_>, programs: {programs}, transcript: &mut T) -> Result<{artifacts}, {error}> ",
        transcript_bound = transcript_generic_bound(types.transcript_trait, types.field_type),
        protocol = types.protocol_snake,
        proof = types.proof_type,
        inputs = types.verifier_inputs_type,
        programs = types.verifier_programs_type,
        artifacts = types.verification_artifacts_type,
        error = types.verify_error_type,
    ));
}
