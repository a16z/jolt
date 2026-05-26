use jolt_prover_harness::{
    FeatureMode, FixtureKind, IngestionSurface, ProgramArtifactKind, ProverInputDescriptor,
};

#[test]
fn sdk_inputs_are_normalized_program_artifacts() -> Result<(), String> {
    let input = ProverInputDescriptor::sdk(FeatureMode::Transparent);

    input.validate().map_err(|error| error.to_string())?;
    assert_eq!(input.surface, IngestionSurface::Sdk);
    assert_eq!(input.artifact, ProgramArtifactKind::JoltProgramExecution);
    Ok(())
}

#[test]
fn core_fixture_inputs_still_use_normalized_artifact_boundary() -> Result<(), String> {
    let input = ProverInputDescriptor::harness_core_fixture(
        FixtureKind::MuldivSmall,
        FeatureMode::Transparent,
    );

    input.validate().map_err(|error| error.to_string())?;
    assert!(input.normalized_jolt_program_artifact);
    Ok(())
}

#[test]
fn direct_tracer_internal_inputs_are_rejected() {
    let mut input = ProverInputDescriptor::harness_synthetic_fixture(
        FixtureKind::MuldivSmall,
        FeatureMode::Transparent,
    );
    input.uses_tracer_internals = true;

    assert!(input.validate().is_err());
}
