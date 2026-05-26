use jolt_prover_harness::{
    AcceptanceMode, FeatureMode, FixtureKind, FrontierManifest, FrontierSpec, ParityTarget,
    PerfGate,
};

const FIXTURES: &[FixtureKind] = &[FixtureKind::MuldivSmall];
const FEATURES: &[FeatureMode] = &[FeatureMode::Transparent];
const PARITY: &[ParityTarget] = &[
    ParityTarget::VerifierAcceptance,
    ParityTarget::CoreCommitments,
];
const OPTS: &[&str] = &["OPT-COM-001"];

#[test]
fn frontier_manifest_requires_verifier_acceptance() {
    let frontier = FrontierSpec {
        name: "bad_frontier",
        mode: AcceptanceMode::PrefixCheckpoint,
        fixtures: FIXTURES,
        features: FEATURES,
        parity: &[ParityTarget::CoreCommitments],
        perf: None,
        optimization_ids: OPTS,
    };

    assert!(frontier.validate().is_err());
}

#[test]
fn frontier_manifest_registers_unique_valid_frontiers() -> Result<(), String> {
    let mut manifest = FrontierManifest::new();
    let frontier = FrontierSpec {
        name: "stage0_commitments",
        mode: AcceptanceMode::FullProofGraft,
        fixtures: FIXTURES,
        features: FEATURES,
        parity: PARITY,
        perf: Some(PerfGate::canonical_frontier()),
        optimization_ids: OPTS,
    };

    manifest
        .register(frontier)
        .map_err(|error| error.to_string())?;
    assert!(manifest.find("stage0_commitments").is_some());
    assert!(manifest.register(frontier).is_err());
    Ok(())
}
