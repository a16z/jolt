use jolt_prover_harness::{
    compare_named_values, AcceptanceMode, CommitmentCheckpoint, FeatureMode, FixtureArtifacts,
    FixtureKind, FixtureProvider, FixtureRequest, FixtureSource, FrontierCheckpoint, FrontierSpec,
    GraftPlan, GraftRecord, GraftSurface, NamedValue, ParityTarget, StaticFixtureProvider,
};

const FIXTURES: &[FixtureKind] = &[FixtureKind::MuldivSmall];
const FEATURES: &[FeatureMode] = &[FeatureMode::Transparent];
const PARITY: &[ParityTarget] = &[
    ParityTarget::VerifierAcceptance,
    ParityTarget::CoreCommitments,
    ParityTarget::CoreProofShape,
];
const OPTS: &[&str] = &["OPT-COM-001", "OPT-COM-006"];

fn stage0_frontier() -> FrontierSpec {
    FrontierSpec {
        name: "stage0_commitments",
        mode: AcceptanceMode::FullProofGraft,
        fixtures: FIXTURES,
        features: FEATURES,
        parity: PARITY,
        perf: None,
        optimization_ids: OPTS,
    }
}

#[test]
fn stage0_frontier_accepts_commitment_graft_plan() -> Result<(), String> {
    let frontier = stage0_frontier();
    let plan = GraftPlan::new(vec![GraftRecord::new(
        GraftSurface::Commitments,
        "replace verifier JoltCommitments",
    )]);

    let validated = plan
        .validate_for(&frontier)
        .map_err(|error| error.to_string())?;
    assert_eq!(validated.records.len(), 1);
    Ok(())
}

#[test]
fn commitment_checkpoint_compares_by_logical_name() {
    let expected = FrontierCheckpoint::Commitments(CommitmentCheckpoint {
        commitments: vec![
            NamedValue::new("RdInc", "c_rd"),
            NamedValue::new("RamInc", "c_ram"),
        ],
        opening_hints: vec![NamedValue::new("RdIncHint", "h_rd")],
    });
    let actual = FrontierCheckpoint::Commitments(CommitmentCheckpoint {
        commitments: vec![
            NamedValue::new("RamInc", "c_ram"),
            NamedValue::new("RdInc", "c_rd"),
        ],
        opening_hints: vec![NamedValue::new("RdIncHint", "h_rd")],
    });

    let report = compare_named_values(
        ParityTarget::CoreCommitments,
        &expected.named_values(),
        &actual.named_values(),
    );
    assert!(report.is_success());
}

#[test]
fn commitment_checkpoint_rejects_duplicate_logical_names() {
    let expected = FrontierCheckpoint::Commitments(CommitmentCheckpoint {
        commitments: vec![NamedValue::new("RdInc", "c_rd")],
        opening_hints: Vec::new(),
    });
    let actual = FrontierCheckpoint::Commitments(CommitmentCheckpoint {
        commitments: vec![
            NamedValue::new("RdInc", "c_rd"),
            NamedValue::new("RdInc", "c_rd_again"),
        ],
        opening_hints: Vec::new(),
    });

    let report = compare_named_values(
        ParityTarget::CoreCommitments,
        &expected.named_values(),
        &actual.named_values(),
    );
    assert!(!report.is_success());
    assert_eq!(report.mismatches[0].path, "actual.RdInc");
}

#[test]
fn fixture_provider_returns_typed_artifacts() -> Result<(), String> {
    let provider = StaticFixtureProvider;
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let artifacts = provider.load(&request).map_err(|error| error.to_string())?;

    assert_eq!(
        artifacts,
        FixtureArtifacts::new(
            FixtureKind::MuldivSmall,
            FeatureMode::Transparent,
            FixtureSource::ModularSynthetic,
        )
        .with_note("static harness fixture placeholder")
    );
    Ok(())
}
