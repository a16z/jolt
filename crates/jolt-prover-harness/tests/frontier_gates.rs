use jolt_prover_harness::{
    AcceptanceMode, FeatureMode, FixtureKind, FrontierSpec, ParityTarget, PerfGate,
};

const FIXTURES: &[FixtureKind] = &[FixtureKind::MuldivSmall];
const FEATURES: &[FeatureMode] = &[FeatureMode::Transparent];
const PARITY: &[ParityTarget] = &[ParityTarget::VerifierAcceptance];
const PERF_PARITY: &[ParityTarget] = &[ParityTarget::VerifierAcceptance, ParityTarget::Performance];
const OPTS: &[&str] = &["OPT-COM-001"];

fn frontier_with(
    fixtures: &'static [FixtureKind],
    features: &'static [FeatureMode],
    parity: &'static [ParityTarget],
    perf: Option<PerfGate>,
    optimization_ids: &'static [&'static str],
) -> FrontierSpec {
    FrontierSpec {
        name: "frontier_under_test",
        mode: AcceptanceMode::PrefixCheckpoint,
        fixtures,
        features,
        parity,
        perf,
        optimization_ids,
    }
}

#[test]
fn frontier_manifest_requires_perf_gate_for_performance_target() {
    let frontier = frontier_with(FIXTURES, FEATURES, PERF_PARITY, None, OPTS);

    assert!(frontier.validate().is_err());
}

#[test]
fn frontier_manifest_rejects_duplicate_dimensions() {
    const DUP_FIXTURES: &[FixtureKind] = &[FixtureKind::MuldivSmall, FixtureKind::MuldivSmall];
    const DUP_FEATURES: &[FeatureMode] = &[FeatureMode::Transparent, FeatureMode::Transparent];
    const DUP_PARITY: &[ParityTarget] = &[
        ParityTarget::VerifierAcceptance,
        ParityTarget::VerifierAcceptance,
    ];
    const DUP_OPTS: &[&str] = &["commitment-streaming", "commitment-streaming"];

    assert!(frontier_with(DUP_FIXTURES, FEATURES, PARITY, None, OPTS)
        .validate()
        .is_err());
    assert!(frontier_with(FIXTURES, DUP_FEATURES, PARITY, None, OPTS)
        .validate()
        .is_err());
    assert!(frontier_with(FIXTURES, FEATURES, DUP_PARITY, None, OPTS)
        .validate()
        .is_err());
    assert!(frontier_with(FIXTURES, FEATURES, PARITY, None, DUP_OPTS)
        .validate()
        .is_err());
}

#[test]
fn frontier_manifest_requires_optimization_inventory_accounting() {
    let frontier = frontier_with(FIXTURES, FEATURES, PARITY, None, &[]);

    assert!(frontier.validate().is_err());
}

#[test]
fn perf_gate_rejects_invalid_thresholds() {
    assert!(PerfGate {
        warn_ratio: 1.10,
        fail_ratio: 1.05,
        min_samples: 3,
        confirmation_size: Some(1),
    }
    .validate()
    .is_err());
    assert!(PerfGate {
        warn_ratio: 1.05,
        fail_ratio: 1.10,
        min_samples: 0,
        confirmation_size: Some(1),
    }
    .validate()
    .is_err());
}

#[test]
#[cfg(not(feature = "core-fixtures"))]
fn core_fixture_provider_is_feature_gated() {
    use jolt_prover_harness::{CoreFixtureProvider, FixtureProvider, FixtureRequest};

    let provider = CoreFixtureProvider;
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);

    assert!(provider.load(&request).is_err());
}

#[test]
#[cfg(feature = "field-inline")]
fn field_inline_facts_distinguish_fr_on_from_fr_off() {
    use jolt_prover_harness::field_inline::FieldInlineFixtureFacts;

    assert!(!FieldInlineFixtureFacts::fr_off().has_field_activity());
    assert!(FieldInlineFixtureFacts::fr_on(2, 1).has_field_activity());
}
