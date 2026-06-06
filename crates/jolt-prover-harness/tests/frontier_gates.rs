use jolt_prover_harness::{FeatureMode, FixtureKind, FrontierGate, FrontierSpec, PerfGate};

const FIXTURES: &[FixtureKind] = &[FixtureKind::MuldivSmall];
const FEATURES: &[FeatureMode] = &[FeatureMode::Transparent];
const GATES: &[FrontierGate] = &[FrontierGate::VerifierCorrectness];
const PERF_GATES: &[FrontierGate] = &[
    FrontierGate::VerifierCorrectness,
    FrontierGate::CorePerformanceParity,
];
const OPTS: &[&str] = &["OPT-COM-001"];
const NON_PERF: &[&str] = &["NON-PERF"];
const KERNELS: &[&str] = &["cpu_streaming_commitments"];

fn frontier_with(
    fixtures: &'static [FixtureKind],
    features: &'static [FeatureMode],
    gates: &'static [FrontierGate],
    perf: Option<PerfGate>,
    optimization_ids: &'static [&'static str],
) -> FrontierSpec {
    FrontierSpec {
        name: "frontier_under_test",
        fixtures,
        features,
        gates,
        perf,
        optimization_ids,
        backend_kernel_ports: KERNELS,
    }
}

#[test]
fn frontier_manifest_requires_perf_gate_for_core_performance_gate() {
    let frontier = frontier_with(FIXTURES, FEATURES, PERF_GATES, None, OPTS);

    assert!(frontier.validate().is_err());
}

#[test]
fn frontier_manifest_requires_jolt_verifier_correctness_gate() {
    let frontier = frontier_with(
        FIXTURES,
        FEATURES,
        &[FrontierGate::CorePerformanceParity],
        Some(PerfGate::canonical_frontier()),
        OPTS,
    );

    assert!(frontier.validate().is_err());
}

#[test]
fn frontier_manifest_requires_core_performance_gate() {
    let frontier = frontier_with(FIXTURES, FEATURES, GATES, None, OPTS);

    assert!(frontier.validate().is_err());
}

#[test]
fn frontier_manifest_rejects_non_perf_marker() {
    let frontier = frontier_with(
        FIXTURES,
        FEATURES,
        PERF_GATES,
        Some(PerfGate::canonical_frontier()),
        NON_PERF,
    );

    assert!(frontier.validate().is_err());
}

#[test]
fn frontier_manifest_rejects_duplicate_dimensions() {
    const DUP_FIXTURES: &[FixtureKind] = &[FixtureKind::MuldivSmall, FixtureKind::MuldivSmall];
    const DUP_FEATURES: &[FeatureMode] = &[FeatureMode::Transparent, FeatureMode::Transparent];
    const DUP_GATES: &[FrontierGate] = &[
        FrontierGate::VerifierCorrectness,
        FrontierGate::VerifierCorrectness,
        FrontierGate::CorePerformanceParity,
    ];
    const DUP_OPTS: &[&str] = &["commitment-streaming", "commitment-streaming"];

    assert!(frontier_with(
        DUP_FIXTURES,
        FEATURES,
        PERF_GATES,
        Some(PerfGate::canonical_frontier()),
        OPTS
    )
    .validate()
    .is_err());
    assert!(frontier_with(
        FIXTURES,
        DUP_FEATURES,
        PERF_GATES,
        Some(PerfGate::canonical_frontier()),
        OPTS
    )
    .validate()
    .is_err());
    assert!(frontier_with(
        FIXTURES,
        FEATURES,
        DUP_GATES,
        Some(PerfGate::canonical_frontier()),
        OPTS
    )
    .validate()
    .is_err());
    assert!(frontier_with(
        FIXTURES,
        FEATURES,
        PERF_GATES,
        Some(PerfGate::canonical_frontier()),
        DUP_OPTS
    )
    .validate()
    .is_err());
}

#[test]
fn frontier_manifest_requires_optimization_inventory_accounting() {
    let frontier = frontier_with(
        FIXTURES,
        FEATURES,
        PERF_GATES,
        Some(PerfGate::canonical_frontier()),
        &[],
    );

    assert!(frontier.validate().is_err());
}

#[test]
fn perf_gate_rejects_invalid_thresholds() {
    assert!(PerfGate {
        warn_ratio: 1.10,
        fail_ratio: 1.05,
        min_samples: 3,
        confirmation_size: Some(1),
        require_time: true,
        require_peak_rss: true,
    }
    .validate()
    .is_err());
    assert!(PerfGate {
        warn_ratio: 1.05,
        fail_ratio: 1.10,
        min_samples: 0,
        confirmation_size: Some(1),
        require_time: true,
        require_peak_rss: true,
    }
    .validate()
    .is_err());
    assert!(PerfGate {
        warn_ratio: 1.05,
        fail_ratio: 1.10,
        min_samples: 3,
        confirmation_size: Some(1),
        require_time: false,
        require_peak_rss: false,
    }
    .validate()
    .is_err());
}

#[test]
fn canonical_perf_gate_fails_beyond_15_percent_regression() {
    use jolt_prover_harness::{evaluate_perf, GateStatus, RunMetrics};

    let gate = PerfGate::canonical_frontier();
    let core = RunMetrics::new(Some(100.0), Some(1_000), None);

    let within_threshold = RunMetrics::new(Some(115.0), Some(1_150), None);
    let over_threshold = RunMetrics::new(Some(116.0), Some(1_160), None);

    assert_ne!(
        evaluate_perf(gate, &core, &within_threshold).status,
        GateStatus::Fail
    );
    assert_eq!(
        evaluate_perf(gate, &core, &over_threshold).status,
        GateStatus::Fail
    );
}

#[test]
fn perf_gate_fails_when_required_axes_are_not_measured() {
    use jolt_prover_harness::{evaluate_perf, GateStatus, RunMetrics};

    let gate = PerfGate::canonical_frontier();
    let core = RunMetrics::new(Some(100.0), None, None);
    let modular = RunMetrics::new(Some(100.0), None, None);

    let evaluation = evaluate_perf(gate, &core, &modular);
    assert_eq!(evaluation.status, GateStatus::Fail);
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
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn transparent_core_fixture_smoke_is_verifier_accepted() -> Result<(), String> {
    use jolt_prover_harness::{
        CoreFixtureProvider, FixtureProvider, FixtureRequest, FixtureSource,
    };

    let provider = CoreFixtureProvider;

    for fixture in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let artifacts = provider
            .load(&FixtureRequest::new(fixture, FeatureMode::Transparent))
            .map_err(|error| error.to_string())?;
        assert_eq!(artifacts.source, FixtureSource::CoreCompatibility);
        assert!(artifacts
            .notes
            .iter()
            .any(|note| note.contains("verified by the modular verifier")));
    }

    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_core_fixture_smoke_is_verifier_accepted() -> Result<(), String> {
    std::thread::Builder::new()
        .name("zk-core-fixture-smoke".to_owned())
        .stack_size(128 * 1024 * 1024)
        .spawn(zk_core_fixture_smoke_is_verifier_accepted_inner)
        .map_err(|error| error.to_string())?
        .join()
        .map_err(|_| "zk core fixture smoke test thread panicked".to_owned())?
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_core_fixture_smoke_is_verifier_accepted_inner() -> Result<(), String> {
    use jolt_prover_harness::{
        CoreFixtureProvider, FixtureProvider, FixtureRequest, FixtureSource,
    };

    let provider = CoreFixtureProvider;
    let _lock = acquire_zk_core_fixture_process_lock()?;

    for fixture in [FixtureKind::ZkMuldivSmall, FixtureKind::ZkAdviceConsumer] {
        let artifacts = provider
            .load(&FixtureRequest::new(fixture, FeatureMode::Zk))
            .map_err(|error| error.to_string())?;
        assert_eq!(artifacts.source, FixtureSource::CoreCompatibility);
        assert!(artifacts
            .notes
            .iter()
            .any(|note| note.contains("verified by the modular verifier")));
    }

    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_core_fixture_initializes_prover_owned_transcript() -> Result<(), String> {
    std::thread::Builder::new()
        .name("zk-core-fixture-transcript".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(zk_core_fixture_initializes_prover_owned_transcript_inner)
        .map_err(|error| error.to_string())?
        .join()
        .map_err(|_| "zk core fixture transcript test thread panicked".to_owned())?
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_core_fixture_initializes_prover_owned_transcript_inner() -> Result<(), String> {
    use jolt_field::Fr;
    use jolt_prover::initialize_proof_transcript;
    use jolt_prover_harness::{load_zk_core_verifier_fixture, FixtureRequest};
    use jolt_transcript::{Blake2bTranscript, Transcript};

    let _lock = acquire_zk_core_fixture_process_lock()?;
    for kind in [FixtureKind::ZkMuldivSmall, FixtureKind::ZkAdviceConsumer] {
        let request = FixtureRequest::new(kind, FeatureMode::Zk);
        let fixture = load_zk_core_verifier_fixture(&request).map_err(|error| error.to_string())?;
        let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
        let checked = initialize_proof_transcript(
            &fixture.preprocessing,
            &fixture.public_io,
            &fixture.proof,
            fixture.trusted_advice_commitment.as_ref(),
            true,
            &mut transcript,
        )
        .map_err(|error| error.to_string())?;

        assert!(checked.zk);
        assert_eq!(checked.trace_length, fixture.proof.trace_length);
        assert_eq!(checked.ram_K, fixture.proof.ram_K);
        assert!(checked.vc_capacity.is_some());
        assert_eq!(
            checked.trusted_advice_commitment_present,
            kind == FixtureKind::ZkAdviceConsumer
        );
    }
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_stage1_committed_boundary_is_native_verifier_accepted() -> Result<(), String> {
    std::thread::Builder::new()
        .name("zk-stage1-boundary".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(zk_stage1_committed_boundary_is_native_verifier_accepted_inner)
        .map_err(|error| error.to_string())?
        .join()
        .map_err(|_| "zk Stage 1 boundary test thread panicked".to_owned())?
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_stage1_committed_boundary_is_native_verifier_accepted_inner() -> Result<(), String> {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_prover_harness::{load_zk_core_verifier_fixture, FixtureRequest};
    use jolt_transcript::Blake2bTranscript;
    use jolt_verifier::{
        stages::stage1::{self, Stage1Output},
        verify_until_stage1,
    };

    let _lock = acquire_zk_core_fixture_process_lock()?;
    for kind in [FixtureKind::ZkMuldivSmall, FixtureKind::ZkAdviceConsumer] {
        let request = FixtureRequest::new(kind, FeatureMode::Zk);
        let fixture = load_zk_core_verifier_fixture(&request).map_err(|error| error.to_string())?;
        let mut pre_stage1 =
            verify_until_stage1::<DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &fixture.preprocessing,
                &fixture.public_io,
                &fixture.proof,
                fixture.trusted_advice_commitment.as_ref(),
                true,
            )
            .map_err(|error| error.to_string())?;
        let stage1 = stage1::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage1, Stage1Output::Zk(_)));
    }

    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_stage2_committed_boundary_is_native_verifier_accepted() -> Result<(), String> {
    std::thread::Builder::new()
        .name("zk-stage2-boundary".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(zk_stage2_committed_boundary_is_native_verifier_accepted_inner)
        .map_err(|error| error.to_string())?
        .join()
        .map_err(|_| "zk Stage 2 boundary test thread panicked".to_owned())?
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_stage2_committed_boundary_is_native_verifier_accepted_inner() -> Result<(), String> {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_prover_harness::{load_zk_core_verifier_fixture, FixtureRequest};
    use jolt_transcript::Blake2bTranscript;
    use jolt_verifier::{
        stages::{
            stage1::{self, Stage1Output},
            stage2::{self, inputs::deps, Stage2Output},
        },
        verify_until_stage1,
    };

    let _lock = acquire_zk_core_fixture_process_lock()?;
    for kind in [FixtureKind::ZkMuldivSmall, FixtureKind::ZkAdviceConsumer] {
        let request = FixtureRequest::new(kind, FeatureMode::Zk);
        let fixture = load_zk_core_verifier_fixture(&request).map_err(|error| error.to_string())?;
        let mut pre_stage1 =
            verify_until_stage1::<DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &fixture.preprocessing,
                &fixture.public_io,
                &fixture.proof,
                fixture.trusted_advice_commitment.as_ref(),
                true,
            )
            .map_err(|error| error.to_string())?;
        let stage1 = stage1::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage1, Stage1Output::Zk(_)));

        let stage2 = stage2::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
            deps(&stage1),
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage2, Stage2Output::Zk(_)));
    }

    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_stage3_committed_boundary_is_native_verifier_accepted() -> Result<(), String> {
    std::thread::Builder::new()
        .name("zk-stage3-boundary".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(zk_stage3_committed_boundary_is_native_verifier_accepted_inner)
        .map_err(|error| error.to_string())?
        .join()
        .map_err(|_| "zk Stage 3 boundary test thread panicked".to_owned())?
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_stage3_committed_boundary_is_native_verifier_accepted_inner() -> Result<(), String> {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_prover_harness::{load_zk_core_verifier_fixture, FixtureRequest};
    use jolt_transcript::Blake2bTranscript;
    use jolt_verifier::{
        stages::{
            stage1::{self, Stage1Output},
            stage2::{self, inputs::deps as stage2_deps, Stage2Output},
            stage3::{self, inputs::deps as stage3_deps, Stage3Output},
        },
        verify_until_stage1,
    };

    let _lock = acquire_zk_core_fixture_process_lock()?;
    for kind in [FixtureKind::ZkMuldivSmall, FixtureKind::ZkAdviceConsumer] {
        let request = FixtureRequest::new(kind, FeatureMode::Zk);
        let fixture = load_zk_core_verifier_fixture(&request).map_err(|error| error.to_string())?;
        let mut pre_stage1 =
            verify_until_stage1::<DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &fixture.preprocessing,
                &fixture.public_io,
                &fixture.proof,
                fixture.trusted_advice_commitment.as_ref(),
                true,
            )
            .map_err(|error| error.to_string())?;
        let stage1 = stage1::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage1, Stage1Output::Zk(_)));

        let stage2 = stage2::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
            stage2_deps(&stage1),
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage2, Stage2Output::Zk(_)));

        let stage3 = stage3::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
            stage3_deps(&stage1, &stage2).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage3, Stage3Output::Zk(_)));
    }

    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_stage8_committed_boundary_is_native_verifier_accepted() -> Result<(), String> {
    std::thread::Builder::new()
        .name("zk-stage8-boundary".to_owned())
        .stack_size(128 * 1024 * 1024)
        .spawn(zk_stage8_committed_boundary_is_native_verifier_accepted_inner)
        .map_err(|error| error.to_string())?
        .join()
        .map_err(|_| "zk Stage 8 boundary test thread panicked".to_owned())?
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn zk_stage8_committed_boundary_is_native_verifier_accepted_inner() -> Result<(), String> {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_prover_harness::{load_zk_core_verifier_fixture, FixtureRequest};
    use jolt_transcript::Blake2bTranscript;
    use jolt_verifier::{
        stages::{
            stage1::{self, Stage1Output},
            stage2::{self, inputs::deps as stage2_deps, Stage2Output},
            stage3::{self, inputs::deps as stage3_deps, Stage3Output},
            stage4::{self, inputs::deps as stage4_deps, Stage4Output},
            stage5::{self, inputs::deps as stage5_deps, Stage5Output},
            stage6::{self, inputs::deps as stage6_deps, Stage6Output},
            stage7::{self, inputs::deps as stage7_deps, Stage7Output},
            stage8::{self, inputs::deps as stage8_deps, Stage8Output},
        },
        verify_until_stage1,
    };

    let _lock = acquire_zk_core_fixture_process_lock()?;
    for kind in [FixtureKind::ZkMuldivSmall, FixtureKind::ZkAdviceConsumer] {
        let request = FixtureRequest::new(kind, FeatureMode::Zk);
        let fixture = load_zk_core_verifier_fixture(&request).map_err(|error| error.to_string())?;
        let mut pre_stage1 =
            verify_until_stage1::<DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &fixture.preprocessing,
                &fixture.public_io,
                &fixture.proof,
                fixture.trusted_advice_commitment.as_ref(),
                true,
            )
            .map_err(|error| error.to_string())?;
        let stage1 = stage1::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage1, Stage1Output::Zk(_)));

        let stage2 = stage2::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
            stage2_deps(&stage1),
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage2, Stage2Output::Zk(_)));

        let stage3 = stage3::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
            stage3_deps(&stage1, &stage2).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage3, Stage3Output::Zk(_)));

        let stage4 = stage4::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
            stage4_deps(&stage2, &stage3).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage4, Stage4Output::Zk(_)));

        let stage5 = stage5::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
            stage5_deps(&stage2, &stage4).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage5, Stage5Output::Zk(_)));

        let stage6 = stage6::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
            stage6_deps(&stage1, &stage2, &stage3, &stage4, &stage5)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage6, Stage6Output::Zk(_)));

        let stage7 = stage7::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            &mut pre_stage1.transcript,
            stage7_deps(&stage4, &stage6).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage7, Stage7Output::Zk(_)));

        let stage8 = stage8::verify(
            &pre_stage1.checked,
            &fixture.preprocessing,
            &fixture.proof,
            fixture.trusted_advice_commitment.as_ref(),
            &mut pre_stage1.transcript,
            stage8_deps(&stage6, &stage7).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage8, Stage8Output::Zk(_)));
    }

    Ok(())
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
struct ZkCoreFixtureProcessLock {
    path: std::path::PathBuf,
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
impl Drop for ZkCoreFixtureProcessLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn acquire_zk_core_fixture_process_lock() -> Result<ZkCoreFixtureProcessLock, String> {
    let path = std::env::temp_dir().join("jolt-prover-harness-zk-core-fixture.lock");
    for _ in 0..1800 {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => return Ok(ZkCoreFixtureProcessLock { path }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(error) => return Err(error.to_string()),
        }
    }
    Err(format!(
        "timed out waiting for ZK core fixture process lock at {}",
        path.display()
    ))
}

#[test]
#[cfg(feature = "field-inline")]
fn field_inline_facts_distinguish_fr_on_from_fr_off() {
    use jolt_prover_harness::field_inline::FieldInlineFixtureFacts;

    assert!(!FieldInlineFixtureFacts::fr_off().has_field_activity());
    assert!(FieldInlineFixtureFacts::fr_on(2, 1).has_field_activity());
}
