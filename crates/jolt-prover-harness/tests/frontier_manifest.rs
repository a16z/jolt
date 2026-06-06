use jolt_prover_harness::{
    registered_frontiers, FeatureMode, FixtureKind, FrontierGate, FrontierManifest, FrontierSpec,
    PerfGate,
};

const FIXTURES: &[FixtureKind] = &[FixtureKind::MuldivSmall];
const FEATURES: &[FeatureMode] = &[FeatureMode::Transparent];
const PERF_GATES: &[FrontierGate] = &[
    FrontierGate::VerifierCorrectness,
    FrontierGate::CorePerformanceParity,
];
const OPTS: &[&str] = &["OPT-COM-001"];
const KERNELS: &[&str] = &["cpu_streaming_commitments"];

#[test]
fn frontier_manifest_requires_verifier_correctness_gate() {
    let frontier = FrontierSpec {
        name: "bad_frontier",
        fixtures: FIXTURES,
        features: FEATURES,
        gates: &[FrontierGate::CorePerformanceParity],
        perf: Some(PerfGate::canonical_frontier()),
        optimization_ids: OPTS,
        backend_kernel_ports: KERNELS,
    };

    assert!(frontier.validate().is_err());
}

#[test]
fn frontier_manifest_registers_unique_valid_frontiers() -> Result<(), String> {
    let mut manifest = FrontierManifest::new();
    let frontier = FrontierSpec {
        name: "stage0_commitments",
        fixtures: FIXTURES,
        features: FEATURES,
        gates: PERF_GATES,
        perf: Some(PerfGate::canonical_frontier()),
        optimization_ids: OPTS,
        backend_kernel_ports: KERNELS,
    };

    manifest
        .register(frontier)
        .map_err(|error| error.to_string())?;
    assert!(manifest.find("stage0_commitments").is_some());
    assert!(manifest.register(frontier).is_err());
    Ok(())
}

#[test]
fn registered_frontier_manifest_requires_correctness_and_perf_gates() -> Result<(), String> {
    let manifest = registered_frontiers().map_err(|error| error.to_string())?;

    for frontier in manifest.iter() {
        assert!(
            frontier.requires_verifier_correctness(),
            "{}",
            frontier.name
        );
        for gate in frontier.gates {
            assert!(
                matches!(
                    gate,
                    FrontierGate::VerifierCorrectness | FrontierGate::CorePerformanceParity
                ),
                "{}",
                frontier.name
            );
        }
        assert!(frontier.requires_core_performance(), "{}", frontier.name);
        assert!(frontier.perf.is_some(), "{}", frontier.name);
    }

    Ok(())
}

#[test]
fn registered_frontier_manifest_tracks_implemented_slices() -> Result<(), String> {
    let manifest = registered_frontiers().map_err(|error| error.to_string())?;

    let stage0 = manifest
        .find("stage0_commitments")
        .ok_or_else(|| "stage0 commitment frontier is missing".to_owned())?;
    assert_eq!(stage0.fixtures, &[FixtureKind::MuldivSmall]);
    assert_eq!(stage0.features, &[FeatureMode::Transparent]);
    assert!(stage0.requires_core_performance());

    let advice = manifest
        .find("stage0_advice_commitments")
        .ok_or_else(|| "stage0 advice frontier is missing".to_owned())?;
    assert_eq!(advice.fixtures, &[FixtureKind::AdviceConsumer]);
    assert!(advice.requires_core_performance());

    let field_inline = manifest
        .find("stage0_field_inline_commitments")
        .ok_or_else(|| "stage0 field-inline frontier is missing".to_owned())?;
    assert_eq!(field_inline.fixtures, &[FixtureKind::FieldInlineSmall]);
    assert_eq!(field_inline.features, &[FeatureMode::FieldInline]);
    assert!(field_inline.requires_core_performance());

    let zk = manifest
        .find("zk_blindfold_core_fixture")
        .ok_or_else(|| "ZK BlindFold frontier is missing".to_owned())?;
    assert_eq!(
        zk.fixtures,
        &[FixtureKind::ZkMuldivSmall, FixtureKind::ZkAdviceConsumer]
    );
    assert_eq!(zk.features, &[FeatureMode::Zk]);
    assert_eq!(
        zk.optimization_ids,
        &["OPT-ZK-001", "OPT-ZK-002", "OPT-ZK-003", "OPT-ZK-006"]
    );
    assert_eq!(
        zk.backend_kernel_ports,
        &[
            "cpu_blindfold_round_commitments",
            "cpu_blindfold_backend_kernels"
        ]
    );
    assert!(zk.requires_core_performance());

    for name in [
        "stage1_spartan_outer_requests",
        "stage2_product_uniskip",
        "stage2_regular_batch_inputs",
        "stage2_regular_batch_sumcheck",
        "stage2_ram_read_write_openings",
        "stage2_ram_terminal_openings",
        "stage2_product_remainder_openings",
        "stage2_instruction_claim_openings",
        "stage3_regular_batch_inputs",
        "stage3_regular_batch_sumcheck",
        "stage4_regular_batch_inputs",
        "stage4_regular_batch_sumcheck",
        "stage5_regular_batch_inputs",
        "stage5_regular_batch_sumcheck",
        "stage6_regular_batch_inputs",
        "stage6_regular_batch_sumcheck",
        "stage7_regular_batch_inputs",
        "stage7_regular_batch_sumcheck",
        "stage8_final_opening",
    ] {
        let frontier = manifest
            .find(name)
            .ok_or_else(|| format!("{name} frontier is missing"))?;
        assert_eq!(frontier.features, &[FeatureMode::Transparent]);
        assert_eq!(
            frontier.fixtures,
            &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer]
        );
        assert!(frontier.requires_core_performance(), "{name}");
        assert!(frontier.perf.is_some(), "{name}");
    }

    let stage8_zk = manifest
        .find("stage8_zk_final_opening")
        .ok_or_else(|| "stage8 ZK final-opening frontier is missing".to_owned())?;
    assert_eq!(
        stage8_zk.fixtures,
        &[FixtureKind::ZkMuldivSmall, FixtureKind::ZkAdviceConsumer]
    );
    assert_eq!(stage8_zk.features, &[FeatureMode::Zk]);

    let stage8_field_inline = manifest
        .find("stage8_field_inline_final_opening")
        .ok_or_else(|| "stage8 field-inline final-opening frontier is missing".to_owned())?;
    assert_eq!(
        stage8_field_inline.fixtures,
        &[FixtureKind::FieldInlineSmall]
    );
    assert_eq!(stage8_field_inline.features, &[FeatureMode::FieldInline]);

    Ok(())
}
