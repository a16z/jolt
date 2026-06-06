#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
use jolt_prover_harness::{FeatureMode, FixtureKind};

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_regular_batch_input_frontier_requires_correctness_and_performance_gates(
) -> Result<(), String> {
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let frontier = manifest
        .find("stage2_regular_batch_inputs")
        .ok_or_else(|| "stage2 regular batch input frontier is missing".to_owned())?;

    assert_eq!(frontier.features, &[FeatureMode::Transparent]);
    assert_eq!(
        frontier.fixtures,
        &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer]
    );
    assert!(frontier.requires_verifier_correctness());
    assert!(frontier.requires_core_performance());
    assert!(frontier.perf.is_some());
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_regular_batch_sumcheck_frontier_requires_correctness_and_performance_gates(
) -> Result<(), String> {
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let frontier = manifest
        .find("stage2_regular_batch_sumcheck")
        .ok_or_else(|| "stage2 regular batch sumcheck frontier is missing".to_owned())?;

    assert_eq!(frontier.features, &[FeatureMode::Transparent]);
    assert_eq!(
        frontier.fixtures,
        &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer]
    );
    assert!(frontier.requires_verifier_correctness());
    assert!(frontier.requires_core_performance());
    assert!(frontier.perf.is_some());
    Ok(())
}

#[test]
fn stage2_regular_batch_input_frontier_is_replacement_ready_with_certified_kernel_evidence(
) -> Result<(), String> {
    let known = jolt_prover_harness::KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .map_err(|error| error.to_string())?;
    let ledger =
        jolt_prover_harness::registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let frontier = manifest
        .find("stage2_regular_batch_inputs")
        .ok_or_else(|| "stage2 regular-batch input frontier is missing".to_owned())?;
    let port = ledger
        .find("cpu_stage2_regular_batch_input_claims")
        .ok_or_else(|| {
            "cpu_stage2_regular_batch_input_claims ledger entry is missing".to_owned()
        })?;
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or_else(|| "failed to locate workspace root".to_owned())?;
    let evidence = port
        .certification_evidence_files
        .iter()
        .map(|path| {
            jolt_prover_harness::KernelBenchmarkEvidence::read_json(&workspace_root.join(path))
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;

    jolt_prover_harness::validate_frontier_replacement_ready(*frontier, &known, &ledger, &evidence)
        .map_err(|error| error.to_string())
}

#[test]
fn stage2_regular_batch_sumcheck_frontier_is_replacement_ready_with_certified_kernel_evidence(
) -> Result<(), String> {
    let known = jolt_prover_harness::KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .map_err(|error| error.to_string())?;
    let ledger =
        jolt_prover_harness::registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let frontier = manifest
        .find("stage2_regular_batch_sumcheck")
        .ok_or_else(|| "stage2 regular-batch sumcheck frontier is missing".to_owned())?;
    let port = ledger
        .find("cpu_stage2_regular_batch_sumcheck")
        .ok_or_else(|| "cpu_stage2_regular_batch_sumcheck ledger entry is missing".to_owned())?;
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or_else(|| "failed to locate workspace root".to_owned())?;
    let evidence = port
        .certification_evidence_files
        .iter()
        .map(|path| {
            jolt_prover_harness::KernelBenchmarkEvidence::read_json(&workspace_root.join(path))
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;

    jolt_prover_harness::validate_frontier_replacement_ready(*frontier, &known, &ledger, &evidence)
        .map_err(|error| error.to_string())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_product_remainder_opening_frontier_requires_correctness_and_performance_gates(
) -> Result<(), String> {
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let frontier = manifest
        .find("stage2_product_remainder_openings")
        .ok_or_else(|| "stage2 product-remainder opening frontier is missing".to_owned())?;

    assert_eq!(frontier.features, &[FeatureMode::Transparent]);
    assert_eq!(
        frontier.fixtures,
        &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer]
    );
    assert!(frontier.requires_verifier_correctness());
    assert!(frontier.requires_core_performance());
    assert!(frontier.perf.is_some());
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_ram_read_write_opening_frontier_requires_correctness_and_performance_gates(
) -> Result<(), String> {
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let frontier = manifest
        .find("stage2_ram_read_write_openings")
        .ok_or_else(|| "stage2 RAM read-write opening frontier is missing".to_owned())?;

    assert_eq!(frontier.features, &[FeatureMode::Transparent]);
    assert_eq!(
        frontier.fixtures,
        &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer]
    );
    assert!(frontier.requires_verifier_correctness());
    assert!(frontier.requires_core_performance());
    assert!(frontier.perf.is_some());
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_ram_terminal_opening_frontier_requires_correctness_and_performance_gates(
) -> Result<(), String> {
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let frontier = manifest
        .find("stage2_ram_terminal_openings")
        .ok_or_else(|| "stage2 RAM terminal opening frontier is missing".to_owned())?;

    assert_eq!(frontier.features, &[FeatureMode::Transparent]);
    assert_eq!(
        frontier.fixtures,
        &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer]
    );
    assert!(frontier.requires_verifier_correctness());
    assert!(frontier.requires_core_performance());
    assert!(frontier.perf.is_some());
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_instruction_claim_opening_frontier_requires_correctness_and_performance_gates(
) -> Result<(), String> {
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let frontier = manifest
        .find("stage2_instruction_claim_openings")
        .ok_or_else(|| "stage2 instruction-claim opening frontier is missing".to_owned())?;

    assert_eq!(frontier.features, &[FeatureMode::Transparent]);
    assert_eq!(
        frontier.fixtures,
        &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer]
    );
    assert!(frontier.requires_verifier_correctness());
    assert!(frontier.requires_core_performance());
    assert!(frontier.perf.is_some());
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_ram_terminal_opening_checkpoint_matches_core_fixtures() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let checkpoint =
            jolt_prover_harness::load_stage2_ram_terminal_opening_checkpoint_fixture(&request)
                .map_err(|error| error.to_string())?;

        assert_eq!(checkpoint.modular, checkpoint.expected);
        assert_eq!(
            checkpoint.ram_raf_opening_point.len(),
            checkpoint.fixture.proof.trace_length.ilog2() as usize
                + checkpoint.fixture.proof.ram_K.ilog2() as usize
        );
        assert_eq!(
            checkpoint.ram_output_check_opening_point.len(),
            checkpoint.fixture.proof.ram_K.ilog2() as usize
        );
        checkpoint
            .fixture
            .verify()
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_regular_batch_input_checkpoint_matches_core_fixtures() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let checkpoint =
            jolt_prover_harness::load_stage2_regular_batch_input_checkpoint_fixture(&request)
                .map_err(|error| error.to_string())?;

        assert_eq!(checkpoint.modular.input_claims, checkpoint.expected);
        checkpoint
            .fixture
            .verify()
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_regular_batch_verifier_replay_accepts_modular_proof() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let checkpoint =
            jolt_prover_harness::load_stage2_regular_batch_verifier_replay_fixture(&request)
                .map_err(|error| error.to_string())?;

        assert_eq!(checkpoint.modular, checkpoint.verified);
    }
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_regular_batch_sumcheck_kernel_fixture_matches_core_fixtures() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let fixture =
            jolt_prover_harness::load_stage2_regular_batch_sumcheck_kernel_benchmark_fixture(
                &request,
            )
            .map_err(|error| error.to_string())?;
        let modular = fixture
            .run_modular_sumcheck()
            .map_err(|error| error.to_string())?;
        let reference_rounds = fixture
            .run_reference_sumcheck()
            .map_err(|error| error.to_string())?;

        assert_eq!(modular.proof, fixture.expected.proof);
        assert_eq!(modular.challenges, fixture.expected.challenges);
        assert_eq!(
            modular.batching_coefficients,
            fixture.expected.batching_coefficients
        );
        assert_eq!(modular.output_claim, fixture.expected.output_claim);
        assert_eq!(reference_rounds, fixture.expected.challenges.len() * 2);
    }
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_ram_read_write_opening_checkpoint_matches_core_fixtures() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let checkpoint =
            jolt_prover_harness::load_stage2_ram_read_write_opening_checkpoint_fixture(&request)
                .map_err(|error| error.to_string())?;

        assert_eq!(checkpoint.modular, checkpoint.expected);
        assert_eq!(
            checkpoint.opening_point.len(),
            checkpoint.fixture.proof.trace_length.ilog2() as usize
                + checkpoint.fixture.proof.ram_K.ilog2() as usize
        );
        checkpoint
            .fixture
            .verify()
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_instruction_claim_opening_checkpoint_matches_core_fixtures() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let checkpoint =
            jolt_prover_harness::load_stage2_instruction_claim_opening_checkpoint_fixture(&request)
                .map_err(|error| error.to_string())?;

        assert_eq!(checkpoint.modular, checkpoint.expected);
        assert_eq!(
            checkpoint.opening_point.len(),
            checkpoint.fixture.proof.trace_length.ilog2() as usize
        );
        checkpoint
            .fixture
            .verify()
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_product_remainder_opening_checkpoint_matches_core_fixtures() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let checkpoint =
            jolt_prover_harness::load_stage2_product_remainder_opening_checkpoint_fixture(&request)
                .map_err(|error| error.to_string())?;

        assert_eq!(checkpoint.modular, checkpoint.expected);
        assert_eq!(
            checkpoint.opening_point.len(),
            checkpoint.fixture.proof.trace_length.ilog2() as usize
        );
        checkpoint
            .fixture
            .verify()
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}
