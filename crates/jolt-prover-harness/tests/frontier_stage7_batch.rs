#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
use jolt_prover_harness::{FeatureMode, FixtureKind};

#[test]
fn stage7_regular_batch_input_frontier_is_replacement_ready_with_certified_kernel_evidence(
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
        .find("stage7_regular_batch_inputs")
        .ok_or_else(|| "stage7 regular-batch input frontier is missing".to_owned())?;
    let port = ledger
        .find("cpu_stage7_regular_batch_input_claims")
        .ok_or_else(|| {
            "cpu_stage7_regular_batch_input_claims ledger entry is missing".to_owned()
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
fn stage7_regular_batch_sumcheck_frontier_is_replacement_ready_with_certified_kernel_evidence(
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
        .find("stage7_regular_batch_sumcheck")
        .ok_or_else(|| "stage7 regular-batch sumcheck frontier is missing".to_owned())?;
    let port = ledger
        .find("cpu_stage7_regular_batch_sumcheck")
        .ok_or_else(|| "cpu_stage7_regular_batch_sumcheck ledger entry is missing".to_owned())?;
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
fn stage7_regular_batch_input_checkpoint_matches_core_fixtures() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let checkpoint =
            jolt_prover_harness::load_stage7_regular_batch_input_checkpoint_fixture(&request)
                .map_err(|error| error.to_string())?;

        assert_eq!(checkpoint.modular.input_claims, checkpoint.expected);
        assert_eq!(
            checkpoint.modular.hamming_gamma,
            checkpoint.expected_hamming_gamma
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
fn stage7_regular_batch_verifier_replay_verifies_against_core_fixtures() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let checkpoint =
            jolt_prover_harness::load_stage7_regular_batch_verifier_replay_fixture(&request)
                .map_err(|error| error.to_string())?;

        assert_eq!(checkpoint.modular, checkpoint.expected);
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
fn stage7_regular_batch_sumcheck_matches_core_fixtures() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let fixture =
            jolt_prover_harness::load_stage7_regular_batch_sumcheck_kernel_benchmark_fixture(
                &request,
            )
            .map_err(|error| error.to_string())?;
        assert_eq!(
            fixture
                .run_reference_sumcheck()
                .map_err(|error| error.to_string())?,
            fixture.expected.challenges.len() * 2
        );
        let proof = fixture
            .run_modular_sumcheck()
            .map_err(|error| error.to_string())?;
        assert_eq!(proof.stage7_sumcheck_proof, fixture.expected.proof);
        assert_eq!(
            proof.verifier_output.public.challenges,
            fixture.expected.challenges
        );
        assert_eq!(
            proof.verifier_output.public.batching_coefficients,
            fixture.expected.batching_coefficients
        );
        assert_eq!(
            proof.verifier_output.batch.sumcheck_final_claim,
            fixture.expected.output_claim
        );
    }
    Ok(())
}
