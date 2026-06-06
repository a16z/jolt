#[test]
fn stage2_product_uniskip_frontier_is_replacement_ready_with_certified_kernel_evidence(
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
        .find("stage2_product_uniskip")
        .ok_or_else(|| "stage2 product uni-skip frontier is missing".to_owned())?;
    let port = ledger
        .find("cpu_spartan_product_uniskip")
        .ok_or_else(|| "cpu_spartan_product_uniskip ledger entry is missing".to_owned())?;
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
fn stage2_product_uniskip_frontier_requires_correctness_and_performance_gates() -> Result<(), String>
{
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let frontier = manifest
        .find("stage2_product_uniskip")
        .ok_or_else(|| "stage2 product uni-skip frontier is missing".to_owned())?;

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
fn stage2_cpu_product_uniskip_verifier_replay_verifies_against_core_fixtures() -> Result<(), String>
{
    for kind in [
        jolt_prover_harness::FixtureKind::MuldivSmall,
        jolt_prover_harness::FixtureKind::AdviceConsumer,
    ] {
        let request = jolt_prover_harness::FixtureRequest::new(
            kind,
            jolt_prover_harness::FeatureMode::Transparent,
        );
        let fixture =
            jolt_prover_harness::load_stage2_product_uniskip_verifier_replay_fixture(&request)
                .map_err(|error| error.to_string())?;

        fixture.verify().map_err(|error| error.to_string())?;
    }
    Ok(())
}
