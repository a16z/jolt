const OPENING_FRONTIERS: &[&str] = &[
    "stage2_ram_read_write_openings",
    "stage2_ram_terminal_openings",
    "stage2_product_remainder_openings",
    "stage2_instruction_claim_openings",
    "stage3_output_openings",
    "stage4_output_openings",
    "stage5_output_openings",
    "stage6_output_openings",
];

#[test]
fn materialized_opening_frontiers_are_replacement_ready_with_certified_kernel_evidence(
) -> Result<(), String> {
    let known = jolt_prover_harness::KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .map_err(|error| error.to_string())?;
    let ledger =
        jolt_prover_harness::registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let port = ledger
        .find("cpu_materialized_opening_evaluations")
        .ok_or_else(|| "cpu_materialized_opening_evaluations ledger entry is missing".to_owned())?;
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

    for frontier_name in OPENING_FRONTIERS {
        let frontier = manifest
            .find(frontier_name)
            .ok_or_else(|| format!("{frontier_name} frontier is missing"))?;

        jolt_prover_harness::validate_frontier_replacement_ready(
            *frontier, &known, &ledger, &evidence,
        )
        .map_err(|error| format!("{frontier_name}: {error}"))?;
    }

    Ok(())
}
