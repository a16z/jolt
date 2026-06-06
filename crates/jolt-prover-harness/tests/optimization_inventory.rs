use jolt_prover_harness::{
    registered_backend_kernel_ports, registered_frontiers, validate_frontier_kernel_accounting,
    validate_frontier_optimization_ids, validate_frontier_replacement_ready,
    validate_global_cpu_backend_inventory_coverage, validate_parity_certified_kernel_evidence,
    validate_parity_certified_kernel_evidence_files, BackendKernelFamily, BackendKernelPortLedger,
    BackendKernelPortSpec, FeatureMode, FixtureKind, FrontierGate, FrontierSpec,
    KernelBenchmarkEvidence, KernelMemoryBudget, KernelPortStatus, KnownOptimizationIds, PerfGate,
    RunMetrics,
};

const INVENTORY: &str = include_str!("../../../specs/jolt-core-prover-optimization-inventory.md");
const FRONTIER_PERF_BENCH: &str = include_str!("../benches/frontier_perf.rs");
const SUMCHECK_KERNEL_BENCH: &str = include_str!("../../jolt-backends/benches/sumcheck_kernels.rs");
const FIXTURES: &[FixtureKind] = &[FixtureKind::MuldivSmall];
const FEATURES: &[FeatureMode] = &[FeatureMode::Transparent];
const GATES: &[FrontierGate] = &[
    FrontierGate::VerifierCorrectness,
    FrontierGate::CorePerformanceParity,
];
const KERNELS: &[&str] = &["cpu_streaming_commitments"];
const TEST_KERNELS: &[&str] = &["cpu_test_certified_kernel"];
const TEST_OPTIMIZATIONS: &[&str] = &["OPT-SC-007"];
const TEST_BENCHMARKS: &[&str] = &["cpu_sumcheck/test_certified_kernel"];

fn frontier(optimization_ids: &'static [&'static str]) -> FrontierSpec {
    FrontierSpec {
        name: "optimization_inventory_test",
        fixtures: FIXTURES,
        features: FEATURES,
        gates: GATES,
        perf: Some(PerfGate::canonical_frontier()),
        optimization_ids,
        backend_kernel_ports: KERNELS,
    }
}

fn certified_frontier() -> FrontierSpec {
    FrontierSpec {
        name: "certified_frontier_test",
        fixtures: FIXTURES,
        features: FEATURES,
        gates: GATES,
        perf: Some(PerfGate::canonical_frontier()),
        optimization_ids: TEST_OPTIMIZATIONS,
        backend_kernel_ports: TEST_KERNELS,
    }
}

fn certified_port() -> BackendKernelPortSpec {
    BackendKernelPortSpec {
        name: "cpu_test_certified_kernel",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: TEST_OPTIMIZATIONS,
        source_locations: &["jolt-core/src/zkvm/spartan/product.rs"],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck::test_certified_kernel"],
        microbenchmarks: TEST_BENCHMARKS,
        certification_evidence_files: &["target/frontier-metrics/cpu_test_certified_kernel.json"],
        status: KernelPortStatus::ParityCertified,
    }
}

fn certified_ledger(known: &KnownOptimizationIds) -> Result<BackendKernelPortLedger, String> {
    let mut ledger = BackendKernelPortLedger::new();
    ledger
        .register(certified_port(), known)
        .map_err(|error| error.to_string())?;
    Ok(ledger)
}

fn benchmark_evidence(
    kernel: &str,
    benchmark: &str,
    optimization_ids: &[&str],
    samples: u32,
    modular_time_ms: f64,
    modular_peak_rss_bytes: u64,
) -> KernelBenchmarkEvidence {
    KernelBenchmarkEvidence {
        kernel: kernel.to_owned(),
        benchmark: benchmark.to_owned(),
        samples,
        optimization_ids: optimization_ids.iter().map(|id| (*id).to_owned()).collect(),
        core: RunMetrics::new(Some(100.0), Some(1_000), None),
        modular: RunMetrics::new(Some(modular_time_ms), Some(modular_peak_rss_bytes), None),
        memory: KernelMemoryBudget::new(512, 1_200, 2_000),
    }
}

#[test]
fn optimization_inventory_parser_finds_known_ids() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;

    assert!(known.contains("OPT-COM-001"));
    assert!(known.contains("OPT-OPEN-003"));
    assert!(known.contains("OPT-ZK-001"));
    assert!(known.requires_cpu_backend("OPT-COM-001"));
    Ok(())
}

#[test]
fn frontier_optimization_ids_must_exist_in_inventory() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;

    validate_frontier_optimization_ids(frontier(&["OPT-COM-001"]), &known)
        .map_err(|e| e.to_string())?;
    assert!(validate_frontier_optimization_ids(frontier(&["OPT-NOPE-000"]), &known).is_err());
    Ok(())
}

#[test]
fn registered_frontier_optimization_ids_must_exist_in_inventory() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;
    let manifest = registered_frontiers().map_err(|e| e.to_string())?;

    for frontier in manifest.iter() {
        validate_frontier_optimization_ids(*frontier, &known).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[test]
fn backend_kernel_ledger_accounts_for_registered_frontier_cpu_optimizations() -> Result<(), String>
{
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;
    let ledger = registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let manifest = registered_frontiers().map_err(|e| e.to_string())?;

    for frontier in manifest.iter() {
        validate_frontier_kernel_accounting(*frontier, &known, &ledger, KernelPortStatus::Required)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[test]
fn backend_kernel_ledger_covers_every_cpu_backend_inventory_id() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;
    let ledger = registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;

    validate_global_cpu_backend_inventory_coverage(&known, &ledger)
        .map_err(|error| error.to_string())
}

#[test]
fn prover_ready_frontiers_require_ported_or_certified_cpu_kernels() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;
    let ledger = registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let manifest = registered_frontiers().map_err(|e| e.to_string())?;

    for name in [
        "stage0_commitments",
        "stage0_advice_commitments",
        "stage1_spartan_outer_requests",
        "stage2_product_uniskip",
        "stage2_ram_read_write_openings",
        "stage2_ram_terminal_openings",
        "stage2_product_remainder_openings",
        "stage2_instruction_claim_openings",
        "stage3_output_openings",
        "stage4_output_openings",
        "stage5_output_openings",
        "stage6_output_openings",
    ] {
        let frontier = manifest
            .find(name)
            .ok_or_else(|| format!("{name} frontier is missing"))?;
        validate_frontier_kernel_accounting(*frontier, &known, &ledger, KernelPortStatus::Ported)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[test]
fn blindfold_backend_rows_are_certified_by_focused_evidence() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;
    let ledger = registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;

    for (name, microbenchmarks, evidence_files) in [
        (
            "cpu_blindfold_round_commitments",
            &["frontier_perf/zk_blindfold_core_fixture"][..],
            &["target/frontier-metrics/kernel-evidence/cpu_blindfold_round_commitments/frontier_perf_zk_blindfold_core_fixture.json"][..],
        ),
        (
            "cpu_blindfold_backend_kernels",
            &["frontier_perf/blindfold_witness_rows"][..],
            &["target/frontier-metrics/kernel-evidence/cpu_blindfold_backend_kernels/frontier_perf_blindfold_witness_rows.json"][..],
        ),
    ] {
        let port = ledger
            .find(name)
            .ok_or_else(|| format!("{name} ledger row is missing"))?;
        assert_eq!(port.family, BackendKernelFamily::BlindFold);
        assert_eq!(
            port.status,
            KernelPortStatus::ParityCertified,
            "{name} should be replacement-ready after focused benchmark evidence"
        );
        assert_eq!(port.microbenchmarks, microbenchmarks);
        assert_eq!(port.certification_evidence_files, evidence_files);
    }

    Ok(())
}

#[test]
fn ported_backend_kernel_microbenchmarks_are_declared_in_bench_sources() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;
    let ledger = registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let bench_sources = [FRONTIER_PERF_BENCH, SUMCHECK_KERNEL_BENCH].join("\n");

    for port in ledger
        .iter()
        .filter(|port| port.status >= KernelPortStatus::Ported)
    {
        for benchmark in port.microbenchmarks {
            assert!(
                bench_sources.contains(benchmark),
                "{benchmark} is not declared in bench sources for {}",
                port.name
            );
        }
    }
    Ok(())
}

#[test]
fn registered_parity_certified_kernel_evidence_files_are_valid() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;
    let ledger = registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or_else(|| "failed to locate workspace root".to_owned())?;

    validate_parity_certified_kernel_evidence_files(
        workspace_root,
        &ledger,
        PerfGate::canonical_frontier(),
    )
    .map_err(|error| error.to_string())
}

#[test]
fn kernel_benchmark_evidence_uses_canonical_artifact_path() {
    let evidence = benchmark_evidence(
        "cpu_test_certified_kernel",
        "cpu_sumcheck/test certified kernel",
        TEST_OPTIMIZATIONS,
        3,
        100.0,
        1_000,
    );
    let path = evidence.canonical_artifact_path(std::path::Path::new("/workspace"));

    assert_eq!(
        path,
        std::path::Path::new("/workspace")
            .join("target/frontier-metrics/kernel-evidence")
            .join("cpu_test_certified_kernel")
            .join("cpu_sumcheck_test_certified_kernel.json")
    );
}

#[test]
fn parity_certified_kernel_status_requires_evidence_files() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;
    let mut ledger = BackendKernelPortLedger::new();

    let missing_evidence_files = BackendKernelPortSpec {
        certification_evidence_files: &[],
        ..certified_port()
    };

    assert!(ledger.register(missing_evidence_files, &known).is_err());
    Ok(())
}

#[test]
fn parity_certified_kernel_requires_passing_benchmark_evidence() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;
    let ledger = certified_ledger(&known)?;
    let gate = PerfGate::canonical_frontier();
    let passing_warn = benchmark_evidence(
        "cpu_test_certified_kernel",
        "cpu_sumcheck/test_certified_kernel",
        TEST_OPTIMIZATIONS,
        3,
        114.0,
        1_140,
    );

    validate_parity_certified_kernel_evidence(&ledger, &[passing_warn], gate)
        .map_err(|error| error.to_string())?;

    assert!(validate_parity_certified_kernel_evidence(&ledger, &[], gate).is_err());

    let too_slow = benchmark_evidence(
        "cpu_test_certified_kernel",
        "cpu_sumcheck/test_certified_kernel",
        TEST_OPTIMIZATIONS,
        3,
        116.0,
        1_140,
    );
    assert!(validate_parity_certified_kernel_evidence(&ledger, &[too_slow], gate).is_err());

    let too_few_samples = benchmark_evidence(
        "cpu_test_certified_kernel",
        "cpu_sumcheck/test_certified_kernel",
        TEST_OPTIMIZATIONS,
        2,
        100.0,
        1_000,
    );
    assert!(validate_parity_certified_kernel_evidence(&ledger, &[too_few_samples], gate).is_err());

    let wrong_optimization_ids = benchmark_evidence(
        "cpu_test_certified_kernel",
        "cpu_sumcheck/test_certified_kernel",
        &["OPT-EQ-004"],
        3,
        100.0,
        1_000,
    );
    assert!(
        validate_parity_certified_kernel_evidence(&ledger, &[wrong_optimization_ids], gate)
            .is_err()
    );
    Ok(())
}

#[test]
fn frontier_replacement_requires_certified_kernel_evidence() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;
    let ledger = certified_ledger(&known)?;
    let passing = benchmark_evidence(
        "cpu_test_certified_kernel",
        "cpu_sumcheck/test_certified_kernel",
        TEST_OPTIMIZATIONS,
        3,
        100.0,
        1_000,
    );

    validate_frontier_replacement_ready(certified_frontier(), &known, &ledger, &[passing])
        .map_err(|error| error.to_string())?;

    let registered_ledger = registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let manifest = registered_frontiers().map_err(|e| e.to_string())?;
    let current_frontier = manifest
        .find("stage1_spartan_outer_requests")
        .ok_or_else(|| "stage1 frontier is missing".to_owned())?;

    assert!(validate_frontier_replacement_ready(
        *current_frontier,
        &known,
        &registered_ledger,
        &[]
    )
    .is_err());
    Ok(())
}
