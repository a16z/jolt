use std::sync::Arc;

use clap::Parser;
use tracing::info;

use jolt_eval::invariant::completeness_prover::ProverCompletenessInvariant;
use jolt_eval::invariant::completeness_verifier::VerifierCompletenessInvariant;
use jolt_eval::invariant::determinism::DeterminismInvariant;
use jolt_eval::invariant::serialization_roundtrip::SerializationRoundtripInvariant;
use jolt_eval::invariant::soundness::SoundnessInvariant;
use jolt_eval::invariant::synthesis::SynthesisRegistry;
use jolt_eval::invariant::zk_consistency::ZkConsistencyInvariant;
use jolt_eval::invariant::{DynInvariant, InvariantReport};
use jolt_eval::TestCase;

#[derive(Parser)]
#[command(name = "check-invariants")]
#[command(about = "Run Jolt invariant checks")]
struct Cli {
    /// Only run the named invariant (default: all)
    #[arg(long)]
    invariant: Option<String>,

    /// Number of random inputs per invariant
    #[arg(long, default_value = "10")]
    num_random: usize,

    /// Path to a pre-compiled guest ELF
    #[arg(long)]
    elf: Option<String>,

    /// Max trace length for the test program
    #[arg(long, default_value = "65536")]
    max_trace_length: usize,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let test_case = if let Some(elf_path) = &cli.elf {
        let elf_bytes = std::fs::read(elf_path)?;
        let memory_config = common::jolt_device::MemoryConfig {
            max_input_size: 4096,
            max_output_size: 4096,
            max_untrusted_advice_size: 0,
            max_trusted_advice_size: 0,
            stack_size: 65536,
            heap_size: 32768,
            program_size: None,
        };
        Arc::new(TestCase {
            elf_contents: elf_bytes,
            memory_config,
            max_trace_length: cli.max_trace_length,
        })
    } else {
        eprintln!("Error: --elf <path> is required. Provide a pre-compiled guest ELF.");
        eprintln!(
            "Example: compile with `cargo build -p <guest> --release` then pass the ELF path."
        );
        std::process::exit(1);
    };

    let default_inputs = vec![];

    let mut registry = SynthesisRegistry::new();
    register_invariants(&mut registry, &test_case, &default_inputs);

    let invariants: Vec<&dyn DynInvariant> = if let Some(name) = &cli.invariant {
        registry
            .invariants()
            .iter()
            .filter(|inv| inv.name() == name.as_str())
            .map(|inv| inv.as_ref())
            .collect()
    } else {
        registry
            .invariants()
            .iter()
            .map(|inv| inv.as_ref())
            .collect()
    };

    if invariants.is_empty() {
        eprintln!("No matching invariants found.");
        if let Some(name) = &cli.invariant {
            eprintln!("Available: soundness, verifier_completeness, prover_completeness, determinism, serialization_roundtrip, zk_consistency");
            eprintln!("Requested: {name}");
        }
        std::process::exit(1);
    }

    let mut all_passed = true;
    for inv in &invariants {
        info!("Running invariant: {}", inv.name());
        let results = inv.run_checks(cli.num_random);
        let report = InvariantReport::from_results(inv.name(), &results);
        print_report(&report);
        if report.failed > 0 {
            all_passed = false;
        }
    }

    if all_passed {
        info!("All invariants passed.");
    } else {
        eprintln!("Some invariants FAILED.");
        std::process::exit(1);
    }

    Ok(())
}

fn register_invariants(
    registry: &mut SynthesisRegistry,
    test_case: &Arc<TestCase>,
    default_inputs: &[u8],
) {
    registry.register(Box::new(SoundnessInvariant::new(
        Arc::clone(test_case),
        default_inputs.to_vec(),
    )));
    registry.register(Box::new(VerifierCompletenessInvariant::new(Arc::clone(
        test_case,
    ))));
    registry.register(Box::new(ProverCompletenessInvariant::new(Arc::clone(
        test_case,
    ))));
    registry.register(Box::new(DeterminismInvariant::new(Arc::clone(test_case))));
    registry.register(Box::new(SerializationRoundtripInvariant::new(
        Arc::clone(test_case),
        default_inputs.to_vec(),
    )));
    registry.register(Box::new(ZkConsistencyInvariant::new(Arc::clone(test_case))));
}

fn print_report(report: &InvariantReport) {
    println!(
        "  {} — {}/{} passed",
        report.name, report.passed, report.total
    );
    for violation in &report.violations {
        println!("    FAIL: {violation}");
    }
}
