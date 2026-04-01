use clap::Parser;
use tracing::info;

use jolt_eval::guests;
use jolt_eval::invariant::synthesis::{invariant_names, SynthesisRegistry};
use jolt_eval::invariant::{DynInvariant, InvariantReport};

#[derive(Parser)]
#[command(name = "check-invariants")]
#[command(about = "Run Jolt invariant checks")]
struct Cli {
    /// Guest program to evaluate (e.g. muldiv, fibonacci, sha2)
    #[arg(long)]
    guest: Option<String>,

    /// Path to a pre-compiled guest ELF (alternative to --guest)
    #[arg(long)]
    elf: Option<String>,

    /// Only run the named invariant (default: all)
    #[arg(long)]
    invariant: Option<String>,

    /// Number of random inputs per invariant
    #[arg(long, default_value = "10")]
    num_random: usize,

    /// Max trace length override
    #[arg(long)]
    max_trace_length: Option<usize>,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let (test_case, default_inputs) = guests::resolve_test_case(
        cli.guest.as_deref(),
        cli.elf.as_deref(),
        cli.max_trace_length,
    );

    let registry = SynthesisRegistry::from_inventory(Some(test_case), default_inputs);

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
            eprintln!("Available: {}", invariant_names().join(", "));
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

fn print_report(report: &InvariantReport) {
    println!(
        "  {} — {}/{} passed",
        report.name, report.passed, report.total
    );
    for violation in &report.violations {
        println!("    FAIL: {violation}");
    }
}
