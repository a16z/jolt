use clap::Parser;
use tracing::info;

use jolt_eval::invariant::{InvariantReport, JoltInvariants};

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
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let all = JoltInvariants::all();
    let invariants: Vec<_> = if let Some(name) = &cli.invariant {
        let filtered: Vec<_> = all
            .into_iter()
            .filter(|inv| inv.name().contains(name.as_str()))
            .collect();
        if filtered.is_empty() {
            let all_inv = JoltInvariants::all();
            let names: Vec<_> = all_inv.iter().map(|i| i.name()).collect();
            eprintln!(
                "Invariant '{name}' not found. Available: {}",
                names.join(", ")
            );
            std::process::exit(1);
        }
        filtered
    } else {
        all
    };

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
