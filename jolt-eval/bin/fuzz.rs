use std::time::{Duration, Instant};

use clap::Parser;

use jolt_eval::guests;
use jolt_eval::invariant::synthesis::{invariant_names, SynthesisRegistry};
use jolt_eval::invariant::{DynInvariant, InvariantReport, SynthesisTarget};

#[derive(Parser)]
#[command(name = "fuzz")]
#[command(about = "Fuzz-test Jolt invariants with random inputs")]
struct Cli {
    /// Guest program to evaluate (e.g. muldiv, fibonacci, sha2)
    #[arg(long)]
    guest: Option<String>,

    /// Path to a pre-compiled guest ELF (alternative to --guest)
    #[arg(long)]
    elf: Option<String>,

    /// Only fuzz the named invariant (default: all fuzzable)
    #[arg(long)]
    invariant: Option<String>,

    /// Total number of fuzz iterations (across all invariants)
    #[arg(long, default_value = "1000")]
    iterations: usize,

    /// Maximum wall-clock duration (e.g. "60s", "5m", "1h")
    #[arg(long)]
    duration: Option<String>,

    /// Max trace length override
    #[arg(long)]
    max_trace_length: Option<usize>,

    /// List available fuzzable invariants and exit
    #[arg(long)]
    list: bool,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    if cli.list {
        println!("Fuzzable invariants:");
        for name in invariant_names() {
            println!("  {name}");
        }
        return Ok(());
    }

    let (test_case, default_inputs) = guests::resolve_test_case(
        cli.guest.as_deref(),
        cli.elf.as_deref(),
        cli.max_trace_length,
    );

    let registry = SynthesisRegistry::from_inventory(Some(test_case), default_inputs);

    let fuzzable: Vec<&dyn DynInvariant> = if let Some(name) = &cli.invariant {
        let matches: Vec<_> = registry
            .for_target(SynthesisTarget::Fuzz)
            .into_iter()
            .filter(|inv| inv.name() == name.as_str())
            .collect();
        if matches.is_empty() {
            eprintln!("Invariant '{name}' not found or not fuzzable.");
            eprintln!("Run with --list to see available invariants.");
            std::process::exit(1);
        }
        matches
    } else {
        registry.for_target(SynthesisTarget::Fuzz)
    };

    if fuzzable.is_empty() {
        eprintln!("No fuzzable invariants registered.");
        std::process::exit(1);
    }

    let deadline = cli.duration.as_deref().map(|s| {
        let dur = parse_duration(s).unwrap_or_else(|| {
            eprintln!("Invalid duration '{s}'. Use e.g. 60s, 5m, 1h.");
            std::process::exit(1);
        });
        Instant::now() + dur
    });

    println!(
        "Fuzzing {} invariant(s), {} iterations",
        fuzzable.len(),
        cli.iterations,
    );
    if let Some(d) = &cli.duration {
        println!("Time limit: {d}");
    }
    println!();

    let mut total_checks = 0usize;
    let mut total_violations = 0usize;
    let start = Instant::now();

    for inv in &fuzzable {
        println!("  {} — setting up...", inv.name());

        let per_invariant = cli.iterations / fuzzable.len();
        let mut checks = 0usize;
        let mut violations = Vec::new();

        let batch_size = per_invariant.min(100);
        let mut remaining = per_invariant;

        while remaining > 0 {
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    println!("  (time limit reached)");
                    break;
                }
            }

            let n = remaining.min(batch_size);
            let results = inv.run_checks(n);
            for r in &results {
                checks += 1;
                if let Err(e) = r {
                    violations.push(e.to_string());
                }
            }
            remaining = remaining.saturating_sub(n);
        }

        let report = InvariantReport {
            name: inv.name().to_string(),
            total: checks,
            passed: checks - violations.len(),
            failed: violations.len(),
            violations: violations.clone(),
        };
        print_report(&report);

        total_checks += checks;
        total_violations += violations.len();
    }

    let elapsed = start.elapsed();
    println!();
    println!(
        "Done: {} checks in {:.1}s, {} violations",
        total_checks,
        elapsed.as_secs_f64(),
        total_violations,
    );

    if total_violations > 0 {
        std::process::exit(1);
    }

    Ok(())
}

fn print_report(report: &InvariantReport) {
    if report.failed == 0 {
        println!(
            "  {} — {}/{} passed",
            report.name, report.passed, report.total
        );
    } else {
        println!(
            "  {} — FAILED {}/{} checks",
            report.name, report.failed, report.total
        );
        for (i, v) in report.violations.iter().enumerate().take(5) {
            println!("    [{i}] {v}");
        }
        if report.violations.len() > 5 {
            println!("    ... and {} more", report.violations.len() - 5);
        }
    }
}

fn parse_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if let Some(n) = s.strip_suffix('s') {
        n.parse::<u64>().ok().map(Duration::from_secs)
    } else if let Some(n) = s.strip_suffix('m') {
        n.parse::<u64>().ok().map(|m| Duration::from_secs(m * 60))
    } else if let Some(n) = s.strip_suffix('h') {
        n.parse::<u64>().ok().map(|h| Duration::from_secs(h * 3600))
    } else {
        s.parse::<u64>().ok().map(Duration::from_secs)
    }
}
