use std::path::Path;

use clap::Parser;

use jolt_eval::objective::performance::read_criterion_estimate;
use jolt_eval::objective::{OptimizationObjective, PerformanceObjective, StaticAnalysisObjective};

#[derive(Parser)]
#[command(name = "measure-objectives")]
#[command(about = "Measure Jolt code quality and performance objectives")]
struct Cli {
    /// Only measure the named objective (default: all). Also accepts
    /// string-keyed objectives: `telemetry:<workload>:<metric>` (modular
    /// prover summary.json) and `callgrind:<bench-name>:instructions`
    /// (iai-callgrind; opt-in, requires Valgrind).
    #[arg(long)]
    objective: Option<String>,

    /// Skip Criterion benchmarks (only show static-analysis objectives)
    #[arg(long)]
    no_bench: bool,
}

fn print_header() {
    println!("{:<35} {:>15} {:>8}", "Objective", "Value", "Units");
    println!("{}", "-".repeat(60));
}

fn print_row(name: &str, val: f64, units: &str) {
    println!("{:<35} {:>15.6} {:>8}", name, val, units);
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    // String-keyed telemetry/callgrind objectives: parse, measure in the
    // current directory (deterministic artifact paths — the objective owns
    // the workload/scale), print one row, done.
    if let Some(keyed) = cli
        .objective
        .as_ref()
        .and_then(|k| OptimizationObjective::from_key(k))
    {
        let objective = keyed.map_err(|e| eyre::eyre!("{e}"))?;
        eprintln!("Measuring {} ...", objective.name());
        print_header();
        match objective.measure_keyed_in(Path::new(".")) {
            Ok(value) => print_row(objective.name(), value, objective.units().unwrap_or("-")),
            Err(e) => {
                println!("{:<35} {:>15}", objective.name(), format!("ERROR: {e}"));
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    // Performance objectives (from Criterion)
    if !cli.no_bench {
        let perf = PerformanceObjective::all();
        let run_bench = cli
            .objective
            .as_ref()
            .is_none_or(|name| perf.iter().any(|p| p.name() == name.as_str()));

        if run_bench {
            eprintln!("Running Criterion benchmarks...");
            let mut any_succeeded = false;
            for p in &perf {
                if let Some(ref filter) = cli.objective {
                    if p.name() != filter.as_str() {
                        continue;
                    }
                }
                let status = std::process::Command::new("cargo")
                    .args([
                        "bench",
                        "-p",
                        "jolt-eval",
                        "--bench",
                        p.name(),
                        "--",
                        "--quick",
                    ])
                    .status();
                if matches!(status, Ok(s) if s.success()) {
                    any_succeeded = true;
                }
            }

            if any_succeeded {
                println!();
                print_header();
                for p in &perf {
                    if let Some(ref filter) = cli.objective {
                        if p.name() != filter.as_str() {
                            continue;
                        }
                    }
                    match read_criterion_estimate(Path::new("."), p.name(), "new") {
                        Some(secs) => print_row(p.name(), secs, "s"),
                        None => {
                            println!("{:<35} {:>15}", p.name(), "NO DATA");
                        }
                    }
                }
            }
        }
    } else {
        println!();
        print_header();
    }

    // Static-analysis objectives
    for sa in StaticAnalysisObjective::all() {
        if let Some(ref name) = cli.objective {
            if sa.name() != name.as_str() {
                continue;
            }
        }
        match sa.collect_measurement() {
            Ok(val) => {
                let units = sa.units().unwrap_or("-");
                print_row(sa.name(), val, units);
            }
            Err(e) => {
                println!("{:<35} {:>15}", sa.name(), format!("ERROR: {e}"));
            }
        }
    }

    Ok(())
}
