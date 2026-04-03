use std::path::Path;

use clap::Parser;

use jolt_eval::objective::{perf_objective_names, Objective};

#[derive(Parser)]
#[command(name = "measure-objectives")]
#[command(about = "Measure Jolt code quality and performance objectives")]
struct Cli {
    /// Only measure the named objective (default: all)
    #[arg(long)]
    objective: Option<String>,

    /// Skip Criterion benchmarks (only show static-analysis objectives)
    #[arg(long)]
    no_bench: bool,
}

fn print_header() {
    println!(
        "{:<35} {:>15} {:>8} {:>10}",
        "Objective", "Value", "Units", "Direction"
    );
    println!("{}", "-".repeat(70));
}

fn print_row(name: &str, val: f64, units: &str, dir: &str) {
    println!("{:<35} {:>15.6} {:>8} {:>10}", name, val, units, dir);
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let repo_root = std::env::current_dir()?;

    // Performance objectives (from Criterion)
    if !cli.no_bench {
        let perf_names = perf_objective_names();
        let run_bench = cli
            .objective
            .as_ref()
            .is_none_or(|name| perf_names.contains(&name.as_str()));

        if run_bench {
            eprintln!("Running Criterion benchmarks...");
            let mut any_succeeded = false;
            for &name in perf_names {
                if let Some(ref filter) = cli.objective {
                    if name != filter.as_str() {
                        continue;
                    }
                }
                let status = std::process::Command::new("cargo")
                    .args(["bench", "-p", "jolt-eval", "--bench", name, "--", "--quick"])
                    .status();
                if matches!(status, Ok(s) if s.success()) {
                    any_succeeded = true;
                }
            }

            if any_succeeded {
                println!();
                print_header();
                for &name in perf_names {
                    if let Some(ref filter) = cli.objective {
                        if name != filter.as_str() {
                            continue;
                        }
                    }
                    match read_criterion_estimate(name) {
                        Some(secs) => print_row(name, secs, "s", "min"),
                        None => {
                            println!("{:<35} {:>15}", name, "NO DATA");
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
    let objectives = Objective::all(&repo_root);
    for obj in &objectives {
        if let Some(ref name) = cli.objective {
            if obj.name() != name.as_str() {
                continue;
            }
        }
        match obj.collect_measurement() {
            Ok(val) => {
                let dir = match obj.direction() {
                    jolt_eval::Direction::Minimize => "min",
                    jolt_eval::Direction::Maximize => "max",
                };
                let units = obj.units().unwrap_or("-");
                print_row(obj.name(), val, units, dir);
            }
            Err(e) => {
                println!("{:<35} {:>15}", obj.name(), format!("ERROR: {e}"));
            }
        }
    }

    Ok(())
}

/// Read the point estimate (mean, in seconds) from Criterion's output.
///
/// Criterion writes to `target/criterion/<name>/new/estimates.json`.
fn read_criterion_estimate(bench_name: &str) -> Option<f64> {
    let path = Path::new("target/criterion")
        .join(bench_name)
        .join("new")
        .join("estimates.json");
    let data = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    // Criterion stores times in nanoseconds
    let nanos = json.get("mean")?.get("point_estimate")?.as_f64()?;
    Some(nanos / 1e9)
}
