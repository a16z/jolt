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
        "{:<35} {:>15} {:>8}",
        "Objective", "Value", "Units"
    );
    println!("{}", "-".repeat(60));
}

fn print_row(name: &str, val: f64, units: &str) {
    println!("{:<35} {:>15.6} {:>8}", name, val, units);
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
                        Some(secs) => print_row(name, secs, "s"),
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
                let units = obj.units().unwrap_or("-");
                print_row(obj.name(), val, units);
            }
            Err(e) => {
                println!("{:<35} {:>15}", obj.name(), format!("ERROR: {e}"));
            }
        }
    }

    Ok(())
}

/// Read the point estimate (mean, in seconds) from Criterion's output.
fn read_criterion_estimate(bench_name: &str) -> Option<f64> {
    let path = Path::new("target/criterion")
        .join(bench_name)
        .join("new")
        .join("estimates.json");
    let data = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    let nanos = json.get("mean")?.get("point_estimate")?.as_f64()?;
    Some(nanos / 1e9)
}
