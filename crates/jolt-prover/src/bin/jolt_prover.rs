//! The modular prover's profile CLI (`required-features = ["profiling"]`).
//!
//! ```text
//! cargo run --release -p jolt-prover --features profiling -- \
//!     profile --name fibonacci --format chrome
//! ```
//!
//! `main` is a thin wrapper over [`jolt_prover::profile::run`] so the
//! profiling smoke test can drive the same entry point in-process.

use clap::{Parser, Subcommand};
use jolt_prover::profile::{BenchmarkArgs, ProfileArgs};

#[derive(Parser)]
#[command(name = "jolt-prover", about = "Modular Jolt prover telemetry harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Prove a named workload and emit the telemetry artifacts.
    Profile(ProfileArgs),
    /// Sweep workloads across scales (one `profile` subprocess per run),
    /// accumulating benchmark-runs/results/modular_timings.csv.
    Benchmark(BenchmarkArgs),
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Profile(args) => {
            let _artifacts = jolt_prover::profile::run(&args);
        }
        Command::Benchmark(args) => {
            if !jolt_prover::profile::run_sweep(&args) {
                std::process::exit(1);
            }
        }
    }
}
