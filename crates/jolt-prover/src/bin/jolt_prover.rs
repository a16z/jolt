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
use jolt_prover::profile::{BenchmarkArgs, PrepareArgs, ProfileArgs};

#[derive(Parser)]
#[command(name = "jolt-prover", about = "Modular Jolt prover telemetry harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build a guest and serialize fixed inputs without executing a proof.
    Prepare(PrepareArgs),
    /// Prove a named workload and emit the telemetry artifacts.
    Profile(ProfileArgs),
    /// Sweep workloads across scales (one `profile` subprocess per run),
    /// accumulating benchmark-runs/results/modular_timings.csv.
    Benchmark(BenchmarkArgs),
}

#[expect(
    clippy::print_stderr,
    reason = "the CLI reports a rejected run on stderr"
)]
fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Prepare(args) => jolt_prover::profile::prepare(&args),
        Command::Profile(args) => {
            if let Err(error) = jolt_prover::profile::run(&args) {
                eprintln!("error: {error}");
                std::process::exit(1);
            }
        }
        Command::Benchmark(args) => {
            if !jolt_prover::profile::run_sweep(&args) {
                std::process::exit(1);
            }
        }
    }
}
