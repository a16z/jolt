//! Akita acceptance-matrix harness.
//!
//! The campaign CLI (`--name`, `--scale` or `--target-trace-size`,
//! `--backend optimized|metal`, `--format chrome`) over
//! [`jolt_prover::profile`]. Beyond the profile artifacts it writes the
//! scorer's files: `benchmark-runs/results/akita_{name}_{scale}_{backend}.csv`
//! (`name,scale,prove_s,padded_len,padded_hz,0,0`) and a copy of the chrome
//! trace at `benchmark-runs/perfetto_traces/akita_{name}_{scale}_{backend}.json`.
//! The profile module prints `PROOF_VERIFIED backend=<backend> value=true`
//! after the verifier accepts.
//!
//! ```text
//! cargo build --release -p jolt-prover --example modular_benchmark --features profiling,metal
//! ./target/release/examples/modular_benchmark --name fibonacci --scale 25 --backend metal
//! ./target/release/examples/modular_benchmark --name btreemap --scale 28 \
//!     --target-trace-size 150000000 --backend metal
//! ```

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::expect_used,
    reason = "benchmark harness: fail loudly and report to stdout"
)]

use std::fs;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use jolt_prover::profile::{run, BackendKind, OutputFormat, ProfileArgs, Workload};

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum Format {
    Default,
    Chrome,
    None,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum Backend {
    Reference,
    Optimized,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    MetalCommitOnly,
}

impl Backend {
    const fn kind(self) -> BackendKind {
        match self {
            Self::Reference => BackendKind::Reference,
            Self::Optimized => BackendKind::Optimized,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal => BackendKind::Metal,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::MetalCommitOnly => BackendKind::MetalCommitOnly,
        }
    }
}

#[derive(Parser, Debug)]
struct Cli {
    #[clap(long, value_enum)]
    name: Workload,

    /// log2 of the max (padded) trace length; derived from
    /// `--target-trace-size` when omitted.
    #[clap(short, long)]
    scale: Option<u32>,

    /// Guest input sized to this many trace cycles.
    #[clap(short, long)]
    target_trace_size: Option<usize>,

    /// Subscriber stack; `chrome` also copies the trace to
    /// `benchmark-runs/perfetto_traces/`. Repeated values are accepted for
    /// compatibility; the strongest one wins.
    #[clap(short, long, value_enum)]
    format: Option<Vec<Format>>,

    #[clap(short, long, value_enum, default_value = "optimized")]
    backend: Backend,
}

fn main() {
    let cli = Cli::parse();
    let scale = match (cli.scale, cli.target_trace_size) {
        (Some(scale), _) => scale,
        (None, Some(target)) => target.next_power_of_two().trailing_zeros(),
        (None, None) => {
            eprintln!("Error: Must provide either --scale or --target-trace-size");
            std::process::exit(1);
        }
    };
    let formats = cli.format.unwrap_or_default();
    let format = if formats.contains(&Format::Chrome) {
        OutputFormat::Chrome
    } else if formats.contains(&Format::Default) {
        OutputFormat::Default
    } else {
        OutputFormat::None
    };
    let backend = cli.backend.kind();

    let artifacts = run(&ProfileArgs {
        name: cli.name,
        scale: Some(scale),
        format,
        backend,
        target_trace_size: cli.target_trace_size,
    });

    let stem = format!(
        "akita_{}_{scale}_{}",
        cli.name.as_str().replace('-', "_"),
        backend.as_str()
    );
    let padded = artifacts.trace_length.next_power_of_two();
    let padded_hz = padded as f64 / artifacts.prove_seconds;
    let results_dir = PathBuf::from("benchmark-runs/results");
    fs::create_dir_all(&results_dir).expect("create results directory");
    let results_path = results_dir.join(format!(
        "akita_{}_{scale}_{}.csv",
        cli.name.as_str(),
        backend.as_str()
    ));
    fs::write(
        &results_path,
        format!(
            "{},{scale},{:.2},{padded},{padded_hz:.2},0,0\n",
            cli.name.as_str(),
            artifacts.prove_seconds
        ),
    )
    .expect("write results csv");
    println!("Results: {}", results_path.display());

    if let Some(trace_path) = artifacts.trace_path {
        let traces_dir = PathBuf::from("benchmark-runs/perfetto_traces");
        fs::create_dir_all(&traces_dir).expect("create perfetto_traces directory");
        let target = traces_dir.join(format!("{stem}.json"));
        fs::copy(&trace_path, &target).expect("copy chrome trace");
        println!("Trace copy: {}", target.display());
    }
}
