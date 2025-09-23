use clap::{Args, Parser, Subcommand, ValueEnum};

#[path = "../../benches/e2e_profiling.rs"]
mod e2e_profiling;
use e2e_profiling::{benchmarks, master_benchmark, BenchType};

use std::any::Any;

use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{self, fmt::format::FmtSpan, prelude::*, EnvFilter};

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Profile(ProfileArgs),
    Benchmark(BenchmarkArgs),
}

#[derive(Args, Debug, Clone)]
struct ProfileArgs {
    /// Output formats
    #[clap(short, long, value_enum)]
    format: Option<Vec<Format>>,

    /// Type of benchmark to run
    #[clap(long, value_enum)]
    name: BenchType,
}

#[derive(Args, Debug)]
struct BenchmarkArgs {
    #[clap(flatten)]
    profile_args: ProfileArgs,

    /// Max trace length to use (as 2^scale)
    #[clap(short, long, default_value_t = 20)]
    scale: usize,

    /// Target trace size to use. If not supplied, will be set to 90% of 2^scale.
    #[clap(short, long)]
    target_trace_size: Option<usize>,
}

#[derive(Debug, Clone, ValueEnum, PartialEq)]
enum Format {
    Default,
    Chrome,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Profile(args) => trace_only(args),
        Commands::Benchmark(args) => benchmark_and_trace(args),
    }
}

fn trace(
    args: ProfileArgs,
    benchmark_fn: Vec<(tracing::Span, Box<dyn FnOnce()>)>,
    trace_file: Option<String>,
) {
    let mut layers = Vec::new();

    let log_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let log_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_filter(log_filter)
        .boxed();
    layers.push(log_layer);

    let mut guards: Vec<Box<dyn Any>> = vec![];

    if let Some(format) = &args.format {
        if format.contains(&Format::Default) {
            let collector_layer = tracing_subscriber::fmt::layer()
                .with_span_events(FmtSpan::CLOSE)
                .compact()
                .with_target(false)
                .with_file(false)
                .with_line_number(false)
                .with_thread_ids(false)
                .with_thread_names(false)
                .boxed();
            layers.push(collector_layer);
        }
        if format.contains(&Format::Chrome) {
            let (chrome_layer, guard) = if let Some(file) = &trace_file {
                ChromeLayerBuilder::new()
                    .file(file)
                    .include_args(true)
                    .build()
            } else {
                ChromeLayerBuilder::new().include_args(true).build()
            };
            layers.push(chrome_layer.boxed());
            guards.push(Box::new(guard));
            if trace_file.is_some() {
                tracing::info!("Running tracing-chrome. Files will be saved in benchmark-runs/perfetto_traces/ and can be viewed in https://ui.perfetto.dev/");
            } else {
                tracing::info!("Running tracing-chrome. Files will be saved as trace-<some timestamp>.json and can be viewed in https://ui.perfetto.dev/");
            }
        }
    }

    tracing_subscriber::registry().with(layers).init();
    for (span, bench) in benchmark_fn.into_iter() {
        span.to_owned().in_scope(|| {
            bench();
            tracing::info!("Bench Complete");
        });
    }
}

// Fixed benchmarks
fn trace_only(args: ProfileArgs) {
    trace(args.clone(), benchmarks(args.name), None)
}

// Dynamically-sized benchmarks
fn benchmark_and_trace(args: BenchmarkArgs) {
    // Generate trace filename
    let bench_name = match args.profile_args.name {
        BenchType::Fibonacci => "fibonacci",
        BenchType::Sha2Chain => "sha2_chain",
        BenchType::Sha3Chain => "sha3_chain",
        BenchType::Btreemap => "btreemap",
        BenchType::Sha2 => panic!("Use sha2-chain instead"),
        BenchType::Sha3 => panic!("Use sha3-chain instead"),
    };
    let trace_file = format!(
        "benchmark-runs/perfetto_traces/{}_{}.json",
        bench_name, args.scale
    );
    std::fs::create_dir_all("benchmark-runs/perfetto_traces").ok();

    trace(
        args.profile_args.clone(),
        master_benchmark(args.profile_args.name, args.scale, args.target_trace_size),
        Some(trace_file),
    )
}
