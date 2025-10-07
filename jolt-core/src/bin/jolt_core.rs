use clap::{Args, Parser, Subcommand, ValueEnum};

#[path = "../../benches/e2e_profiling.rs"]
mod e2e_profiling;
use e2e_profiling::{benchmarks, BenchType};

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
    /// Benchmark type
    #[clap(long, value_enum)]
    name: BenchType,

    /// Max trace length as 2^scale
    #[clap(short, long)]
    scale: usize,

    /// Target specific cycle count (optional, defaults to 90% of 2^scale)
    #[clap(short, long)]
    target_trace_size: Option<usize>,

    /// Output formats
    #[clap(short, long, value_enum)]
    format: Option<Vec<Format>>,
}

#[derive(Debug, Clone, ValueEnum, PartialEq)]
enum Format {
    Default,
    Chrome,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Profile(args) => trace(args),
        Commands::Benchmark(args) => run_benchmark(args),
    }
}

fn trace(args: ProfileArgs) {
    let mut layers = Vec::new();

    let log_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_filter(EnvFilter::from_default_env()) // reads RUST_LOG
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
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            layers.push(chrome_layer.boxed());
            guards.push(Box::new(guard));
            tracing::info!("Running tracing-chrome. Files will be saved as trace-<some timestamp>.json and can be viewed in https://ui.perfetto.dev/");
        }
    }

    tracing_subscriber::registry().with(layers).init();
    for (span, bench) in benchmarks(args.name).into_iter() {
        span.to_owned().in_scope(|| {
            bench();
            tracing::info!("Bench Complete");
        });
    }
}

fn run_benchmark(args: BenchmarkArgs) {
    // For now, bridge to existing env-var-based master_benchmark
    // Will be improved in later steps
    std::env::set_var("BENCH_TYPE", format!("{}", args.name).to_lowercase());
    std::env::set_var("BENCH_SCALE", args.scale.to_string());
    if let Some(target) = args.target_trace_size {
        std::env::set_var("TARGET_TRACE_SIZE", target.to_string());
    }
    
    // Setup tracing with optional custom filename
    let profile_args = ProfileArgs {
        format: args.format,
        name: BenchType::MasterBenchmark,
    };
    trace(profile_args);
}
