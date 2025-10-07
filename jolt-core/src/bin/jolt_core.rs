use clap::{Args, Parser, Subcommand, ValueEnum};

#[path = "../../benches/e2e_profiling.rs"]
mod e2e_profiling;
use e2e_profiling::{benchmarks, master_benchmark, BenchType};

use std::any::Any;

use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{self, fmt::format::FmtSpan, prelude::*, EnvFilter};
use chrono::Local;

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

fn normalize_bench_name(name: &str) -> String {
    name.to_lowercase().replace(" ", "_")
}

fn setup_tracing(formats: Option<Vec<Format>>, trace_name: &str) -> Vec<Box<dyn Any>> {
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

    if let Some(format) = formats {
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
            let trace_file = format!("benchmark-runs/perfetto_traces/{}.json", trace_name);
            std::fs::create_dir_all("benchmark-runs/perfetto_traces").ok();
            let (chrome_layer, guard) = ChromeLayerBuilder::new().include_args(true).file(trace_file).build();
            layers.push(chrome_layer.boxed());
            guards.push(Box::new(guard));
            tracing::info!("Running tracing-chrome. Files will be saved as trace-<some timestamp>.json and can be viewed in https://ui.perfetto.dev/");
        }
    }

    tracing_subscriber::registry().with(layers).init();
    guards
}

fn trace(args: ProfileArgs) {
    let bench_name = normalize_bench_name(&args.name.to_string());
    let timestamp = Local::now().format("%Y%m%d-%H%M");
    let trace_name = format!("{}_{}", bench_name, timestamp);
    let _guards = setup_tracing(args.format, &trace_name);
    
    for (span, bench) in benchmarks(args.name).into_iter() {
        span.in_scope(|| {
            bench();
            tracing::info!("Bench Complete");
        });
    }
}

fn run_benchmark(args: BenchmarkArgs) {
    let bench_name = normalize_bench_name(&args.name.to_string());
    let trace_name = format!("{}_{}", bench_name, args.scale);
    let _guards = setup_tracing(args.format, &trace_name);
    
    // Call master_benchmark with parameters
    for (span, bench) in master_benchmark(
        vec![args.name],
        vec![args.scale],
        args.target_trace_size,
    ).into_iter() {
        span.in_scope(|| {
            bench();
            tracing::info!("Benchmark Complete");
        });
    }
}
