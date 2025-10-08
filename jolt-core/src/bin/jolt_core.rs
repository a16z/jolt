use clap::{Args, Parser, Subcommand, ValueEnum};

#[path = "../../benches/e2e_profiling.rs"]
mod e2e_profiling;
use e2e_profiling::{benchmarks, master_benchmark, BenchType};

use std::any::Any;

use chrono::Local;
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

    /// Max trace length as 2^scale (optional if target-trace-size is provided)
    #[clap(short, long)]
    scale: Option<usize>,

    /// Target specific cycle count (optional, defaults to 90% of 2^scale)
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
        Commands::Profile(args) => trace(args),
        Commands::Benchmark(args) => run_benchmark(args),
    }
}

fn normalize_bench_name(name: &str) -> String {
    name.to_lowercase().replace(" ", "_")
}

fn setup_tracing(formats: Option<Vec<Format>>, trace_name: &str) -> Vec<Box<dyn Any>> {
    if std::env::var("PPROF_PREFIX").is_err() {
        std::env::set_var(
            "PPROF_PREFIX",
            format!("benchmark-runs/pprof/{trace_name}_"),
        );
    }

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
            let trace_file = format!("benchmark-runs/perfetto_traces/{trace_name}.json");
            std::fs::create_dir_all("benchmark-runs/perfetto_traces").ok();
            let (chrome_layer, guard) = ChromeLayerBuilder::new()
                .include_args(true)
                .file(trace_file)
                .build();
            layers.push(chrome_layer.boxed());
            guards.push(Box::new(guard));
            tracing::info!("Running tracing-chrome. Files will be saved as trace-<some timestamp>.json and can be viewed in https://ui.perfetto.dev/");
        }
    }

    tracing_subscriber::registry().with(layers).init();

    #[cfg(feature = "monitor")]
    guards.push(Box::new({
        use jolt_core::utils::monitor::MetricsMonitor;
        tracing::info!("Starting MetricsMonitor - remember to run python3 scripts/postprocess_trace.py trace-*.json");
        MetricsMonitor::start(
            std::env::var("MONITOR_INTERVAL")
                .unwrap_or("0.1".to_string())
                .parse::<f64>()
                .unwrap(),
        )
    }));

    guards
}

fn trace(args: ProfileArgs) {
    let bench_name = normalize_bench_name(&args.name.to_string());
    let timestamp = Local::now().format("%Y%m%d-%H%M");
    let trace_name = format!("{bench_name}_{timestamp}");
    let _guards = setup_tracing(args.format, &trace_name);

    for (span, bench) in benchmarks(args.name).into_iter() {
        span.in_scope(|| {
            bench();
            tracing::info!("Bench Complete");
        });
    }
}

fn run_benchmark(args: BenchmarkArgs) {
    let scale = match (args.scale, args.target_trace_size) {
        (Some(s), _) => s,  // Scale provided, use it
        (None, Some(target)) => target.next_power_of_two().trailing_zeros() as usize,
        (None, None) => {
            eprintln!("Error: Must provide either --scale or --target-trace-size");
            std::process::exit(1);
        }
    };

    let bench_name = normalize_bench_name(&args.profile_args.name.to_string());
    let trace_name = format!("{}_{}", bench_name, scale);
    let _guards = setup_tracing(args.profile_args.format, &trace_name);

    // Call master_benchmark with parameters
    for (span, bench) in
        master_benchmark(args.profile_args.name, scale, args.target_trace_size).into_iter()
    {
        span.in_scope(|| {
            bench();
            tracing::info!("Benchmark Complete");
        });
    }
}
