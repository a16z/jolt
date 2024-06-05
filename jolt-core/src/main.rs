use clap::{Args, Parser, Subcommand, ValueEnum};

use jolt_core::benches::{
    bench::{benchmarks, BenchType, PCSType},
    sum_timer::CumulativeTimingLayer,
};

use std::any::Any;

use tracing_chrome::ChromeLayerBuilder;
use tracing_flame::FlameLayer;
use tracing_subscriber::{self, fmt::format::FmtSpan, prelude::*};
use tracing_texray::TeXRayLayer;

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Trace(TraceArgs),
}

#[derive(Args, Debug)]
struct TraceArgs {
    /// Output formats
    #[clap(short, long, value_enum)]
    format: Option<Vec<Format>>,

    #[clap(long, value_enum)]
    pcs: PCSType,

    /// Type of benchmark to run
    #[clap(long, value_enum)]
    name: BenchType,

    /// Number of cycles to run the benchmark for
    #[clap(short, long)]
    num_cycles: Option<usize>,
}

#[derive(Args, Debug)]
struct PlotArgs {
    /// Type of benchmark to run
    #[clap(long, value_enum, num_args = 1..)]
    bench: Vec<BenchType>,

    /// SVG output file path
    #[clap(short, long)]
    out: Option<String>,

    /// Number of cycles to run benchmarks for
    #[clap(short, long, num_args = 1..)]
    num_cycles: Vec<usize>,

    /// Size of read-write memory to run benchmarks with
    #[clap(short, long, num_args = 1..)]
    memory_size: Vec<usize>,

    /// Size of bytecode to run benchmarks with
    #[clap(short, long, num_args = 1..)]
    bytecode_size: Vec<usize>,
}

#[derive(Debug, Clone, ValueEnum, PartialEq)]
enum Format {
    Default,
    Texray,
    Flamegraph,
    Chrome,
    Sum,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Trace(args) => trace(args),
    }
}

fn trace(args: TraceArgs) {
    let mut layers = Vec::new();

    let mut guards: Vec<Box<dyn Any>> = vec![];

    if let Some(format) = &args.format {
        if format.contains(&Format::Default) {
            let collector_layer = tracing_subscriber::fmt::layer()
                .with_span_events(FmtSpan::CLOSE)
                .boxed();
            layers.push(collector_layer);
        }
        if format.contains(&Format::Texray) {
            let texray_layer = TeXRayLayer::new();
            layers.push(texray_layer.boxed());
        }
        if format.contains(&Format::Flamegraph) {
            let (flame_layer, guard) = FlameLayer::with_file("./tracing.folded").unwrap();
            layers.push(flame_layer.boxed());
            guards.push(Box::new(guard));
        }
        if format.contains(&Format::Chrome) {
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            layers.push(chrome_layer.boxed());
            guards.push(Box::new(guard));
            println!("Running tracing-chrome. Files will be saved as trace-<some timestamp>.json and can be viewed in chrome://tracing.");
        }
        if format.contains(&Format::Sum) {
            let (sum_timing_layer, guard) = CumulativeTimingLayer::new(None);
            layers.push(sum_timing_layer.boxed());
            guards.push(Box::new(guard));
        }
    }

    tracing_subscriber::registry().with(layers).init();
    for (span, bench) in benchmarks(args.pcs, args.name, args.num_cycles, None, None).into_iter() {
        span.to_owned().in_scope(|| {
            bench();
            tracing::info!("Bench Complete");
        });
    }
}
