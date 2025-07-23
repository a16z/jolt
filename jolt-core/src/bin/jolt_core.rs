use clap::{Args, Parser, Subcommand, ValueEnum};

use jolt_core::benches::bench::{benchmarks, BenchType, PCSType};

use std::any::Any;

use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{self, fmt::format::FmtSpan, prelude::*};

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Profile(ProfileArgs),
}

#[derive(Args, Debug)]
struct ProfileArgs {
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

#[derive(Debug, Clone, ValueEnum, PartialEq)]
enum Format {
    Default,
    Chrome,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Profile(args) => trace(args),
    }
}

fn trace(args: ProfileArgs) {
    let mut layers = Vec::new();

    let mut guards: Vec<Box<dyn Any>> = vec![];

    if let Some(format) = &args.format {
        if format.contains(&Format::Default) {
            let collector_layer = tracing_subscriber::fmt::layer()
                .with_span_events(FmtSpan::CLOSE)
                .boxed();
            layers.push(collector_layer);
        }
        if format.contains(&Format::Chrome) {
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            layers.push(chrome_layer.boxed());
            guards.push(Box::new(guard));
            println!("Running tracing-chrome. Files will be saved as trace-<some timestamp>.json and can be viewed in chrome://tracing.");
        }
    }

    tracing_subscriber::registry().with(layers).init();
    for (span, bench) in benchmarks(args.pcs, args.name).into_iter() {
        span.to_owned().in_scope(|| {
            bench();
            tracing::info!("Bench Complete");
        });
    }
}
