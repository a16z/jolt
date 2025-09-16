use clap::{Args, Parser, Subcommand, ValueEnum};

use jolt_core::benches::bench::{benchmarks, create_benchmark_plot, BenchType};

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
}

#[derive(Args, Debug)]
struct ProfileArgs {
    /// Output formats
    #[clap(short, long, value_enum)]
    format: Option<Vec<Format>>,

    /// Type of benchmark to run
    #[clap(long, value_enum)]
    name: BenchType,
}

#[derive(Debug, Clone, ValueEnum, PartialEq)]
enum Format {
    Default,
    Chrome,
}

fn main() {
    let _ = create_benchmark_plot();
    // let cli = Cli::parse();
    // match cli.command {
    //     Commands::Profile(args) => trace(args),
    // }
}

fn trace(args: ProfileArgs) {
    let mut layers = Vec::new();

    let log_layer = tracing_subscriber::fmt::layer()
        .pretty()
        .with_filter(EnvFilter::from_default_env()) // reads RUST_LOG
        .boxed();
    layers.push(log_layer);

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
            println!("Running tracing-chrome. Files will be saved as trace-<some timestamp>.json and can be viewed in https://ui.perfetto.dev/");
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
