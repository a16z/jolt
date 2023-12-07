use clap::{Parser, ValueEnum};
use liblasso::benches::bench::{benchmarks, BenchType};
use std::{fs::File, io::BufWriter};
use tracing_flame::FlameLayer;
use tracing_subscriber::{self, fmt, fmt::format::FmtSpan, prelude::*, registry::Registry};

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser, Debug)]
struct Cli {
    /// Output format
    #[clap(short, long, value_enum, default_value_t = Format::Default)]
    format: Format,

    /// Type of benchmark to run
    #[clap(long, value_enum)]
    name: BenchType,
}

#[derive(Debug, Clone, ValueEnum)]
enum Format {
    Default,
    Texray,
    Flamegraph,
}

fn setup_global_subscriber() -> impl Drop {
    let fmt_layer = fmt::Layer::default();

    let (flame_layer, _guard) = FlameLayer::with_file("./tracing.folded").unwrap();

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(flame_layer)
        .init();

    _guard
}

fn main() {
    let args = Cli::parse();
    match args.format {
        Format::Default => {
            let collector = tracing_subscriber::fmt()
                .with_max_level(tracing::Level::TRACE)
                .with_span_events(FmtSpan::CLOSE)
                .finish();
            tracing::subscriber::set_global_default(collector)
                .expect("setting tracing default failed");
            for (span, bench) in benchmarks(args.name).into_iter() {
                span.to_owned().in_scope(|| {
                    bench();
                    tracing::info!("Bench Complete");
                });
            }
        }
        Format::Texray => {
            tracing_texray::init();
            for (span, bench) in benchmarks(args.name).into_iter() {
                tracing_texray::examine(span.to_owned()).in_scope(bench);
            }
        }
        Format::Flamegraph => {
            let _guard = setup_global_subscriber();
            for (span, bench) in benchmarks(args.name).into_iter() {
                span.to_owned().in_scope(|| {
                    bench();
                    tracing::info!("Bench Complete");
                });
            }
        }
    }
}
