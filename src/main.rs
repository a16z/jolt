use liblasso::benches::bench::{benchmarks, BenchType};
use tracing_subscriber::{self, fmt::format::FmtSpan};

use clap::Parser;

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser, Debug)]
struct Cli {
  /// Whether to present in chart format
  #[clap(long, short, action)]
  chart: bool,

  /// Type of benchmark to run
  #[clap(long, value_enum)]
  name: BenchType,
}

fn main() {
  let args = Cli::parse();
  if args.chart {
    tracing_texray::init();
    for (span, bench) in benchmarks(args.name).iter() {
      tracing_texray::examine(span.to_owned()).in_scope(bench);
    }
  } else {
    let collector = tracing_subscriber::fmt()
      .with_max_level(tracing::Level::TRACE)
      .with_span_events(FmtSpan::CLOSE)
      .finish();
    tracing::subscriber::set_global_default(collector).expect("setting tracing default failed");
    for (span, bench) in benchmarks(args.name).iter() {
      span.to_owned().in_scope(|| {
        bench();
        tracing::info!("Bench Complete");
      });
    }
  }
}
