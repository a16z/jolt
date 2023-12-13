use clap::{Args, Parser, Subcommand, ValueEnum};
use itertools::izip;
use liblasso::benches::bench::{benchmarks, BenchType};
use rgb::RGB8;
use std::{fs::File, io::BufWriter, time::Instant};
use textplots::{Chart, ColorPlot, Plot, Shape};
use tracing_chrome::ChromeLayerBuilder;
use tracing_flame::FlameLayer;
use tracing_subscriber::{self, fmt, fmt::format::FmtSpan, prelude::*};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Trace(TraceArgs),
    Plot(PlotArgs),
}

#[derive(Args, Debug)]
struct TraceArgs {
    /// Output format
    #[clap(short, long, value_enum, default_value_t = Format::Default)]
    format: Format,

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

    /// Number of cycles to run the benchmark for
    #[clap(short, long, num_args = 1..)]
    num_cycles: Vec<usize>,

    #[clap(short, long, num_args = 1..)]
    memory_size: Vec<usize>,

    #[clap(short, long, num_args = 1..)]
    bytecode_size: Vec<usize>,
}

#[derive(Debug, Clone, ValueEnum)]
enum Format {
    Default,
    Texray,
    Flamegraph,
    Chrome,
}

fn main() {
    let cli = Cli::parse();

    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    match cli.command {
        Commands::Trace(args) => trace(args),
        Commands::Plot(args) => plot(args),
    }
}

fn bench_to_color(bench_type: &BenchType) -> RGB8 {
    match bench_type {
        BenchType::EverythingExceptR1CS => RGB8 {
            r: 255,
            g: 255,
            b: 255,
        }, // white
        BenchType::ReadWriteMemory => RGB8 { r: 255, g: 0, b: 0 }, // red
        BenchType::Bytecode => RGB8 { r: 0, g: 0, b: 255 },        // blue
        BenchType::InstructionLookups => RGB8 { r: 0, g: 255, b: 0 }, // green
        _ => panic!("Unsupported bench type"),
    }
}

fn plot(args: PlotArgs) {
    let mut x: Vec<Vec<f32>> = Vec::with_capacity(args.bench.len());
    let mut y: Vec<Vec<f32>> = Vec::with_capacity(args.bench.len());

    for bench_type in &args.bench {
        if args.num_cycles.len() > 1 {
            assert!(args.memory_size.len() == 1);
            assert!(args.bytecode_size.len() == 1);

            let x_i = args.num_cycles.iter().map(|z| *z as f32).collect();
            let mut y_i = Vec::with_capacity(args.num_cycles.len());
            for num_cycles in &args.num_cycles {
                for (_, bench) in benchmarks(
                    *bench_type,
                    Some(*num_cycles),
                    Some(args.memory_size[0]),
                    Some(args.bytecode_size[0]),
                )
                .into_iter()
                {
                    let now = Instant::now();
                    bench();
                    y_i.push(now.elapsed().as_micros() as f32 / 1_000_000.0); // time in seconds
                }
            }
            x.push(x_i);
            y.push(y_i);
        }

        if args.memory_size.len() > 1 {
            assert!(args.num_cycles.len() == 1);
            assert!(args.bytecode_size.len() == 1);

            let x_i = args.memory_size.iter().map(|z| *z as f32).collect();
            let mut y_i = Vec::with_capacity(args.memory_size.len());
            for memory_size in &args.memory_size {
                for (_, bench) in benchmarks(
                    *bench_type,
                    Some(args.num_cycles[0]),
                    Some(*memory_size),
                    Some(args.bytecode_size[0]),
                )
                .into_iter()
                {
                    let now = Instant::now();
                    bench();
                    y_i.push(now.elapsed().as_micros() as f32 / 1_000_000.0); // time in seconds
                }
            }
            x.push(x_i);
            y.push(y_i);
        }

        if args.bytecode_size.len() > 1 {
            assert!(args.num_cycles.len() == 1);
            assert!(args.memory_size.len() == 1);

            let x_i = args.bytecode_size.iter().map(|z| *z as f32).collect();
            let mut y_i = Vec::with_capacity(args.bytecode_size.len());
            for bytecode_size in &args.bytecode_size {
                for (_, bench) in benchmarks(
                    *bench_type,
                    Some(args.num_cycles[0]),
                    Some(args.memory_size[0]),
                    Some(*bytecode_size),
                )
                .into_iter()
                {
                    let now = Instant::now();
                    bench();
                    y_i.push(now.elapsed().as_micros() as f32 / 1_000_000.0); // time in seconds
                }
            }
            x.push(x_i);
            y.push(y_i);
        }
    }

    let xmin = x
        .iter()
        .flatten()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let xmax = x
        .iter()
        .flatten()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let ymax = y
        .iter()
        .flatten()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mut chart = &mut Chart::new_with_y_range(120, 80, *xmin, *xmax, 0.0, *ymax);
    let points: Vec<_> = izip!(x, y)
        .map(|(x_i, y_i)| izip!(x_i, y_i).collect::<Vec<(f32, f32)>>())
        .collect();
    let lines: Vec<_> = points.iter().map(|p| Shape::Lines(p)).collect();
    for (i, bench_type) in args.bench.iter().enumerate() {
        chart = chart.linecolorplot(&lines[i], bench_to_color(bench_type));
    }
    chart.display();
}

fn trace(args: TraceArgs) {
    match args.format {
        Format::Default => {
            let collector = tracing_subscriber::fmt()
                .with_max_level(tracing::Level::TRACE)
                .with_span_events(FmtSpan::CLOSE)
                .finish();
            tracing::subscriber::set_global_default(collector)
                .expect("setting tracing default failed");
            for (span, bench) in benchmarks(args.name, args.num_cycles, None, None).into_iter() {
                span.to_owned().in_scope(|| {
                    bench();
                    tracing::info!("Bench Complete");
                });
            }
        }
        Format::Texray => {
            tracing_texray::init();
            for (span, bench) in benchmarks(args.name, args.num_cycles, None, None).into_iter() {
                tracing_texray::examine(span.to_owned()).in_scope(bench);
            }
        }
        Format::Flamegraph => {
            let fmt_layer = fmt::Layer::default();
            let (flame_layer, _guard) = FlameLayer::with_file("./tracing.folded").unwrap();
            tracing_subscriber::registry()
                .with(fmt_layer)
                .with(flame_layer)
                .init();

            for (span, bench) in benchmarks(args.name, args.num_cycles, None, None).into_iter() {
                span.to_owned().in_scope(|| {
                    bench();
                    tracing::info!("Bench Complete");
                });
            }
        }
        Format::Chrome => {
            let (chrome_layer, _guard) = ChromeLayerBuilder::new().build();
            tracing_subscriber::registry().with(chrome_layer).init();
            println!("Running tracing-chrome. Files will be saved as trace-<some timestamp>.json and can be viewed in chrome://tracing.");
            for (span, bench) in benchmarks(args.name, args.num_cycles, None, None).into_iter() {
                span.to_owned().in_scope(|| {
                    bench();
                    tracing::info!("Bench Complete");
                });
            }
            drop(_guard);
        }
    }
}
