use clap::Parser;

use jolt_eval::guests;
use jolt_eval::objective::{build_objectives_from_inventory, registered_objectives};
use jolt_eval::SharedSetup;

#[derive(Parser)]
#[command(name = "measure-objectives")]
#[command(about = "Measure Jolt performance objectives")]
struct Cli {
    /// Guest program to evaluate (e.g. muldiv, fibonacci, sha2)
    #[arg(long)]
    guest: Option<String>,

    /// Path to a pre-compiled guest ELF (alternative to --guest)
    #[arg(long)]
    elf: Option<String>,

    /// Only measure the named objective (default: all)
    #[arg(long)]
    objective: Option<String>,

    /// Number of samples per objective
    #[arg(long)]
    samples: Option<usize>,

    /// Max trace length override
    #[arg(long)]
    max_trace_length: Option<usize>,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let (test_case, default_inputs) = guests::resolve_test_case(
        cli.guest.as_deref(),
        cli.elf.as_deref(),
        cli.max_trace_length,
    );

    let setup = SharedSetup::new_from_arc(test_case);
    let objectives = build_objectives_from_inventory(Some(&setup), default_inputs);

    let filtered: Vec<_> = if let Some(name) = &cli.objective {
        objectives
            .into_iter()
            .filter(|o| o.name() == name.as_str())
            .collect()
    } else {
        objectives
    };

    if filtered.is_empty() {
        let all_names: Vec<_> = registered_objectives().map(|e| e.name).collect();
        eprintln!(
            "No matching objectives. Available: {}",
            all_names.join(", ")
        );
        std::process::exit(1);
    }

    println!("{:<25} {:>15} {:>10}", "Objective", "Value", "Direction");
    println!("{}", "-".repeat(52));

    for obj in &filtered {
        let samples = cli.samples.unwrap_or(1);
        let mut measurements = Vec::new();

        for _ in 0..samples {
            match obj.collect_measurement() {
                Ok(val) => measurements.push(val),
                Err(e) => {
                    println!("{:<25} {:>15}", obj.name(), format!("ERROR: {e}"));
                    continue;
                }
            }
        }

        if !measurements.is_empty() {
            let mean = measurements.iter().sum::<f64>() / measurements.len() as f64;
            let dir = match obj.direction() {
                jolt_eval::Direction::Minimize => "min",
                jolt_eval::Direction::Maximize => "max",
            };
            println!("{:<25} {:>15.2} {:>10}", obj.name(), mean, dir);
        }
    }

    Ok(())
}
