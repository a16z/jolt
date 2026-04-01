use clap::Parser;

use jolt_eval::objective::{build_objectives_from_inventory, registered_objectives};
use jolt_eval::{SharedSetup, TestCase};

#[derive(Parser)]
#[command(name = "measure-objectives")]
#[command(about = "Measure Jolt performance objectives")]
struct Cli {
    /// Only measure the named objective (default: all)
    #[arg(long)]
    objective: Option<String>,

    /// Number of samples per objective
    #[arg(long)]
    samples: Option<usize>,

    /// Path to a pre-compiled guest ELF
    #[arg(long)]
    elf: Option<String>,

    /// Max trace length for the test program
    #[arg(long, default_value = "65536")]
    max_trace_length: usize,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let test_case = if let Some(elf_path) = &cli.elf {
        let elf_bytes = std::fs::read(elf_path)?;
        let memory_config = common::jolt_device::MemoryConfig {
            max_input_size: 4096,
            max_output_size: 4096,
            max_untrusted_advice_size: 0,
            max_trusted_advice_size: 0,
            stack_size: 65536,
            heap_size: 32768,
            program_size: None,
        };
        TestCase {
            elf_contents: elf_bytes,
            memory_config,
            max_trace_length: cli.max_trace_length,
        }
    } else {
        eprintln!("Error: --elf <path> is required. Provide a pre-compiled guest ELF.");
        std::process::exit(1);
    };

    let setup = SharedSetup::new(test_case);
    let objectives = build_objectives_from_inventory(&setup, vec![]);

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
        eprintln!("No matching objectives. Available: {}", all_names.join(", "));
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
