use std::sync::Arc;

use clap::Parser;

use jolt_eval::objective::guest_cycles::GuestCycleCountObjective;
use jolt_eval::objective::inline_lengths::InlineLengthsObjective;
use jolt_eval::objective::peak_rss::PeakRssObjective;
use jolt_eval::objective::proof_size::ProofSizeObjective;
use jolt_eval::objective::prover_time::ProverTimeObjective;
use jolt_eval::objective::verifier_time::VerifierTimeObjective;
use jolt_eval::objective::wrapping_cost::WrappingCostObjective;
use jolt_eval::objective::Objective;
use jolt_eval::TestCase;

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
        Arc::new(TestCase {
            elf_contents: elf_bytes,
            memory_config,
            max_trace_length: cli.max_trace_length,
        })
    } else {
        eprintln!("Error: --elf <path> is required. Provide a pre-compiled guest ELF.");
        std::process::exit(1);
    };

    let inputs = vec![];
    let prover_pp = Arc::new(test_case.prover_preprocessing());
    let verifier_pp = Arc::new(TestCase::verifier_preprocessing(&prover_pp));

    let objectives = build_objectives(&test_case, &prover_pp, &verifier_pp, &inputs);

    let filtered: Vec<&Objective> = if let Some(name) = &cli.objective {
        objectives
            .iter()
            .filter(|o| o.name() == name.as_str())
            .collect()
    } else {
        objectives.iter().collect()
    };

    if filtered.is_empty() {
        eprintln!("No matching objectives found.");
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

fn build_objectives(
    test_case: &Arc<TestCase>,
    prover_pp: &Arc<jolt_eval::ProverPreprocessing>,
    verifier_pp: &Arc<jolt_eval::VerifierPreprocessing>,
    inputs: &[u8],
) -> Vec<Objective> {
    vec![
        Objective::PeakRss(PeakRssObjective::new(
            Arc::clone(test_case),
            Arc::clone(prover_pp),
            inputs.to_vec(),
        )),
        Objective::ProverTime(ProverTimeObjective::new(
            Arc::clone(test_case),
            Arc::clone(prover_pp),
            inputs.to_vec(),
        )),
        Objective::ProofSize(ProofSizeObjective::new(
            Arc::clone(test_case),
            Arc::clone(prover_pp),
            inputs.to_vec(),
        )),
        Objective::VerifierTime(VerifierTimeObjective::new(
            Arc::clone(test_case),
            Arc::clone(prover_pp),
            Arc::clone(verifier_pp),
            inputs.to_vec(),
        )),
        Objective::GuestCycleCount(GuestCycleCountObjective::new(
            Arc::clone(test_case),
            inputs.to_vec(),
        )),
        Objective::InlineLengths(InlineLengthsObjective::new(Arc::clone(test_case))),
        Objective::WrappingCost(WrappingCostObjective::new(
            Arc::clone(test_case),
            Arc::clone(prover_pp),
        )),
    ]
}
