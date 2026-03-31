use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;

use clap::Parser;

use jolt_eval::agent::ClaudeCodeAgent;
use jolt_eval::invariant::completeness_prover::ProverCompletenessInvariant;
use jolt_eval::invariant::completeness_verifier::VerifierCompletenessInvariant;
use jolt_eval::invariant::determinism::DeterminismInvariant;
use jolt_eval::invariant::serialization_roundtrip::SerializationRoundtripInvariant;
use jolt_eval::invariant::soundness::SoundnessInvariant;
use jolt_eval::invariant::synthesis::SynthesisRegistry;
use jolt_eval::invariant::zk_consistency::ZkConsistencyInvariant;
use jolt_eval::objective::guest_cycles::GuestCycleCountObjective;
use jolt_eval::objective::inline_lengths::InlineLengthsObjective;
use jolt_eval::objective::optimize::{auto_optimize, OptimizeConfig, OptimizeEnv};
use jolt_eval::objective::peak_rss::PeakRssObjective;
use jolt_eval::objective::proof_size::ProofSizeObjective;
use jolt_eval::objective::prover_time::ProverTimeObjective;
use jolt_eval::objective::verifier_time::VerifierTimeObjective;
use jolt_eval::objective::wrapping_cost::WrappingCostObjective;
use jolt_eval::objective::{measure_objectives, Direction, Objective};
use jolt_eval::TestCase;

#[derive(Parser)]
#[command(name = "optimize")]
#[command(about = "AI-driven optimization of Jolt objectives")]
struct Cli {
    /// Objectives to optimize (comma-separated). Default: all.
    #[arg(long)]
    objectives: Option<String>,

    /// Number of optimization iterations
    #[arg(long, default_value = "5")]
    iterations: usize,

    /// AI model to use
    #[arg(long, default_value = "claude-sonnet-4-20250514")]
    model: String,

    /// Path to a pre-compiled guest ELF
    #[arg(long)]
    elf: String,

    /// Max trace length for the test program
    #[arg(long, default_value = "65536")]
    max_trace_length: usize,

    /// Maximum number of Claude agentic turns per iteration
    #[arg(long, default_value = "30")]
    max_turns: usize,

    /// Extra context to include in the optimization prompt
    #[arg(long)]
    hint: Option<String>,
}

/// Real environment backed by Jolt objectives, invariants, and git.
struct RealEnv {
    objectives: Vec<Objective>,
    registry: SynthesisRegistry,
    repo_dir: std::path::PathBuf,
}

impl OptimizeEnv for RealEnv {
    fn measure(&mut self) -> HashMap<String, f64> {
        measure_objectives(&self.objectives)
    }

    fn check_invariants(&mut self) -> bool {
        self.registry.invariants().iter().all(|inv| {
            let results = inv.run_checks(0);
            results.iter().all(|r| r.is_ok())
        })
    }

    fn directions(&self) -> HashMap<String, Direction> {
        self.objectives
            .iter()
            .map(|o| (o.name().to_string(), o.direction()))
            .collect()
    }

    fn apply_diff(&mut self, diff: &str) {
        if let Err(e) = jolt_eval::agent::apply_diff(&self.repo_dir, diff) {
            tracing::warn!("Failed to apply diff: {e}");
        }
    }

    fn accept(&mut self, iteration: usize) {
        println!("  Improvement found -- keeping changes.");
        let _ = Command::new("git")
            .current_dir(&self.repo_dir)
            .args(["add", "-A"])
            .status();
        let msg = format!("perf(auto-optimize): iteration {iteration}");
        let _ = Command::new("git")
            .current_dir(&self.repo_dir)
            .args(["commit", "-m", &msg, "--allow-empty"])
            .status();
    }

    fn reject(&mut self) {
        println!("  Reverting changes.");
        let _ = Command::new("git")
            .current_dir(&self.repo_dir)
            .args(["checkout", "."])
            .status();
    }
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let elf_bytes = std::fs::read(&cli.elf)?;
    let memory_config = common::jolt_device::MemoryConfig {
        max_input_size: 4096,
        max_output_size: 4096,
        max_untrusted_advice_size: 0,
        max_trusted_advice_size: 0,
        stack_size: 65536,
        heap_size: 32768,
        program_size: None,
    };
    let test_case = Arc::new(TestCase {
        elf_contents: elf_bytes,
        memory_config,
        max_trace_length: cli.max_trace_length,
    });

    let inputs = vec![];
    let prover_pp = Arc::new(test_case.prover_preprocessing());
    let verifier_pp = Arc::new(TestCase::verifier_preprocessing(&prover_pp));

    let all_objectives = build_objectives(&test_case, &prover_pp, &verifier_pp, &inputs);
    let objective_names: Vec<String> = if let Some(names) = &cli.objectives {
        names.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        all_objectives
            .iter()
            .map(|o| o.name().to_string())
            .collect()
    };

    let objectives: Vec<Objective> = all_objectives
        .into_iter()
        .filter(|o| objective_names.contains(&o.name().to_string()))
        .collect();

    if objectives.is_empty() {
        eprintln!("No matching objectives. Available: peak_rss, prover_time, proof_size, verifier_time, guest_cycle_count, inline_lengths, wrapping_cost");
        std::process::exit(1);
    }

    let default_inputs = vec![];
    let mut registry = SynthesisRegistry::new();
    register_invariants(&mut registry, &test_case, &default_inputs);

    let repo_dir = std::env::current_dir()?;
    let agent = ClaudeCodeAgent::new(&cli.model, cli.max_turns);
    let config = OptimizeConfig {
        num_iterations: cli.iterations,
        hint: cli.hint.clone(),
    };

    let mut env = RealEnv {
        objectives,
        registry,
        repo_dir,
    };

    println!("=== Baseline measurements ===");
    let baseline = env.measure();
    print_measurements(&env.directions(), &baseline);
    println!();

    let result = auto_optimize(&agent, &mut env, &config, &std::env::current_dir()?);

    println!("=== Optimization summary ===");
    println!(
        "{}/{} iterations produced improvements.",
        result
            .attempts
            .iter()
            .filter(|a| a.invariants_passed
                && a.measurements
                    .iter()
                    .any(|(name, &val)| { result.baseline.get(name).is_some_and(|&b| val != b) }))
            .count(),
        result.attempts.len()
    );
    println!();
    println!("Final measurements:");
    print_measurements(&env.directions(), &result.best);

    Ok(())
}

fn print_measurements(directions: &HashMap<String, Direction>, measurements: &HashMap<String, f64>) {
    let mut names: Vec<_> = directions.keys().collect();
    names.sort();
    for name in names {
        let val = measurements
            .get(name)
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "N/A".to_string());
        let dir = match directions[name] {
            Direction::Minimize => "min",
            Direction::Maximize => "max",
        };
        println!("  {:<25} {:>15} {:>6}", name, val, dir);
    }
}

fn register_invariants(
    registry: &mut SynthesisRegistry,
    test_case: &Arc<TestCase>,
    default_inputs: &[u8],
) {
    registry.register(Box::new(SoundnessInvariant::new(Arc::clone(test_case), default_inputs.to_vec())));
    registry.register(Box::new(VerifierCompletenessInvariant::new(Arc::clone(test_case))));
    registry.register(Box::new(ProverCompletenessInvariant::new(Arc::clone(test_case))));
    registry.register(Box::new(DeterminismInvariant::new(Arc::clone(test_case))));
    registry.register(Box::new(SerializationRoundtripInvariant::new(Arc::clone(test_case), default_inputs.to_vec())));
    registry.register(Box::new(ZkConsistencyInvariant::new(Arc::clone(test_case))));
}

fn build_objectives(
    test_case: &Arc<TestCase>,
    prover_pp: &Arc<jolt_eval::ProverPreprocessing>,
    verifier_pp: &Arc<jolt_eval::VerifierPreprocessing>,
    inputs: &[u8],
) -> Vec<Objective> {
    vec![
        Objective::PeakRss(PeakRssObjective::new(Arc::clone(test_case), Arc::clone(prover_pp), inputs.to_vec())),
        Objective::ProverTime(ProverTimeObjective::new(Arc::clone(test_case), Arc::clone(prover_pp), inputs.to_vec())),
        Objective::ProofSize(ProofSizeObjective::new(Arc::clone(test_case), Arc::clone(prover_pp), inputs.to_vec())),
        Objective::VerifierTime(VerifierTimeObjective::new(Arc::clone(test_case), Arc::clone(prover_pp), Arc::clone(verifier_pp), inputs.to_vec())),
        Objective::GuestCycleCount(GuestCycleCountObjective::new(Arc::clone(test_case), inputs.to_vec())),
        Objective::InlineLengths(InlineLengthsObjective::new(Arc::clone(test_case))),
        Objective::WrappingCost(WrappingCostObjective::new(Arc::clone(test_case), Arc::clone(prover_pp))),
    ]
}
