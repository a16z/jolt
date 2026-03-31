use std::sync::Arc;

use clap::Parser;
use tracing::info;

use jolt_eval::agent::ClaudeCodeAgent;
use jolt_eval::invariant::completeness_prover::ProverCompletenessInvariant;
use jolt_eval::invariant::completeness_verifier::VerifierCompletenessInvariant;
use jolt_eval::invariant::determinism::DeterminismInvariant;
use jolt_eval::invariant::serialization_roundtrip::SerializationRoundtripInvariant;
use jolt_eval::invariant::soundness::SoundnessInvariant;
use jolt_eval::invariant::synthesis::redteam::{auto_redteam, RedTeamConfig, RedTeamResult};
use jolt_eval::invariant::synthesis::SynthesisRegistry;
use jolt_eval::invariant::zk_consistency::ZkConsistencyInvariant;
use jolt_eval::invariant::SynthesisTarget;
use jolt_eval::TestCase;

#[derive(Parser)]
#[command(name = "redteam")]
#[command(about = "AI-driven red team testing of Jolt invariants")]
struct Cli {
    /// Name of the invariant to test
    #[arg(long)]
    invariant: String,

    /// Number of red-team iterations
    #[arg(long, default_value = "10")]
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

    /// Number of random fuzz inputs to run after each agent iteration
    #[arg(long, default_value = "100")]
    num_fuzz: usize,

    /// Maximum number of Claude agentic turns per iteration
    #[arg(long, default_value = "30")]
    max_turns: usize,

    /// List available red-teamable invariants and exit
    #[arg(long)]
    list: bool,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    if cli.list {
        println!("Red-teamable invariants:");
        println!("  soundness");
        println!("  verifier_completeness");
        println!("  prover_completeness");
        println!("  determinism");
        println!("  serialization_roundtrip");
        println!("  zk_consistency");
        return Ok(());
    }

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

    let default_inputs = vec![];
    let mut registry = SynthesisRegistry::new();
    register_invariants(&mut registry, &test_case, &default_inputs);

    let invariant = registry
        .for_target(SynthesisTarget::RedTeam)
        .into_iter()
        .find(|inv| inv.name() == cli.invariant.as_str());

    let Some(invariant) = invariant else {
        eprintln!("Invariant '{}' not found or not red-teamable.", cli.invariant);
        eprintln!("Run with --list to see available invariants.");
        std::process::exit(1);
    };

    let config = RedTeamConfig {
        num_iterations: cli.iterations,
        num_fuzz_per_iteration: cli.num_fuzz,
    };

    let agent = ClaudeCodeAgent::new(&cli.model, cli.max_turns);
    let repo_dir = std::env::current_dir()?;

    info!(
        "Starting red team: invariant={}, iterations={}, model={}, fuzz_per_iter={}",
        cli.invariant, cli.iterations, cli.model, cli.num_fuzz
    );

    let result = auto_redteam(invariant, &config, &agent, &repo_dir);

    match result {
        RedTeamResult::Violation { description, error } => {
            println!();
            println!("==== VIOLATION FOUND ====");
            println!("Approach: {description}");
            println!("Error:    {error}");
            std::process::exit(1);
        }
        RedTeamResult::NoViolation { attempts } => {
            println!();
            println!(
                "No violations found after {} iterations.",
                attempts.len()
            );
            for attempt in &attempts {
                println!(
                    "  {}: {} -- {}",
                    attempt.description, attempt.approach, attempt.failure_reason
                );
            }
        }
    }

    Ok(())
}

fn register_invariants(
    registry: &mut SynthesisRegistry,
    test_case: &Arc<TestCase>,
    default_inputs: &[u8],
) {
    registry.register(Box::new(SoundnessInvariant::new(
        Arc::clone(test_case),
        default_inputs.to_vec(),
    )));
    registry.register(Box::new(VerifierCompletenessInvariant::new(Arc::clone(
        test_case,
    ))));
    registry.register(Box::new(ProverCompletenessInvariant::new(Arc::clone(
        test_case,
    ))));
    registry.register(Box::new(DeterminismInvariant::new(Arc::clone(test_case))));
    registry.register(Box::new(SerializationRoundtripInvariant::new(
        Arc::clone(test_case),
        default_inputs.to_vec(),
    )));
    registry.register(Box::new(ZkConsistencyInvariant::new(Arc::clone(
        test_case,
    ))));
}
