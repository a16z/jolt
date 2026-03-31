use std::sync::Arc;

use clap::Parser;
use tracing::info;

use jolt_eval::invariant::soundness::SoundnessInvariant;
use jolt_eval::invariant::synthesis::redteam::{auto_redteam, RedTeamConfig, RedTeamResult};
use jolt_eval::invariant::synthesis::SynthesisRegistry;
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

    let mut registry = SynthesisRegistry::new();
    registry.register(Box::new(SoundnessInvariant::new(
        Arc::clone(&test_case),
        vec![],
    )));

    let invariant = registry
        .invariants()
        .iter()
        .find(|inv| inv.name() == cli.invariant.as_str())
        .map(|inv| inv.as_ref());

    let Some(invariant) = invariant else {
        eprintln!("Invariant '{}' not found.", cli.invariant);
        eprintln!("Available: soundness");
        std::process::exit(1);
    };

    let config = RedTeamConfig {
        invariant_name: cli.invariant.clone(),
        num_iterations: cli.iterations,
        model: cli.model.clone(),
        working_dir: std::env::current_dir()?,
    };

    info!(
        "Starting red team: invariant={}, iterations={}, model={}",
        cli.invariant, cli.iterations, cli.model
    );

    // The invoke_agent callback is a placeholder for actual AI interaction.
    // In production, this would shell out to `claude` CLI or use the API.
    let result = auto_redteam(invariant, &config, |description, failed_attempts| {
        info!(
            "Agent prompt: find violation of: {}",
            &description[..description.len().min(100)]
        );
        info!("Previous failed attempts: {}", failed_attempts.len());
        // Placeholder: return None (no candidate produced)
        // Real implementation would invoke Claude Code in a worktree
        None
    });

    match result {
        RedTeamResult::Violation { description, error } => {
            println!("VIOLATION FOUND!");
            println!("  Approach: {description}");
            println!("  Error: {error}");
            std::process::exit(1);
        }
        RedTeamResult::NoViolation { attempts } => {
            println!("No violations found after {} attempts.", attempts.len());
            for attempt in &attempts {
                println!(
                    "  {}: {} — {}",
                    attempt.description, attempt.approach, attempt.failure_reason
                );
            }
        }
    }

    Ok(())
}
