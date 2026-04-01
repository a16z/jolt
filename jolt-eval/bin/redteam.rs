use clap::Parser;
use tracing::info;

use jolt_eval::agent::ClaudeCodeAgent;
use jolt_eval::guests;
use jolt_eval::invariant::synthesis::redteam::{auto_redteam, RedTeamConfig, RedTeamResult};
use jolt_eval::invariant::synthesis::{invariant_names, SynthesisRegistry};
use jolt_eval::invariant::SynthesisTarget;

#[derive(Parser)]
#[command(name = "redteam")]
#[command(about = "AI-driven red team testing of Jolt invariants")]
struct Cli {
    /// Guest program to evaluate (e.g. muldiv, fibonacci, sha2)
    #[arg(long)]
    guest: Option<String>,

    /// Path to a pre-compiled guest ELF (alternative to --guest)
    #[arg(long)]
    elf: Option<String>,

    /// Name of the invariant to test
    #[arg(long)]
    invariant: String,

    /// Number of red-team iterations
    #[arg(long, default_value = "10")]
    iterations: usize,

    /// AI model to use
    #[arg(long, default_value = "claude-sonnet-4-20250514")]
    model: String,

    /// Maximum number of Claude agentic turns per iteration
    #[arg(long, default_value = "30")]
    max_turns: usize,

    /// Max trace length override
    #[arg(long)]
    max_trace_length: Option<usize>,

    /// List available red-teamable invariants and exit
    #[arg(long)]
    list: bool,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    if cli.list {
        println!("Red-teamable invariants:");
        for name in invariant_names() {
            println!("  {name}");
        }
        return Ok(());
    }

    let (test_case, default_inputs) = guests::resolve_test_case(
        cli.guest.as_deref(),
        cli.elf.as_deref(),
        cli.max_trace_length,
    );

    let registry = SynthesisRegistry::from_inventory(Some(test_case), default_inputs);

    let invariant = registry
        .for_target(SynthesisTarget::RedTeam)
        .into_iter()
        .find(|inv| inv.name() == cli.invariant.as_str());

    let Some(invariant) = invariant else {
        eprintln!(
            "Invariant '{}' not found or not red-teamable.",
            cli.invariant
        );
        eprintln!("Run with --list to see available invariants.");
        std::process::exit(1);
    };

    let config = RedTeamConfig {
        num_iterations: cli.iterations,
    };
    let agent = ClaudeCodeAgent::new(&cli.model, cli.max_turns);
    let repo_dir = std::env::current_dir()?;

    info!(
        "Starting red team: invariant={}, iterations={}, model={}",
        cli.invariant, cli.iterations, cli.model
    );

    let result = auto_redteam(invariant, &config, &agent, &repo_dir);

    match result {
        RedTeamResult::Violation {
            approach,
            input_json,
            error,
        } => {
            println!();
            println!("==== VIOLATION FOUND ====");
            println!("Approach:  {approach}");
            println!("Input:     {input_json}");
            println!("Error:     {error}");
            std::process::exit(1);
        }
        RedTeamResult::NoViolation { attempts } => {
            println!();
            println!("No violations found after {} iterations.", attempts.len());
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
