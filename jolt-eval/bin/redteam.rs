use clap::Parser;
use tracing::info;

use jolt_eval::agent::ClaudeCodeAgent;
use jolt_eval::invariant::sort_e2e;
use jolt_eval::invariant::synthesis::redteam::{auto_redteam, RedTeamConfig, RedTeamResult};
use jolt_eval::invariant::{JoltInvariants, SynthesisTarget};

#[derive(Parser)]
#[command(name = "redteam")]
#[command(about = "AI-driven red team testing of Jolt invariants")]
struct Cli {
    /// Name of the invariant to test (mutually exclusive with --test).
    #[arg(long, conflicts_with = "test")]
    invariant: Option<String>,

    /// Run the built-in e2e sort test instead of a named invariant.
    #[arg(long, conflicts_with = "invariant")]
    test: bool,

    /// List all red-teamable invariants and exit.
    #[arg(long)]
    list: bool,

    /// Number of red-team iterations
    #[arg(long, default_value = "10")]
    iterations: usize,

    /// AI model to use
    #[arg(long, default_value = "claude-sonnet-4-20250514")]
    model: String,

    /// Maximum number of Claude agentic turns per iteration
    #[arg(long, default_value = "30")]
    max_turns: usize,

    /// Extra context or guidance for the red-team agent
    #[arg(long)]
    hint: Option<String>,

    /// Print agent prompts and responses to stderr.
    #[arg(long)]
    verbose: bool,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    if cli.list {
        println!("Red-teamable invariants:");
        for inv in &JoltInvariants::all() {
            if inv.targets().contains(SynthesisTarget::RedTeam) {
                println!("  {}", inv.name());
            }
        }
        println!("\nBuilt-in e2e targets (use --test):");
        println!("  candidate_sort");
        return Ok(());
    }

    if cli.test {
        sort_e2e::run_redteam_test(
            &cli.model,
            cli.max_turns,
            cli.iterations,
            cli.hint,
            cli.verbose,
        );
        return Ok(());
    }

    let invariant_name = cli
        .invariant
        .as_deref()
        .expect("--invariant or --test is required (use --list to see options)");

    let all = JoltInvariants::all();
    let invariant = all
        .iter()
        .filter(|inv| inv.targets().contains(SynthesisTarget::RedTeam))
        .find(|inv| inv.name() == invariant_name);

    let Some(invariant) = invariant else {
        eprintln!("Invariant '{invariant_name}' not found or not red-teamable.");
        eprintln!("Run with --list to see available invariants.");
        std::process::exit(1);
    };

    let config = RedTeamConfig {
        num_iterations: cli.iterations,
        hint: cli.hint,
        verbose: cli.verbose,
    };
    let agent = ClaudeCodeAgent::new(&cli.model, cli.max_turns);
    let repo_dir = std::env::current_dir()?;

    info!(
        "Starting red team: invariant={invariant_name}, iterations={}, model={}",
        cli.iterations, cli.model
    );

    let result = match invariant {
        JoltInvariants::SplitEqBindLowHigh(inv) => auto_redteam(inv, &config, &agent, &repo_dir),
        JoltInvariants::SplitEqBindHighLow(inv) => auto_redteam(inv, &config, &agent, &repo_dir),
        JoltInvariants::Soundness(inv) => auto_redteam(inv, &config, &agent, &repo_dir),
    };

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
