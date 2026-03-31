use std::process::Command;
use std::sync::Arc;

use clap::Parser;
use tracing::info;

use jolt_eval::invariant::completeness_prover::ProverCompletenessInvariant;
use jolt_eval::invariant::completeness_verifier::VerifierCompletenessInvariant;
use jolt_eval::invariant::determinism::DeterminismInvariant;
use jolt_eval::invariant::serialization_roundtrip::SerializationRoundtripInvariant;
use jolt_eval::invariant::soundness::SoundnessInvariant;
use jolt_eval::invariant::synthesis::redteam::{
    auto_redteam, create_worktree, remove_worktree, RedTeamConfig, RedTeamResult,
};
use jolt_eval::invariant::synthesis::SynthesisRegistry;
use jolt_eval::invariant::zk_consistency::ZkConsistencyInvariant;
use jolt_eval::invariant::{FailedAttempt, SynthesisTarget};
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

    let working_dir = std::env::current_dir()?;
    let config = RedTeamConfig {
        invariant_name: cli.invariant.clone(),
        num_iterations: cli.iterations,
        model: cli.model.clone(),
        working_dir: working_dir.clone(),
        num_fuzz_per_iteration: cli.num_fuzz,
    };

    info!(
        "Starting red team: invariant={}, iterations={}, model={}, fuzz_per_iter={}",
        cli.invariant, cli.iterations, cli.model, cli.num_fuzz
    );

    let model = cli.model.clone();
    let max_turns = cli.max_turns;

    let result = auto_redteam(invariant, &config, |description, failed_attempts| {
        invoke_claude_agent(&working_dir, description, failed_attempts, &model, max_turns)
    });

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

/// Invoke the Claude Code CLI in an isolated worktree to attempt to find
/// an invariant violation.
///
/// Flow:
/// 1. Create a detached git worktree so the agent has a full repo copy
/// 2. Build a prompt with the invariant description + past failed attempts
/// 3. Run `claude -p <prompt> --model <model> --max-turns <N>` in the worktree
/// 4. Capture the agent's analysis as the approach description
/// 5. Clean up the worktree
fn invoke_claude_agent(
    repo_dir: &std::path::Path,
    invariant_description: &str,
    failed_attempts: &[FailedAttempt],
    model: &str,
    max_turns: usize,
) -> Option<(String, Vec<u8>)> {
    // 1. Create worktree
    let worktree_dir = match create_worktree(repo_dir, "redteam") {
        Ok(dir) => {
            info!("Created worktree at {}", dir.display());
            dir
        }
        Err(e) => {
            tracing::error!("Failed to create worktree: {e}");
            return None;
        }
    };

    // 2. Build prompt
    let prompt = build_prompt(invariant_description, failed_attempts);

    // 3. Run Claude
    info!("Invoking claude (model={model}, max_turns={max_turns})...");
    let result = Command::new("claude")
        .current_dir(&worktree_dir)
        .arg("-p")
        .arg(&prompt)
        .arg("--model")
        .arg(model)
        .arg("--max-turns")
        .arg(max_turns.to_string())
        .arg("--verbose")
        .output();

    // 4. Clean up worktree regardless of outcome
    info!("Cleaning up worktree...");
    remove_worktree(repo_dir, &worktree_dir);
    // Also remove the temp directory itself
    let _ = std::fs::remove_dir_all(&worktree_dir);

    // 5. Parse result
    match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if !output.status.success() {
                tracing::warn!("claude exited with status {}", output.status);
                if !stderr.is_empty() {
                    tracing::warn!("stderr: {}", &stderr[..stderr.len().min(500)]);
                }
            }

            let approach = if stdout.trim().is_empty() {
                if stderr.trim().is_empty() {
                    return None;
                }
                // Sometimes useful output goes to stderr
                truncate(&stderr, 2000).to_string()
            } else {
                truncate(&stdout, 2000).to_string()
            };

            info!(
                "Agent response ({} chars): {}...",
                approach.len(),
                &approach[..approach.len().min(200)]
            );

            Some((approach, vec![]))
        }
        Err(e) => {
            tracing::error!("Failed to invoke claude: {e}");
            tracing::error!(
                "Make sure the `claude` CLI is installed and on your PATH. \
                 Install via: npm install -g @anthropic-ai/claude-code"
            );
            None
        }
    }
}

fn build_prompt(invariant_description: &str, failed_attempts: &[FailedAttempt]) -> String {
    let mut prompt = String::new();

    prompt.push_str(
        "You are a security researcher red-teaming a zkVM (Jolt). \
         Your goal is to find an input or scenario that VIOLATES the following invariant.\n\n",
    );

    prompt.push_str("## Invariant\n\n");
    prompt.push_str(invariant_description);
    prompt.push_str("\n\n");

    prompt.push_str(
        "## Instructions\n\n\
         1. Read the relevant source code in this repository to understand how the \
            invariant is enforced.\n\
         2. Look for edge cases, off-by-one errors, missing checks, or assumptions \
            that could be violated.\n\
         3. If you find a potential weakness, describe it clearly.\n\
         4. Try to construct a concrete input or scenario that triggers the violation.\n\
         5. Summarize your approach and findings.\n\n\
         Focus on finding REAL bugs, not theoretical concerns. The invariant will be \
         mechanically checked after your analysis, so only genuine violations count.\n\n",
    );

    if !failed_attempts.is_empty() {
        prompt.push_str("## Previous Failed Attempts\n\n");
        prompt.push_str(
            "The following approaches have already been tried and did NOT find a violation. \
             Try a fundamentally different approach.\n\n",
        );
        for attempt in failed_attempts {
            prompt.push_str(&format!(
                "- **{}**: {}\n  Reason for failure: {}\n",
                attempt.description, attempt.approach, attempt.failure_reason
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str(
        "## Output\n\n\
         End your response with a clear summary of:\n\
         - What you investigated\n\
         - What you found (if anything)\n\
         - Whether you believe the invariant holds or can be violated\n",
    );

    prompt
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

fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        // Find a char boundary
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}
