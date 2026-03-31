use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;

use clap::Parser;
use tracing::info;

use jolt_eval::invariant::completeness_prover::ProverCompletenessInvariant;
use jolt_eval::invariant::completeness_verifier::VerifierCompletenessInvariant;
use jolt_eval::invariant::determinism::DeterminismInvariant;
use jolt_eval::invariant::serialization_roundtrip::SerializationRoundtripInvariant;
use jolt_eval::invariant::soundness::SoundnessInvariant;
use jolt_eval::invariant::synthesis::redteam::{create_worktree, remove_worktree};
use jolt_eval::invariant::synthesis::SynthesisRegistry;
use jolt_eval::invariant::zk_consistency::ZkConsistencyInvariant;
use jolt_eval::objective::guest_cycles::GuestCycleCountObjective;
use jolt_eval::objective::inline_lengths::InlineLengthsObjective;
use jolt_eval::objective::peak_rss::PeakRssObjective;
use jolt_eval::objective::proof_size::ProofSizeObjective;
use jolt_eval::objective::prover_time::ProverTimeObjective;
use jolt_eval::objective::verifier_time::VerifierTimeObjective;
use jolt_eval::objective::wrapping_cost::WrappingCostObjective;
use jolt_eval::objective::{measure_objectives, Objective, OptimizationAttempt};
use jolt_eval::TestCase;

#[derive(Parser)]
#[command(name = "optimize")]
#[command(about = "AI-driven optimization of Jolt objectives")]
struct Cli {
    /// Objectives to optimize (comma-separated). Default: all.
    /// Available: peak_rss, prover_time, proof_size, verifier_time,
    ///            guest_cycle_count, inline_lengths, wrapping_cost
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

    // Build objectives
    let all_objectives = build_objectives(&test_case, &prover_pp, &verifier_pp, &inputs);
    let objective_names: Vec<String> = if let Some(names) = &cli.objectives {
        names.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        all_objectives.iter().map(|o| o.name().to_string()).collect()
    };

    let objectives: Vec<Objective> = all_objectives
        .into_iter()
        .filter(|o| objective_names.contains(&o.name().to_string()))
        .collect();

    if objectives.is_empty() {
        eprintln!("No matching objectives found.");
        eprintln!("Available: peak_rss, prover_time, proof_size, verifier_time, guest_cycle_count, inline_lengths, wrapping_cost");
        std::process::exit(1);
    }

    // Build invariants for safety checking
    let default_inputs = vec![];
    let mut registry = SynthesisRegistry::new();
    register_invariants(&mut registry, &test_case, &default_inputs);

    // Measure baseline
    let baseline = measure_objectives(&objectives);
    println!("=== Baseline measurements ===");
    print_measurements(&objectives, &baseline);
    println!();

    let repo_dir = std::env::current_dir()?;
    let mut attempts: Vec<OptimizationAttempt> = Vec::new();
    let mut best = baseline.clone();

    for iteration in 0..cli.iterations {
        println!("=== Iteration {}/{} ===", iteration + 1, cli.iterations);

        // Invoke Claude in a worktree to make optimizations
        let diff = match invoke_optimize_agent(
            &repo_dir,
            &objectives,
            &best,
            &attempts,
            &cli.model,
            cli.max_turns,
            cli.hint.as_deref(),
        ) {
            Some(d) => d,
            None => {
                info!("Agent produced no changes, stopping.");
                break;
            }
        };

        // Re-measure after the agent's changes
        let new_measurements = measure_objectives(&objectives);
        println!("  Measurements after changes:");
        print_measurements(&objectives, &new_measurements);

        // Check invariants
        let invariants_passed = registry.invariants().iter().all(|inv| {
            let results = inv.run_checks(0);
            results.iter().all(|r| r.is_ok())
        });

        if !invariants_passed {
            println!("  Invariants FAILED -- reverting.");
            revert_changes(&repo_dir);
        }

        // Check if score improved (lower is better for all default objectives)
        let improved = if invariants_passed {
            objective_names.iter().any(|name| {
                let old = best.get(name);
                let new = new_measurements.get(name);
                match (old, new) {
                    (Some(&o), Some(&n)) => {
                        let obj = objectives.iter().find(|obj| obj.name() == name);
                        match obj.map(|o| o.direction()) {
                            Some(jolt_eval::Direction::Minimize) => n < o,
                            Some(jolt_eval::Direction::Maximize) => n > o,
                            None => false,
                        }
                    }
                    _ => false,
                }
            })
        } else {
            false
        };

        let attempt = OptimizationAttempt {
            description: format!("iteration {}", iteration + 1),
            diff: truncate(&diff, 5000).to_string(),
            measurements: new_measurements.clone(),
            invariants_passed,
        };
        attempts.push(attempt);

        if improved {
            println!("  Improvement found -- keeping changes.");
            best = new_measurements;
            // Commit the successful optimization
            commit_changes(&repo_dir, iteration + 1);
        } else if invariants_passed {
            println!("  No improvement -- reverting.");
            revert_changes(&repo_dir);
        }

        println!();
    }

    // Summary
    println!("=== Optimization summary ===");
    println!(
        "{}/{} iterations produced improvements.",
        attempts
            .iter()
            .filter(|a| a.invariants_passed
                && a.measurements.iter().any(|(name, &val)| {
                    let baseline_val = baseline.get(name);
                    baseline_val.is_some_and(|&b| val != b)
                }))
            .count(),
        attempts.len()
    );
    println!();
    println!("Final measurements:");
    print_measurements(&objectives, &best);

    Ok(())
}

/// Invoke Claude in an isolated worktree to attempt an optimization.
/// Returns the agent's output (approach description) or None.
fn invoke_optimize_agent(
    repo_dir: &std::path::Path,
    objectives: &[Objective],
    current_best: &HashMap<String, f64>,
    past_attempts: &[OptimizationAttempt],
    model: &str,
    max_turns: usize,
    hint: Option<&str>,
) -> Option<String> {
    // Create worktree
    let worktree_dir = match create_worktree(repo_dir, "optimize") {
        Ok(dir) => {
            info!("Created worktree at {}", dir.display());
            dir
        }
        Err(e) => {
            tracing::error!("Failed to create worktree: {e}");
            return None;
        }
    };

    let prompt = build_prompt(objectives, current_best, past_attempts, hint);

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

    // Capture any diff the agent produced in the worktree
    let diff = Command::new("git")
        .current_dir(&worktree_dir)
        .args(["diff", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).to_string();
            if s.trim().is_empty() {
                None
            } else {
                Some(s)
            }
        });

    // Apply the agent's changes to the real repo (if any)
    if let Some(diff_text) = &diff {
        info!("Agent produced a diff ({} bytes), applying to repo...", diff_text.len());
        let mut child = Command::new("git")
            .current_dir(repo_dir)
            .args(["apply", "--allow-empty"])
            .stdin(std::process::Stdio::piped())
            .spawn()
            .ok();
        if let Some(ref mut c) = child {
            use std::io::Write;
            if let Some(stdin) = c.stdin.as_mut() {
                let _ = stdin.write_all(diff_text.as_bytes());
            }
            let _ = c.wait();
        }
    }

    // Clean up worktree
    info!("Cleaning up worktree...");
    remove_worktree(repo_dir, &worktree_dir);
    let _ = std::fs::remove_dir_all(&worktree_dir);

    // Parse agent output
    match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if !output.status.success() {
                tracing::warn!("claude exited with status {}", output.status);
                if !stderr.is_empty() {
                    tracing::warn!("stderr: {}", truncate(&stderr, 500));
                }
            }

            let response = if stdout.trim().is_empty() {
                truncate(&stderr, 2000).to_string()
            } else {
                truncate(&stdout, 2000).to_string()
            };

            if response.trim().is_empty() && diff.is_none() {
                return None;
            }

            info!("Agent response ({} chars)", response.len());
            Some(diff.unwrap_or(response))
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

fn build_prompt(
    objectives: &[Objective],
    current_best: &HashMap<String, f64>,
    past_attempts: &[OptimizationAttempt],
    hint: Option<&str>,
) -> String {
    let mut prompt = String::new();

    prompt.push_str(
        "You are an expert performance engineer optimizing a zkVM (Jolt). \
         Your goal is to make code changes that improve the following objectives.\n\n",
    );

    prompt.push_str("## Objectives to optimize\n\n");
    for obj in objectives {
        let dir = match obj.direction() {
            jolt_eval::Direction::Minimize => "lower is better",
            jolt_eval::Direction::Maximize => "higher is better",
        };
        let current = current_best
            .get(obj.name())
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "unknown".to_string());
        prompt.push_str(&format!(
            "- **{}**: current = {}, direction = {}\n",
            obj.name(),
            current,
            dir,
        ));
    }
    prompt.push('\n');

    prompt.push_str(
        "## Instructions\n\n\
         1. Read the relevant source code (especially `jolt-core/src/`) to understand \
            hot paths and potential optimization opportunities.\n\
         2. Make targeted code changes that you believe will improve the objectives.\n\
         3. Focus on changes to `jolt-core/` -- do NOT modify `jolt-eval/`.\n\
         4. Prefer changes that are safe, correct, and unlikely to break invariants.\n\
         5. Run `cargo clippy -p jolt-core --features host --message-format=short -q` \
            to verify your changes compile.\n\
         6. Summarize what you changed and why you expect it to improve the objectives.\n\n",
    );

    if let Some(h) = hint {
        prompt.push_str("## Hint\n\n");
        prompt.push_str(h);
        prompt.push_str("\n\n");
    }

    if !past_attempts.is_empty() {
        prompt.push_str("## Previous attempts\n\n");
        for attempt in past_attempts {
            let status = if attempt.invariants_passed {
                "invariants passed"
            } else {
                "INVARIANTS FAILED"
            };
            prompt.push_str(&format!("- **{}** ({}): ", attempt.description, status));
            for (name, val) in &attempt.measurements {
                prompt.push_str(&format!("{name}={val:.4} "));
            }
            prompt.push('\n');
        }
        prompt.push('\n');
    }

    prompt.push_str(
        "## Output\n\n\
         Make your code changes directly. After you're done, summarize:\n\
         - What you changed\n\
         - Why you expect improvement\n\
         - Any risks or trade-offs\n",
    );

    prompt
}

fn revert_changes(repo_dir: &std::path::Path) {
    let _ = Command::new("git")
        .current_dir(repo_dir)
        .args(["checkout", "."])
        .status();
}

fn commit_changes(repo_dir: &std::path::Path, iteration: usize) {
    let _ = Command::new("git")
        .current_dir(repo_dir)
        .args(["add", "-A"])
        .status();
    let msg = format!("perf(auto-optimize): iteration {iteration}");
    let _ = Command::new("git")
        .current_dir(repo_dir)
        .args(["commit", "-m", &msg, "--allow-empty"])
        .status();
}

fn print_measurements(objectives: &[Objective], measurements: &HashMap<String, f64>) {
    for obj in objectives {
        let val = measurements
            .get(obj.name())
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "N/A".to_string());
        let dir = match obj.direction() {
            jolt_eval::Direction::Minimize => "min",
            jolt_eval::Direction::Maximize => "max",
        };
        println!("  {:<25} {:>15} {:>6}", obj.name(), val, dir);
    }
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

fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}
