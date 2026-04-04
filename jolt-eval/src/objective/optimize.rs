use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use crate::agent::{truncate, AgentHarness, DiffScope};

use super::objective_fn::ObjectiveFunction;
use super::OptimizationObjective;

/// Configuration for an optimization run.
pub struct OptimizeConfig {
    pub num_iterations: usize,
    pub hint: Option<String>,
    pub verbose: bool,
    pub diff_scope: DiffScope,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            num_iterations: 5,
            hint: None,
            verbose: false,
            diff_scope: DiffScope::Exclude(vec!["jolt-eval/".into()]),
        }
    }
}

/// Result of a complete optimization run.
pub struct OptimizeResult {
    pub attempts: Vec<OptimizationAttempt>,
    pub baseline_score: f64,
    pub best_score: f64,
    pub best_measurements: HashMap<OptimizationObjective, f64>,
}

/// Record of a single optimization attempt.
pub struct OptimizationAttempt {
    pub description: String,
    pub diff: String,
    pub measurements: HashMap<OptimizationObjective, f64>,
    pub score: f64,
    pub invariants_passed: bool,
}

/// Environment trait that decouples the optimization loop from side effects.
pub trait OptimizeEnv {
    /// Measure all raw objectives. Returns objective -> value.
    fn measure(&mut self) -> HashMap<OptimizationObjective, f64>;

    /// Check all invariants. Returns `true` if they all pass.
    fn check_invariants(&mut self) -> bool;

    /// Apply an agent-produced diff to the working tree.
    fn apply_diff(&mut self, diff: &str);

    /// Called when a change is accepted.
    fn accept(&mut self, iteration: usize);

    /// Called when a change is rejected.
    fn reject(&mut self);
}

/// Run an AI-driven optimization loop.
///
/// The agent tries to minimize `objective.evaluate(measurements)`.
/// Each iteration: invoke agent, apply diff, re-measure, accept/reject.
pub fn auto_optimize<A: AgentHarness, E: OptimizeEnv>(
    agent: &A,
    env: &mut E,
    objective: &ObjectiveFunction,
    config: &OptimizeConfig,
    repo_dir: &Path,
) -> OptimizeResult {
    // Create a branch for this optimization run. Silently ignored if
    // repo_dir is not a git repository (e.g. in tests).
    let branch = format!("jolt-eval/optimize/{}", objective.name);
    let _ = Command::new("git")
        .current_dir(repo_dir)
        .args(["checkout", "-b", &branch])
        .status();

    let baseline = env.measure();
    let baseline_score = (objective.evaluate)(&baseline);
    let mut best_score = baseline_score;
    let mut best_measurements = baseline.clone();
    let mut attempts = Vec::new();

    for iteration in 0..config.num_iterations {
        let prompt = build_optimize_prompt(
            objective,
            best_score,
            &best_measurements,
            &attempts,
            config.hint.as_deref(),
        );

        if config.verbose {
            eprintln!("── Iteration {} prompt ──", iteration + 1);
            eprintln!("{prompt}");
            eprintln!("────────────────────────");
        }

        let response = match agent.invoke(repo_dir, &prompt, &config.diff_scope) {
            Ok(r) => r,
            Err(e) => {
                tracing::info!("Agent error: {e}");
                break;
            }
        };

        if config.verbose {
            eprintln!("── Iteration {} response ──", iteration + 1);
            eprintln!("{}", response.text);
            if let Some(ref d) = response.diff {
                eprintln!("── diff ({} bytes) ──", d.len());
                eprintln!("{}", truncate(d, 2000));
            } else {
                eprintln!("(no diff)");
            }
            eprintln!("──────────────────────────");
        }

        let diff_text = match &response.diff {
            Some(d) => {
                env.apply_diff(d);
                d.clone()
            }
            None => {
                tracing::info!("Agent produced no code changes, stopping.");
                break;
            }
        };

        let new_measurements = env.measure();
        let new_score = (objective.evaluate)(&new_measurements);
        let invariants_passed = env.check_invariants();

        let improved = invariants_passed && new_score < best_score;

        let attempt = OptimizationAttempt {
            description: format!("iteration {}", iteration + 1),
            diff: truncate(&diff_text, 5000).to_string(),
            measurements: new_measurements.clone(),
            score: new_score,
            invariants_passed,
        };
        attempts.push(attempt);

        let iter = iteration + 1;
        if improved {
            eprintln!("  ✓ iteration {iter} ACCEPTED — score {best_score:.10} → {new_score:.10}",);
            best_score = new_score;
            best_measurements = new_measurements;
            env.accept(iter);
            let msg = format!(
                "perf(auto-optimize): {} iteration {iter} (score {new_score:.10})",
                objective.name,
            );
            let _ = Command::new("git")
                .current_dir(repo_dir)
                .args(["add", "-A"])
                .status();
            let _ = Command::new("git")
                .current_dir(repo_dir)
                .args(["commit", "-m", &msg])
                .status();
        } else if !invariants_passed {
            eprintln!("  ✗ iteration {iter} REJECTED (invariants failed) — score {new_score:.10}",);
            env.reject();
            let _ = Command::new("git")
                .current_dir(repo_dir)
                .args(["checkout", "."])
                .status();
        } else {
            eprintln!(
                "  ✗ iteration {iter} REJECTED (no improvement) — score {new_score:.10} ≥ best {best_score:.10}",
            );
            env.reject();
            let _ = Command::new("git")
                .current_dir(repo_dir)
                .args(["checkout", "."])
                .status();
        }
    }

    OptimizeResult {
        attempts,
        baseline_score,
        best_score,
        best_measurements,
    }
}

fn build_optimize_prompt(
    objective: &ObjectiveFunction,
    current_best_score: f64,
    current_best_measurements: &HashMap<OptimizationObjective, f64>,
    past_attempts: &[OptimizationAttempt],
    hint: Option<&str>,
) -> String {
    let mut prompt = String::new();

    prompt.push_str(
        "You are an expert performance engineer optimizing a zkVM (Jolt). \
         Your goal is to make code changes that MINIMIZE the objective function.\n\n",
    );

    prompt.push_str("## Objective\n\n");
    prompt.push_str(&format!("Minimize: **{}**\n", objective.name));

    let inputs = objective.inputs;
    prompt.push_str("Inputs: ");
    for (i, input) in inputs.iter().enumerate() {
        if i > 0 {
            prompt.push_str(", ");
        }
        prompt.push_str(input.name());
    }
    prompt.push_str(&format!(
        "\nCurrent best score: {current_best_score:.6}\n\n"
    ));
    prompt.push_str(
        "The objective function is defined in `jolt-eval/src/objective/objective_fn/`. \
         Read the implementation to understand exactly what you are optimizing.\n\n",
    );

    prompt.push_str("## Current measurements\n\n");
    let mut entries: Vec<_> = current_best_measurements.iter().collect();
    entries.sort_by_key(|(k, _)| k.name());
    for (key, val) in &entries {
        prompt.push_str(&format!("- **{}**: {val:.6}\n", key.name()));
    }
    prompt.push('\n');

    prompt.push_str(
        "## Instructions\n\n\
         1. Read the relevant source code (especially `jolt-core/src/`) to understand \
            hot paths and potential optimization opportunities.\n\
         2. Make targeted code changes that you believe will reduce the objective function.\n\
         3. Focus on changes to `jolt-core/` -- do NOT modify `jolt-eval/`.\n\
         4. Prefer changes that are safe, correct, and unlikely to break invariants.\n\
         5. Run `cargo clippy -p jolt-core --features host --message-format=short -q` \
            to verify your changes compile.\n\
         6. Summarize what you changed and why you expect improvement.\n\n",
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
            prompt.push_str(&format!(
                "- **{}** ({}, score={:.6}): ",
                attempt.description, status, attempt.score
            ));
            let mut keys: Vec<_> = attempt.measurements.iter().collect();
            keys.sort_by_key(|(k, _)| k.name());
            for (key, val) in keys {
                prompt.push_str(&format!("{}={val:.6} ", key.name()));
            }
            prompt.push('\n');
        }
        prompt.push('\n');
    }

    prompt.push_str(
        "## Output\n\n\
         Make your code changes directly. After you're done, summarize:\n\
         - What you changed\n\
         - Why you expect the objective function to decrease\n\
         - Any risks or trade-offs\n",
    );

    prompt
}
