use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use crate::agent::{truncate, AgentHarness};

use super::objective_fn::ObjectiveFunction;
use super::OptimizationObjective;

/// Configuration for an optimization run.
pub struct OptimizeConfig {
    pub num_iterations: usize,
    pub hint: Option<String>,
    pub verbose: bool,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            num_iterations: 5,
            hint: None,
            verbose: false,
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
    pub iteration: usize,
    pub score: f64,
    pub invariants_passed: bool,
    /// Relative path to the persisted attempt directory, if available.
    pub path: Option<String>,
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

#[allow(clippy::too_many_arguments)]
fn write_attempt_files(
    dir: &Path,
    diff: &str,
    response_text: &str,
    measurements: &HashMap<OptimizationObjective, f64>,
    score: f64,
    accepted: bool,
    invariants_passed: bool,
) -> Option<()> {
    std::fs::write(dir.join("diff.patch"), diff).ok()?;
    std::fs::write(dir.join("response.md"), response_text).ok()?;

    let meas: HashMap<String, f64> = measurements
        .iter()
        .map(|(k, &v)| (k.name().to_string(), v))
        .collect();
    let meas_json = serde_json::to_string_pretty(&meas).ok()?;
    std::fs::write(dir.join("measurements.json"), meas_json).ok()?;

    let status = serde_json::json!({
        "accepted": accepted,
        "score": score,
        "invariants_passed": invariants_passed,
    });
    std::fs::write(
        dir.join("status.json"),
        serde_json::to_string_pretty(&status).ok()?,
    )
    .ok()?;

    Some(())
}

#[allow(clippy::too_many_arguments)]
fn persist_attempt(
    repo_dir: &Path,
    objective_name: &str,
    iteration: usize,
    diff: &str,
    response_text: &str,
    measurements: &HashMap<OptimizationObjective, f64>,
    score: f64,
    accepted: bool,
    invariants_passed: bool,
) -> Option<String> {
    let dir = repo_dir
        .join("jolt-eval/optimize-history")
        .join(objective_name)
        .join(format!("attempt-{iteration}"));
    std::fs::create_dir_all(&dir).ok()?;
    write_attempt_files(
        &dir,
        diff,
        response_text,
        measurements,
        score,
        accepted,
        invariants_passed,
    )?;
    Some(
        dir.strip_prefix(repo_dir)
            .ok()?
            .to_string_lossy()
            .to_string(),
    )
}

fn persist_baseline(
    repo_dir: &Path,
    objective_name: &str,
    measurements: &HashMap<OptimizationObjective, f64>,
    score: f64,
) {
    let dir = repo_dir
        .join("jolt-eval/optimize-history")
        .join(objective_name)
        .join("baseline");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let _ = write_attempt_files(&dir, "", "", measurements, score, true, true);
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
    let baseline_score = (objective.evaluate)(&baseline, &baseline);
    persist_baseline(repo_dir, objective.name, &baseline, baseline_score);
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

        let response = match agent.invoke(repo_dir, &prompt, &objective.diff_scope()) {
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

        let response_text = response.text.clone();
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
        let new_score = (objective.evaluate)(&new_measurements, &baseline);
        let invariants_passed = env.check_invariants();

        let improved = invariants_passed && new_score < best_score;
        let iter = iteration + 1;

        let attempt_path = persist_attempt(
            repo_dir,
            objective.name,
            iter,
            &diff_text,
            &response_text,
            &new_measurements,
            new_score,
            improved,
            invariants_passed,
        );

        let attempt = OptimizationAttempt {
            iteration: iter,
            score: new_score,
            invariants_passed,
            path: attempt_path,
        };
        attempts.push(attempt);

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
            let _ = Command::new("git")
                .current_dir(repo_dir)
                .args(["clean", "-fd"])
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
            let _ = Command::new("git")
                .current_dir(repo_dir)
                .args(["clean", "-fd"])
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

    prompt.push_str("## What you are optimizing\n\n");
    for input in inputs {
        let units_str = input
            .units()
            .map(|u| format!(" (units: {u})"))
            .unwrap_or_default();
        prompt.push_str(&format!(
            "- **{}**{units_str}: {}\n",
            input.name(),
            input.description()
        ));
    }
    prompt.push('\n');

    prompt.push_str("## Current measurements\n\n");
    let mut entries: Vec<_> = current_best_measurements.iter().collect();
    entries.sort_by_key(|(k, _)| k.name());
    for (key, val) in &entries {
        let units_str = key.units().map(|u| format!(" {u}")).unwrap_or_default();
        prompt.push_str(&format!("- **{}**: {val:.6}{units_str}\n", key.name()));
    }
    prompt.push('\n');

    let paths_list = match objective.diff_scope() {
        crate::agent::DiffScope::Include(paths) => paths.join(", "),
        _ => "jolt-core/".to_string(),
    };
    prompt.push_str("## Instructions\n\n");
    prompt.push_str(&format!(
        "1. Read the relevant source code in: {paths_list}. Also read \
         `jolt-eval/src/objective/objective_fn/` to understand the exact scoring formula.\n"
    ));
    prompt.push_str(
        "2. Make targeted code changes that you believe will reduce the objective function.\n\
         3. Focus your changes on the paths listed above -- do NOT modify `jolt-eval/` unless \
            it is explicitly listed.\n\
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
            let status_label = match (
                attempt.invariants_passed,
                attempt.score < current_best_score,
            ) {
                (true, true) => "ACCEPTED",
                (false, _) => "REJECTED (invariants failed)",
                _ => "REJECTED (no improvement)",
            };
            if let Some(ref path) = attempt.path {
                prompt.push_str(&format!(
                    "- **Iteration {}** — {status_label}, score={:.6}. Details: {path}/\n",
                    attempt.iteration, attempt.score,
                ));
            } else {
                prompt.push_str(&format!(
                    "- **Iteration {}** — {status_label}, score={:.6}\n",
                    attempt.iteration, attempt.score,
                ));
            }
        }
        prompt.push('\n');
        prompt.push_str(
            "Read the attempt directories for full diffs, measurements, and agent responses.\n\
             If previous attempts failed or showed no improvement, try a fundamentally \
             different approach.\n\n",
        );
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
