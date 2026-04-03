use std::collections::HashMap;
use std::path::Path;

use crate::agent::{truncate, AgentHarness, DiffScope};

/// A function that combines raw objective measurements into a single
/// scalar value to minimize.
///
/// The optimizer always minimizes. To maximize something, negate it
/// in your implementation.
pub trait ObjectiveFunction: Send + Sync {
    /// Human-readable description of what this function optimizes,
    /// included in the agent prompt.
    fn description(&self) -> String;

    /// Combine raw measurements into a single scalar to minimize.
    fn evaluate(&self, measurements: &HashMap<String, f64>) -> f64;
}

/// A simple objective function that returns a single named measurement.
pub struct SingleObjective {
    pub name: String,
}

impl ObjectiveFunction for SingleObjective {
    fn description(&self) -> String {
        format!("Minimize {}", self.name)
    }

    fn evaluate(&self, measurements: &HashMap<String, f64>) -> f64 {
        measurements.get(&self.name).copied().unwrap_or(f64::INFINITY)
    }
}

/// Configuration for an optimization run.
pub struct OptimizeConfig {
    pub num_iterations: usize,
    pub hint: Option<String>,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            num_iterations: 5,
            hint: None,
        }
    }
}

/// Result of a complete optimization run.
pub struct OptimizeResult {
    pub attempts: Vec<OptimizationAttempt>,
    pub baseline_score: f64,
    pub best_score: f64,
    pub best_measurements: HashMap<String, f64>,
}

/// Record of a single optimization attempt.
pub struct OptimizationAttempt {
    pub description: String,
    pub diff: String,
    pub measurements: HashMap<String, f64>,
    pub score: f64,
    pub invariants_passed: bool,
}

/// Environment trait that decouples the optimization loop from side effects.
pub trait OptimizeEnv {
    /// Measure all raw objectives. Returns name -> value.
    fn measure(&mut self) -> HashMap<String, f64>;

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
    objective: &dyn ObjectiveFunction,
    config: &OptimizeConfig,
    repo_dir: &Path,
) -> OptimizeResult {
    let baseline = env.measure();
    let baseline_score = objective.evaluate(&baseline);
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

        let diff_scope = DiffScope::Exclude(vec!["jolt-eval/".into()]);
        let response = match agent.invoke(repo_dir, &prompt, &diff_scope) {
            Ok(r) => r,
            Err(e) => {
                tracing::info!("Agent error: {e}");
                break;
            }
        };

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
        let new_score = objective.evaluate(&new_measurements);
        let invariants_passed = env.check_invariants();

        if !invariants_passed {
            env.reject();
        }

        let improved = invariants_passed && new_score < best_score;

        let attempt = OptimizationAttempt {
            description: format!("iteration {}", iteration + 1),
            diff: truncate(&diff_text, 5000).to_string(),
            measurements: new_measurements.clone(),
            score: new_score,
            invariants_passed,
        };
        attempts.push(attempt);

        if improved {
            best_score = new_score;
            best_measurements = new_measurements;
            env.accept(iteration + 1);
        } else if invariants_passed {
            env.reject();
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
    objective: &dyn ObjectiveFunction,
    current_best_score: f64,
    current_best_measurements: &HashMap<String, f64>,
    past_attempts: &[OptimizationAttempt],
    hint: Option<&str>,
) -> String {
    let mut prompt = String::new();

    prompt.push_str(
        "You are an expert performance engineer optimizing a zkVM (Jolt). \
         Your goal is to make code changes that MINIMIZE the objective function.\n\n",
    );

    prompt.push_str("## Objective function\n\n");
    prompt.push_str(&objective.description());
    prompt.push_str(&format!(
        "\n\nCurrent best score: {current_best_score:.6}\n\n"
    ));

    prompt.push_str("## Current measurements\n\n");
    let mut names: Vec<_> = current_best_measurements.keys().collect();
    names.sort();
    for name in &names {
        let val = current_best_measurements[*name];
        prompt.push_str(&format!("- **{name}**: {val:.6}\n"));
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
            let mut keys: Vec<_> = attempt.measurements.keys().collect();
            keys.sort();
            for name in keys {
                let val = attempt.measurements[name];
                prompt.push_str(&format!("{name}={val:.6} "));
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
