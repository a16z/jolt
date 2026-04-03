use std::collections::HashMap;
use std::path::Path;

use crate::agent::{truncate, AgentHarness};
use crate::objective::{Direction, OptimizationAttempt};

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
    pub baseline: HashMap<String, f64>,
    pub best: HashMap<String, f64>,
}

/// Environment trait that decouples the optimization loop from side effects.
///
/// The real binary implements this with actual measurement, invariant
/// checking, and git operations.  Tests supply a mock implementation.
pub trait OptimizeEnv {
    /// Measure all objectives.  Returns name -> value.
    fn measure(&mut self) -> HashMap<String, f64>;

    /// Check all invariants.  Returns `true` if they all pass.
    fn check_invariants(&mut self) -> bool;

    /// Return the direction for each objective (name -> direction).
    fn directions(&self) -> HashMap<String, Direction>;

    /// Apply an agent-produced diff to the working tree.
    fn apply_diff(&mut self, diff: &str);

    /// Called when a change is accepted (measurements improved, invariants passed).
    fn accept(&mut self, iteration: usize);

    /// Called when a change is rejected (no improvement, or invariants failed).
    fn reject(&mut self);
}

/// Run an AI-driven optimization loop.
///
/// Each iteration:
/// 1. Builds a prompt from objective directions, current best measurements,
///    past attempts, and an optional hint.
/// 2. Invokes the agent via [`AgentHarness`].
/// 3. If the agent produced a diff, applies it via [`OptimizeEnv::apply_diff`].
/// 4. Re-measures objectives and checks invariants.
/// 5. Accepts or rejects the change.
pub fn auto_optimize<A: AgentHarness, E: OptimizeEnv>(
    agent: &A,
    env: &mut E,
    config: &OptimizeConfig,
    repo_dir: &Path,
) -> OptimizeResult {
    let directions = env.directions();
    let baseline = env.measure();
    let mut best = baseline.clone();
    let mut attempts = Vec::new();

    for iteration in 0..config.num_iterations {
        let prompt = build_optimize_prompt(&directions, &best, &attempts, config.hint.as_deref());

        let response = match agent.invoke(repo_dir, &prompt) {
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
        let invariants_passed = env.check_invariants();

        if !invariants_passed {
            env.reject();
        }

        let improved = if invariants_passed {
            directions.iter().any(|(name, dir)| {
                let old = best.get(name);
                let new = new_measurements.get(name);
                match (old, new) {
                    (Some(&o), Some(&n)) => match dir {
                        Direction::Minimize => n < o,
                        Direction::Maximize => n > o,
                    },
                    _ => false,
                }
            })
        } else {
            false
        };

        let attempt = OptimizationAttempt {
            description: format!("iteration {}", iteration + 1),
            diff: truncate(&diff_text, 5000).to_string(),
            measurements: new_measurements.clone(),
            invariants_passed,
        };
        attempts.push(attempt);

        if improved {
            best = new_measurements;
            env.accept(iteration + 1);
        } else if invariants_passed {
            env.reject();
        }
    }

    OptimizeResult {
        attempts,
        baseline,
        best,
    }
}

fn build_optimize_prompt(
    directions: &HashMap<String, Direction>,
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
    let mut names: Vec<_> = directions.keys().collect();
    names.sort();
    for name in &names {
        let dir = match directions[*name] {
            Direction::Minimize => "lower is better",
            Direction::Maximize => "higher is better",
        };
        let current = current_best
            .get(*name)
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "unknown".to_string());
        prompt.push_str(&format!(
            "- **{name}**: current = {current}, direction = {dir}\n",
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
            let mut keys: Vec<_> = attempt.measurements.keys().collect();
            keys.sort();
            for name in keys {
                let val = attempt.measurements[name];
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
