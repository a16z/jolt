use std::collections::HashMap;

use super::{measure_objectives, Objective};
use crate::invariant::DynInvariant;

/// Record of an optimization attempt.
pub struct OptimizationAttempt {
    pub description: String,
    pub diff: String,
    pub measurements: HashMap<String, f64>,
    pub invariants_passed: bool,
}

/// Run an AI-driven optimization loop.
///
/// The objective function maps measured values to a single scalar score.
/// Each iteration:
/// 1. Measures all objectives
/// 2. Checks that invariants still hold
/// 3. If the score improved and invariants pass, commits the change
/// 4. Otherwise reverts
///
/// This function provides the measurement and comparison infrastructure.
/// The actual AI interaction (telling Claude to optimize) is handled by
/// the caller.
pub fn auto_optimize<F>(
    objectives: &[Objective],
    invariants: &[Box<dyn DynInvariant>],
    objective_function: F,
    num_iterations: usize,
    mut on_iteration: impl FnMut(usize, f64, &HashMap<String, f64>) -> Option<String>,
) -> Vec<OptimizationAttempt>
where
    F: Fn(&HashMap<String, f64>) -> f64,
{
    let baseline_measurements = measure_objectives(objectives);
    let mut baseline_score = objective_function(&baseline_measurements);
    let mut attempts = Vec::new();

    for i in 0..num_iterations {
        // Let the caller drive the optimization (e.g. invoke Claude)
        let diff = match on_iteration(i, baseline_score, &baseline_measurements) {
            Some(d) => d,
            None => break,
        };

        let new_measurements = measure_objectives(objectives);
        let new_score = objective_function(&new_measurements);

        // Check that all invariants still hold
        let invariants_passed = invariants
            .iter()
            .all(|inv| inv.run_checks(0).iter().all(|r| r.is_ok()));

        let attempt = OptimizationAttempt {
            description: format!("iteration {i}"),
            diff,
            measurements: new_measurements,
            invariants_passed,
        };

        if invariants_passed && new_score > baseline_score {
            baseline_score = new_score;
        }

        attempts.push(attempt);
    }

    attempts
}
