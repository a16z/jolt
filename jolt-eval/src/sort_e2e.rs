//! End-to-end test harnesses for the optimization and red-team loops,
//! using simple sorting functions as the target domain.

use std::collections::HashMap;

use crate::agent::ClaudeCodeAgent;
use crate::invariant::synthesis::redteam::{auto_redteam, RedTeamConfig, RedTeamResult};
use crate::invariant::{CheckError, Invariant, InvariantViolation};
use crate::objective::objective_fn::ObjectiveFunction;
use crate::objective::optimize::{auto_optimize, OptimizeConfig, OptimizeEnv};
use crate::objective::{OptimizationObjective, NAIVE_SORT_TIME};
use crate::sort_targets::{candidate_sort, naive_sort};

/// Invariant: a sort function must preserve all elements (multiset
/// equality) and produce sorted output.
#[jolt_eval_macros::invariant(RedTeam)]
pub struct CandidateSortInvariant;

impl Invariant for CandidateSortInvariant {
    type Setup = ();
    type Input = Vec<i32>;

    fn name(&self) -> &str {
        "candidate_sort"
    }

    fn description(&self) -> String {
        "The sort function `candidate_sort` in \
         jolt-eval/src/sort_targets.rs must return a \
         permutation of its input in non-decreasing order. \
         Any dropped, duplicated, or misplaced elements are a violation."
            .to_string()
    }

    fn setup(&self) {}

    fn check(&self, _: &(), input: Vec<i32>) -> Result<(), CheckError> {
        if input.len() > 1_000 {
            return Err(CheckError::InvalidInput(
                "input too large (max 1000)".into(),
            ));
        }

        let mut output = input.clone();
        candidate_sort(&mut output);

        let mut expected = input;
        expected.sort();

        if output != expected {
            return Err(CheckError::Violation(InvariantViolation::new(format!(
                "sort incorrect: expected {expected:?}, got {output:?}"
            ))));
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<Vec<i32>> {
        vec![
            vec![],
            vec![1],
            vec![3, 1, 2],
            vec![5, 4, 3, 2, 1],
            vec![1, 1, 1],
        ]
    }
}

/// Invariant for the naive (correct) sort — used in the optimization
/// loop to verify that the "optimized" sort is still correct.
pub struct NaiveSortInvariant;

impl Invariant for NaiveSortInvariant {
    type Setup = ();
    type Input = Vec<i32>;

    fn name(&self) -> &str {
        "naive_sort_correctness"
    }

    fn description(&self) -> String {
        "The naive sort must return a permutation of its input in \
         non-decreasing order."
            .to_string()
    }

    fn setup(&self) {}

    fn check(&self, _: &(), input: Vec<i32>) -> Result<(), CheckError> {
        let mut output = input.clone();
        naive_sort(&mut output);

        let mut expected = input;
        expected.sort();

        if output != expected {
            return Err(CheckError::Violation(InvariantViolation::new(format!(
                "naive sort incorrect: expected {expected:?}, got {output:?}"
            ))));
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<Vec<i32>> {
        vec![vec![], vec![1], vec![3, 1, 2], vec![5, 4, 3, 2, 1]]
    }
}

/// An [`OptimizeEnv`] that measures the sort objective by shelling out
/// to `cargo run --bin optimize -- --measure`, which recompiles and
/// runs the (potentially modified) `sort_targets::naive_sort`.
pub struct SortOptimizeEnv {
    repo_dir: std::path::PathBuf,
    last_invariant_ok: bool,
}

impl SortOptimizeEnv {
    pub fn new(repo_dir: &std::path::Path) -> Self {
        Self {
            repo_dir: repo_dir.to_path_buf(),
            last_invariant_ok: true,
        }
    }
}

impl OptimizeEnv for SortOptimizeEnv {
    fn measure(&mut self) -> HashMap<OptimizationObjective, f64> {
        let output = std::process::Command::new("cargo")
            .current_dir(&self.repo_dir)
            .args([
                "run",
                "--release",
                "-p",
                "jolt-eval",
                "--bin",
                "optimize",
                "--",
                "--objective",
                "minimize_naive_sort_time",
                "--measure",
            ])
            .output();

        let mut m = HashMap::new();
        match output {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                if let Ok(json) =
                    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&stdout)
                {
                    for (key, val) in &json {
                        if let Some(v) = val.as_f64() {
                            // Match the key back to an OptimizationObjective.
                            if key == "naive_sort_time" {
                                m.insert(NAIVE_SORT_TIME, v);
                            }
                        }
                    }
                }
                // Check invariant: sort must produce sorted output.
                // The measurement binary already ran the sort successfully,
                // so we just verify the output is sorted by running it locally
                // (cheap, since it's already compiled).
                let mut buf: Vec<i32> = (0..5000i32).rev().collect();
                naive_sort(&mut buf);
                self.last_invariant_ok = buf.windows(2).all(|w| w[0] <= w[1]);
            }
            _ => {
                self.last_invariant_ok = false;
            }
        }
        m
    }

    fn check_invariants(&mut self) -> bool {
        self.last_invariant_ok
    }

    fn apply_diff(&mut self, diff: &str) {
        let _ = crate::agent::apply_diff(&self.repo_dir, diff);
    }

    fn accept(&mut self, _iteration: usize) {}

    fn reject(&mut self) {}
}

const SORT_TARGETS_PATH: &str = "jolt-eval/src/sort_targets.rs";

/// Run the red-team e2e test against `CandidateSortInvariant`.
pub fn run_redteam_test(
    model: &str,
    max_turns: usize,
    iterations: usize,
    hint: Option<String>,
    verbose: bool,
) {
    let invariant = CandidateSortInvariant;
    let agent = ClaudeCodeAgent::new(model, max_turns);
    let repo_dir = std::env::current_dir().expect("current dir");
    let config = RedTeamConfig {
        num_iterations: iterations,
        hint,
        verbose,
    };

    println!("=== Red-team e2e: candidate_sort ===");
    println!("model={model}, max_turns={max_turns}, iterations={iterations}");
    println!();

    let result = auto_redteam(&invariant, &config, &agent, &repo_dir);

    match &result {
        RedTeamResult::Violation {
            approach,
            input_json,
            error,
        } => {
            println!("VIOLATION FOUND");
            println!("  Approach:  {approach}");
            println!("  Input:     {input_json}");
            println!("  Error:     {error}");
        }
        RedTeamResult::NoViolation { attempts } => {
            println!("No violation found after {} attempts.", attempts.len());
            for a in attempts {
                println!(
                    "  {}: {} -- {}",
                    a.description, a.approach, a.failure_reason
                );
            }
        }
    }
}

/// Run the optimization e2e test against the naive bubble sort.
pub fn run_optimize_test(
    model: &str,
    max_turns: usize,
    iterations: usize,
    hint: Option<String>,
    verbose: bool,
) {
    let agent = ClaudeCodeAgent::new(model, max_turns);
    let repo_dir = std::env::current_dir().expect("current dir");

    let mut env = SortOptimizeEnv::new(&repo_dir);

    let baseline = env.measure();
    let baseline_time = baseline[&NAIVE_SORT_TIME];

    let obj = ObjectiveFunction {
        name: "minimize_naive_sort_time",
        inputs: &[NAIVE_SORT_TIME],
        evaluate: |m, _| m.get(&NAIVE_SORT_TIME).copied().unwrap_or(f64::INFINITY),
    };
    let hint = hint.unwrap_or_else(|| {
        format!(
            "The target is the `naive_sort` function in {SORT_TARGETS_PATH}. \
             Replace it with a faster sorting algorithm. \
             You MAY modify that file for this task."
        )
    });
    let config = OptimizeConfig {
        num_iterations: iterations,
        hint: Some(hint),
        verbose,
    };

    println!("=== Optimize e2e: naive bubble sort ===");
    println!("model={model}, max_turns={max_turns}, iterations={iterations}");
    println!("Baseline sort time: {baseline_time:.6}s");
    println!();

    let result = auto_optimize(&agent, &mut env, &obj, &config, &repo_dir);

    println!("Best score: {:.6}s", result.best_score);
    println!(
        "Improvement: {:.1}%",
        (1.0 - result.best_score / baseline_time) * 100.0
    );
    for (i, a) in result.attempts.iter().enumerate() {
        println!(
            "  attempt {}: score={:.6}, invariants={}",
            i + 1,
            a.score,
            a.invariants_passed
        );
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::agent::{AgentResponse, MockAgent};
    use crate::invariant::synthesis::redteam::{auto_redteam, RedTeamConfig, RedTeamResult};
    use crate::objective::optimize::OptimizeEnv;

    use super::*;

    #[test]
    fn redteam_e2e_finds_sort_violation() {
        let invariant = CandidateSortInvariant;

        // 17 elements (exceeds the small-array threshold), with the
        // minimum value at the end — triggers the bug.
        let bad_input: Vec<i32> = (1..=17).rev().collect();
        let response = serde_json::json!({
            "analysis": "Trying a reversed sequence of 17 elements.",
            "counterexample": bad_input,
        });
        let agent = MockAgent::always_ok(&response.to_string());
        let config = RedTeamConfig {
            num_iterations: 5,
            ..Default::default()
        };

        let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

        match result {
            RedTeamResult::Violation { error, .. } => {
                assert!(
                    error.contains("sort incorrect"),
                    "unexpected error: {error}"
                );
            }
            RedTeamResult::NoViolation { .. } => {
                panic!("Expected violation for large reversed input");
            }
        }

        assert_eq!(agent.recorded_prompts().len(), 1);
    }

    #[test]
    fn redteam_e2e_no_violation_for_small_input() {
        let invariant = CandidateSortInvariant;

        let response = serde_json::json!({
            "analysis": "Trying a small permutation.",
            "counterexample": [5, 3, 1, 4, 2],
        });
        let agent = MockAgent::always_ok(&response.to_string());
        let config = RedTeamConfig {
            num_iterations: 3,
            ..Default::default()
        };

        let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

        match result {
            RedTeamResult::NoViolation { attempts } => {
                assert_eq!(attempts.len(), 3);
            }
            RedTeamResult::Violation { .. } => {
                panic!("Small inputs should not trigger a violation");
            }
        }
    }

    #[test]
    #[ignore] // Requires Claude API access
    fn redteam_e2e_real_agent() {
        run_redteam_test("claude-sonnet-4-20250514", 10, 5, None, false);
    }

    #[test]
    fn optimize_e2e_sort_accepts_improvement() {
        use crate::objective::objective_fn::MINIMIZE_NAIVE_SORT_TIME;

        let agent = MockAgent::from_responses(vec![Ok(AgentResponse {
            text: "Replaced bubble sort with merge sort".into(),
            diff: Some("--- a/sort.rs\n+++ b/sort.rs\n-bubble\n+merge".into()),
        })]);

        // Predetermined measurements: baseline 0.01s, after optimization 0.0001s.
        let mut mock = MockEnv {
            measurements: vec![
                HashMap::from([(NAIVE_SORT_TIME, 0.01)]),
                HashMap::from([(NAIVE_SORT_TIME, 0.0001)]),
            ],
            index: 0,
            invariant_ok: true,
        };

        let config = OptimizeConfig {
            num_iterations: 1,
            ..Default::default()
        };

        let result = auto_optimize(
            &agent,
            &mut mock,
            &MINIMIZE_NAIVE_SORT_TIME,
            &config,
            Path::new("/tmp"),
        );

        assert_eq!(result.attempts.len(), 1);
        assert!(result.attempts[0].invariants_passed);
        assert!(result.best_score < 0.01);
    }

    #[test]
    fn optimize_e2e_sort_rejects_broken() {
        use crate::objective::objective_fn::MINIMIZE_NAIVE_SORT_TIME;

        let agent = MockAgent::from_responses(vec![Ok(AgentResponse {
            text: "Removed sorting entirely".into(),
            diff: Some("--- a/sort.rs\n+++ b/sort.rs\n-sort\n+noop".into()),
        })]);

        let mut mock = MockEnv {
            measurements: vec![
                HashMap::from([(NAIVE_SORT_TIME, 0.01)]),
                HashMap::from([(NAIVE_SORT_TIME, 0.0001)]),
            ],
            index: 0,
            invariant_ok: false,
        };

        let config = OptimizeConfig {
            num_iterations: 1,
            ..Default::default()
        };

        let result = auto_optimize(
            &agent,
            &mut mock,
            &MINIMIZE_NAIVE_SORT_TIME,
            &config,
            Path::new("/tmp"),
        );

        assert!(!result.attempts[0].invariants_passed);
    }

    /// Simple mock env for unit tests (no subprocess, no recompilation).
    struct MockEnv {
        measurements: Vec<HashMap<OptimizationObjective, f64>>,
        index: usize,
        invariant_ok: bool,
    }

    impl OptimizeEnv for MockEnv {
        fn measure(&mut self) -> HashMap<OptimizationObjective, f64> {
            let idx = self.index.min(self.measurements.len() - 1);
            self.index += 1;
            self.measurements[idx].clone()
        }
        fn check_invariants(&mut self) -> bool {
            self.invariant_ok
        }
        fn apply_diff(&mut self, _: &str) {}
        fn accept(&mut self, _: usize) {}
        fn reject(&mut self) {}
    }

    #[test]
    #[ignore] // Requires Claude API access
    fn optimize_e2e_real_agent() {
        run_optimize_test("claude-sonnet-4-20250514", 10, 2, None, false);
    }
}
