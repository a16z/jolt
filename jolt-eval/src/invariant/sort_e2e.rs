//! End-to-end test harnesses for the optimization and red-team loops,
//! using simple sorting functions as the target domain.

use std::collections::HashMap;

use super::{CheckError, Invariant, InvariantViolation};
use crate::agent::ClaudeCodeAgent;
use crate::invariant::synthesis::redteam::{auto_redteam, RedTeamConfig, RedTeamResult};
use crate::objective::objective_fn::ObjectiveFunction;
use crate::objective::optimize::{auto_optimize, OptimizeConfig, OptimizeEnv};
use crate::objective::{OptimizationObjective, NAIVE_SORT_TIME};

/// Naive bubble sort — the optimization target.
/// Intentionally O(n²) so a "smarter" sort is measurably faster.
pub fn naive_sort(data: &mut [i32]) {
    let n = data.len();
    for i in 0..n {
        for j in 0..n.saturating_sub(1 + i) {
            if data[j] > data[j + 1] {
                data.swap(j, j + 1);
            }
        }
    }
}

/// A sorting routine used as a red-team target.
pub fn candidate_sort(data: &mut [i32]) {
    if data.len() <= 16 {
        // Small-array path: insertion sort.
        for i in 1..data.len() {
            let key = data[i];
            let mut j = i;
            while j > 0 && data[j - 1] > key {
                data[j] = data[j - 1];
                j -= 1;
            }
            data[j] = key;
        }
    } else {
        // Large-array path: delegate to an optimized routine.
        let last = data.len() - 1;
        data[..last].sort();
    }
}

// ── Red-team invariant ──────────────────────────────────────────────

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
         jolt-eval/src/invariant/sort_e2e.rs must return a \
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

// ── SortOptimizeEnv ─────────────────────────────────────────────────

/// An [`OptimizeEnv`] that measures wall-clock time of a sort function.
/// `apply_diff` simulates optimization by swapping to `slice::sort`.
pub struct SortOptimizeEnv {
    sort_fn: fn(&mut [i32]),
    data: Vec<i32>,
    invariant_ok: bool,
}

impl SortOptimizeEnv {
    pub fn new(data_size: usize) -> Self {
        let data: Vec<i32> = (0..data_size as i32).rev().collect();
        Self {
            sort_fn: naive_sort,
            data,
            invariant_ok: true,
        }
    }
}

impl OptimizeEnv for SortOptimizeEnv {
    fn measure(&mut self) -> HashMap<OptimizationObjective, f64> {
        let mut buf = self.data.clone();
        let start = std::time::Instant::now();
        (self.sort_fn)(&mut buf);
        let elapsed = start.elapsed().as_secs_f64();

        self.invariant_ok = buf.windows(2).all(|w| w[0] <= w[1]);

        let mut m = HashMap::new();
        m.insert(NAIVE_SORT_TIME, elapsed);
        m
    }

    fn check_invariants(&mut self) -> bool {
        self.invariant_ok
    }

    fn apply_diff(&mut self, _diff: &str) {
        self.sort_fn = |d: &mut [i32]| d.sort();
    }

    fn accept(&mut self, _iteration: usize) {}

    fn reject(&mut self) {
        self.sort_fn = naive_sort;
    }
}

// ── CLI-accessible e2e runners ──────────────────────────────────────

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

    let mut env = SortOptimizeEnv::new(5000);

    let baseline = env.measure();
    let baseline_time = baseline[&NAIVE_SORT_TIME];
    env.sort_fn = naive_sort;

    let obj = ObjectiveFunction {
        name: "naive_sort_time",
        inputs: &[NAIVE_SORT_TIME],
        evaluate: |m| m.get(&NAIVE_SORT_TIME).copied().unwrap_or(f64::INFINITY),
    };
    let hint = hint.unwrap_or_else(|| {
        "The target is the `naive_sort` function in \
         jolt-eval/src/invariant/sort_e2e.rs. Replace it with a faster \
         sorting algorithm. You MAY modify jolt-eval/ for this task."
            .into()
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

    // ── Red-team e2e (MockAgent) ────────────────────────────────────

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

    // ── Red-team e2e (real agent) ───────────────────────────────────

    #[test]
    #[ignore] // Requires Claude API access
    fn redteam_e2e_real_agent() {
        run_redteam_test("claude-sonnet-4-20250514", 10, 5, None, false);
    }

    // ── Optimize e2e (MockAgent) ────────────────────────────────────

    #[test]
    fn optimize_e2e_sort_improves() {
        let agent = MockAgent::from_responses(vec![Ok(AgentResponse {
            text: "Replaced bubble sort with merge sort".into(),
            diff: Some("--- a/sort.rs\n+++ b/sort.rs\n-bubble\n+merge".into()),
        })]);

        let mut env = SortOptimizeEnv::new(5000);

        let baseline = env.measure();
        let baseline_time = baseline[&NAIVE_SORT_TIME];
        assert!(baseline_time > 0.0);

        env.sort_fn = naive_sort;

        let obj = ObjectiveFunction {
            name: "naive_sort_time",
            inputs: &[NAIVE_SORT_TIME],
            evaluate: |m| m.get(&NAIVE_SORT_TIME).copied().unwrap_or(f64::INFINITY),
        };
        let config = OptimizeConfig {
            num_iterations: 1,
            hint: None,
            verbose: false,
        };

        let result = auto_optimize(&agent, &mut env, &obj, &config, Path::new("/tmp"));

        assert!(
            result.best_score < baseline_time,
            "expected improvement: baseline={baseline_time:.6}, best={:.6}",
            result.best_score
        );
        assert_eq!(result.attempts.len(), 1);
        assert!(result.attempts[0].invariants_passed);
    }

    #[test]
    fn optimize_e2e_sort_rejects_broken_optimization() {
        let agent = MockAgent::from_responses(vec![Ok(AgentResponse {
            text: "Removed sorting entirely for speed".into(),
            diff: Some("--- a/sort.rs\n+++ b/sort.rs\n-sort\n+noop".into()),
        })]);

        let env = SortOptimizeEnv::new(100);

        struct BrokenSortEnv(SortOptimizeEnv);

        impl OptimizeEnv for BrokenSortEnv {
            fn measure(&mut self) -> HashMap<OptimizationObjective, f64> {
                self.0.measure()
            }
            fn check_invariants(&mut self) -> bool {
                self.0.check_invariants()
            }
            fn apply_diff(&mut self, _diff: &str) {
                self.0.sort_fn = |d: &mut [i32]| {
                    if d.len() > 1 {
                        d.swap(0, d.len() - 1);
                    }
                };
            }
            fn accept(&mut self, i: usize) {
                self.0.accept(i);
            }
            fn reject(&mut self) {
                self.0.reject();
            }
        }

        let mut broken_env = BrokenSortEnv(env);

        let obj = ObjectiveFunction {
            name: "naive_sort_time",
            inputs: &[NAIVE_SORT_TIME],
            evaluate: |m| m.get(&NAIVE_SORT_TIME).copied().unwrap_or(f64::INFINITY),
        };
        let config = OptimizeConfig {
            num_iterations: 1,
            hint: None,
            verbose: false,
        };

        let result = auto_optimize(&agent, &mut broken_env, &obj, &config, Path::new("/tmp"));

        assert!(!result.attempts[0].invariants_passed);
    }

    // ── Optimize e2e (real agent) ───────────────────────────────────

    #[test]
    #[ignore] // Requires Claude API access
    fn optimize_e2e_real_agent() {
        run_optimize_test("claude-sonnet-4-20250514", 10, 2, None, false);
    }
}
