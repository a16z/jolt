//! End-to-end test harnesses for the optimization and red-team loops,
//! using simple sorting functions as the target domain.

use crate::agent::ClaudeCodeAgent;
use crate::invariant::synthesis::redteam::{auto_redteam, RedTeamConfig, RedTeamResult};
use crate::invariant::{CheckError, Invariant, InvariantViolation};
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::agent::MockAgent;
    use crate::invariant::synthesis::redteam::{auto_redteam, RedTeamConfig, RedTeamResult};

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
}
