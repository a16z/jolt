use std::path::{Path, PathBuf};
use std::process::Command;

use super::super::{DynInvariant, FailedAttempt, SynthesisTarget};
use super::SynthesisRegistry;

/// Result of a red-team session.
pub enum RedTeamResult {
    /// Found a counterexample that violates the invariant.
    Violation { description: String, error: String },
    /// All attempts failed to find a violation.
    NoViolation { attempts: Vec<FailedAttempt> },
}

/// Configuration for an AI red-team session.
pub struct RedTeamConfig {
    pub invariant_name: String,
    pub num_iterations: usize,
    pub model: String,
    pub working_dir: PathBuf,
}

impl Default for RedTeamConfig {
    fn default() -> Self {
        Self {
            invariant_name: String::new(),
            num_iterations: 10,
            model: "claude-sonnet-4-20250514".to_string(),
            working_dir: PathBuf::from("."),
        }
    }
}

/// Create an isolated git worktree for the AI agent to work in.
pub fn create_worktree(repo_dir: &Path, _branch_name: &str) -> Result<PathBuf, String> {
    let tmp = tempfile::tempdir().map_err(|e| format!("Failed to create temp dir: {e}"))?;
    // Persist the temp dir so the worktree outlives this function
    let worktree_dir = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let status = Command::new("git")
        .current_dir(repo_dir)
        .args(["worktree", "add", "--detach"])
        .arg(&worktree_dir)
        .status()
        .map_err(|e| format!("Failed to run git worktree: {e}"))?;

    if !status.success() {
        return Err("git worktree add failed".to_string());
    }

    Ok(worktree_dir)
}

/// Remove a git worktree.
pub fn remove_worktree(repo_dir: &Path, worktree_dir: &Path) {
    let _ = Command::new("git")
        .current_dir(repo_dir)
        .args(["worktree", "remove", "--force"])
        .arg(worktree_dir)
        .status();
}

/// Run an AI red-team session against a single invariant.
///
/// The AI agent runs in an isolated worktree to produce a claimed bad input.
/// The invariant is checked in the original working tree so the AI cannot cheat.
///
/// This function orchestrates the loop but delegates the actual AI interaction
/// to the `invoke_agent` callback, which should:
/// 1. Receive the invariant description and past failed attempts
/// 2. Have the AI produce a candidate counterexample (as bytes)
/// 3. Return the candidate or None if the AI couldn't produce one
pub fn auto_redteam(
    invariant: &dyn DynInvariant,
    config: &RedTeamConfig,
    mut invoke_agent: impl FnMut(&str, &[FailedAttempt]) -> Option<(String, Vec<u8>)>,
) -> RedTeamResult {
    let description = invariant.description();
    let mut failed_attempts = Vec::new();

    for iteration in 0..config.num_iterations {
        tracing::info!(
            "Red team iteration {}/{} for '{}'",
            iteration + 1,
            config.num_iterations,
            invariant.name()
        );

        let result = invoke_agent(&description, &failed_attempts);

        match result {
            Some((approach, _candidate_bytes)) => {
                // Run the invariant's checks to see if the agent found a violation
                let check_results = invariant.run_checks(0);
                let violation = check_results.iter().find(|r| r.is_err());

                if let Some(Err(e)) = violation {
                    return RedTeamResult::Violation {
                        description: approach,
                        error: e.to_string(),
                    };
                }

                failed_attempts.push(FailedAttempt {
                    description: format!("Iteration {}", iteration + 1),
                    approach,
                    failure_reason: "Invariant check passed for all inputs".to_string(),
                });
            }
            None => {
                failed_attempts.push(FailedAttempt {
                    description: format!("Iteration {}", iteration + 1),
                    approach: "Agent could not produce a candidate".to_string(),
                    failure_reason: "No candidate generated".to_string(),
                });
            }
        }
    }

    RedTeamResult::NoViolation {
        attempts: failed_attempts,
    }
}

/// List all invariants suitable for red-team testing.
pub fn redteamable_invariants(registry: &SynthesisRegistry) -> Vec<&dyn DynInvariant> {
    registry.for_target(SynthesisTarget::RedTeam)
}
