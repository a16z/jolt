use std::path::Path;

use super::super::{DynInvariant, FailedAttempt, SynthesisTarget};
use super::SynthesisRegistry;
use crate::agent::{truncate, AgentHarness};

/// Result of a red-team session.
pub enum RedTeamResult {
    /// Found a counterexample that violates the invariant.
    Violation { description: String, error: String },
    /// All attempts failed to find a violation.
    NoViolation { attempts: Vec<FailedAttempt> },
}

/// Configuration for an AI red-team session.
pub struct RedTeamConfig {
    pub num_iterations: usize,
    /// Number of random fuzz inputs to run after each agent attempt.
    pub num_fuzz_per_iteration: usize,
}

impl Default for RedTeamConfig {
    fn default() -> Self {
        Self {
            num_iterations: 10,
            num_fuzz_per_iteration: 100,
        }
    }
}

/// Run an AI red-team session against a single invariant.
///
/// Each iteration:
/// 1. Builds a prompt from the invariant description + past failed attempts
/// 2. Invokes the agent (via the [`AgentHarness`] trait) to analyze the code
/// 3. Runs the invariant's seed corpus + random fuzz inputs
/// 4. If a violation is found, returns immediately
/// 5. Otherwise records the failed attempt and continues
///
/// The `agent` is responsible for its own isolation (e.g. worktrees).
pub fn auto_redteam(
    invariant: &dyn DynInvariant,
    config: &RedTeamConfig,
    agent: &dyn AgentHarness,
    repo_dir: &Path,
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

        let prompt = build_redteam_prompt(&description, &failed_attempts);

        match agent.invoke(repo_dir, &prompt) {
            Ok(response) => {
                let approach = truncate(&response.text, 2000).to_string();
                tracing::info!(
                    "Agent response ({} chars): {}...",
                    approach.len(),
                    truncate(&approach, 200)
                );

                // Run the invariant's seed corpus + random fuzz inputs
                let check_results = invariant.run_checks(config.num_fuzz_per_iteration);
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
            Err(e) => {
                tracing::warn!("Agent invocation failed: {e}");
                failed_attempts.push(FailedAttempt {
                    description: format!("Iteration {}", iteration + 1),
                    approach: "Agent invocation failed".to_string(),
                    failure_reason: e.to_string(),
                });
            }
        }
    }

    RedTeamResult::NoViolation {
        attempts: failed_attempts,
    }
}

fn build_redteam_prompt(
    invariant_description: &str,
    failed_attempts: &[FailedAttempt],
) -> String {
    let mut prompt = String::new();

    prompt.push_str(
        "You are a security researcher red-teaming a zkVM (Jolt). \
         Your goal is to find an input or scenario that VIOLATES the following invariant.\n\n",
    );

    prompt.push_str("## Invariant\n\n");
    prompt.push_str(invariant_description);
    prompt.push_str("\n\n");

    prompt.push_str(
        "## Instructions\n\n\
         1. Read the relevant source code in this repository to understand how the \
            invariant is enforced.\n\
         2. Look for edge cases, off-by-one errors, missing checks, or assumptions \
            that could be violated.\n\
         3. If you find a potential weakness, describe it clearly.\n\
         4. Try to construct a concrete input or scenario that triggers the violation.\n\
         5. Summarize your approach and findings.\n\n\
         Focus on finding REAL bugs, not theoretical concerns. The invariant will be \
         mechanically checked after your analysis, so only genuine violations count.\n\n",
    );

    if !failed_attempts.is_empty() {
        prompt.push_str("## Previous Failed Attempts\n\n");
        prompt.push_str(
            "The following approaches have already been tried and did NOT find a violation. \
             Try a fundamentally different approach.\n\n",
        );
        for attempt in failed_attempts {
            prompt.push_str(&format!(
                "- **{}**: {}\n  Reason for failure: {}\n",
                attempt.description, attempt.approach, attempt.failure_reason
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str(
        "## Output\n\n\
         End your response with a clear summary of:\n\
         - What you investigated\n\
         - What you found (if anything)\n\
         - Whether you believe the invariant holds or can be violated\n",
    );

    prompt
}

/// List all invariants suitable for red-team testing.
pub fn redteamable_invariants(registry: &SynthesisRegistry) -> Vec<&dyn DynInvariant> {
    registry.for_target(SynthesisTarget::RedTeam)
}
