use std::path::Path;

use super::super::{extract_json, CheckJsonResult, DynInvariant, FailedAttempt, SynthesisTarget};
use super::SynthesisRegistry;
use crate::agent::AgentHarness;

/// Result of a red-team session.
pub enum RedTeamResult {
    /// The agent produced a counterexample that violates the invariant.
    Violation {
        approach: String,
        input_json: String,
        error: String,
    },
    /// All attempts failed to find a violation.
    NoViolation { attempts: Vec<FailedAttempt> },
}

/// Configuration for an AI red-team session.
pub struct RedTeamConfig {
    pub num_iterations: usize,
}

impl Default for RedTeamConfig {
    fn default() -> Self {
        Self { num_iterations: 10 }
    }
}

/// Run an AI red-team session against a single invariant.
///
/// Each iteration:
/// 1. Builds a prompt that includes the invariant description, a JSON
///    example of the `Input` type, and past failed attempts.
/// 2. Invokes the agent (via [`AgentHarness`]) to analyze the code and
///    produce a candidate counterexample as a JSON object.
/// 3. Extracts the JSON from the agent's response, deserializes it into
///    the invariant's `Input` type, and runs [`Invariant::check`] on it.
/// 4. If the check fails, the counterexample is genuine — return it.
/// 5. Otherwise records the failed attempt and continues.
///
/// The `agent` is responsible for its own isolation (e.g. worktrees).
pub fn auto_redteam(
    invariant: &dyn DynInvariant,
    config: &RedTeamConfig,
    agent: &dyn AgentHarness,
    repo_dir: &Path,
) -> RedTeamResult {
    let description = invariant.description();
    let input_example = invariant.input_json_example();
    let setup = invariant.dyn_setup();
    let mut failed_attempts = Vec::new();

    for iteration in 0..config.num_iterations {
        tracing::info!(
            "Red team iteration {}/{} for '{}'",
            iteration + 1,
            config.num_iterations,
            invariant.name()
        );

        let prompt = build_redteam_prompt(&description, input_example.as_deref(), &failed_attempts);

        let response = match agent.invoke(repo_dir, &prompt) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("Agent invocation failed: {e}");
                failed_attempts.push(FailedAttempt {
                    description: format!("Iteration {}", iteration + 1),
                    approach: "Agent invocation failed".to_string(),
                    failure_reason: e.to_string(),
                });
                continue;
            }
        };

        let approach = response.text;
        let Some(json) = extract_json(&approach) else {
            tracing::info!("No JSON found in agent response");
            failed_attempts.push(FailedAttempt {
                description: format!("Iteration {}", iteration + 1),
                approach,
                failure_reason: "Agent response did not contain a JSON counterexample".to_string(),
            });
            continue;
        };

        match invariant.check_json_input(&*setup, &json) {
            CheckJsonResult::Violation(violation) => {
                tracing::info!("Counterexample CONFIRMED: {violation}");
                return RedTeamResult::Violation {
                    approach,
                    input_json: json,
                    error: violation.to_string(),
                };
            }
            CheckJsonResult::Pass => {
                failed_attempts.push(FailedAttempt {
                    description: format!("Iteration {}", iteration + 1),
                    approach,
                    failure_reason: format!(
                        "Candidate input did not violate the invariant: {json}"
                    ),
                });
            }
            CheckJsonResult::BadInput(parse_err) => {
                tracing::info!("Agent produced unparseable input: {parse_err}");
                failed_attempts.push(FailedAttempt {
                    description: format!("Iteration {}", iteration + 1),
                    approach,
                    failure_reason: format!(
                        "Could not deserialize agent JSON into Input type: {parse_err}"
                    ),
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
    input_example: Option<&str>,
    failed_attempts: &[FailedAttempt],
) -> String {
    let mut prompt = String::new();

    prompt.push_str(
        "You are a security researcher red-teaming a zkVM (Jolt). \
         Your goal is to find a concrete input that VIOLATES the following invariant.\n\n",
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
         3. Construct a concrete JSON input that you believe will trigger a violation.\n\
         4. The input will be deserialized and checked mechanically — only genuine \
            violations count.\n\n",
    );

    if let Some(example) = input_example {
        prompt.push_str("## Input format\n\n");
        prompt.push_str(
            "The counterexample must be a JSON object matching this schema. \
             Here is an example of a valid input:\n\n```json\n",
        );
        prompt.push_str(example);
        prompt.push_str("\n```\n\n");
    }

    if !failed_attempts.is_empty() {
        prompt.push_str("## Previous failed attempts\n\n");
        prompt.push_str(
            "The following approaches have already been tried and did NOT produce a \
             valid counterexample. Try a fundamentally different approach.\n\n",
        );
        for attempt in failed_attempts {
            prompt.push_str(&format!(
                "- **{}**: {}\n  Failure: {}\n",
                attempt.description, attempt.approach, attempt.failure_reason
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str(
        "## Required output\n\n\
         End your response with a JSON code block containing your candidate \
         counterexample. Use exactly this format:\n\n\
         ```json\n{ ... }\n```\n\n\
         The JSON must match the input schema above. If after thorough analysis \
         you believe no violation exists, still provide your best-effort candidate.\n",
    );

    prompt
}

/// List all invariants suitable for red-team testing.
pub fn redteamable_invariants(registry: &SynthesisRegistry) -> Vec<&dyn DynInvariant> {
    registry.for_target(SynthesisTarget::RedTeam)
}
