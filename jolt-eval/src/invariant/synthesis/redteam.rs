use std::path::Path;

use super::super::{FailedAttempt, Invariant, InvariantViolation};
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
    pub hint: Option<String>,
}

impl Default for RedTeamConfig {
    fn default() -> Self {
        Self {
            num_iterations: 10,
            hint: None,
        }
    }
}

/// Run an AI red-team session against a single invariant.
pub fn auto_redteam<I: Invariant>(
    invariant: &I,
    config: &RedTeamConfig,
    agent: &dyn AgentHarness,
    repo_dir: &Path,
) -> RedTeamResult {
    let description = invariant.description();
    let input_example: Option<String> = invariant
        .seed_corpus()
        .into_iter()
        .next()
        .and_then(|input| serde_json::to_string_pretty(&input).ok());
    let input_schema = serde_json::to_value(schemars::schema_for!(I::Input)).unwrap();
    let envelope_schema = build_envelope_schema(&input_schema);
    let setup = invariant.setup();
    let mut failed_attempts = Vec::new();

    for iteration in 0..config.num_iterations {
        tracing::info!(
            "Red team iteration {}/{} for '{}'",
            iteration + 1,
            config.num_iterations,
            invariant.name()
        );

        let prompt = build_redteam_prompt(
            &description,
            input_example.as_deref(),
            config.hint.as_deref(),
            &failed_attempts,
        );

        let response = match agent.invoke_structured(repo_dir, &prompt, &envelope_schema) {
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

        let (analysis, counterexample_json) = match parse_envelope(&response.text) {
            Some(pair) => pair,
            None => match super::super::extract_json(&response.text) {
                Some(json) => match parse_envelope(&json) {
                    Some(pair) => pair,
                    None => (response.text.clone(), json),
                },
                None => {
                    failed_attempts.push(FailedAttempt {
                        description: format!("Iteration {}", iteration + 1),
                        approach: response.text,
                        failure_reason: "Agent response did not contain valid JSON".to_string(),
                    });
                    continue;
                }
            },
        };

        match check_counterexample(invariant, &setup, &counterexample_json) {
            Ok(()) => {
                failed_attempts.push(FailedAttempt {
                    description: format!("Iteration {}", iteration + 1),
                    approach: analysis,
                    failure_reason: format!(
                        "Candidate input did not violate the invariant: {counterexample_json}"
                    ),
                });
            }
            Err(CheckError::Violation(violation)) => {
                tracing::info!("Counterexample CONFIRMED: {violation}");
                return RedTeamResult::Violation {
                    approach: analysis,
                    input_json: counterexample_json,
                    error: violation.to_string(),
                };
            }
            Err(CheckError::BadInput(parse_err)) => {
                tracing::info!("Agent produced unparseable input: {parse_err}");
                failed_attempts.push(FailedAttempt {
                    description: format!("Iteration {}", iteration + 1),
                    approach: analysis,
                    failure_reason: format!(
                        "Could not deserialize response JSON into Input type: {parse_err}"
                    ),
                });
            }
        }
    }

    RedTeamResult::NoViolation {
        attempts: failed_attempts,
    }
}

enum CheckError {
    Violation(InvariantViolation),
    BadInput(String),
}

fn check_counterexample<I: Invariant>(
    inv: &I,
    setup: &I::Setup,
    json: &str,
) -> Result<(), CheckError> {
    let input: I::Input =
        serde_json::from_str(json).map_err(|e| CheckError::BadInput(e.to_string()))?;
    inv.check(setup, input).map_err(CheckError::Violation)
}

fn build_envelope_schema(input_schema: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "Your analysis of the invariant and approach to finding a violation"
            },
            "counterexample": input_schema
        },
        "required": ["analysis", "counterexample"]
    })
}

fn parse_envelope(text: &str) -> Option<(String, String)> {
    let val: serde_json::Value = serde_json::from_str(text).ok()?;
    let analysis = val.get("analysis")?.as_str()?.to_string();
    let counterexample = val.get("counterexample")?;
    Some((analysis, serde_json::to_string(counterexample).ok()?))
}

fn build_redteam_prompt(
    invariant_description: &str,
    input_example: Option<&str>,
    hint: Option<&str>,
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
         3. Construct a concrete counterexample input that you believe will trigger \
            a violation.\n\
         4. The input will be deserialized and checked mechanically — only genuine \
            violations count.\n\n",
    );

    if let Some(example) = input_example {
        prompt.push_str("## Input format\n\n");
        prompt.push_str(
            "The counterexample must be a JSON value matching the schema. \
             Here is an example of a valid input:\n\n```json\n",
        );
        prompt.push_str(example);
        prompt.push_str("\n```\n\n");
    }

    if let Some(hint) = hint {
        prompt.push_str("## Hint\n\n");
        prompt.push_str(hint);
        prompt.push_str("\n\n");
    }

    if !failed_attempts.is_empty() {
        prompt.push_str("## Previous failed attempts\n\n");
        prompt.push_str(
            "The following approaches have already been tried and did NOT produce a \
             valid counterexample.\n\n",
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
         Respond with a JSON object containing:\n\
         - `analysis`: your reasoning and what you investigated\n\
         - `counterexample`: the candidate input matching the schema above\n",
    );

    prompt
}
