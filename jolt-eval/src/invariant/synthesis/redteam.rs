use std::path::Path;

use super::super::{CheckError, FailedAttempt, Invariant};
use crate::agent::{AgentHarness, DiffScope};

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
    pub verbose: bool,
}

impl Default for RedTeamConfig {
    fn default() -> Self {
        Self {
            num_iterations: 10,
            hint: None,
            verbose: false,
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
        let iter = iteration + 1;
        tracing::info!(
            "Red team iteration {iter}/{} for '{}'",
            config.num_iterations,
            invariant.name()
        );

        let prompt = build_redteam_prompt(
            &description,
            input_example.as_deref(),
            &input_schema,
            config.hint.as_deref(),
            &failed_attempts,
            iter,
            config.num_iterations,
        );

        if config.verbose {
            eprintln!("── Iteration {iter} prompt ──");
            eprintln!("{prompt}");
            eprintln!("────────────────────────");
        }

        let diff_scope = DiffScope::Include(vec!["jolt-eval/guest-sandbox/".into()]);
        let response =
            match agent.invoke_structured(repo_dir, &prompt, &envelope_schema, &diff_scope) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("Agent invocation failed: {e}");
                    let path = persist_redteam_attempt(
                        repo_dir,
                        invariant.name(),
                        iter,
                        "Agent invocation failed",
                        &e.to_string(),
                    );
                    failed_attempts.push(FailedAttempt {
                        description: format!("Iteration {iter}"),
                        approach: "Agent invocation failed".to_string(),
                        approach_summary: "Agent invocation failed".to_string(),
                        failure_reason: e.to_string(),
                        path,
                    });
                    continue;
                }
            };

        if config.verbose {
            eprintln!("── Iteration {iter} response ──");
            eprintln!("{}", response.text);
            if let Some(ref d) = response.diff {
                eprintln!("── diff ({} bytes) ──", d.len());
                eprintln!("{d}");
            }
            eprintln!("──────────────────────────");
        }

        let (analysis, approach_summary, counterexample_json) =
            match parse_envelope(&response.text) {
                Some(triple) => triple,
                None => match super::super::extract_json(&response.text) {
                    Some(json) => match parse_envelope(&json) {
                        Some(triple) => triple,
                        None => (response.text.clone(), String::new(), json),
                    },
                    None => {
                        let failure = "Agent response did not contain valid JSON".to_string();
                        let path = persist_redteam_attempt(
                            repo_dir,
                            invariant.name(),
                            iter,
                            &response.text,
                            &failure,
                        );
                        failed_attempts.push(FailedAttempt {
                            description: format!("Iteration {iter}"),
                            approach: response.text,
                            approach_summary: String::new(),
                            failure_reason: failure,
                            path,
                        });
                        continue;
                    }
                },
            };

        let input: I::Input = match serde_json::from_str(&counterexample_json) {
            Ok(v) => v,
            Err(e) => {
                tracing::info!("Agent produced unparsable input: {e}");
                let failure = format!("Could not deserialize response JSON into Input type: {e}");
                let path =
                    persist_redteam_attempt(repo_dir, invariant.name(), iter, &analysis, &failure);
                failed_attempts.push(FailedAttempt {
                    description: format!("Iteration {iter}"),
                    approach: analysis,
                    approach_summary,
                    failure_reason: failure,
                    path,
                });
                continue;
            }
        };

        // Let the invariant fill in fields from the agent's worktree diff
        // (e.g. SoundnessInvariant uses it to populate the patch field).
        let input = invariant.enrich_input(input, response.diff.as_deref());

        match invariant.check(&setup, input) {
            Ok(()) => {
                let failure =
                    format!("Candidate input did not violate the invariant: {counterexample_json}");
                let path =
                    persist_redteam_attempt(repo_dir, invariant.name(), iter, &analysis, &failure);
                failed_attempts.push(FailedAttempt {
                    description: format!("Iteration {iter}"),
                    approach: analysis,
                    approach_summary,
                    failure_reason: failure,
                    path,
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
            Err(CheckError::InvalidInput(reason)) => {
                let failure = format!("Invalid input: {reason}");
                let path =
                    persist_redteam_attempt(repo_dir, invariant.name(), iter, &analysis, &failure);
                failed_attempts.push(FailedAttempt {
                    description: format!("Iteration {iter}"),
                    approach: analysis,
                    approach_summary,
                    failure_reason: failure,
                    path,
                });
            }
        }
    }

    RedTeamResult::NoViolation {
        attempts: failed_attempts,
    }
}

/// Persist a red-team attempt's approach to disk and return the relative path.
fn persist_redteam_attempt(
    repo_dir: &Path,
    invariant_name: &str,
    iteration: usize,
    approach: &str,
    failure_reason: &str,
) -> Option<String> {
    let dir = repo_dir
        .join("jolt-eval/redteam-history")
        .join(invariant_name)
        .join(format!("attempt-{iteration}"));
    std::fs::create_dir_all(&dir).ok()?;
    std::fs::write(dir.join("approach.md"), approach).ok()?;
    std::fs::write(dir.join("failure_reason.txt"), failure_reason).ok()?;
    Some(
        dir.strip_prefix(repo_dir)
            .ok()?
            .to_string_lossy()
            .to_string(),
    )
}

fn build_envelope_schema(input_schema: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "Your analysis of the invariant and approach to finding a violation"
            },
            "approach_summary": {
                "type": "string",
                "description": "A short summary of your approach (at most 4 sentences)"
            },
            "counterexample": input_schema
        },
        "required": ["analysis", "approach_summary", "counterexample"],
        "additionalProperties": false
    })
}

/// Parsed envelope: (analysis, approach_summary, counterexample_json).
fn parse_envelope(text: &str) -> Option<(String, String, String)> {
    let val: serde_json::Value = serde_json::from_str(text).ok()?;
    let analysis = val.get("analysis")?.as_str()?.to_string();
    let approach_summary = val
        .get("approach_summary")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let counterexample = val.get("counterexample")?;
    Some((
        analysis,
        approach_summary,
        serde_json::to_string(counterexample).ok()?,
    ))
}

fn build_redteam_prompt(
    invariant_description: &str,
    input_example: Option<&str>,
    input_schema: &serde_json::Value,
    hint: Option<&str>,
    failed_attempts: &[FailedAttempt],
    current_iteration: usize,
    total_iterations: usize,
) -> String {
    let mut prompt = String::new();

    prompt.push_str(&format!(
        "You are a security researcher red-teaming a zkVM (Jolt). \
         Your goal is to find a concrete input that VIOLATES the following invariant.\n\
         Iteration {current_iteration}/{total_iterations}.\n\n"
    ));

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
            violations count.\n\n\
         IMPORTANT: Do NOT ask questions or request clarification. Do NOT create \
         plans or outlines. You MUST produce a concrete counterexample JSON on \
         every iteration. Spend your turns reading code and reasoning, then output \
         your best counterexample. A wrong guess is always better than no guess.\n\n",
    );

    prompt.push_str("## Input format\n\n");
    if let Some(example) = input_example {
        prompt.push_str(
            "The counterexample must be a JSON value matching the schema. \
             Here is an example of a valid input:\n\n```json\n",
        );
        prompt.push_str(example);
        prompt.push_str("\n```\n\n");
    }
    prompt.push_str("JSON schema for the counterexample:\n\n```json\n");
    prompt.push_str(&serde_json::to_string_pretty(input_schema).unwrap_or_default());
    prompt.push_str("\n```\n\n");

    if let Some(hint) = hint {
        prompt.push_str("## Hint\n\n");
        prompt.push_str(hint);
        prompt.push_str("\n\n");
    }

    if !failed_attempts.is_empty() {
        prompt.push_str("## Previous failed attempts\n\n");
        prompt.push_str(
            "The following approaches have already been tried and did NOT produce a \
             valid counterexample. Try a fundamentally different strategy.\n\n",
        );
        for attempt in failed_attempts {
            let path_ref = attempt
                .path
                .as_deref()
                .map(|p| format!(" (full details in `{p}/`)"))
                .unwrap_or_default();
            prompt.push_str(&format!(
                "- **{}**: {}{path_ref}\n  Failure: {}\n",
                attempt.description, attempt.approach_summary, attempt.failure_reason,
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str(
        "## Required output\n\n\
         You MUST respond with a JSON object containing exactly two fields:\n\
         - `analysis`: a brief summary of what you investigated and why you \
           chose this counterexample\n\
         - `counterexample`: the candidate input matching the schema above\n\n\
         Do NOT respond with anything other than this JSON object. No questions, \
         no plans, no markdown outside the JSON.\n",
    );

    prompt
}
