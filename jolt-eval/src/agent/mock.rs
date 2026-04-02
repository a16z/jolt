use std::path::Path;

use super::{AgentError, AgentHarness, AgentResponse};

/// A mock agent for testing. Returns pre-configured responses and records
/// every prompt it receives.
///
/// # Usage
///
/// ```ignore
/// use jolt_eval::agent::{MockAgent, AgentResponse};
///
/// // Agent that always succeeds with a fixed response
/// let agent = MockAgent::always_ok("I found nothing.");
///
/// // Agent that returns a sequence of responses
/// let agent = MockAgent::from_responses(vec![
///     Ok(AgentResponse { text: "attempt 1".into(), diff: None }),
///     Err(AgentError::new("network timeout")),
///     Ok(AgentResponse { text: "attempt 3".into(), diff: Some("diff".into()) }),
/// ]);
///
/// // After invoking, inspect the prompts the agent received
/// let prompts = agent.recorded_prompts();
/// ```
pub struct MockAgent {
    responses: std::sync::Mutex<Vec<Result<AgentResponse, AgentError>>>,
    prompts: std::sync::Mutex<Vec<String>>,
}

impl MockAgent {
    /// Create a mock that always returns `Ok` with the given text and no diff.
    pub fn always_ok(text: &str) -> Self {
        let text = text.to_string();
        Self {
            responses: std::sync::Mutex::new(vec![Ok(AgentResponse { text, diff: None })]),
            prompts: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Create a mock that always returns `Err`.
    pub fn always_err(message: &str) -> Self {
        Self {
            responses: std::sync::Mutex::new(vec![Err(AgentError::new(message))]),
            prompts: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Create a mock that returns responses from a queue.
    /// After the queue is exhausted, subsequent calls return an error.
    pub fn from_responses(responses: Vec<Result<AgentResponse, AgentError>>) -> Self {
        let mut reversed = responses;
        reversed.reverse(); // so we can pop from the back
        Self {
            responses: std::sync::Mutex::new(reversed),
            prompts: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Return all prompts that were passed to `invoke`, in order.
    pub fn recorded_prompts(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
    }
}

impl AgentHarness for MockAgent {
    fn invoke(&self, _repo_dir: &Path, prompt: &str) -> Result<AgentResponse, AgentError> {
        self.prompts.lock().unwrap().push(prompt.to_string());

        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            return Err(AgentError::new("MockAgent: no more responses"));
        }
        // If only one response left, clone it (repeating) instead of popping
        if responses.len() == 1 {
            return match &responses[0] {
                Ok(r) => Ok(AgentResponse {
                    text: r.text.clone(),
                    diff: r.diff.clone(),
                }),
                Err(e) => Err(AgentError::new(&e.message)),
            };
        }
        responses.pop().unwrap()
    }
}
