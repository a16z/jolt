pub mod claude;
pub mod mock;

use std::fmt;
use std::path::Path;

pub use claude::ClaudeCodeAgent;
pub use mock::MockAgent;

/// Output from an agent invocation.
#[derive(Debug)]
pub struct AgentResponse {
    /// The agent's textual output/analysis.
    pub text: String,
    /// A unified diff of code changes the agent produced, if any.
    pub diff: Option<String>,
}

/// Error during agent invocation.
#[derive(Debug, Clone)]
pub struct AgentError {
    pub message: String,
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AgentError {}

impl AgentError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// A coding agent that can analyze or modify a repository given a prompt.
///
/// Implementors are responsible for their own isolation strategy (worktrees,
/// containers, API calls, etc.). The `repo_dir` parameter indicates the
/// repository root so the agent can set up whatever sandbox it needs.
pub trait AgentHarness: Send + Sync {
    fn invoke(&self, repo_dir: &Path, prompt: &str) -> Result<AgentResponse, AgentError>;

    /// Invoke the agent with a JSON Schema constraint on the response.
    ///
    /// Agents that support structured output (e.g. Claude Code with
    /// `--output-format json --json-schema`) should override this to
    /// guarantee the response conforms to `schema`.
    ///
    /// The default falls back to [`invoke`](Self::invoke).
    fn invoke_structured(
        &self,
        repo_dir: &Path,
        prompt: &str,
        _schema: &serde_json::Value,
    ) -> Result<AgentResponse, AgentError> {
        self.invoke(repo_dir, prompt)
    }
}

/// Apply a unified diff to `repo_dir`.
pub fn apply_diff(repo_dir: &Path, diff: &str) -> Result<(), AgentError> {
    use std::process::Command;

    let mut child = Command::new("git")
        .current_dir(repo_dir)
        .args(["apply", "--allow-empty"])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| AgentError::new(format!("git apply spawn: {e}")))?;

    if let Some(stdin) = child.stdin.as_mut() {
        use std::io::Write;
        let _ = stdin.write_all(diff.as_bytes());
    }

    let status = child
        .wait()
        .map_err(|e| AgentError::new(format!("git apply wait: {e}")))?;

    if !status.success() {
        return Err(AgentError::new("git apply failed"));
    }
    Ok(())
}

pub fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        return s;
    }
    let mut end = max_len;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
