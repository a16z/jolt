use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

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
///
/// # Examples
///
/// The built-in [`ClaudeCodeAgent`] creates a git worktree and invokes the
/// `claude` CLI. A multi-agent harness could fan out to several agents in
/// parallel and merge results. An API-based agent could call a remote
/// service without any local isolation.
pub trait AgentHarness: Send + Sync {
    fn invoke(&self, repo_dir: &Path, prompt: &str) -> Result<AgentResponse, AgentError>;

    /// Invoke the agent with a JSON Schema constraint on the response.
    ///
    /// Agents that support structured output (e.g. Claude Code with
    /// `--output-format json --json-schema`) should override this to
    /// guarantee the response conforms to `schema`.  The returned
    /// [`AgentResponse::text`] must be the validated JSON string.
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

/// Agent implementation that invokes the Claude Code CLI in an isolated
/// git worktree.
pub struct ClaudeCodeAgent {
    pub model: String,
    pub max_turns: usize,
}

impl ClaudeCodeAgent {
    pub fn new(model: impl Into<String>, max_turns: usize) -> Self {
        Self {
            model: model.into(),
            max_turns,
        }
    }
}

impl ClaudeCodeAgent {
    fn run_cli(
        &self,
        worktree_dir: &Path,
        prompt: &str,
        extra_args: &[&str],
    ) -> Result<std::process::Output, AgentError> {
        tracing::info!(
            "Invoking claude (model={}, max_turns={})...",
            self.model,
            self.max_turns
        );
        let mut cmd = Command::new("claude");
        cmd.current_dir(worktree_dir)
            .arg("-p")
            .arg(prompt)
            .arg("--model")
            .arg(&self.model)
            .arg("--max-turns")
            .arg(self.max_turns.to_string())
            .arg("--verbose");
        for arg in extra_args {
            cmd.arg(arg);
        }
        cmd.output().map_err(|e| {
            AgentError::new(format!(
                "Failed to invoke claude: {e}. \
                 Make sure the `claude` CLI is installed and on your PATH. \
                 Install via: npm install -g @anthropic-ai/claude-code"
            ))
        })
    }
}

impl AgentHarness for ClaudeCodeAgent {
    fn invoke(&self, repo_dir: &Path, prompt: &str) -> Result<AgentResponse, AgentError> {
        let worktree_dir = create_worktree(repo_dir)?;
        tracing::info!("Created worktree at {}", worktree_dir.display());

        let result = self.run_cli(&worktree_dir, prompt, &[]);

        // Capture diff before cleanup
        let diff = Command::new("git")
            .current_dir(&worktree_dir)
            .args(["diff", "HEAD"])
            .output()
            .ok()
            .and_then(|o| {
                let s = String::from_utf8_lossy(&o.stdout).to_string();
                if s.trim().is_empty() {
                    None
                } else {
                    Some(s)
                }
            });

        tracing::info!("Cleaning up worktree...");
        remove_worktree(repo_dir, &worktree_dir);
        let _ = std::fs::remove_dir_all(&worktree_dir);

        let output = result?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !output.status.success() {
            tracing::warn!("claude exited with status {}", output.status);
            if !stderr.is_empty() {
                tracing::warn!("stderr: {}", truncate(&stderr, 500));
            }
        }

        let text = if stdout.trim().is_empty() {
            stderr.to_string()
        } else {
            stdout.to_string()
        };

        if text.trim().is_empty() && diff.is_none() {
            return Err(AgentError::new("Agent produced no output"));
        }

        Ok(AgentResponse { text, diff })
    }

    fn invoke_structured(
        &self,
        repo_dir: &Path,
        prompt: &str,
        schema: &serde_json::Value,
    ) -> Result<AgentResponse, AgentError> {
        let worktree_dir = create_worktree(repo_dir)?;
        tracing::info!("Created worktree at {}", worktree_dir.display());

        let schema_str = serde_json::to_string(schema)
            .map_err(|e| AgentError::new(format!("schema serialization: {e}")))?;

        let result = self.run_cli(
            &worktree_dir,
            prompt,
            &["--output-format", "json", "--json-schema", &schema_str],
        );

        tracing::info!("Cleaning up worktree...");
        remove_worktree(repo_dir, &worktree_dir);
        let _ = std::fs::remove_dir_all(&worktree_dir);

        let output = result?;
        let stdout = String::from_utf8_lossy(&output.stdout);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AgentError::new(format!(
                "claude exited with status {}: {}",
                output.status,
                truncate(&stderr, 500)
            )));
        }

        // Parse the CLI JSON envelope and extract structured_output
        let envelope: serde_json::Value = serde_json::from_str(&stdout)
            .map_err(|e| AgentError::new(format!("failed to parse CLI JSON envelope: {e}")))?;

        let text = if let Some(structured) = envelope.get("structured_output") {
            serde_json::to_string(structured)
                .map_err(|e| AgentError::new(format!("re-serialize structured_output: {e}")))?
        } else if let Some(result) = envelope.get("result") {
            match result {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other)
                    .map_err(|e| AgentError::new(format!("re-serialize result: {e}")))?,
            }
        } else {
            return Err(AgentError::new(
                "CLI JSON envelope contained neither structured_output nor result",
            ));
        };

        Ok(AgentResponse { text, diff: None })
    }
}

/// Create an isolated detached git worktree from `repo_dir`.
pub fn create_worktree(repo_dir: &Path) -> Result<PathBuf, AgentError> {
    let tmp = tempfile::tempdir().map_err(|e| AgentError::new(format!("tempdir: {e}")))?;
    let worktree_dir = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let status = Command::new("git")
        .current_dir(repo_dir)
        .args(["worktree", "add", "--detach"])
        .arg(&worktree_dir)
        .status()
        .map_err(|e| AgentError::new(format!("git worktree: {e}")))?;

    if !status.success() {
        return Err(AgentError::new("git worktree add failed"));
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

/// Apply a unified diff to `repo_dir`.
pub fn apply_diff(repo_dir: &Path, diff: &str) -> Result<(), AgentError> {
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
