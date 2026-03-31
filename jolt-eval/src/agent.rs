use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Output from an agent invocation.
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

impl AgentHarness for ClaudeCodeAgent {
    fn invoke(&self, repo_dir: &Path, prompt: &str) -> Result<AgentResponse, AgentError> {
        // 1. Create worktree
        let worktree_dir = create_worktree(repo_dir)?;
        tracing::info!("Created worktree at {}", worktree_dir.display());

        // 2. Run Claude
        tracing::info!(
            "Invoking claude (model={}, max_turns={})...",
            self.model,
            self.max_turns
        );
        let result = Command::new("claude")
            .current_dir(&worktree_dir)
            .arg("-p")
            .arg(prompt)
            .arg("--model")
            .arg(&self.model)
            .arg("--max-turns")
            .arg(self.max_turns.to_string())
            .arg("--verbose")
            .output();

        // 3. Capture diff before cleanup
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

        // 4. Clean up worktree
        tracing::info!("Cleaning up worktree...");
        remove_worktree(repo_dir, &worktree_dir);
        let _ = std::fs::remove_dir_all(&worktree_dir);

        // 5. Parse result
        match result {
            Ok(output) => {
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
            Err(e) => Err(AgentError::new(format!(
                "Failed to invoke claude: {e}. \
                 Make sure the `claude` CLI is installed and on your PATH. \
                 Install via: npm install -g @anthropic-ai/claude-code"
            ))),
        }
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
