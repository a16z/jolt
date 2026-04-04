use std::path::{Path, PathBuf};
use std::process::Command;

use super::{AgentError, AgentHarness, AgentResponse, DiffScope};

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

    fn run_cli(
        &self,
        worktree_dir: &Path,
        prompt: &str,
        extra_args: &[&str],
        verbose: bool,
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
            .arg("--dangerously-skip-permissions");
        if verbose {
            cmd.arg("--verbose");
        }
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
    fn invoke(
        &self,
        repo_dir: &Path,
        prompt: &str,
        diff_scope: &DiffScope,
    ) -> Result<AgentResponse, AgentError> {
        let worktree_dir = create_worktree(repo_dir)?;
        tracing::info!("Created worktree at {}", worktree_dir.display());

        let result = self.run_cli(&worktree_dir, prompt, &[], true);

        let diff = capture_diff(&worktree_dir, diff_scope);

        tracing::info!("Cleaning up worktree...");
        remove_worktree(repo_dir, &worktree_dir);
        let _ = std::fs::remove_dir_all(&worktree_dir);

        let output = result?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !output.status.success() {
            tracing::warn!("claude exited with status {}", output.status);
            if !stderr.is_empty() {
                tracing::warn!("stderr: {}", super::truncate(&stderr, 500));
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
        diff_scope: &DiffScope,
    ) -> Result<AgentResponse, AgentError> {
        let worktree_dir = create_worktree(repo_dir)?;
        tracing::info!("Created worktree at {}", worktree_dir.display());

        let schema_str = serde_json::to_string(schema)
            .map_err(|e| AgentError::new(format!("schema serialization: {e}")))?;

        let result = self.run_cli(
            &worktree_dir,
            prompt,
            &["--output-format", "json", "--json-schema", &schema_str],
            false,
        );

        let diff = capture_diff(&worktree_dir, diff_scope);

        tracing::info!("Cleaning up worktree...");
        remove_worktree(repo_dir, &worktree_dir);
        let _ = std::fs::remove_dir_all(&worktree_dir);

        let output = result?;
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse the JSON envelope — even on non-zero exit (e.g. max_turns
        // reached), Claude may still have produced structured output.
        let envelope: serde_json::Value = match serde_json::from_str(&stdout) {
            Ok(v) => v,
            Err(e) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let detail = if stderr.trim().is_empty() {
                        super::truncate(&stdout, 1000)
                    } else {
                        super::truncate(&stderr, 1000)
                    };
                    return Err(AgentError::new(format!(
                        "claude exited with status {}: {}",
                        output.status, detail
                    )));
                }
                return Err(AgentError::new(format!(
                    "failed to parse CLI JSON envelope: {e}"
                )));
            }
        };

        let text = if let Some(structured) = envelope.get("structured_output") {
            serde_json::to_string(structured)
                .map_err(|e| AgentError::new(format!("re-serialize structured_output: {e}")))?
        } else if let Some(result) = envelope.get("result") {
            match result {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other)
                    .map_err(|e| AgentError::new(format!("re-serialize result: {e}")))?,
            }
        } else if !output.status.success() {
            let errors = envelope
                .get("errors")
                .and_then(|e| serde_json::to_string(e).ok())
                .unwrap_or_default();
            return Err(AgentError::new(format!(
                "claude exited with status {}: {}",
                output.status, errors
            )));
        } else {
            return Err(AgentError::new(
                "CLI JSON envelope contained neither structured_output nor result",
            ));
        };

        Ok(AgentResponse { text, diff })
    }
}

/// Capture a unified diff of changes in a worktree relative to HEAD,
/// filtered by the given [`DiffScope`].
fn capture_diff(worktree_dir: &Path, scope: &DiffScope) -> Option<String> {
    let mut cmd = Command::new("git");
    cmd.current_dir(worktree_dir).args(["diff", "HEAD", "--"]);
    match scope {
        DiffScope::All => {}
        DiffScope::Include(paths) => {
            for p in paths {
                cmd.arg(p);
            }
        }
        DiffScope::Exclude(paths) => {
            for p in paths {
                cmd.arg(format!(":!{p}"));
            }
        }
    }
    cmd.output().ok().and_then(|o| {
        let s = String::from_utf8_lossy(&o.stdout).to_string();
        if s.trim().is_empty() {
            None
        } else {
            Some(s)
        }
    })
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
