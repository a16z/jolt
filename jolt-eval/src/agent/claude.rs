use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

use super::{AgentError, AgentHarness, AgentResponse, DiffScope};

/// Agent implementation that invokes the Claude Code CLI in an isolated
/// git worktree.
///
/// Runs the CLI with `--output-format stream-json`, parses the
/// newline-delimited event stream as it arrives, and pretty-prints a
/// running log of assistant text, thinking, and tool calls to stderr.
/// The final `result` event (which carries either `structured_output` or
/// `result`) is preserved for the caller.
pub struct ClaudeCodeAgent {
    pub model: String,
    pub max_turns: usize,
    /// When true, tool results are also printed. Text, thinking, and
    /// tool-use summaries are always streamed.
    pub verbose: bool,
}

impl ClaudeCodeAgent {
    pub fn new(model: impl Into<String>, max_turns: usize, verbose: bool) -> Self {
        Self {
            model: model.into(),
            max_turns,
            verbose,
        }
    }

    fn run_streaming(
        &self,
        worktree_dir: &Path,
        prompt: &str,
        extra_args: &[&str],
    ) -> Result<StreamOutcome, AgentError> {
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
            .arg("--dangerously-skip-permissions")
            .arg("--output-format")
            .arg("stream-json")
            // stream-json requires --verbose on the CLI side; this is
            // independent of our printer verbosity.
            .arg("--verbose");
        for arg in extra_args {
            cmd.arg(arg);
        }
        cmd.stdout(Stdio::piped()).stderr(Stdio::inherit());

        let mut child = cmd.spawn().map_err(|e| {
            AgentError::new(format!(
                "Failed to invoke claude: {e}. \
                 Make sure the `claude` CLI is installed and on your PATH. \
                 Install via: npm install -g @anthropic-ai/claude-code"
            ))
        })?;

        let stdout = child.stdout.take().expect("stdout was piped");
        let reader = BufReader::new(stdout);

        let mut final_event: Option<Value> = None;
        let mut accumulated_text = String::new();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!("claude stdout read error: {e}");
                    continue;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            let event: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        "unparsable claude event: {e}; line: {}",
                        super::truncate(&line, 200)
                    );
                    continue;
                }
            };

            accumulate_text(&event, &mut accumulated_text);
            print_event(&event, self.verbose);

            if event.get("type").and_then(Value::as_str) == Some("result") {
                final_event = Some(event);
            }
        }

        let status = child
            .wait()
            .map_err(|e| AgentError::new(format!("wait for claude: {e}")))?;

        Ok(StreamOutcome {
            status,
            final_event,
            accumulated_text,
        })
    }
}

struct StreamOutcome {
    status: std::process::ExitStatus,
    final_event: Option<Value>,
    accumulated_text: String,
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

        let result = self.run_streaming(&worktree_dir, prompt, &[]);

        let diff = capture_diff(&worktree_dir, diff_scope);

        tracing::info!("Cleaning up worktree...");
        remove_worktree(repo_dir, &worktree_dir);
        let _ = std::fs::remove_dir_all(&worktree_dir);

        let outcome = result?;

        if !outcome.status.success() {
            tracing::warn!("claude exited with status {}", outcome.status);
        }

        // Prefer the final `result` event's text (canonical), fall back to
        // accumulated assistant text if missing.
        let text = outcome
            .final_event
            .as_ref()
            .and_then(|e| e.get("result"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or(outcome.accumulated_text);

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

        let result = self.run_streaming(&worktree_dir, prompt, &["--json-schema", &schema_str]);

        let diff = capture_diff(&worktree_dir, diff_scope);

        tracing::info!("Cleaning up worktree...");
        remove_worktree(repo_dir, &worktree_dir);
        let _ = std::fs::remove_dir_all(&worktree_dir);

        let outcome = result?;

        let Some(final_event) = outcome.final_event.as_ref() else {
            return Err(AgentError::new(format!(
                "claude exited with status {} before emitting a result event",
                outcome.status
            )));
        };

        if let Some(structured) = final_event.get("structured_output") {
            let text = serde_json::to_string(structured)
                .map_err(|e| AgentError::new(format!("re-serialize structured_output: {e}")))?;
            return Ok(AgentResponse { text, diff });
        }

        if let Some(result_val) = final_event.get("result") {
            let text = match result_val {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other)
                    .map_err(|e| AgentError::new(format!("re-serialize result: {e}")))?,
            };
            return Ok(AgentResponse { text, diff });
        }

        if !outcome.status.success() {
            let errors = final_event
                .get("errors")
                .and_then(|e| serde_json::to_string(e).ok())
                .unwrap_or_default();
            return Err(AgentError::new(format!(
                "claude exited with status {}: {}",
                outcome.status, errors
            )));
        }

        Err(AgentError::new(
            "result event contained neither structured_output nor result",
        ))
    }
}

/// Accumulate the text content of assistant messages for the
/// fallback-when-missing-result-event path in [`ClaudeCodeAgent::invoke`].
fn accumulate_text(event: &Value, out: &mut String) {
    if event.get("type").and_then(Value::as_str) != Some("assistant") {
        return;
    }
    let Some(content) = event
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
    else {
        return;
    };
    for block in content {
        if block.get("type").and_then(Value::as_str) == Some("text") {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                out.push_str(text);
            }
        }
    }
}

/// Pretty-print an event to stderr.
///
/// Always printed: assistant text, thinking blocks, and a one-line
/// summary of tool-use calls. Only printed when `verbose`: tool results.
fn print_event(event: &Value, verbose: bool) {
    let Some(ty) = event.get("type").and_then(Value::as_str) else {
        return;
    };
    match ty {
        "assistant" => print_assistant(event),
        "user" if verbose => print_tool_result(event),
        "result" => print_result(event),
        _ => {}
    }
}

fn print_assistant(event: &Value) {
    let Some(content) = event
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
    else {
        return;
    };
    for block in content {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                for line in text.lines() {
                    eprintln!("│ {line}");
                }
            }
            Some("thinking") => {
                let text = block.get("thinking").and_then(Value::as_str).unwrap_or("");
                for line in text.lines() {
                    eprintln!("┊ {line}");
                }
            }
            Some("tool_use") => {
                let name = block.get("name").and_then(Value::as_str).unwrap_or("?");
                let summary = tool_input_summary(block.get("input").unwrap_or(&Value::Null));
                eprintln!("◇ {name}({summary})");
            }
            _ => {}
        }
    }
}

fn print_tool_result(event: &Value) {
    let Some(content) = event
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
    else {
        return;
    };
    for block in content {
        if block.get("type").and_then(Value::as_str) == Some("tool_result") {
            let text = tool_result_text(block);
            let preview = super::truncate(&text, 400);
            let ellipsis = if preview.len() < text.len() {
                "…"
            } else {
                ""
            };
            eprintln!("  → {preview}{ellipsis}");
        }
    }
}

fn print_result(event: &Value) {
    let is_err = event
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if is_err {
        eprintln!("✗ run reported error");
    } else {
        eprintln!("✓ done");
    }
}

/// Render tool input as a one-line summary. Prefer a well-known field
/// over dumping the whole JSON object.
fn tool_input_summary(input: &Value) -> String {
    for key in [
        "file_path",
        "path",
        "command",
        "query",
        "pattern",
        "url",
        "description",
    ] {
        if let Some(v) = input.get(key).and_then(Value::as_str) {
            return format!("{key}={}", super::truncate(v, 120));
        }
    }
    let s = input.to_string();
    super::truncate(&s, 120).to_string()
}

fn tool_result_text(block: &Value) -> String {
    match block.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => {
            let mut out = String::new();
            for item in arr {
                if let Some(t) = item.get("text").and_then(Value::as_str) {
                    out.push_str(t);
                }
            }
            out
        }
        _ => String::new(),
    }
}

/// Capture a unified diff of changes in a worktree relative to HEAD,
/// filtered by the given [`DiffScope`].
///
/// Stages intent-to-add for untracked files first so that `git diff HEAD`
/// includes newly created files (not just edits to tracked ones).
fn capture_diff(worktree_dir: &Path, scope: &DiffScope) -> Option<String> {
    // Mark untracked files with intent-to-add so `git diff HEAD` sees them.
    let _ = Command::new("git")
        .current_dir(worktree_dir)
        .args(["add", "--intent-to-add", "."])
        .status();

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
///
/// After creating the worktree, symlinks gitignored directories (like
/// `redteam-history/`) so the agent can read cross-iteration state that
/// only exists in the main working tree.
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

    // Symlink gitignored directories so the agent can read them.
    #[cfg(unix)]
    for subpath in ["jolt-eval/redteam-history", "jolt-eval/optimize-history"] {
        let src = repo_dir.join(subpath);
        if src.is_dir() {
            let _ = std::os::unix::fs::symlink(&src, worktree_dir.join(subpath));
        }
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
