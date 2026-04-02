use std::path::{Path, PathBuf};
use std::process::Command;

use super::{AbstractObjective, Direction, MeasurementError};

/// Total lines of Rust code (excluding comments and blanks) across
/// `jolt-core/src/`, as reported by `tokei`.
pub struct LlocObjective {
    root: PathBuf,
}

impl LlocObjective {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
        }
    }
}

impl AbstractObjective for LlocObjective {
    fn name(&self) -> &str {
        "lloc"
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let stats = tokei_rust_stats(&self.root.join("jolt-core/src"))?;
        Ok(stats.code as f64)
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }
}

pub(crate) struct TokeiStats {
    pub code: u64,
    pub comments: u64,
}

/// Run `tokei --type Rust -o json` on a directory and parse the result.
pub(crate) fn tokei_rust_stats(dir: &Path) -> Result<TokeiStats, MeasurementError> {
    let output = Command::new("tokei")
        .arg(dir)
        .args(["--type", "Rust", "-o", "json"])
        .output()
        .map_err(|e| {
            MeasurementError::new(format!(
                "tokei: {e}. Install via: cargo install tokei"
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MeasurementError::new(format!("tokei failed: {stderr}")));
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| MeasurementError::new(format!("tokei JSON parse: {e}")))?;

    let rust = json
        .get("Rust")
        .ok_or_else(|| MeasurementError::new("no Rust section in tokei output"))?;

    Ok(TokeiStats {
        code: rust["code"].as_u64().unwrap_or(0),
        comments: rust["comments"].as_u64().unwrap_or(0),
    })
}
