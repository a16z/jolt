use std::path::{Path, PathBuf};

use super::lloc::TokeiStats;
use super::{AbstractObjective, Direction, MeasurementError};

/// Comment density (comments / code) across `jolt-core/src/`.
///
/// Higher is better — more documentation relative to code.
pub struct CommentDensityObjective {
    root: PathBuf,
}

impl CommentDensityObjective {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
        }
    }
}

impl AbstractObjective for CommentDensityObjective {
    fn name(&self) -> &str {
        "comment_density"
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let TokeiStats {
            code, comments, ..
        } = super::lloc::tokei_rust_stats(&self.root.join("jolt-core/src"))?;
        if code == 0 {
            return Ok(0.0);
        }
        Ok(comments as f64 / code as f64)
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }
}
