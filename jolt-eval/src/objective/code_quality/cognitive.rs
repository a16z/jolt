use std::path::{Path, PathBuf};

use rust_code_analysis::FuncSpace;

use super::lloc::{analyze_rust_file, rust_files};
use crate::objective::{AbstractObjective, MeasurementError};

/// Average cognitive complexity per function across all Rust files under
/// `jolt-core/src/`.
///
/// Cognitive complexity measures how difficult code is to understand,
/// penalizing deeply nested control flow, recursion, and breaks in
/// linear flow. Lower is better.
pub struct CognitiveComplexityObjective {
    root: PathBuf,
}

impl CognitiveComplexityObjective {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
        }
    }
}

impl AbstractObjective for CognitiveComplexityObjective {
    fn name(&self) -> &str {
        "cognitive_complexity_avg"
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let src_dir = self.root.join("jolt-core/src");
        let mut total = 0.0;
        let mut count = 0usize;
        for path in rust_files(&src_dir)? {
            if let Some(space) = analyze_rust_file(&path) {
                collect_leaf_cognitive(&space, &mut total, &mut count);
            }
        }
        if count == 0 {
            return Ok(0.0);
        }
        Ok(total / count as f64)
    }

}

/// Walk the function-space tree and collect cognitive complexity from
/// leaf functions (functions with no child spaces).
fn collect_leaf_cognitive(space: &FuncSpace, total: &mut f64, count: &mut usize) {
    if space.spaces.is_empty() {
        let c = space.metrics.cognitive.cognitive();
        if c > 0.0 {
            *total += c;
            *count += 1;
        }
    } else {
        for child in &space.spaces {
            collect_leaf_cognitive(child, total, count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cognitive_on_jolt_core() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let obj = CognitiveComplexityObjective::new(root);
        let val = obj.collect_measurement().unwrap();
        assert!(val > 0.0, "avg cognitive should be > 0, got {val}");
        assert!(val < 100.0, "avg cognitive should be < 100, got {val}");
    }

    #[test]
    fn cognitive_on_single_file() {
        let source = b"fn simple() { let x = 1; }".to_vec();
        let path = Path::new("test.rs");
        let space =
            rust_code_analysis::get_function_spaces(&rust_code_analysis::LANG::Rust, source, path, None)
                .unwrap();
        let mut total = 0.0;
        let mut count = 0;
        collect_leaf_cognitive(&space, &mut total, &mut count);
        // A straight-line function has 0 cognitive complexity
        assert_eq!(total, 0.0);
    }
}
