use std::path::Path;

use rust_code_analysis::FuncSpace;

use super::lloc::{analyze_rust_file, rust_files};
use crate::objective::{
    MeasurementError, Objective, OptimizationObjective, StaticAnalysisObjective,
};

pub const COGNITIVE_COMPLEXITY: OptimizationObjective = OptimizationObjective::StaticAnalysis(
    StaticAnalysisObjective::CognitiveComplexity(CognitiveComplexityObjective {
        target_dir: "jolt-core/src",
    }),
);

/// Average cognitive complexity per function across all Rust files under
/// a target directory.
#[derive(Clone, Copy)]
pub struct CognitiveComplexityObjective {
    pub(crate) target_dir: &'static str,
}

impl Objective for CognitiveComplexityObjective {
    type Setup = ();

    fn name(&self) -> &str {
        "cognitive_complexity_avg"
    }

    fn description(&self) -> &str {
        "Average cognitive complexity per function in jolt-core/src/"
    }

    fn setup(&self) {}

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let src_dir = repo_root.join(self.target_dir);
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
        let obj = CognitiveComplexityObjective {
            target_dir: "jolt-core/src",
        };
        let val = obj.collect_measurement().unwrap();
        assert!(val > 0.0, "avg cognitive should be > 0, got {val}");
        assert!(val < 100.0, "avg cognitive should be < 100, got {val}");
    }

    #[test]
    fn cognitive_on_single_file() {
        let source = b"fn simple() { let x = 1; }".to_vec();
        let path = Path::new("test.rs");
        let space = rust_code_analysis::get_function_spaces(
            &rust_code_analysis::LANG::Rust,
            source,
            path,
            None,
        )
        .unwrap();
        let mut total = 0.0;
        let mut count = 0;
        collect_leaf_cognitive(&space, &mut total, &mut count);
        assert_eq!(total, 0.0);
    }
}
