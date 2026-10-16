use std::path::Path;

use rust_code_analysis::FuncSpace;

use super::lloc::{analyze_rust_file, rust_files_in_crates};
use super::PROOF_SYSTEM_CRATE_DIRS;
use crate::objective::{
    MeasurementError, Objective, OptimizationObjective, StaticAnalysisObjective,
};

pub const COGNITIVE_COMPLEXITY: OptimizationObjective = OptimizationObjective::StaticAnalysis(
    StaticAnalysisObjective::CognitiveComplexity(CognitiveComplexityObjective {
        crate_dirs: PROOF_SYSTEM_CRATE_DIRS,
    }),
);

/// Average cognitive complexity per function across all Rust files under
/// the modular proof system.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CognitiveComplexityObjective {
    pub(crate) crate_dirs: &'static [&'static str],
}

impl CognitiveComplexityObjective {
    pub fn collect_measurement_in(&self, repo_root: &Path) -> Result<f64, MeasurementError> {
        let mut total = 0.0;
        let mut count = 0usize;
        for path in rust_files_in_crates(repo_root, self.crate_dirs)? {
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

impl Objective for CognitiveComplexityObjective {
    type Setup = ();

    fn name(&self) -> &str {
        "cognitive_complexity_avg"
    }

    fn description(&self) -> String {
        "Average cognitive complexity per function in the modular proof system".to_string()
    }

    fn setup(&self) {}

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        self.collect_measurement_in(repo_root)
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
