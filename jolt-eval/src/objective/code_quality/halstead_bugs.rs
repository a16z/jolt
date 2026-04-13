use std::path::Path;

use rust_code_analysis::FuncSpace;

use super::lloc::{analyze_rust_file, rust_files};
use crate::objective::{
    MeasurementError, Objective, OptimizationObjective, StaticAnalysisObjective,
};

pub const HALSTEAD_BUGS: OptimizationObjective = OptimizationObjective::StaticAnalysis(
    StaticAnalysisObjective::HalsteadBugs(HalsteadBugsObjective {
        target_dir: "jolt-core/src",
    }),
);

/// Estimated number of delivered bugs across all Rust files under
/// a target directory, based on Halstead's bug prediction formula
/// (B = V / 3000, where V is program volume).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct HalsteadBugsObjective {
    pub(crate) target_dir: &'static str,
}

impl HalsteadBugsObjective {
    pub fn collect_measurement_in(&self, repo_root: &Path) -> Result<f64, MeasurementError> {
        let src_dir = repo_root.join(self.target_dir);
        let mut total = 0.0;
        for path in rust_files(&src_dir)? {
            if let Some(space) = analyze_rust_file(&path) {
                total += sum_bugs(&space);
            }
        }
        Ok(total)
    }
}

impl Objective for HalsteadBugsObjective {
    type Setup = ();

    fn name(&self) -> &str {
        "halstead_bugs"
    }

    fn description(&self) -> String {
        format!(
            "Estimated delivered bugs (Halstead volume / 3000) in {}",
            self.target_dir
        )
    }

    fn setup(&self) {}

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        self.collect_measurement_in(repo_root)
    }
}

fn sum_bugs(space: &FuncSpace) -> f64 {
    let b = space.metrics.halstead.bugs();
    let mut total = if b.is_finite() { b } else { 0.0 };
    for child in &space.spaces {
        total += sum_bugs(child);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halstead_bugs_on_jolt_core() {
        let obj = HalsteadBugsObjective {
            target_dir: "jolt-core/src",
        };
        let val = obj.collect_measurement().unwrap();
        assert!(val > 0.0, "halstead bugs should be > 0, got {val}");
    }

    #[test]
    fn halstead_bugs_on_trivial_code() {
        let source = b"fn f() { let x = 1 + 2; }".to_vec();
        let path = Path::new("test.rs");
        let space = rust_code_analysis::get_function_spaces(
            &rust_code_analysis::LANG::Rust,
            source,
            path,
            None,
        )
        .unwrap();
        let bugs = sum_bugs(&space);
        assert!(bugs < 1.0, "trivial code bugs should be < 1, got {bugs}");
    }
}
