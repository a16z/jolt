use std::path::{Path, PathBuf};

use rust_code_analysis::FuncSpace;

use super::lloc::{analyze_rust_file, rust_files};
use super::{AbstractObjective, Direction, MeasurementError};

/// Estimated number of delivered bugs across all Rust files under
/// `jolt-core/src/`, based on Halstead's bug prediction formula
/// (B = V / 3000, where V is program volume).
///
/// Lower is better.
pub struct HalsteadBugsObjective {
    root: PathBuf,
}

impl HalsteadBugsObjective {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
        }
    }
}

impl AbstractObjective for HalsteadBugsObjective {
    fn name(&self) -> &str {
        "halstead_bugs"
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let src_dir = self.root.join("jolt-core/src");
        let mut total = 0.0;
        for path in rust_files(&src_dir)? {
            if let Some(space) = analyze_rust_file(&path) {
                total += sum_bugs(&space);
            }
        }
        Ok(total)
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }
}

/// Sum Halstead bugs across all function spaces in the tree,
/// skipping NaN values (empty functions produce 0/0).
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
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let obj = HalsteadBugsObjective::new(root);
        let val = obj.collect_measurement().unwrap();
        assert!(val > 0.0, "halstead bugs should be > 0, got {val}");
    }

    #[test]
    fn halstead_bugs_on_trivial_code() {
        let source = b"fn f() { let x = 1 + 2; }".to_vec();
        let path = Path::new("test.rs");
        let space =
            rust_code_analysis::get_function_spaces(&rust_code_analysis::LANG::Rust, source, path, None)
                .unwrap();
        let bugs = sum_bugs(&space);
        // Trivial code should have very low estimated bugs
        assert!(bugs < 1.0, "trivial code bugs should be < 1, got {bugs}");
    }
}
