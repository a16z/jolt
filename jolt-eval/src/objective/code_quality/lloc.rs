use std::path::{Path, PathBuf};

use rust_code_analysis::{get_function_spaces, FuncSpace, LANG};

use crate::objective::code_quality::PROOF_SYSTEM_CRATE_DIRS;
use crate::objective::{
    MeasurementError, Objective, OptimizationObjective, StaticAnalysisObjective,
};

pub const LLOC: OptimizationObjective =
    OptimizationObjective::StaticAnalysis(StaticAnalysisObjective::Lloc(LlocObjective {
        crate_dirs: PROOF_SYSTEM_CRATE_DIRS,
    }));

/// Total logical lines of code (LLOC) across all Rust files under
/// the modular proof system.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LlocObjective {
    pub(crate) crate_dirs: &'static [&'static str],
}

impl LlocObjective {
    pub fn collect_measurement_in(&self, repo_root: &Path) -> Result<f64, MeasurementError> {
        let mut total = 0.0;
        for path in rust_files_in_crates(repo_root, self.crate_dirs)? {
            if let Some(space) = analyze_rust_file(&path) {
                total += space.metrics.loc.lloc();
            }
        }
        Ok(total)
    }
}

impl Objective for LlocObjective {
    type Setup = ();

    fn name(&self) -> &str {
        "lloc"
    }

    fn description(&self) -> String {
        "Total logical lines of code in the modular proof system".to_string()
    }

    fn setup(&self) {}

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        self.collect_measurement_in(repo_root)
    }

    fn units(&self) -> Option<&str> {
        Some("lines")
    }
}

pub(crate) fn rust_files_in_crates(
    repo_root: &Path,
    crate_dirs: &[&str],
) -> Result<Vec<PathBuf>, MeasurementError> {
    let mut files = Vec::new();
    for crate_dir in crate_dirs {
        files.extend(rust_files(&repo_root.join(crate_dir).join("src"))?);
    }
    Ok(files)
}

pub(crate) fn rust_files(dir: &Path) -> Result<Vec<PathBuf>, MeasurementError> {
    let mut files = Vec::new();
    walk_rust_files(dir, &mut files)
        .map_err(|e| MeasurementError::new(format!("walking {}: {e}", dir.display())))?;
    Ok(files)
}

fn walk_rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_rust_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

pub(crate) fn analyze_rust_file(path: &Path) -> Option<FuncSpace> {
    let source = std::fs::read(path).ok()?;
    get_function_spaces(&LANG::Rust, source, path, None)
}
