use std::path::{Path, PathBuf};

use rust_code_analysis::{get_function_spaces, FuncSpace, LANG};

use super::{AbstractObjective, Direction, MeasurementError};

/// Total logical lines of code (LLOC) across all Rust files under
/// `jolt-core/src/`.
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
        let src_dir = self.root.join("jolt-core/src");
        let mut total = 0.0;
        for path in rust_files(&src_dir)? {
            if let Some(space) = analyze_rust_file(&path) {
                total += space.metrics.loc.lloc();
            }
        }
        Ok(total)
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lloc_on_jolt_core() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let obj = LlocObjective::new(root);
        let val = obj.collect_measurement().unwrap();
        assert!(val > 1000.0, "LLOC should be > 1000, got {val}");
    }

    #[test]
    fn lloc_on_inline_source() {
        let source = b"fn f() { let x = 1; let y = 2; }".to_vec();
        let path = Path::new("test.rs");
        let space = get_function_spaces(&LANG::Rust, source, path, None).unwrap();
        let lloc = space.metrics.loc.lloc();
        assert!(lloc >= 2.0, "two statements should give lloc >= 2, got {lloc}");
    }

    #[test]
    fn rust_files_finds_rs_files() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let files = rust_files(&src).unwrap();
        assert!(!files.is_empty());
        assert!(files.iter().all(|f| f.extension().unwrap() == "rs"));
    }
}
