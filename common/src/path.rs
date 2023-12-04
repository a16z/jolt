#[allow(non_snake_case)]
/// For clean importing
pub mod JoltPaths {
    use std::{env, path::PathBuf};

    // Traces:    <target dir>/<TRACE_DIR_NAME>/<program name>.<TRACE_FILE_SUFFIX>
    // Programs:  <target dir>/<TRACE_DIR_NAME>/<program name>.<ELF_TRACE_FILE_SUFFIX>
    const TRACE_DIR_NAME: &'static str = "traces";
    const TRACE_FILE_SUFFIX: &'static str = "jolttrace";
    const BYTECODE_FILE_SUFFIX: &'static str = "joltbytecode";

    fn root() -> PathBuf {
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
    }
    fn target_dir() -> PathBuf {
        // Note: ../target paths are hacks because we don't have access to the workspace root programatically
        root().join("../target").canonicalize().unwrap()
    }
    pub fn elf_path(program_name: &str) -> PathBuf {
        target_dir().join(format!(
            "riscv32i-unknown-none-elf/release/{}",
            program_name
        ))
    }

    pub fn trace_path(program_name: &str) -> PathBuf {
        target_dir()
            .join(TRACE_DIR_NAME)
            .join(format!("{}.{}", program_name, TRACE_FILE_SUFFIX))
    }

    pub fn bytecode_path(program_name: &str) -> PathBuf {
        target_dir()
            .join(TRACE_DIR_NAME)
            .join(format!("{}.{}", program_name, BYTECODE_FILE_SUFFIX))
    }
}
