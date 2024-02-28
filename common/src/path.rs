#[allow(non_snake_case)]
/// For clean importing
pub mod JoltPaths {
    use std::{env, path::PathBuf};

    // Traces:    <target dir>/<TRACE_DIR_NAME>/<program name>.<TRACE_FILE_SUFFIX>
    // Programs:  <target dir>/<TRACE_DIR_NAME>/<program name>.<ELF_TRACE_FILE_SUFFIX>
    const TRACE_DIR_NAME: &'static str = "traces";
    const TRACE_FILE_SUFFIX: &'static str = "jolttrace";
    const BYTECODE_FILE_SUFFIX: &'static str = "joltbytecode";
    const CIRCUIT_DIR_NAME: &'static str = "circuits";

    fn root() -> PathBuf {
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
    }
    fn target_dir() -> PathBuf {
        // Note: ../target paths are hacks because we don't have access to the workspace root programatically
        root().join("../target").canonicalize().unwrap()
    }
    pub fn elf_path(program_name: &str) -> PathBuf {
        target_dir().join(format!(
            "riscv32i-unknown-none-elf/guest/{}",
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

    pub fn circuit_path() -> PathBuf {
        root().join("../jolt-core/src/r1cs/circuits/jolt_single_step.circom").canonicalize().unwrap()
    }

    pub fn circom_build_script_path() -> PathBuf {
        root().join("../jolt-core/src/r1cs/scripts/compile_jolt.sh").canonicalize().unwrap()
    }

    pub fn circuit_artifacts_path() -> PathBuf {
        target_dir().join(CIRCUIT_DIR_NAME)
    }

    pub fn circuit_cpp_wnts_path() -> PathBuf {
        circuit_artifacts_path().join("jolt_single_step_cpp/jolt_single_step.cpp")
    }

    pub fn circuit_dat_path() -> PathBuf {
        circuit_artifacts_path().join("jolt_single_step_cpp/jolt_single_step.dat")
    }

    pub fn r1cs_path() -> PathBuf {
        circuit_artifacts_path().join("jolt_single_step.r1cs")
    }

    pub fn witness_generator_path() -> PathBuf {
        circuit_artifacts_path().join("jolt_single_step_js/jolt_single_step.wasm")
    }
}
