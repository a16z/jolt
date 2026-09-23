//! Shared support for the trace_* example harnesses: guest builds via the
//! jolt CLI, mirroring `jolt_host::Program`'s defaults
//! (no-std, backtrace off, default memory layout, `--release`, feature
//! `guest`) so the two paths produce byte-identical guest ELFs and reuse the
//! same cached builds under /tmp/jolt-guest-targets.

use common::constants::{
    DEFAULT_HEAP_SIZE, DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE,
    DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE, DEFAULT_STACK_SIZE,
    RAM_START_ADDRESS,
};
use common::jolt_device::MemoryConfig;
use jolt_riscv::RV64IMAC_JOLT_ALL_INLINES;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_TARGET_DIR: &str = "/tmp/jolt-guest-targets";

/// Build `package` with the jolt CLI (`JOLT_PATH` overrides the binary) and
/// return its ELF bytes, ELF path, and the memory config the tracer runs it
/// with — the counterpart of `host::Program::new(package)` + `trace`'s
/// internal setup in `jolt-host`.
pub fn build_guest(package: &str) -> (Vec<u8>, PathBuf, MemoryConfig) {
    let jolt_cmd = std::env::var("JOLT_PATH").unwrap_or_else(|_| "jolt".to_string());
    // Same layout as host::Program::build_with_features; the trailing '-' is
    // its `{guest}-{func}` naming with func unset.
    let guest_target_dir = format!("{DEFAULT_TARGET_DIR}/{package}-");
    let output = Command::new(&jolt_cmd)
        .args([
            "build",
            "-p",
            package,
            "--backtrace",
            "off",
            "--stack-size",
            &DEFAULT_STACK_SIZE.to_string(),
            "--heap-size",
            &DEFAULT_HEAP_SIZE.to_string(),
            "--",
            "--release",
            "--target-dir",
            &guest_target_dir,
            "--features",
            "guest",
        ])
        .output()
        .expect("failed to run jolt - make sure it's installed (cargo install --path .)");
    if !output.status.success() {
        panic!(
            "failed to compile guest {package} with jolt:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let elf_path = PathBuf::from(&guest_target_dir)
        .join("riscv64imac-unknown-none-elf/release")
        .join(package);
    let (elf, memory_config) = load_guest(&elf_path);
    (elf, elf_path, memory_config)
}

/// Read a built guest ELF and derive its memory config — the per-trace setup
/// `host::Program::trace`/`execute` repeated on every call. trace_bench keeps
/// this inside its timed region so reported MHz stay comparable with numbers
/// measured through the legacy harness.
pub fn load_guest(elf_path: &Path) -> (Vec<u8>, MemoryConfig) {
    let elf = std::fs::read(elf_path)
        .unwrap_or_else(|err| panic!("built ELF not found at {}: {err}", elf_path.display()));
    let image = jolt_program::image::decode_elf(&elf, RV64IMAC_JOLT_ALL_INLINES)
        .expect("program ELF decoding failed");
    let memory_config = MemoryConfig {
        heap_size: DEFAULT_HEAP_SIZE,
        stack_size: DEFAULT_STACK_SIZE,
        max_input_size: DEFAULT_MAX_INPUT_SIZE,
        max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
        max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
        max_output_size: DEFAULT_MAX_OUTPUT_SIZE,
        program_size: Some(image.program_end - RAM_START_ADDRESS),
    };
    (elf, memory_config)
}

/// Input encoding for the `(input: [u8; 32], num_iters: u32)` chain guests.
pub fn chain_input(iters: u32) -> Vec<u8> {
    let mut input = postcard::to_stdvec(&[5u8; 32]).unwrap();
    input.extend(postcard::to_stdvec(&iters).unwrap());
    input
}
