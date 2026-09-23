//! Shared guest-build helper for the integration tests.
//!
//! Guest ELFs are produced by the `jolt` CLI, which is not present in every
//! CI job. `setup` returns `None` when the CLI cannot be spawned so that
//! guest-dependent tests skip instead of failing; a build that starts and
//! then fails is a real error and panics. The `x86-tracer` workflow job
//! installs the CLI, so the skip path never hides a regression there.

#![expect(clippy::expect_used, clippy::panic, clippy::print_stderr)]

use jolt_program::execution::{JoltProgram, TraceInputs};

/// Build a guest and wrap it in a `JoltProgram` plus `TraceInputs`.
///
/// `None` means the `jolt` CLI is unavailable and the caller should skip.
pub fn setup(package: &str, func: &str, input: Vec<u8>) -> Option<(JoltProgram, TraceInputs)> {
    let elf = guest_elf(package, func)?;
    let mut provider = tracer::TracerInlineExpansionProvider::new();
    let program = jolt_program::build_jolt_program_with_inline_provider(
        &elf,
        &mut provider,
        jolt_riscv::RV64IMAC_JOLT_ALL_INLINES,
    )
    .expect("failed to build Jolt program");
    let memory_config = common::jolt_device::MemoryConfig {
        program_size: Some(program.program_end - common::constants::RAM_START_ADDRESS),
        ..Default::default()
    };
    let inputs = TraceInputs::new(input, Vec::new(), Vec::new(), memory_config);
    Some((program, inputs))
}

fn guest_elf(package: &str, func: &str) -> Option<Vec<u8>> {
    let target_dir = format!("/tmp/jolt-guest-targets/{package}-{func}");
    let output = match std::process::Command::new("jolt")
        .args([
            "build",
            "-p",
            package,
            "--stack-size",
            &common::constants::DEFAULT_STACK_SIZE.to_string(),
            "--heap-size",
            &common::constants::DEFAULT_HEAP_SIZE.to_string(),
            "--",
            "--release",
            "--target-dir",
            &target_dir,
            "--features",
            "guest",
        ])
        .env("JOLT_FUNC_NAME", func)
        .output()
    {
        Ok(output) => output,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "SKIP: the `jolt` CLI is not installed, so guest `{package}` cannot be built. \
                 Install it with `cargo install --path .` to run this test."
            );
            return None;
        }
        Err(e) => panic!("failed to run the `jolt` CLI: {e}"),
    };
    assert!(
        output.status.success(),
        "failed to build {package}:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let elf_path = format!("{target_dir}/riscv64imac-unknown-none-elf/release/{package}");
    Some(
        std::fs::read(&elf_path)
            .unwrap_or_else(|e| panic!("failed to read guest ELF at {elf_path}: {e}")),
    )
}
