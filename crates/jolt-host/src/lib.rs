//! Host-side guest construction and execution.

pub mod analyze;
#[cfg(feature = "field-inline")]
pub mod field_inline;
mod program;

use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Error as IoError;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use common::constants::RAM_START_ADDRESS;
use jolt_program::{JoltProgram, ProgramError};
use jolt_riscv::{JoltInstructionProfile, JoltInstructionRow, RV64IMAC_JOLT_ALL_INLINES};
use serde::Deserialize;
use tracer::TracerInlineExpansionProvider;

#[derive(Deserialize)]
struct CargoMetadata {
    workspace_root: PathBuf,
    target_directory: PathBuf,
}

impl CargoMetadata {
    fn guest_target_dir(&self) -> PathBuf {
        let mut workspace = DefaultHasher::new();
        self.workspace_root.hash(&mut workspace);
        self.target_directory
            .join("jolt-guest-targets")
            .join(format!("{:016x}", workspace.finish()))
    }
}

/// Guest build cache used when no target dir is given, for the cargo workspace
/// that contains the current directory: a directory named after a hash of the
/// workspace root, under `jolt-guest-targets` in the workspace's cargo target
/// directory.
///
/// A guest target dir must serve one workspace root only. Cargo names a
/// workspace member's units by its workspace-relative path and rebuilds one
/// only when a source file is newer than the cached unit, so a second checkout
/// building into the same directory can be handed the first checkout's ELF.
/// The root is part of the path because cargo configuration can point several
/// checkouts at one target directory.
///
/// # Panics
///
/// If `cargo metadata` fails, as it does outside a cargo workspace.
#[expect(
    clippy::expect_used,
    reason = "the infallible host build API reports cargo failures by panicking"
)]
pub fn default_target_dir() -> PathBuf {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()
        .expect("failed to run cargo metadata");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: CargoMetadata =
        serde_json::from_slice(&output.stdout).expect("parse the cargo metadata output");
    metadata.guest_target_dir()
}

#[derive(Clone)]
pub struct Program {
    guest: String,
    func: Option<String>,
    profile: Option<String>,
    instruction_profile: JoltInstructionProfile,
    guest_features: Vec<String>,
    heap_size: u64,
    stack_size: u64,
    max_input_size: u64,
    max_untrusted_advice_size: u64,
    max_trusted_advice_size: u64,
    max_output_size: u64,
    std: bool,
    backtrace: Option<String>,
    pub elf: Option<GuestElf>,
    pub elf_compute_advice: Option<GuestElf>,
}

/// A guest ELF and the bytes read from it once.
///
/// Guest builds replace the file at `path` even when nothing was recompiled
/// (cargo removes and re-links its uplifted binary), so the bytes are read a
/// single time and every later consumer uses them instead of the path.
#[derive(Clone)]
pub struct GuestElf {
    path: PathBuf,
    bytes: Arc<[u8]>,
}

impl GuestElf {
    pub fn read(path: PathBuf) -> Result<Self, IoError> {
        let bytes = std::fs::read(&path)?.into();
        Ok(Self { path, bytes })
    }

    /// Where the ELF was read from; the tracer uses it only to symbolize
    /// guest panics.
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// An ELF-backed source accepted by SDK-generated preprocessing and proving APIs.
pub trait JoltProgramSource {
    fn get_elf_contents(&self) -> Option<Vec<u8>>;
    fn get_elf_compute_advice_contents(&self) -> Option<Vec<u8>>;

    fn instruction_profile(&self) -> JoltInstructionProfile {
        RV64IMAC_JOLT_ALL_INLINES
    }

    #[expect(
        clippy::expect_used,
        reason = "the source trait preserves the infallible missing-ELF contract"
    )]
    fn build_jolt_program(&self) -> Result<JoltProgram, ProgramError> {
        let elf = self.get_elf_contents().expect("ELF contents not available");
        let mut inline_provider = TracerInlineExpansionProvider::new();
        jolt_program::build_jolt_program_with_inline_provider(
            &elf,
            &mut inline_provider,
            self.instruction_profile(),
        )
    }

    #[expect(
        clippy::expect_used,
        reason = "the source trait preserves the infallible decode contract"
    )]
    fn decode(&mut self) -> (Vec<JoltInstructionRow>, Vec<(u64, u8)>, u64, u64) {
        let program = self
            .build_jolt_program()
            .expect("failed to build Jolt program");
        (
            program.expanded_bytecode,
            program.memory_init,
            program.program_end - RAM_START_ADDRESS,
            program.entry_address,
        )
    }
}

impl JoltProgramSource for Program {
    fn get_elf_contents(&self) -> Option<Vec<u8>> {
        Program::get_elf_contents(self)
    }

    fn get_elf_compute_advice_contents(&self) -> Option<Vec<u8>> {
        Program::get_elf_compute_advice_contents(self)
    }

    fn instruction_profile(&self) -> JoltInstructionProfile {
        Program::instruction_profile(self)
    }

    fn decode(&mut self) -> (Vec<JoltInstructionRow>, Vec<(u64, u8)>, u64, u64) {
        Program::decode(self)
    }
}

#[cfg(test)]
mod tests {
    use super::CargoMetadata;
    use std::path::{Path, PathBuf};

    #[test]
    fn workspace_roots_sharing_a_cargo_target_dir_get_distinct_guest_dirs() {
        let guest_target_dir = |workspace_root: &str| {
            CargoMetadata {
                workspace_root: PathBuf::from(workspace_root),
                target_directory: PathBuf::from("/shared/target"),
            }
            .guest_target_dir()
        };
        let first = guest_target_dir("/checkouts/first");
        let second = guest_target_dir("/checkouts/second");

        assert_ne!(first, second);
        // CI jobs that save a cache of target/ delete this directory by name.
        let cache_root = Some(Path::new("/shared/target/jolt-guest-targets"));
        assert_eq!(first.parent(), cache_root);
        assert_eq!(second.parent(), cache_root);
    }
}
