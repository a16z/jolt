//! Host-side guest construction and execution.

pub mod analyze;
#[cfg(feature = "field-inline")]
pub mod field_inline;
mod program;

use std::io::Error as IoError;
use std::path::PathBuf;
use std::sync::Arc;

use common::constants::RAM_START_ADDRESS;
use jolt_program::{JoltProgram, ProgramError};
use jolt_riscv::{JoltInstructionProfile, JoltInstructionRow, RV64IMAC_JOLT_ALL_INLINES};
use tracer::TracerInlineExpansionProvider;

/// Guest build cache used when no target dir is given: `jolt-guest-targets`
/// under the system temp dir (`TMPDIR`, else `/tmp`).
pub fn default_target_dir() -> PathBuf {
    std::env::temp_dir().join("jolt-guest-targets")
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
