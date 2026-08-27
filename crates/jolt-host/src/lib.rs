//! Host-side guest construction and execution.

pub mod analyze;
mod program;

use std::path::PathBuf;

use jolt_riscv::{JoltInstructionProfile, JoltInstructionRow, RV64IMAC_JOLT_ALL_INLINES};

pub const DEFAULT_TARGET_DIR: &str = "/tmp/jolt-guest-targets";

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
    pub elf: Option<PathBuf>,
    pub elf_compute_advice: Option<PathBuf>,
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
    fn build_jolt_program(&self) -> Result<jolt_program::JoltProgram, jolt_program::ProgramError> {
        let elf = self.get_elf_contents().expect("ELF contents not available");
        let mut inline_provider = tracer::TracerInlineExpansionProvider::new();
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
            program.program_end - common::constants::RAM_START_ADDRESS,
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
