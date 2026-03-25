#![allow(clippy::type_complexity)]

use std::path::PathBuf;

#[cfg(feature = "host")]
pub mod analyze;
#[cfg(feature = "host")]
pub mod program;

pub trait JoltProgramSource {
    fn get_elf_contents(&self) -> Option<Vec<u8>>;
    fn get_elf_compute_advice_contents(&self) -> Option<Vec<u8>>;
    fn decode(
        &mut self,
    ) -> (
        Vec<tracer::instruction::Instruction>,
        Vec<(u64, u8)>,
        u64,
        u64,
    ) {
        let elf = self.get_elf_contents().expect("ELF contents not available");
        crate::guest::program::decode(&elf)
    }
}

impl JoltProgramSource for Program {
    fn get_elf_contents(&self) -> Option<Vec<u8>> {
        Program::get_elf_contents(self)
    }

    fn get_elf_compute_advice_contents(&self) -> Option<Vec<u8>> {
        Program::get_elf_compute_advice_contents(self)
    }

    fn decode(
        &mut self,
    ) -> (
        Vec<tracer::instruction::Instruction>,
        Vec<(u64, u8)>,
        u64,
        u64,
    ) {
        Program::decode(self)
    }
}

#[derive(Clone)]
pub struct Program {
    guest: String,
    func: Option<String>,
    profile: Option<String>,
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

pub const DEFAULT_TARGET_DIR: &str = "/tmp/jolt-guest-targets";
