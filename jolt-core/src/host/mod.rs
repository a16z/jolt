#![allow(clippy::type_complexity)]

use core::str::FromStr;
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::PathBuf,
    process::Command,
};

use common::{
    constants::{
        DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE, DEFAULT_MEMORY_SIZE, DEFAULT_STACK_SIZE,
    },
    jolt_device::{JoltDevice, MemoryConfig},
};
use rayon::prelude::*;
use tracer::{
    emulator::memory::Memory,
    instruction::{RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence},
};

use crate::field::JoltField;

use self::analyze::ProgramSummary;
#[cfg(not(target_arch = "wasm32"))]
use self::toolchain::{install_no_std_toolchain, install_toolchain};

pub mod analyze;
#[cfg(not(target_arch = "wasm32"))]
pub mod toolchain;

pub const DEFAULT_TARGET_DIR: &str = "/tmp/jolt-guest-targets";

#[derive(Clone)]
pub struct Program {
    guest: String,
    func: Option<String>,
    memory_size: u64,
    stack_size: u64,
    max_input_size: u64,
    max_output_size: u64,
    std: bool,
    pub elf: Option<PathBuf>,
}

impl Program {
    pub fn new(guest: &str) -> Self {
        Self {
            guest: guest.to_string(),
            func: None,
            memory_size: DEFAULT_MEMORY_SIZE,
            stack_size: DEFAULT_STACK_SIZE,
            max_input_size: DEFAULT_MAX_INPUT_SIZE,
            max_output_size: DEFAULT_MAX_OUTPUT_SIZE,
            std: false,
            elf: None,
        }
    }

    pub fn set_std(&mut self, std: bool) {
        self.std = std;
    }

    pub fn set_func(&mut self, func: &str) {
        self.func = Some(func.to_string())
    }

    pub fn set_memory_size(&mut self, len: u64) {
        self.memory_size = len;
    }

    pub fn set_stack_size(&mut self, len: u64) {
        self.stack_size = len;
    }

    pub fn set_max_input_size(&mut self, size: u64) {
        self.max_input_size = size;
    }

    pub fn set_max_output_size(&mut self, size: u64) {
        self.max_output_size = size;
    }

    #[tracing::instrument(skip_all, name = "Program::build")]
    pub fn build(&mut self, target_dir: &str) {
        if self.elf.is_none() {
            #[cfg(not(target_arch = "wasm32"))]
            install_toolchain().unwrap();
            #[cfg(not(target_arch = "wasm32"))]
            install_no_std_toolchain().unwrap();

            self.save_linker();

            let rust_flags = [
                "-C",
                &format!("link-arg=-T{}", self.linker_path()),
                "-C",
                "passes=lower-atomic",
                "-C",
                "panic=abort",
                "-C",
                "strip=symbols",
                "-C",
                "opt-level=z",
            ];

            let toolchain = if self.std {
                "riscv32im-jolt-zkvm-elf"
            } else {
                "riscv32im-unknown-none-elf"
            };

            let mut envs = vec![("CARGO_ENCODED_RUSTFLAGS", rust_flags.join("\x1f"))];

            if self.std {
                envs.push(("RUSTUP_TOOLCHAIN", toolchain.to_string()));
            }

            if let Some(func) = &self.func {
                envs.push(("JOLT_FUNC_NAME", func.to_string()));
            }

            let target = format!(
                "{}/{}-{}",
                target_dir,
                self.guest,
                self.func.as_ref().unwrap_or(&"".to_string())
            );

            let output = Command::new("cargo")
                .envs(envs)
                .args([
                    "build",
                    "--release",
                    "--features",
                    "guest",
                    "-p",
                    &self.guest,
                    "--target-dir",
                    &target,
                    "--target",
                    toolchain,
                ])
                .output()
                .expect("failed to build guest");

            if !output.status.success() {
                io::stderr().write_all(&output.stderr).unwrap();
                panic!("failed to compile guest");
            }

            let elf = format!("{}/{}/release/{}", target, toolchain, self.guest);
            self.elf = Some(PathBuf::from_str(&elf).unwrap());
        }
    }

    pub fn decode(&mut self) -> (Vec<RV32IMInstruction>, Vec<(u64, u8)>) {
        self.build(DEFAULT_TARGET_DIR);
        let elf = self.elf.as_ref().unwrap();
        let mut elf_file =
            File::open(elf).unwrap_or_else(|_| panic!("could not open elf file: {elf:?}"));
        let mut elf_contents = Vec::new();
        elf_file.read_to_end(&mut elf_contents).unwrap();
        let (mut instructions, raw_bytes) = tracer::decode(&elf_contents);
        // Expand virtual sequences
        instructions = instructions
            .into_par_iter()
            .flat_map_iter(|instr| match instr {
                RV32IMInstruction::DIV(div) => div.virtual_sequence(),
                RV32IMInstruction::DIVU(divu) => divu.virtual_sequence(),
                RV32IMInstruction::LB(lb) => lb.virtual_sequence(),
                RV32IMInstruction::LBU(lbu) => lbu.virtual_sequence(),
                RV32IMInstruction::LH(lh) => lh.virtual_sequence(),
                RV32IMInstruction::LHU(lhu) => lhu.virtual_sequence(),
                RV32IMInstruction::MULH(mulh) => mulh.virtual_sequence(),
                RV32IMInstruction::MULHSU(mulhsu) => mulhsu.virtual_sequence(),
                RV32IMInstruction::REM(rem) => rem.virtual_sequence(),
                RV32IMInstruction::REMU(remu) => remu.virtual_sequence(),
                RV32IMInstruction::SB(sb) => sb.virtual_sequence(),
                RV32IMInstruction::SH(sh) => sh.virtual_sequence(),
                RV32IMInstruction::SLL(sll) => sll.virtual_sequence(),
                RV32IMInstruction::SLLI(slli) => slli.virtual_sequence(),
                RV32IMInstruction::SRA(sra) => sra.virtual_sequence(),
                RV32IMInstruction::SRAI(srai) => srai.virtual_sequence(),
                RV32IMInstruction::SRL(srl) => srl.virtual_sequence(),
                RV32IMInstruction::SRLI(srli) => srli.virtual_sequence(),
                RV32IMInstruction::SHA256(sha256) => sha256.virtual_sequence(),
                RV32IMInstruction::SHA256INIT(sha256init) => sha256init.virtual_sequence(),
                RV32IMInstruction::KECCAK256(keccak256) => keccak256.virtual_sequence(),
                _ => vec![instr],
            })
            .collect();

        (instructions, raw_bytes)
    }

    // TODO(moodlezoup): Make this generic over InstructionSet
    #[tracing::instrument(skip_all, name = "Program::trace")]
    pub fn trace(&mut self, inputs: &[u8]) -> (Vec<RV32IMCycle>, Memory, JoltDevice) {
        self.build(DEFAULT_TARGET_DIR);
        let elf = self.elf.as_ref().unwrap();
        let mut elf_file =
            File::open(elf).unwrap_or_else(|_| panic!("could not open elf file: {elf:?}"));
        let mut elf_contents = Vec::new();
        elf_file.read_to_end(&mut elf_contents).unwrap();
        let memory_config = MemoryConfig {
            memory_size: self.memory_size,
            stack_size: self.stack_size,
            max_input_size: self.max_input_size,
            max_output_size: self.max_output_size,
        };
        tracer::trace(elf_contents, inputs, &memory_config)
    }

    pub fn trace_analyze<F: JoltField>(mut self, inputs: &[u8]) -> ProgramSummary {
        let (bytecode, init_memory_state) = self.decode();
        let (trace, _, io_device) = self.trace(inputs);

        ProgramSummary {
            trace,
            bytecode,
            memory_init: init_memory_state,
            io_device,
        }
    }

    fn save_linker(&self) {
        let linker_path = PathBuf::from_str(&self.linker_path()).unwrap();
        if let Some(parent) = linker_path.parent() {
            fs::create_dir_all(parent).expect("could not create linker file");
        }

        let linker_script = LINKER_SCRIPT_TEMPLATE
            .replace("{MEMORY_SIZE}", &self.memory_size.to_string())
            .replace("{STACK_SIZE}", &self.stack_size.to_string());

        let mut file = File::create(linker_path).expect("could not create linker file");
        file.write_all(linker_script.as_bytes())
            .expect("could not save linker");
    }

    fn linker_path(&self) -> String {
        format!("/tmp/jolt-guest-linkers/{}.ld", self.guest)
    }
}

const LINKER_SCRIPT_TEMPLATE: &str = r#"
MEMORY {
  program (rwx) : ORIGIN = 0x80000000, LENGTH = {MEMORY_SIZE}
}

SECTIONS {
  .text.boot : {
    *(.text.boot)
  } > program

  .text : {
    *(.text)
  } > program

  .data : {
    *(.data)
  } > program

  .bss : {
    *(.bss)
  } > program

  . = ALIGN(8);
  . = . + {STACK_SIZE};
  _STACK_PTR = .;
  . = ALIGN(8);
  _HEAP_PTR = .;
}
"#;
