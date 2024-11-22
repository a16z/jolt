#![allow(clippy::type_complexity)]

use core::str::FromStr;
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::PathBuf,
    process::Command,
};

use postcard;
use rayon::prelude::*;
use serde::Serialize;

use common::{
    constants::{
        DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE, DEFAULT_MEMORY_SIZE, DEFAULT_STACK_SIZE,
    },
    rv_trace::JoltDevice,
};
pub use tracer::ELFInstruction;

use crate::{
    field::JoltField,
    jolt::{
        instruction::{
            div::DIVInstruction, divu::DIVUInstruction, lb::LBInstruction, lbu::LBUInstruction,
            lh::LHInstruction, lhu::LHUInstruction, mulh::MULHInstruction,
            mulhsu::MULHSUInstruction, rem::REMInstruction, remu::REMUInstruction,
            sb::SBInstruction, sh::SHInstruction, VirtualInstructionSequence,
        },
        vm::{bytecode::BytecodeRow, rv32i_vm::RV32I, JoltTraceStep},
    },
};

use self::analyze::ProgramSummary;
#[cfg(not(target_arch = "wasm32"))]
use self::toolchain::{install_no_std_toolchain, install_toolchain};

pub mod analyze;
#[cfg(not(target_arch = "wasm32"))]
pub mod toolchain;

#[derive(Clone)]
pub struct Program {
    guest: String,
    func: Option<String>,
    input: Vec<u8>,
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
            input: Vec::new(),
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

    pub fn set_input<T: Serialize>(&mut self, input: &T) {
        let mut serialized = postcard::to_stdvec(input).unwrap();
        self.input.append(&mut serialized);
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
    pub fn build(&mut self) {
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
                "/tmp/jolt-guest-target-{}-{}",
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

    pub fn decode(&mut self) -> (Vec<ELFInstruction>, Vec<(u64, u8)>) {
        self.build();
        let elf = self.elf.as_ref().unwrap();
        let mut elf_file =
            File::open(elf).unwrap_or_else(|_| panic!("could not open elf file: {:?}", elf));
        let mut elf_contents = Vec::new();
        elf_file.read_to_end(&mut elf_contents).unwrap();
        tracer::decode(&elf_contents)
    }

    // TODO(moodlezoup): Make this generic over InstructionSet
    #[tracing::instrument(skip_all, name = "Program::trace")]
    pub fn trace(&mut self) -> (JoltDevice, Vec<JoltTraceStep<RV32I>>) {
        self.build();
        let elf = self.elf.clone().unwrap();
        let (raw_trace, io_device) =
            tracer::trace(&elf, &self.input, self.max_input_size, self.max_output_size);

        let trace: Vec<_> = raw_trace
            .into_par_iter()
            .flat_map(|row| match row.instruction.opcode {
                tracer::RV32IM::MULH => MULHInstruction::<32>::virtual_trace(row),
                tracer::RV32IM::MULHSU => MULHSUInstruction::<32>::virtual_trace(row),
                tracer::RV32IM::DIV => DIVInstruction::<32>::virtual_trace(row),
                tracer::RV32IM::DIVU => DIVUInstruction::<32>::virtual_trace(row),
                tracer::RV32IM::REM => REMInstruction::<32>::virtual_trace(row),
                tracer::RV32IM::REMU => REMUInstruction::<32>::virtual_trace(row),
                tracer::RV32IM::SH => SHInstruction::<32>::virtual_trace(row),
                tracer::RV32IM::SB => SBInstruction::<32>::virtual_trace(row),
                tracer::RV32IM::LBU => LBUInstruction::<32>::virtual_trace(row),
                tracer::RV32IM::LHU => LHUInstruction::<32>::virtual_trace(row),
                tracer::RV32IM::LB => LBInstruction::<32>::virtual_trace(row),
                tracer::RV32IM::LH => LHInstruction::<32>::virtual_trace(row),
                _ => vec![row],
            })
            .map(|row| {
                let instruction_lookup = if let Ok(jolt_instruction) = RV32I::try_from(&row) {
                    Some(jolt_instruction)
                } else {
                    // Instruction does not use lookups
                    None
                };

                JoltTraceStep {
                    instruction_lookup,
                    bytecode_row: BytecodeRow::from_instruction::<RV32I>(&row.instruction),
                    memory_ops: (&row).into(),
                    circuit_flags: row.instruction.to_circuit_flags(),
                }
            })
            .collect();

        (io_device, trace)
    }

    pub fn trace_analyze<F: JoltField>(mut self) -> ProgramSummary {
        self.build();
        let elf = self.elf.as_ref().unwrap();
        let (raw_trace, _) =
            tracer::trace(elf, &self.input, self.max_input_size, self.max_output_size);

        let (bytecode, memory_init) = self.decode();
        let (io_device, processed_trace) = self.trace();

        ProgramSummary {
            raw_trace,
            bytecode,
            memory_init,
            io_device,
            processed_trace,
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
