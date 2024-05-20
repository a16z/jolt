#![allow(clippy::type_complexity)]

use core::{str::FromStr, u8};
use std::{
    fs::{self, File},
    io::{self, Write},
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
    rv_trace::{JoltDevice, NUM_CIRCUIT_FLAGS},
};
use strum::EnumCount;
use tracer::ELFInstruction;

use crate::{
    jolt::{
        instruction::{mulh::MULHInstruction, VirtualInstructionSequence},
        vm::{bytecode::BytecodeRow, rv32i_vm::RV32I, JoltTraceStep},
    },
    poly::field::JoltField,
    utils::thread::unsafe_allocate_zero_vec,
};

use self::{analyze::ProgramSummary, toolchain::install_toolchain};

pub mod analyze;
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
            install_toolchain().unwrap();
            self.save_linker();

            let rust_flags = [
                "-C",
                &format!("link-arg=-T{}", self.linker_path()),
                "-C",
                "passes=loweratomic",
                "-C",
                "panic=abort",
            ];

            let toolchain = "riscv32i-jolt-zkvm-elf";
            let mut envs = vec![
                ("CARGO_ENCODED_RUSTFLAGS", rust_flags.join("\x1f")),
                ("RUSTUP_TOOLCHAIN", toolchain.to_string()),
            ];

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
                    "--bin",
                    "guest",
                ])
                .output()
                .expect("failed to build guest");

            if !output.status.success() {
                io::stderr().write_all(&output.stderr).unwrap();
                panic!("failed to compile guest");
            }

            let elf = format!("{}/{}/release/guest", target, toolchain);
            self.elf = Some(PathBuf::from_str(&elf).unwrap());
        }
    }

    pub fn decode(&mut self) -> (Vec<ELFInstruction>, Vec<(u64, u8)>) {
        self.build();
        let elf = self.elf.as_ref().unwrap();
        tracer::decode(elf)
    }

    // TODO(moodlezoup): Make this generic over InstructionSet
    #[tracing::instrument(skip_all, name = "Program::trace")]
    pub fn trace<F: JoltField>(mut self) -> (JoltDevice, Vec<JoltTraceStep<RV32I>>, Vec<F>) {
        self.build();
        let elf = self.elf.unwrap();
        let (raw_trace, io_device) =
            tracer::trace(&elf, &self.input, self.max_input_size, self.max_output_size);

        let trace: Vec<_> = raw_trace
            .into_par_iter()
            .flat_map(|row| match row.instruction.opcode {
                tracer::RV32IM::MULH => MULHInstruction::<32>::virtual_sequence(row),
                tracer::RV32IM::MULHSU => todo!(),
                tracer::RV32IM::DIV => todo!(),
                tracer::RV32IM::DIVU => todo!(),
                tracer::RV32IM::REM => todo!(),
                tracer::RV32IM::REMU => todo!(),
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
                }
            })
            .collect();
        let padded_trace_len = trace.len().next_power_of_two();

        let mut circuit_flag_trace = unsafe_allocate_zero_vec(padded_trace_len * NUM_CIRCUIT_FLAGS);
        circuit_flag_trace
            .par_chunks_mut(padded_trace_len)
            .enumerate()
            .for_each(|(flag_index, chunk)| {
                chunk.iter_mut().zip(trace.iter()).for_each(|(flag, row)| {
                    let packed_circuit_flags = row.bytecode_row.bitflags >> RV32I::COUNT;
                    // Check if the flag is set in the packed representation
                    if (packed_circuit_flags >> (NUM_CIRCUIT_FLAGS - flag_index - 1)) & 1 != 0 {
                        *flag = F::one();
                    }
                });
            });

        (io_device, trace, circuit_flag_trace)
    }

    pub fn trace_analyze<F: JoltField>(mut self) -> ProgramSummary {
        self.build();
        let elf = self.elf.as_ref().unwrap();
        let (raw_trace, _) =
            tracer::trace(elf, &self.input, self.max_input_size, self.max_output_size);

        let (bytecode, memory_init) = self.decode();
        let (io_device, processed_trace, circuit_flags) = self.trace();
        let circuit_flags: Vec<bool> = circuit_flags
            .into_iter()
            .map(|flag: F| flag.is_one())
            .collect();

        ProgramSummary {
            raw_trace,
            bytecode,
            memory_init,
            io_device,
            processed_trace,
            circuit_flags,
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
