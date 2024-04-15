use core::{str::FromStr, u8};
use std::{
    fs::{self, File},
    io::{self, Write},
    path::PathBuf,
    process::Command,
};

use ark_ff::PrimeField;
use postcard;
use rayon::prelude::*;
use serde::Serialize;

use common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{JoltDevice, MemoryOp, NUM_CIRCUIT_FLAGS},
};
use tracer::ELFInstruction;

use crate::{
    jolt::vm::{bytecode::BytecodeRow, rv32i_vm::RV32I},
    utils::thread::unsafe_allocate_zero_vec,
};

use self::analyze::ProgramSummary;

pub mod analyze;

const DEFAULT_MEMORY_SIZE: usize = 10 * 1024 * 1024;
const DEFAULT_STACK_SIZE: usize = 4096;

#[derive(Clone)]
pub struct Program {
    guest: String,
    func: Option<String>,
    input: Vec<u8>,
    memory_size: usize,
    stack_size: usize,
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
            elf: None,
        }
    }

    pub fn set_func(&mut self, func: &str) {
        self.func = Some(func.to_string())
    }

    pub fn set_input<T: Serialize>(&mut self, input: &T) {
        let mut serialized = postcard::to_stdvec(input).unwrap();
        self.input.append(&mut serialized);
    }

    pub fn set_memory_size(&mut self, len: usize) {
        self.memory_size = len;
    }

    pub fn set_stack_size(&mut self, len: usize) {
        self.stack_size = len;
    }

    #[tracing::instrument(skip_all, name = "Program::build")]
    pub fn build(&mut self) {
        if self.elf.is_none() {
            self.save_linker();

            let mut envs = vec![("RUSTFLAGS", format!("-C link-arg=-T{}", self.linker_path()))];
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
                .args(&[
                    "build",
                    "--release",
                    "--features",
                    "guest",
                    "-p",
                    &self.guest,
                    "--target-dir",
                    &target,
                    "--target",
                    "riscv32i-unknown-none-elf",
                    "--bin",
                    "guest",
                ])
                .output()
                .expect("failed to build guest");

            if !output.status.success() {
                io::stderr().write(&output.stderr).unwrap();
                panic!("failed to compile guest");
            }

            let elf = format!("{}/riscv32i-unknown-none-elf/release/guest", target,);
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
    pub fn trace<F: PrimeField>(
        mut self,
    ) -> (
        JoltDevice,
        Vec<BytecodeRow>,
        Vec<Option<RV32I>>,
        Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
        Vec<F>,
    ) {
        self.build();
        let elf = self.elf.unwrap();
        let (trace, io_device) = tracer::trace(&elf, self.input);

        let bytecode_trace: Vec<BytecodeRow> = trace
            .par_iter()
            .map(|row| BytecodeRow::from_instruction::<RV32I>(&row.instruction))
            .collect();

        let instruction_trace: Vec<Option<RV32I>> = trace
            .par_iter()
            .map(|row| {
                if let Ok(jolt_instruction) = RV32I::try_from(row) {
                    Some(jolt_instruction)
                } else {
                    // Instruction does not use lookups
                    None
                }
            })
            .collect();

        let memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]> =
            trace.par_iter().map(|row| row.into()).collect();

        let padded_trace_len = trace.len().next_power_of_two();
        let mut circuit_flag_trace = unsafe_allocate_zero_vec(padded_trace_len * NUM_CIRCUIT_FLAGS);
        circuit_flag_trace
            .par_chunks_mut(padded_trace_len)
            .enumerate()
            .for_each(|(flag_index, chunk)| {
                chunk.iter_mut().zip(trace.iter()).for_each(|(flag, row)| {
                    if row.instruction.to_circuit_flags()[flag_index] {
                        *flag = F::one();
                    }
                });
            });

        (
            io_device,
            bytecode_trace,
            instruction_trace,
            memory_trace,
            circuit_flag_trace,
        )
    }

    pub fn trace_analyze<F: PrimeField>(mut self) -> ProgramSummary {
        self.build();
        let elf = self.elf.as_ref().unwrap();
        let (raw_trace, _) = tracer::trace(&elf, self.input.clone());

        let (bytecode, memory_init) = self.decode();
        let (io_device, bytecode_trace, instruction_trace, memory_trace, circuit_flags) =
            self.trace();
        let circuit_flags: Vec<bool> = circuit_flags
            .into_iter()
            .map(|flag: F| flag.is_one())
            .collect();

        let program_summary = ProgramSummary {
            raw_trace,
            bytecode,
            memory_init,
            io_device,
            bytecode_trace,
            instruction_trace,
            memory_trace,
            circuit_flags,
        };

        program_summary
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
        file.write(linker_script.as_bytes())
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
