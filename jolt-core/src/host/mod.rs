use core::{str::FromStr, u8};
use std::{
    collections::HashMap, fs::{self, File}, io::{self, Write}, path::PathBuf, process::Command
};

use ark_ff::PrimeField;
use postcard;
use rayon::prelude::*;
use serde::Serialize;

use common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{JoltDevice, MemoryOp, NUM_CIRCUIT_FLAGS, RV32IM},
};
use tracer::ELFInstruction;

use crate::jolt::{
    instruction::add::ADDInstruction,
    vm::{bytecode::BytecodeRow, rv32i_vm::RV32I},
};

const DEFAULT_MEMORY_SIZE: usize = 10 * 1024 * 1024;

#[derive(Clone)]
pub struct Program {
    guest: String,
    input: Vec<u8>,
    memory_size: usize,
    pub elf: Option<PathBuf>,
}

impl Program {
    pub fn new(guest: &str) -> Self {
        Self {
            guest: guest.to_string(),
            input: Vec::new(),
            memory_size: DEFAULT_MEMORY_SIZE,
            elf: None,
        }
    }

    pub fn set_input<T: Serialize>(&mut self, input: &T) {
        let mut serialized = postcard::to_stdvec(input).unwrap();
        self.input.append(&mut serialized);
    }

    pub fn set_memory_size(&mut self, len: usize) {
        self.memory_size = len;
    }

    #[tracing::instrument(skip_all, name = "Program::build")]
    pub fn build(&mut self) {
        if self.elf.is_none() {
            self.save_linker(self.memory_size);
            let output = Command::new("cargo")
                .envs([("RUSTFLAGS", format!("-C link-arg=-T{}", self.linker_path()))])
                .args(&[
                    "build",
                    "--release",
                    "--features",
                    "guest",
                    "-p",
                    &self.guest,
                    "--target-dir",
                    "/tmp/jolt-guest-target",
                    "--target",
                    "riscv32i-unknown-none-elf",
                ])
                .output()
                .expect("failed to build guest");

            io::stdout().write(&output.stdout).unwrap();
            io::stderr().write(&output.stderr).unwrap();

            let elf = format!(
                "/tmp/jolt-guest-target/riscv32i-unknown-none-elf/release/{}",
                self.guest
            );
            self.elf = Some(PathBuf::from_str(&elf).unwrap());
        }
    }


    #[tracing::instrument(skip_all, name = "Program::decode")]
    pub fn decode(&mut self) -> Vec<ELFInstruction> {
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
        Vec<RV32I>,
        Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
        Vec<F>,
    ) {
        self.build();
        let elf = self.elf.unwrap();
        let (trace, io_device) = tracer::trace(&elf, self.input);

        let span_bytecode_trace = tracing::span!(tracing::Level::TRACE, "bytecode_trace");
        let _enter = span_bytecode_trace.enter();
        let bytecode_trace: Vec<BytecodeRow> = trace
            .par_iter()
            .map(|row| BytecodeRow::from_instruction::<RV32I>(&row.instruction))
            .collect();
        drop(_enter);

        let span_instruction_trace = tracing::span!(tracing::Level::TRACE, "instruction_trace");
        let _enter = span_instruction_trace.enter();
        let instruction_trace: Vec<RV32I> = trace
            .par_iter()
            .map(|row| {
                if let Ok(jolt_instruction) = RV32I::try_from(row) {
                    jolt_instruction
                } else {
                    // TODO(moodlezoup): Add a `padding` function to InstructionSet trait
                    ADDInstruction(0_u64, 0_u64).into()
                }
            })
            .collect();
        drop(_enter);

        let span_memory_trace = tracing::span!(tracing::Level::TRACE, "memory_trace");
        let _enter = span_memory_trace.enter();
        let memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]> =
            trace.par_iter().map(|row| row.into()).collect();
        drop(_enter);

        let span_circuit_flag_trace = tracing::span!(tracing::Level::TRACE, "circuit_flag_trace");
        let _enter = span_circuit_flag_trace.enter();
        let mut circuit_flag_trace = vec![F::zero(); trace.len() * NUM_CIRCUIT_FLAGS];
        circuit_flag_trace.par_chunks_mut(NUM_CIRCUIT_FLAGS).enumerate().for_each(|(chunk_index, chunk)| {
            let flags = trace[chunk_index].instruction.to_circuit_flags();
            for (flag_index, flag) in flags.iter().enumerate() {
                if *flag {
                    chunk[flag_index] = F::one();
                }
            }
        });
        drop(_enter);

        (
            io_device,
            bytecode_trace,
            instruction_trace,
            memory_trace,
            circuit_flag_trace,
        )
    }

    pub fn trace_analyze(mut self) -> Vec<u8> {
        self.build();
        let elf = self.elf.unwrap();
        let (rows, device) = tracer::trace(&elf, self.input);

        let mut counts = HashMap::<RV32IM, u64>::new();
        for row in rows {
            let op = row.instruction.opcode;
            if let Some(count) = counts.get(&op) {
                counts.insert(op, count + 1);
            } else {
                counts.insert(op, 1);
            }
        }

        let mut counts: Vec<_> = counts.into_iter().collect();
        counts.sort_by_key(|v| v.1);
        counts.reverse();

        for (op, count) in counts {
            println!("{:?}: {}", op, count)
        }

        device.outputs
    }

    fn save_linker(&self, memory_size: usize) {
        let linker_path = PathBuf::from_str(&self.linker_path()).unwrap();
        if let Some(parent) = linker_path.parent() {
            fs::create_dir_all(parent).expect("could not create linker file");
        }

        let linker_script = LINKER_SCRIPT_TEMPLATE
            .replace("{MEMORY_SIZE}", &memory_size.to_string());

        let mut file = File::create(linker_path).expect("could not create linker file");
        file.write(linker_script.as_bytes()).expect("could not save linker");
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
  . = . + 4096;
  _STACK_PTR = .;
}
"#;
