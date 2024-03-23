use core::{str::FromStr, u8};
use std::{
    collections::HashMap,
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
    rv_trace::{JoltDevice, MemoryOp, RV32IM},
};
use tracer::ELFInstruction;

use crate::jolt::{
    instruction::add::ADDInstruction,
    vm::{bytecode::BytecodeRow, rv32i_vm::RV32I},
};

pub struct Program {
    guest: String,
    input: Vec<u8>,
    pub elf: Option<PathBuf>,
}

impl Program {
    pub fn new(guest: &str) -> Self {
        Self {
            guest: guest.to_string(),
            input: Vec::new(),
            elf: None,
        }
    }

    pub fn input<T: Serialize>(mut self, input: &T) -> Self {
        let mut serialized = postcard::to_stdvec(input).unwrap();
        self.input.append(&mut serialized);

        self
    }

    pub fn build(&mut self) {
        if self.elf.is_none() {
            let output = Command::new("cargo")
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

    pub fn decode(&mut self) -> Vec<ELFInstruction> {
        self.build();
        let elf = self.elf.as_ref().unwrap();
        tracer::decode(elf)
    }

    // TODO(moodlezoup): Make this generic over InstructionSet
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

        let bytecode_trace: Vec<BytecodeRow> = trace
            .par_iter()
            .map(|row| BytecodeRow::from_instruction::<RV32I>(&row.instruction))
            .collect();

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

        let memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]> =
            trace.par_iter().map(|row| row.into()).collect();
        let circuit_flag_trace = trace
            .par_iter()
            .flat_map(|row| {
                row.instruction
                    .to_circuit_flags()
                    .iter()
                    .map(|&flag| flag.into())
                    .collect::<Vec<F>>()
            })
            .collect();

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
}
