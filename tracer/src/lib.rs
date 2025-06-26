#![cfg_attr(not(feature = "std"), no_std)]
#![allow(dead_code)]
#![allow(clippy::legacy_numeric_constants)]

#[cfg(not(feature = "std"))]
extern crate alloc;

use std::vec;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use common::{self, constants::RAM_START_ADDRESS, jolt_device::MemoryConfig};
use emulator::{
    cpu::{self, Xlen},
    default_terminal::DefaultTerminal,
    get_mut_emulator, Emulator, EmulatorState,
};

use instruction::{RV32IMCycle, RV32IMInstruction};
use object::{Object, ObjectSection, SectionKind};

mod emulator;
pub mod instruction;

pub use common::jolt_device::JoltDevice;

/// Executes a RISC-V program and generates its execution trace along with emulator state checkpoints.
///
/// # Details
/// The function performs these steps:
/// 1. Sets up an emulator with the provided program and configuration
/// 2. Runs the program to completion while:
///    - Collecting execution traces of each instruction
///    - Optionally saving periodic checkpoints of the emulator state
///
/// # Arguments
///
/// * `elf_contents`
/// * `inputs`
/// * `memory_config`
/// * `checkpoint_interval` - Number of RV32IMCycle at which to save emulator checkpoints
///                          If None, no checkpoints will be saved
///
/// # Returns
///
/// Returns a tuple containing:
/// * `Vec<RV32IMCycle>` - Complete execution trace
/// * `JoltDevice`
/// * `Option<Vec<LazyTraceIterator>>` - If checkpoint_interval is not None, contains emulator
///                                      checkpoints every n RV32IMCycle. Otherwise None.
///
/// # Example Usage
///
/// let (execution_trace, checkpoints) = trace(elf_contents, inputs, memory_config, Some(5));
///
/// let full_execution_trace = checkpoints.as_ref().unwrap()[0].clone().collect::Vec<RV32IMCycle>();
/// assert!(execution_trace == full_execution_trace);
///
/// let trace_from_checkpoint_1 = checkpoints.as_ref().unwrap()[1].unwrap()[1].clone().collect::Vec<RV32IMCycle>();
/// assert!(trace_from_checkpoint_1 == execution_trace[n..])
///
/// let trace_from_checkpoint_2 = checkpoints.as_ref().unwrap()[2].unwrap()[1].clone().collect::Vec<RV32IMCycle>();
/// assert!(trace_from_checkpoint_2 == execution_trace[2*n..])
///
#[tracing::instrument(skip_all)]
pub fn trace(
    elf_contents: Vec<u8>,
    inputs: &[u8],
    memory_config: &MemoryConfig,
    checkpoint_interval: Option<usize>,
) -> (Vec<RV32IMCycle>, JoltDevice, Option<Vec<LazyTraceIterator>>) {
    let mut emulator_trace_iter =
        LazyTraceIterator::new(setup_emulator(elf_contents, inputs, memory_config));
    let mut checkpoints = Vec::new();

    let trace = match checkpoint_interval {
        Some(n) => {
            let mut trace = Vec::with_capacity(1 << 24); // TODO(moodlezoup): make configurable
            loop {
                checkpoints.push(emulator_trace_iter.clone());
                let trace_n = emulator_trace_iter.by_ref().take(n);
                let prev_len = trace.len();
                trace.extend(trace_n);
                if trace.len() - prev_len < n {
                    break;
                }
            }
            trace
        }

        None => emulator_trace_iter.by_ref().collect(),
    };

    let mut final_emulator_state = emulator_trace_iter.get_emulator_state();
    let mut_jolt_device = &mut final_emulator_state.get_mut_cpu().get_mut_mmu().jolt_device;
    let device = std::mem::take(mut_jolt_device);
    (trace, device, checkpoint_interval.map(|_| checkpoints))
}

#[tracing::instrument(skip_all)]
fn step_emulator(emulator: &mut Emulator, prev_pc: &mut u64, trace: Option<&mut Vec<RV32IMCycle>>) {
    let pc = emulator.get_cpu().read_pc();
    // This is a trick to see if the program has terminated by throwing itself
    // into an infinite loop. It seems to be a good heuristic for now but we
    // should eventually migrate to an explicit shutdown signal.
    if let Some(trace_vec) = trace.as_ref() {
        assert!(trace_vec.is_empty());
    }
    if *prev_pc == pc {
        return;
    }
    emulator.tick(trace);
    *prev_pc = pc;
}

#[tracing::instrument(skip_all)]
fn setup_emulator(elf_contents: Vec<u8>, inputs: &[u8], memory_config: &MemoryConfig) -> Emulator {
    let term = DefaultTerminal::new();
    let mut emulator = Emulator::new(Box::new(term));
    emulator.update_xlen(get_xlen());

    let mut jolt_device = JoltDevice::new(memory_config);
    jolt_device.inputs = inputs.to_vec();
    emulator.get_mut_cpu().get_mut_mmu().jolt_device = jolt_device;

    emulator.setup_program(elf_contents);
    emulator
}

/// An iterator that lazily generates execution traces from a RISC-V emulator checkpoint.
///
/// This iterator produces instruction traces one at a time, executing the emulator
/// as needed rather than generating the entire trace upfront. It buffers traces
/// in `current_traces` since some instructions generate multiple trace entries.
/// When the `current_traces` buffer is exhausted, it executes another emulator tick
/// to generate more.
///
/// # Fields
///
/// * `emulator` - Clone of the checkpoint emulator state to execute from
/// * `prev_pc` - Previous program counter value, used for termination detection
/// * `current_traces` - Buffer of trace entries from the most recent emulator tick
#[derive(Clone)]
pub struct LazyTraceIterator {
    emulator_state: EmulatorState,
    prev_pc: u64,
    current_traces: Vec<RV32IMCycle>,
    // If length is None, the iterator will run until the program ends.
    count: usize, // number of cycles completed
}

impl LazyTraceIterator {
    pub fn new(emulator_state: EmulatorState) -> Self {
        LazyTraceIterator {
            emulator_state,
            prev_pc: 0,
            current_traces: vec![],
            count: 0,
        }
    }

    pub fn at_tick_boundary(self) -> bool {
        self.current_traces.is_empty()
    }

    pub fn get_emulator_state(self) -> EmulatorState {
        self.emulator_state
    }
}

impl Iterator for LazyTraceIterator {
    type Item = RV32IMCycle;
    /// Advances the iterator and returns the next trace entry.
    ///
    /// # Returns
    ///
    /// * `Some(RV32IMCycle)` - The next instruction trace in the execution sequence
    /// * `None` - If program execution has completed.
    ///
    /// # Details
    ///
    /// The function follows this sequence:
    /// 1. Returns any remaining traces from the previous emulator tick
    /// 2. If buffer `current_traces` is empty, and the number of ticks
    ///    is not reached, executes another emulator tick``
    /// 3. Checks for program termination using the heuristic of PC not changing
    /// 4. Buffers new traces in FIFO order
    /// 5. Returns the next trace or None if execution is complete
    fn next(&mut self) -> Option<Self::Item> {
        //Iterate over t returning in FIFO order before calling tick() again.
        if !self.current_traces.is_empty() {
            return self.current_traces.pop();
        }

        // Step the emulator to execute the next instruction till the program ends.
        self.count += 1;
        assert!(self.current_traces.is_empty());
        step_emulator(
            get_mut_emulator(&mut self.emulator_state),
            &mut self.prev_pc,
            Some(&mut self.current_traces),
        );
        if self.current_traces.is_empty() {
            None
        } else {
            self.current_traces.reverse();
            self.current_traces.pop()
        }
    }
}

#[tracing::instrument(skip_all)]
pub fn decode(elf: &[u8]) -> (Vec<RV32IMInstruction>, Vec<(u64, u8)>) {
    let obj = object::File::parse(elf).unwrap();

    let sections = obj
        .sections()
        .filter(|s| s.address() >= RAM_START_ADDRESS)
        .collect::<Vec<_>>();

    let mut instructions = Vec::new();
    let mut data = Vec::new();

    for section in sections {
        let raw_data = section.data().unwrap();

        if let SectionKind::Text = section.kind() {
            for (chunk, word) in raw_data.chunks(4).enumerate() {
                let word = u32::from_le_bytes(word.try_into().unwrap());
                let address = chunk as u64 * 4 + section.address();

                if let Ok(inst) = RV32IMInstruction::decode(word, address) {
                    instructions.push(inst);
                    continue;
                }
                // Unrecognized instruction, or from a ReadOnlyData section
                eprintln!("Warning: word: {word:08X} at address: {address:08X} is not recognized as a valid instruction.");
                instructions.push(RV32IMInstruction::UNIMPL);
            }
        }
        let address = section.address();
        for (offset, byte) in raw_data.iter().enumerate() {
            data.push((address + offset as u64, *byte));
        }
    }

    (instructions, data)
}

fn get_xlen() -> Xlen {
    match common::constants::XLEN {
        32 => cpu::Xlen::Bit32,
        64 => cpu::Xlen::Bit64,
        _ => panic!("Emulator only supports 32 / 64 bit registers."),
    }
}
