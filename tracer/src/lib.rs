#![cfg_attr(not(feature = "std"), no_std)]
#![allow(dead_code)]
#![allow(clippy::legacy_numeric_constants)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use common::{self, constants::RAM_START_ADDRESS, jolt_device::MemoryConfig};
use emulator::{
    cpu::{self, Xlen},
    default_terminal::DefaultTerminal,
    Emulator,
    EmulatorState,
};

use instruction::{RV32IMCycle, RV32IMInstruction};
use object::{Object, ObjectSection, SectionKind};

mod emulator;
pub mod instruction;

pub use common::jolt_device::JoltDevice;

pub type Checkpoint = EmulatorState;
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
/// * `checkpoint_interval` - Optional interval (n) at which to save emulator state checkpoints
///                          If None, no checkpoints will be saved
///
/// # Returns
///
/// Returns a tuple containing:
/// * `Vec<RV32IMCycle>` - Complete execution trace
/// * `JoltDevice`
/// * `Option<Vec<Emulator>>` - If checkpoint_interval was Some(n), contains emulator states saved
///                            every n steps. Otherwise None.
///
#[tracing::instrument(skip_all)]
pub fn trace(
    elf_contents: Vec<u8>,
    inputs: &[u8],
    memory_config: &MemoryConfig,
    checkpoint_interval: Option<usize>,
) -> (Vec<RV32IMCycle>, JoltDevice, Option<Vec<Checkpoint>>) {
    let mut emulator = setup_emulator(elf_contents, inputs, memory_config);
    // checkpoints are emulator states
    let checkpoints = run_and_get_checkpoints(&mut emulator, checkpoint_interval);
    let execution_trace = std::mem::take(&mut emulator.get_mut_cpu().trace);
    let device = std::mem::take(&mut emulator.get_mut_cpu().get_mut_mmu().jolt_device);
    return (execution_trace, device, checkpoints);
}

#[tracing::instrument(skip_all)]
fn step_emulator(emulator: &mut Emulator, prev_pc: &mut u64, tracing: bool) -> bool {
    let pc = emulator.get_cpu().read_pc();
    // This is a trick to see if the program has terminated by throwing itself
    // into an infinite loop. It seems to be a good heuristic for now but we
    // should eventually migrate to an explicit shutdown signal.
    if *prev_pc == pc {
        return true;
    }
    emulator.tick(tracing);
    *prev_pc = pc;
    false
}

/// Executes a RISC-V program while collecting periodic checkpoints of the emulator state.
///
/// This function runs the emulator until program completion, optionally saving snapshots
/// of the emulator state at regular intervals.
///
/// # Arguments
///
/// * `emulator` - Mutable reference to an initialized emulator containing the program to run
/// * `checkpoint_interval` - Optional interval at which to save emulator state:
///   * `Some(n)` - Save a checkpoint every `n` instructions
///   * `None` - Run to completion without saving checkpoints
///
/// # Returns
///
/// Returns `Option<Vec<Emulator>>`:
/// * `Some(vec)` - If `checkpoint_interval` was `Some(n)`, contains vector of emulator
///                 states saved at each interval
/// * `None` - If `checkpoint_interval` was `None`
///
/// # Notes
///
/// - The emulator's state is cloned at each checkpoint, which can be memory-intensive
///   for long-running programs with frequent checkpoints
/// - Program termination is detected via an infinite loop heuristic rather than
///   an explicit shutdown signal
/// - Tracing is always enabled during execution. This can be changed if tracing is not needed during checkpointing.
#[tracing::instrument(skip_all)]
pub fn run_and_get_checkpoints(
    emulator: &mut Emulator,
    checkpoint_interval: Option<usize>,
) -> Option<Vec<Checkpoint>> {
    let mut prev_pc: u64 = 0;
    let mut checkpoints = Vec::new();

    match checkpoint_interval {
        Some(interval) => {
            let mut count = 0;
            loop {
                if count % interval == 0 {
                    checkpoints.push(emulator.save_state());
                }
                count += 1;
                if step_emulator(emulator, &mut prev_pc, true) {
                    break;
                }
            }
        }
        None => loop {
            if step_emulator(emulator, &mut prev_pc, true) {
                break;
            }
        },
    }

    checkpoint_interval.map(|_| checkpoints)
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
    return emulator;
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
pub struct LazyTraceIterator {
    emulator: Emulator,
    prev_pc: u64,
    current_traces: Vec<RV32IMCycle>,
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
    /// 2. If buffer `current_traces` is empty, executes another emulator tick
    /// 3. Checks for program termination using the heuristic of PC not changing
    /// 4. Buffers new traces in FIFO order
    /// 5. Returns the next trace or None if execution is complete
    fn next(&mut self) -> Option<Self::Item> {
        //Iterate over t returning in FIFO order before calling tick() again.
        if let Some(trace) = self.current_traces.pop() {
            return Some(trace);
        }

        if step_emulator(&mut self.emulator, &mut self.prev_pc, true) {
            return None;
        }

        self.current_traces = std::mem::take(&mut self.emulator.get_mut_cpu().trace);
        self.current_traces.reverse();
        self.current_traces.pop()
    }
}

#[tracing::instrument(skip_all)]
pub fn trace_from_checkpoint(checkpoint: &Checkpoint) -> LazyTraceIterator {
    LazyTraceIterator {
        emulator: Emulator::from_state(checkpoint),
        prev_pc: 0,
        current_traces: Vec::new(),
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
