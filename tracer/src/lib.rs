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
    Emulator, EmulatorState, get_mut_emulator
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
/// * `checkpoint_interval` - Number of emulator ticks (n) at which to save emulator checkpoints
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
/// # Example Usage
///
/// let mut trace: Vec<RV32IMCycle> = Vec::new();
/// let mut emulator = setup_emulator(elf_contents, inputs, memory_config);
/// let (execution_trace, checkpoints) = run_and_get_checkpoints(&mut emulator, Some(5));
/// for (i,ci) in checkpoints.unwrap().into_iter().enumerate(){
///     let ci_vec: Vec<RV32IMCycle> = ci.collect();
///     let l = ci_vec.len();
///     trace.extend(ci_vec);
///     println!("ckp #{i:} trace segment length = {l:}")
///     }
/// assert!(trace == execution_trace);
///
#[tracing::instrument(skip_all)]
pub fn trace(
    elf_contents: Vec<u8>,
    inputs: &[u8],
    memory_config: &MemoryConfig,
    checkpoint_interval: Option<usize>,
) -> (Vec<RV32IMCycle>, JoltDevice, Option<Vec<LazyTraceIterator>>) {
    let mut emulator = setup_emulator(elf_contents, inputs, memory_config);
    let (execution_trace, checkpoints) =
        run_and_get_checkpoints(&mut emulator, checkpoint_interval);
    let device: JoltDevice = std::mem::take(&mut emulator.get_mut_cpu().get_mut_mmu().jolt_device);
    return (execution_trace, device, checkpoints);
}

#[tracing::instrument(skip_all)]
fn step_emulator(emulator: &mut Emulator, prev_pc: &mut u64) -> Option<Vec<RV32IMCycle>> {
    let pc = emulator.get_cpu().read_pc();
    // This is a trick to see if the program has terminated by throwing itself
    // into an infinite loop. It seems to be a good heuristic for now but we
    // should eventually migrate to an explicit shutdown signal.
    if *prev_pc == pc {
        return None;
    }
    let trace = emulator.tick();
    *prev_pc = pc;
    Some(trace)
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
///   * `Some(n)` - Save a checkpoint every `n` emulator ticks
///   * `None` - Run to completion without saving checkpoints
///
/// # Returns
///
/// Returns `(Vec<RV32IMCycle>, Option<Vec<LazyTraceIterator>>)`:
/// * `Vec<RV32IMCycle>` - Complete trace vector
/// * `Some(Vec<LazyTraceIterator>)` - If `checkpoint_interval` was `Some(n)`, contains vector of checkpoints
///                 saved at each interval that can be used to resume trace computation
/// * `None` - If `checkpoint_interval` was `None`
///
/// # Notes
///
/// - The emulator's state is cloned at each checkpoint, which is memory-intensive
/// - Program termination is detected via an infinite loop heuristic rather than
///   an explicit shutdown signal
/// - Tracing is always enabled during execution. This can be changed if tracing is not needed
///   during checkpointing.
#[tracing::instrument(skip_all)]
pub fn run_and_get_checkpoints(
    emulator: &mut Emulator,
    checkpoint_interval: Option<usize>,
) -> (Vec<RV32IMCycle>, Option<Vec<LazyTraceIterator>>) {
    let mut prev_pc: u64 = 0;
    let mut checkpoints = Vec::new();
    let mut trace = Vec::with_capacity(1 << 24); // TODO(moodlezoup): make configurable

    match checkpoint_interval {
        Some(n) => {
            let mut count = 0;
            loop {
                if count % n == 0 {
                    checkpoints.push(LazyTraceIterator {
                        state: emulator.save_state(),
                        prev_pc,
                        current_traces: Vec::new(),
                        length: checkpoint_interval,
                    });
                }
                count += 1;
                match step_emulator(emulator, &mut prev_pc) {
                    None => break,
                    Some(cycles) => trace.extend(cycles),
                }
            }
        }
        None => loop {
            match step_emulator(emulator, &mut prev_pc) {
                None => break,
                Some(cycles) => trace.extend(cycles),
            }
        },
    }

    (trace, checkpoint_interval.map(|_| checkpoints))
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
/// * `length` - Length of the iterator. This length is interpreted as the number of
///              emulator ticks. Thus, the length of the generated trace vector is
///              strictly greater than the this length.
pub struct LazyTraceIterator {
    state: EmulatorState,
    prev_pc: u64,
    current_traces: Vec<RV32IMCycle>,
    length: Option<usize>, // number of ticks
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

        // Check if iterator length (checkpoint interval) is exhausted
        match self.length {
            Some(n) if n <= 0 => return None, // If length is set and reached 0, stop iteration
            Some(n) => self.length = Some(n - 1), // Decrement length
            None => (), // If length is not set, continue till the program ends
        }

        // Step the emulator to execute the next instruction till the program ends.
        match step_emulator(get_mut_emulator(&mut self.state), &mut self.prev_pc) {
            None => return None,
            Some(cycles) => {
                assert!(self.current_traces.is_empty());
                self.current_traces = cycles;
                self.current_traces.reverse();
                self.current_traces.pop()
            }
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
