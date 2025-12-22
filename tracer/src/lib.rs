#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
extern crate core;

use itertools::Itertools;
use std::vec;
use tracing::{error, info};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use common::{self, constants::RAM_START_ADDRESS, jolt_device::MemoryConfig};
use emulator::{
    cpu::{self, Xlen},
    default_terminal::DefaultTerminal,
    get_mut_emulator, EmulatorState,
};

use instruction::{Cycle, Instruction};
use object::{Object, ObjectSection, SectionKind};

pub mod emulator;
pub mod instruction;
pub mod utils;

pub use common::jolt_device::JoltDevice;
pub use instruction::inline::{list_registered_inlines, register_inline};

use crate::{
    emulator::{
        memory::{CheckpointingMemory, Memory, MemoryData, ReplayableMemory},
        GeneralizedEmulator,
    },
    instruction::uncompress_instruction,
};

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
/// * `checkpoint_interval` - Number of Cycle at which to save emulator checkpoints
///                          If None, no checkpoints will be saved
///
/// # Returns
///
/// Returns a tuple containing:
/// * `Vec<Cycle>` - Complete execution trace
/// * `JoltDevice`
/// * `Option<Vec<LazyTraceIterator>>` - If checkpoint_interval is not None, contains emulator
///                                      checkpoints every n Cycle. Otherwise None.
///
/// # Example Usage
///
/// let (execution_trace, checkpoints) = trace(elf_contents, inputs, memory_config, Some(5));
///
/// let full_execution_trace = checkpoints.as_ref().unwrap()[0].clone().collect::Vec<Cycle>();
/// assert!(execution_trace == full_execution_trace);
///
/// let trace_from_checkpoint_1 = checkpoints.as_ref().unwrap()[1].clone().collect::Vec<Cycle>();
/// assert!(trace_from_checkpoint_1 == execution_trace[n..])
///
/// let trace_from_checkpoint_2 = checkpoints.as_ref().unwrap()[2].clone().collect::Vec<Cycle>();
/// assert!(trace_from_checkpoint_2 == execution_trace[2*n..])
///
#[tracing::instrument(skip_all)]
pub fn trace(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
) -> (LazyTraceIterator, Vec<Cycle>, Memory, JoltDevice) {
    let mut lazy_trace_iter = LazyTraceIterator::new(setup_emulator_with_backtraces(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
    ));
    let lazy_trace_iter_ = lazy_trace_iter.clone();
    let trace: Vec<Cycle> = lazy_trace_iter.by_ref().collect();
    let final_memory_state = std::mem::take(lazy_trace_iter.final_memory_state.as_mut().unwrap());
    (
        lazy_trace_iter_,
        trace,
        final_memory_state,
        lazy_trace_iter.get_jolt_device(),
    )
}

use crate::utils::trace_writer::{TraceBatchCollector, TraceWriter, TraceWriterConfig};

pub fn trace_to_file(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    out_path: &std::path::PathBuf,
) -> (Memory, JoltDevice) {
    let config = TraceWriterConfig::default();

    let writer =
        TraceWriter::<Cycle>::new(out_path, config).expect("Failed to create trace writer");
    let mut collector = TraceBatchCollector::new(writer);
    let mut lazy = LazyTraceIterator::new(setup_emulator_with_backtraces(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
    ));

    for cycle in &mut lazy {
        collector.push(cycle);
    }

    let total = collector
        .finalize()
        .expect("Failed to finalize trace writer");

    info!("trace length: {total} cycles");

    let final_mem = lazy.final_memory_state.take().unwrap();
    (final_mem, lazy.get_jolt_device())
}

#[tracing::instrument(skip_all)]
pub fn trace_lazy(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
) -> LazyTraceIterator {
    LazyTraceIterator::new(setup_emulator_with_backtraces(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
    ))
}

#[tracing::instrument(skip_all)]
pub fn trace_checkpoints(
    elf_contents: &[u8],
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    checkpoint_interval: usize,
) -> (Vec<std::iter::Take<LazyTraceIterator>>, JoltDevice) {
    let mut emulator_trace_iter = LazyTraceIterator::new(setup_emulator(
        elf_contents,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
    ));
    let mut checkpoints = Vec::new();

    loop {
        let chkpt = emulator_trace_iter.clone().take(checkpoint_interval);
        checkpoints.push(chkpt);
        emulator_trace_iter = emulator_trace_iter.dropping(checkpoint_interval);
        if emulator_trace_iter.is_empty() {
            break;
        }
    }
    (checkpoints, emulator_trace_iter.get_jolt_device())
}

fn step_emulator<D: MemoryData>(
    emulator: &mut GeneralizedEmulator<D>,
    prev_pc: &mut u64,
    trace: Option<&mut Vec<Cycle>>,
) {
    let pc = emulator.get_cpu().read_pc();
    // This is a trick to see if the program has terminated by throwing itself
    // into an infinite loop. It seems to be a good heuristic for now but we
    // should eventually migrate to an explicit shutdown signal.
    if *prev_pc == pc {
        return;
    }
    emulator.tick(trace);
    *prev_pc = pc;
}

#[tracing::instrument(skip_all)]
fn setup_emulator<D: MemoryData>(
    elf_contents: &[u8],
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
) -> GeneralizedEmulator<D> {
    setup_emulator_with_backtraces(
        elf_contents,
        None,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
    )
}

#[tracing::instrument(skip_all)]
/// Sets up an emulator instance with access to the elf-path for symbol loading and de-mangling.
fn setup_emulator_with_backtraces<D: MemoryData>(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
) -> GeneralizedEmulator<D> {
    let term = DefaultTerminal::default();
    let mut emulator = GeneralizedEmulator::new(Box::new(term));
    emulator.update_xlen(get_xlen());

    let mut jolt_device = JoltDevice::new(memory_config);
    jolt_device.inputs = inputs.to_vec();
    jolt_device.trusted_advice = trusted_advice.to_vec();
    jolt_device.untrusted_advice = untrusted_advice.to_vec();
    emulator.get_mut_cpu().get_mut_mmu().jolt_device = Some(jolt_device);
    if let Some(elf_path) = elf_path {
        emulator.set_elf_path(elf_path);
    }
    emulator.setup_program(elf_contents);
    emulator
}

pub struct Checkpoint {
    emulator_state: GeneralizedEmulator<ReplayableMemory>,
    prev_pc: u64,
    current_traces: Vec<Cycle>,
    /// The remaining number of cycles that can be replayed for this checkpoint
    cycles_remaining: usize,
    /// The total number of cycles executed so far, including the ones prior to this checkpoint
    cycle_count: usize,
}

// SAFETY: Checkpoint contains only owned data and can be safely sent between threads
unsafe impl Send for Checkpoint {}

impl Iterator for Checkpoint {
    type Item = Cycle;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.current_traces.is_empty() {
            return self.current_traces.pop();
        }
        debug_assert!(self.current_traces.is_empty());

        if self.cycles_remaining == 0 {
            return None;
        }
        self.cycles_remaining -= 1;
        self.cycle_count += 1;

        step_emulator(
            &mut self.emulator_state,
            &mut self.prev_pc,
            Some(&mut self.current_traces),
        );
        if self.current_traces.is_empty() {
            let cpu = self.emulator_state.get_mut_cpu();
            if cpu.mmu.jolt_device.as_ref().unwrap().panic {
                error!(
                    "Guest program terminated due to panic after {} cycles.",
                    cpu.trace_len
                );
                utils::panic::display_panic_backtrace(&self.emulator_state);
            }
            None
        } else {
            self.current_traces.reverse();
            self.current_traces.pop()
        }
    }
}

pub struct CheckpointingEmulator {
    emulator_state: GeneralizedEmulator<CheckpointingMemory>,
    prev_pc: u64,
    current_traces: Vec<Cycle>,
    cycles_since_last_checkpoint: usize,
    cycle_count: usize,
    finished: bool,
    pub(crate) final_memory_state: Option<Memory>,
}

impl CheckpointingEmulator {
    pub fn new(emulator_state: GeneralizedEmulator<CheckpointingMemory>) -> Self {
        Self {
            emulator_state,
            prev_pc: 0,
            current_traces: vec![],
            cycles_since_last_checkpoint: 0,
            cycle_count: 0,
            finished: false,
            final_memory_state: None,
        }
    }

    pub fn new_for_test() -> Self {
        let minimal_elf = vec![
            0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0xf3, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];

        use crate::MemoryConfig;
        let memory_config = MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        };

        let emulator_state = setup_emulator(&minimal_elf, b"[]", &[], &[], &memory_config);

        Self {
            emulator_state,
            prev_pc: 0,
            current_traces: vec![],
            cycles_since_last_checkpoint: 0,
            cycle_count: 0,
            finished: true,
            final_memory_state: Some(emulator::memory::Memory::default()),
        }
    }

    pub fn at_tick_boundary(&self) -> bool {
        self.current_traces.is_empty()
    }

    pub fn save_checkpoint(&mut self) -> Checkpoint {
        // XXX: What should we do if we're not at a tick boundary?
        assert!(self.at_tick_boundary());

        let res = Checkpoint {
            emulator_state: self.emulator_state.save_checkpoint(),
            prev_pc: self.prev_pc,
            current_traces: vec![],
            cycles_remaining: self.cycles_since_last_checkpoint,
            cycle_count: self.cycle_count,
        };
        self.cycles_since_last_checkpoint = 0;

        res
    }

    pub fn get_jolt_device(mut self) -> JoltDevice {
        self.emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .jolt_device
            .take()
            .expect("JoltDevice was not initialized")
    }

    pub fn is_empty(&self) -> bool {
        self.finished
    }
}

impl Iterator for CheckpointingEmulator {
    type Item = Cycle;

    /// Advances the iterator and returns the next trace entry.
    ///
    /// # Returns
    ///
    /// * `Some(Cycle)` - The next instruction trace in the execution sequence
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
        if !self.current_traces.is_empty() {
            return self.current_traces.pop();
        }
        debug_assert!(self.current_traces.is_empty());

        self.cycle_count += 1;
        self.cycles_since_last_checkpoint += 1;

        step_emulator(
            &mut self.emulator_state,
            &mut self.prev_pc,
            Some(&mut self.current_traces),
        );
        if self.current_traces.is_empty() {
            self.finished = true;
            let emulator = &mut self.emulator_state;
            let cpu = emulator.get_mut_cpu();
            let memory = std::mem::take(&mut cpu.mmu.memory.memory);
            self.final_memory_state = Some(memory.into_vec_memory_backend());

            if cpu.mmu.jolt_device.as_ref().unwrap().panic {
                error!(
                    "Guest program terminated due to panic after {} cycles.",
                    cpu.trace_len
                );
                utils::panic::display_panic_backtrace(&self.emulator_state);
            }
            None
        } else {
            self.current_traces.reverse();
            self.current_traces.pop()
        }
    }
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
#[derive(Clone, Debug)]
pub struct LazyTraceIterator {
    emulator_state: EmulatorState,
    prev_pc: u64,
    current_traces: Vec<Cycle>,
    count: usize, // number of cycles completed
    finished: bool,
    pub(crate) final_memory_state: Option<Memory>,
}

// SAFETY: LazyTraceIterator contains only owned data and can be safely sent between threads
unsafe impl Send for LazyTraceIterator {}

impl LazyTraceIterator {
    pub fn new(emulator_state: EmulatorState) -> Self {
        LazyTraceIterator {
            emulator_state,
            prev_pc: 0,
            current_traces: vec![],
            count: 0,
            finished: false,
            final_memory_state: None,
        }
    }

    pub fn new_for_test() -> Self {
        let minimal_elf = vec![
            0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0xf3, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];

        use crate::MemoryConfig;
        let memory_config = MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        };

        let emulator_state = setup_emulator(&minimal_elf, b"[]", &[], &[], &memory_config);

        LazyTraceIterator {
            emulator_state,
            prev_pc: 0,
            current_traces: vec![],
            count: 0,
            finished: true,
            final_memory_state: Some(emulator::memory::Memory::default()),
        }
    }

    pub fn at_tick_boundary(&self) -> bool {
        self.current_traces.is_empty()
    }

    pub fn get_emulator_state(self) -> EmulatorState {
        self.emulator_state
    }

    pub fn clone_emulator_state(&self) -> EmulatorState {
        self.emulator_state.clone()
    }

    pub fn get_jolt_device(self) -> JoltDevice {
        let mut final_emulator_state = self.get_emulator_state();
        final_emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .jolt_device
            .take()
            .expect("JoltDevice was not initialized")
    }

    pub fn is_empty(&self) -> bool {
        self.finished
    }
}

impl Iterator for LazyTraceIterator {
    type Item = Cycle;
    /// Advances the iterator and returns the next trace entry.
    ///
    /// # Returns
    ///
    /// * `Some(Cycle)` - The next instruction trace in the execution sequence
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
        if !self.current_traces.is_empty() {
            return self.current_traces.pop();
        }

        self.count += 1;
        assert!(self.current_traces.is_empty());
        step_emulator(
            get_mut_emulator(&mut self.emulator_state),
            &mut self.prev_pc,
            Some(&mut self.current_traces),
        );
        if self.current_traces.is_empty() {
            self.finished = true;
            let emulator = get_mut_emulator(&mut self.emulator_state);
            let cpu = emulator.get_mut_cpu();
            let memory = std::mem::take(&mut cpu.mmu.memory.memory);
            self.final_memory_state = Some(memory);

            if cpu.mmu.jolt_device.as_ref().unwrap().panic {
                error!(
                    "Guest program terminated due to panic after {} cycles.",
                    cpu.trace_len
                );
                utils::panic::display_panic_backtrace(&self.emulator_state);
            }
            None
        } else {
            self.current_traces.reverse();
            self.current_traces.pop()
        }
    }
}

#[tracing::instrument(skip_all)]
pub fn decode(elf: &[u8]) -> (Vec<Instruction>, Vec<(u64, u8)>, u64, Xlen) {
    let obj = object::File::parse(elf).unwrap();
    let mut xlen = Xlen::Bit64;
    if let object::File::Elf32(_) = &obj {
        xlen = Xlen::Bit32;
    }

    let sections = obj
        .sections()
        .filter(|s| s.address() >= RAM_START_ADDRESS)
        .collect::<Vec<_>>();

    let mut instructions = Vec::new();
    let mut data = Vec::new();

    // keeps track of the highest address used in the program as the end address
    let mut program_end = RAM_START_ADDRESS;
    for section in sections {
        let start = section.address();
        let length = section.size();
        let end = start + length;
        program_end = program_end.max(end);

        let raw_data = section.data().unwrap();

        if let SectionKind::Text = section.kind() {
            let mut offset = 0;
            while offset < raw_data.len() {
                let address = section.address() + offset as u64;

                // Check if we have at least 2 bytes
                if offset + 1 >= raw_data.len() {
                    break;
                }

                // Read first 2 bytes to determine instruction length
                let first_halfword = u16::from_le_bytes([raw_data[offset], raw_data[offset + 1]]);

                // Check if it's a compressed instruction (lowest 2 bits != 11)
                if (first_halfword & 0b11) != 0b11 {
                    // Compressed 16-bit instruction
                    let compressed_inst = first_halfword;
                    if compressed_inst == 0x0000 {
                        offset += 2;
                        continue;
                    }

                    if let Ok(inst) = Instruction::decode(
                        uncompress_instruction(compressed_inst as u32, xlen),
                        address,
                        true,
                    ) {
                        instructions.push(inst);
                    } else {
                        eprintln!("Warning: compressed instruction {compressed_inst:04X} at address: {address:08X} failed to decode.");
                        instructions.push(Instruction::UNIMPL);
                    }
                    offset += 2;
                } else {
                    // Standard 32-bit instruction
                    if offset + 3 >= raw_data.len() {
                        eprintln!("Warning: incomplete instruction at address: {address:08X}");
                        break;
                    }

                    let word = u32::from_le_bytes([
                        raw_data[offset],
                        raw_data[offset + 1],
                        raw_data[offset + 2],
                        raw_data[offset + 3],
                    ]);

                    if let Ok(inst) = Instruction::decode(word, address, false) {
                        instructions.push(inst);
                    } else {
                        eprintln!("Warning: word: {word:08X} at address: {address:08X} is not recognized as a valid instruction.");
                        instructions.push(Instruction::UNIMPL);
                    }
                    offset += 4;
                }
            }
        }
        let address = section.address();
        for (offset, byte) in raw_data.iter().enumerate() {
            data.push((address + offset as u64, *byte));
        }
    }
    (instructions, data, program_end, xlen)
}

fn get_xlen() -> Xlen {
    match common::constants::XLEN {
        32 => cpu::Xlen::Bit32,
        64 => cpu::Xlen::Bit64,
        _ => panic!("Emulator only supports 32 / 64 bit registers."),
    }
}

pub struct IterChunks<I: Iterator> {
    chunk_size: usize,
    iter: I,
}

pub trait ChunksIterator: Iterator + Sized {
    fn iter_chunks(self, size: usize) -> IterChunks<Self> {
        assert!(size != 0, "chunk size must be non-zero");
        IterChunks {
            chunk_size: size,
            iter: self,
        }
    }
}

impl<I: Iterator + Sized> ChunksIterator for I {}

impl<I: Iterator<Item: Clone>> Iterator for IterChunks<I> {
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Vec::with_capacity(self.chunk_size);
        chunk.extend(self.iter.by_ref().take(self.chunk_size));
        if chunk.is_empty() {
            return None;
        }
        Some(chunk)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    const ELF_CONTENTS: &[u8] = include_bytes!("testfiles/muldiv-guest");
    const INPUTS: [u8; 6] = [0xbd, 0xaa, 0xde, 0x5, 0x11, 0x5c];
    #[test]
    /// Test that the trace function produces the expected number of cycles for a given ELF input.
    /// Test the checkpointing functionality by verifying the number of checkpoints created and
    /// if the traces from checkpoints match the overall execution trace.
    /// The test is based on the muldiv benchmark.
    fn test_trace() {
        use common::jolt_device::MemoryConfig;

        let expected_trace_length = 441;
        let elf: Vec<u8> = ELF_CONTENTS.to_vec();
        let n = 50;
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let (_, execution_trace, _, _) = trace(&elf, None, &INPUTS, &[], &[], &memory_config);
        let (checkpoints, _) = trace_checkpoints(&elf, &INPUTS, &[], &[], &memory_config, n);
        assert_eq!(execution_trace.len(), expected_trace_length);
        assert_eq!(checkpoints.len(), expected_trace_length.div_ceil(n));

        let trace_chunk = execution_trace
            .chunks(n)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();
        for (i, checkpoint) in checkpoints.into_iter().enumerate() {
            let ti: Vec<Cycle> = checkpoint.collect();
            assert_eq!(trace_chunk[i], ti);
        }
    }

    #[test]
    fn test_lazy_iterator() {
        let elf = ELF_CONTENTS.to_vec();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };

        let (_, execution_trace, _, _) = trace(&elf, None, &INPUTS, &[], &[], &memory_config);
        let mut emulator: GeneralizedEmulator<Vec<u64>> = setup_emulator(&elf, &INPUTS, &[], &[], &memory_config);
        let mut prev_pc: u64 = 0;
        let mut trace = vec![];
        let mut prev_trace_len = 0;
        loop {
            step_emulator(&mut emulator, &mut prev_pc, Some(&mut trace));
            if trace.len() - prev_trace_len == 0 {
                break;
            }
            prev_trace_len = trace.len();
        }
        assert_eq!(execution_trace, trace);
    }
}
