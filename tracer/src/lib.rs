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
        memory::{Memory, MemoryData},
        Emulator,
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
    let mut lazy_trace_iter = trace_lazy(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
    );
    let lazy_trace_iter_ = lazy_trace_iter.clone();
    let trace: Vec<Cycle> = lazy_trace_iter.by_ref().collect();
    let final_memory_state =
        std::mem::take(&mut lazy_trace_iter.lazy_tracer.final_memory_state).unwrap();
    (
        lazy_trace_iter_,
        trace,
        final_memory_state,
        lazy_trace_iter.lazy_tracer.get_jolt_device(),
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
    let mut lazy = trace_lazy(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
    );

    for cycle in &mut lazy {
        collector.push(cycle);
    }

    let total = collector
        .finalize()
        .expect("Failed to finalize trace writer");

    info!("trace length: {total} cycles");

    let final_mem = lazy.lazy_tracer.final_memory_state.take().unwrap();
    (final_mem, lazy.lazy_tracer.get_jolt_device())
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
    LazyTraceIterator::new(CheckpointingTracer::new(setup_emulator_with_backtraces(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
    )))
}

#[tracing::instrument(skip_all)]
pub fn trace_checkpoints(
    elf_contents: &[u8],
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    checkpoint_interval: usize,
) -> (Vec<Checkpoint>, JoltDevice) {
    let mut emulator_trace_iter =
        GeneralizedLazyTraceIter::new(CheckpointingTracer::new(setup_emulator(
            elf_contents,
            inputs,
            untrusted_advice,
            trusted_advice,
            memory_config,
        )));
    emulator_trace_iter.lazy_tracer.start_saving_checkpoints();
    let mut checkpoints = Vec::new();

    loop {
        emulator_trace_iter = emulator_trace_iter.dropping(checkpoint_interval);
        let chkpt = emulator_trace_iter.lazy_tracer.save_checkpoint();
        checkpoints.push(chkpt);
        if emulator_trace_iter.lazy_tracer.has_terminated() {
            break;
        }
    }
    (
        checkpoints,
        emulator_trace_iter.lazy_tracer.get_jolt_device(),
    )
}

fn step_emulator(emulator: &mut Emulator, prev_pc: &mut u64, trace: Option<&mut Vec<Cycle>>) {
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
fn setup_emulator(
    elf_contents: &[u8],
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
) -> Emulator {
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
fn setup_emulator_with_backtraces(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
) -> Emulator {
    let term = DefaultTerminal::default();
    let mut emulator = Emulator::new(Box::new(term));
    emulator.update_xlen(get_xlen());

    assert!(
        trusted_advice.len() as u64 <= memory_config.max_trusted_advice_size,
        "Trusted advice too long: got {} bytes, max is {} bytes (set by MemoryConfig.max_trusted_advice_size).",
        trusted_advice.len(),
        memory_config.max_trusted_advice_size,
    );
    assert!(
        untrusted_advice.len() as u64 <= memory_config.max_untrusted_advice_size,
        "Untrusted advice too long: got {} bytes, max is {} bytes (set by MemoryConfig.max_untrusted_advice_size).",
        untrusted_advice.len(),
        memory_config.max_untrusted_advice_size,
    );

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

/// A type that can be used to lazily generate a trace, one [`Cycle`] at a time.
pub trait LazyTracer {
    /// Check if the program execution has terminated.
    fn has_terminated(&self) -> bool;

    /// Check if the program execution has panicked.
    fn has_panicked(&self) -> bool;

    /// Returns whether the next execution of [`LazyTracer::lazy_step_cycle`] will emulate a new
    /// instruction or return the next [`Cycle`] in the last executed instruction.
    fn at_tick_boundary(&self) -> bool;

    /// Print a backtrace, assuming the program has panicked.
    fn print_panic_log(&self);

    /// Get the next [`Cycle`] in the program execution. If the program is at a tick boundary, this
    /// emulates the next instruction. Otherwise, it returns the next cycle within the last
    /// executed instruction.
    fn lazy_step_cycle(&mut self) -> Option<Cycle>;

    /// Take the [`JoltDevice`] from this tracer, consuming the tracer.
    fn get_jolt_device(self) -> JoltDevice;
}

/// An iterator that lazily generates execution traces from a RISC-V emulator checkpoint.
///
/// This iterator produces instruction traces one at a time, executing the emulator
/// as needed rather than generating the entire trace upfront. It buffers traces
/// in `current_traces` since some instructions generate multiple trace entries.
/// When the `current_traces` buffer is exhausted, it executes another emulator tick
/// to generate more.
#[derive(Clone, Debug)]
pub struct GeneralizedLazyTraceIter<T> {
    pub lazy_tracer: T,
}

pub type LazyTraceIterator = GeneralizedLazyTraceIter<CheckpointingTracer>;

unsafe impl<T: Send> Send for GeneralizedLazyTraceIter<T> {}

impl<T: LazyTracer> Iterator for GeneralizedLazyTraceIter<T> {
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
        if self.lazy_tracer.has_terminated() {
            return None;
        }

        let res = self.lazy_tracer.lazy_step_cycle();

        if res.is_none() && self.lazy_tracer.has_panicked() {
            self.lazy_tracer.print_panic_log();
        }

        res
    }
}

impl<T> GeneralizedLazyTraceIter<T> {
    pub fn new(lazy_tracer: T) -> Self {
        Self { lazy_tracer }
    }
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    emulator_state: Emulator,
    prev_pc: u64,
    current_traces: Vec<Cycle>,
    /// The remaining number of cycles that can be replayed for this checkpoint
    trace_steps_remaining: usize,
    /// The total number of cycles executed so far, including the ones prior to this checkpoint
    cycle_count: usize,
}

// SAFETY: Checkpoint contains only owned data and can be safely sent between threads
unsafe impl Send for Checkpoint {}

impl Checkpoint {
    pub(crate) fn new_with_empty_memory(
        emulator_state: &Emulator,
        prev_pc: u64,
        current_traces: &[Cycle],
        cycle_count: usize,
    ) -> Self {
        Self {
            emulator_state: emulator_state.save_state_with_empty_memory(),
            prev_pc,
            current_traces: current_traces.to_vec(),
            trace_steps_remaining: 0,
            cycle_count,
        }
    }

    pub(crate) fn set_memory_state(&mut self, data: MemoryData, cycles_remaining: usize) {
        self.trace_steps_remaining = cycles_remaining;
        self.emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .memory
            .memory
            .data = data;
    }
}

impl LazyTracer for Checkpoint {
    fn has_terminated(&self) -> bool {
        self.trace_steps_remaining == 0
    }

    fn has_panicked(&self) -> bool {
        self.emulator_state
            .get_cpu()
            .mmu
            .jolt_device
            .as_ref()
            .unwrap()
            .panic
    }

    fn at_tick_boundary(&self) -> bool {
        self.current_traces.is_empty()
    }

    fn print_panic_log(&self) {
        error!(
            "Guest program terminated due to panic after {} cycles.",
            self.emulator_state.get_cpu().trace_len
        );
        utils::panic::display_panic_backtrace(&self.emulator_state);
    }

    fn lazy_step_cycle(&mut self) -> Option<Cycle> {
        if !self.current_traces.is_empty() {
            self.trace_steps_remaining -= 1;
            return self.current_traces.pop();
        }

        self.cycle_count += 1;

        step_emulator(
            &mut self.emulator_state,
            &mut self.prev_pc,
            Some(&mut self.current_traces),
        );
        if self.current_traces.is_empty() {
            None
        } else {
            self.trace_steps_remaining -= 1;
            self.current_traces.reverse();
            self.current_traces.pop()
        }
    }

    fn get_jolt_device(mut self) -> JoltDevice {
        self.emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .jolt_device
            .take()
            .unwrap()
    }
}

/// A tracer that uses a `Vec<u64>` memory backend but additionally stores the initial value of
/// each memory access to a [`Checkpoint`], which can be saved and replayed from.
#[derive(Clone, Debug)]
pub struct CheckpointingTracer {
    emulator_state: Emulator,
    prev_pc: u64,
    current_traces: Vec<Cycle>,
    trace_steps_since_last_checkpoint: usize,
    cycle_count: usize,
    finished: bool,
    saved_processor_state: Option<Checkpoint>,
    pub(crate) final_memory_state: Option<Memory>,
}

impl CheckpointingTracer {
    pub fn new(emulator_state: Emulator) -> Self {
        Self {
            emulator_state,
            prev_pc: 0,
            current_traces: vec![],
            trace_steps_since_last_checkpoint: 0,
            cycle_count: 0,
            finished: false,
            saved_processor_state: None,
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

        Self::new(emulator_state)
    }

    /// Start recording memory accesses so that checkpoints can be saved using
    /// [`CheckpointingTracer::save_checkpoint`].
    pub fn start_saving_checkpoints(&mut self) {
        self.saved_processor_state = Some(Checkpoint::new_with_empty_memory(
            &self.emulator_state,
            self.prev_pc,
            &self.current_traces,
            self.cycle_count,
        ));
        self.emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .memory
            .memory
            .data
            .start_saving_checkpoints();
    }

    /// Save the recorded memory traces to a new [`Checkpoint`] and reset the hashmap to which
    /// they're recorded. The chunk of the trace that has been executed since the last call to
    /// [`CheckpointingTracer::save_checkpoint`] or
    /// [`CheckpointingTracer::start_saving_checkpoints`] can be replayed from the resulting
    /// [`Checkpoint`].
    pub fn save_checkpoint(&mut self) -> Checkpoint {
        assert!(self
            .emulator_state
            .get_cpu()
            .mmu
            .memory
            .memory
            .data
            .is_saving_checkpoints());

        // Save the processor state at the start of the current chunk
        let mut new_processor_state = Checkpoint::new_with_empty_memory(
            &self.emulator_state,
            self.prev_pc,
            &self.current_traces,
            self.cycle_count,
        );
        core::mem::swap(
            self.saved_processor_state.as_mut().unwrap(),
            &mut new_processor_state,
        );

        // Store the hashmap of memory assignments since the last chunk
        let data = self
            .emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .memory
            .memory
            .data
            .save_checkpoint();
        new_processor_state.set_memory_state(data, self.trace_steps_since_last_checkpoint);
        self.trace_steps_since_last_checkpoint = 0;

        new_processor_state
    }
}

impl LazyTracer for CheckpointingTracer {
    fn has_terminated(&self) -> bool {
        self.finished
    }

    fn has_panicked(&self) -> bool {
        self.emulator_state
            .get_cpu()
            .mmu
            .jolt_device
            .as_ref()
            .unwrap()
            .panic
    }

    fn at_tick_boundary(&self) -> bool {
        self.current_traces.is_empty()
    }

    fn print_panic_log(&self) {
        error!(
            "Guest program terminated due to panic after {} cycles.",
            self.emulator_state.get_cpu().trace_len
        );
        utils::panic::display_panic_backtrace(&self.emulator_state);
    }

    fn lazy_step_cycle(&mut self) -> Option<Cycle> {
        if !self.current_traces.is_empty() {
            self.trace_steps_since_last_checkpoint += 1;
            return self.current_traces.pop();
        }

        self.cycle_count += 1;

        step_emulator(
            &mut self.emulator_state,
            &mut self.prev_pc,
            Some(&mut self.current_traces),
        );
        if self.current_traces.is_empty() {
            self.finished = true;
            let emulator = &mut self.emulator_state;
            let cpu = emulator.get_mut_cpu();
            self.final_memory_state = Some(cpu.mmu.memory.memory.take_memory());
            None
        } else {
            self.trace_steps_since_last_checkpoint += 1;
            self.current_traces.reverse();
            self.current_traces.pop()
        }
    }

    fn get_jolt_device(mut self) -> JoltDevice {
        self.emulator_state
            .get_mut_cpu()
            .get_mut_mmu()
            .jolt_device
            .take()
            .expect("JoltDevice was not initialized")
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
mod tests {
    use super::*;
    use common::jolt_device::MemoryConfig;

    fn minimal_elf() -> Vec<u8> {
        vec![
            0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0xf3, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
            0x38, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]
    }

    #[test]
    #[should_panic(expected = "Trusted advice too long")]
    fn panics_when_trusted_advice_exceeds_max() {
        let elf = minimal_elf();
        let memory_config = MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 2048,
            ..Default::default()
        };
        let _ = setup_emulator(&elf, b"[]", &[], &[0u8; 4096], &memory_config);
    }

    #[test]
    #[should_panic(expected = "Untrusted advice too long")]
    fn panics_when_untrusted_advice_exceeds_max() {
        let elf = minimal_elf();
        let memory_config = MemoryConfig {
            program_size: Some(1024),
            max_untrusted_advice_size: 128,
            ..Default::default()
        };
        let _ = setup_emulator(&elf, b"[]", &[0u8; 256], &[], &memory_config);
    }

    const ELF_CONTENTS: &[u8] = &[
        127, 69, 76, 70, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 243, 0, 1, 0, 0, 0, 0, 0, 0,
        128, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 160, 23, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 64, 0, 56,
        0, 4, 0, 64, 0, 13, 0, 12, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        128, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 146, 4, 0, 0, 0, 0, 0, 0, 146, 4, 0, 0, 0, 0, 0,
        0, 0, 16, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 146, 20, 0, 0, 0, 0, 0, 0, 146, 4, 0,
        128, 0, 0, 0, 0, 146, 4, 0, 128, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0,
        0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 81, 229, 116, 100, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 112, 4, 0, 0, 0, 9, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 23, 17, 0, 0, 19, 1, 1, 95, 151, 0, 0, 0, 231, 128, 224, 2, 1, 160, 65, 17, 6,
        228, 34, 224, 0, 8, 55, 197, 255, 127, 133, 69, 35, 0, 181, 0, 1, 160, 65, 17, 6, 228, 34,
        224, 0, 8, 151, 0, 0, 0, 231, 128, 64, 254, 1, 17, 6, 236, 55, 165, 255, 127, 131, 5, 5, 0,
        19, 248, 245, 7, 99, 203, 5, 0, 155, 5, 21, 0, 3, 133, 5, 0, 147, 120, 245, 7, 99, 89, 5,
        6, 121, 168, 131, 5, 21, 0, 19, 246, 245, 7, 30, 6, 51, 104, 6, 1, 99, 203, 5, 0, 155, 5,
        37, 0, 3, 133, 5, 0, 147, 120, 245, 7, 99, 87, 5, 4, 173, 168, 5, 37, 3, 6, 21, 0, 147,
        117, 246, 7, 186, 5, 51, 232, 5, 1, 183, 165, 255, 127, 99, 74, 6, 0, 141, 37, 3, 133, 5,
        0, 147, 120, 245, 7, 99, 83, 5, 2, 137, 168, 3, 6, 37, 0, 147, 118, 246, 7, 214, 6, 51,
        232, 6, 1, 99, 76, 6, 0, 145, 37, 3, 133, 5, 0, 147, 120, 245, 7, 99, 74, 5, 2, 19, 133,
        21, 0, 121, 160, 3, 5, 53, 0, 99, 73, 5, 28, 19, 117, 245, 15, 193, 69, 99, 116, 181, 28,
        114, 5, 51, 104, 5, 1, 183, 165, 255, 127, 149, 37, 3, 133, 5, 0, 147, 120, 245, 7, 227,
        90, 5, 252, 3, 133, 21, 0, 19, 118, 245, 7, 30, 6, 179, 104, 22, 1, 99, 69, 5, 0, 19, 133,
        37, 0, 177, 160, 3, 133, 37, 0, 19, 118, 245, 7, 58, 6, 179, 104, 22, 1, 99, 69, 5, 0, 19,
        133, 53, 0, 21, 168, 3, 134, 53, 0, 19, 133, 69, 0, 147, 118, 246, 7, 214, 6, 179, 232, 22,
        1, 99, 80, 6, 2, 3, 5, 5, 0, 99, 65, 5, 22, 19, 118, 245, 15, 65, 69, 99, 124, 166, 20, 19,
        133, 85, 0, 114, 6, 179, 104, 22, 1, 183, 181, 255, 127, 99, 131, 165, 20, 131, 5, 5, 0,
        147, 247, 245, 7, 99, 218, 5, 6, 183, 181, 255, 127, 253, 53, 99, 8, 181, 18, 131, 5, 21,
        0, 19, 246, 245, 7, 30, 6, 209, 143, 99, 221, 5, 4, 183, 181, 255, 127, 249, 53, 99, 11,
        181, 16, 131, 5, 37, 0, 19, 246, 245, 7, 58, 6, 209, 143, 99, 208, 5, 4, 183, 181, 255,
        127, 245, 53, 99, 14, 181, 14, 131, 5, 53, 0, 19, 246, 245, 7, 86, 6, 209, 143, 99, 211, 5,
        2, 183, 181, 255, 127, 241, 53, 99, 1, 181, 14, 3, 5, 69, 0, 99, 77, 5, 12, 19, 117, 245,
        15, 193, 69, 99, 120, 181, 12, 27, 21, 197, 1, 201, 143, 23, 5, 0, 0, 183, 134, 12, 0, 5,
        71, 25, 70, 147, 5, 69, 43, 27, 133, 230, 193, 133, 70, 115, 0, 0, 0, 213, 199, 179, 133,
        8, 3, 19, 8, 193, 0, 151, 6, 0, 0, 187, 215, 245, 2, 25, 70, 147, 133, 6, 41, 62, 198, 131,
        103, 193, 0, 137, 70, 115, 0, 0, 0, 163, 11, 1, 0, 19, 213, 119, 0, 163, 9, 241, 0, 35, 10,
        1, 0, 163, 10, 1, 0, 35, 11, 1, 0, 57, 197, 19, 230, 7, 8, 147, 213, 231, 0, 163, 9, 193,
        0, 35, 10, 161, 0, 9, 71, 141, 205, 19, 102, 5, 8, 19, 213, 87, 1, 35, 10, 193, 0, 163, 10,
        177, 0, 13, 193, 147, 229, 5, 8, 241, 131, 163, 10, 177, 0, 35, 11, 161, 0, 153, 203, 19,
        101, 5, 8, 35, 11, 161, 0, 163, 11, 241, 0, 21, 71, 33, 160, 13, 71, 17, 160, 17, 71, 55,
        181, 255, 127, 147, 5, 49, 1, 58, 134, 151, 0, 0, 0, 231, 128, 0, 2, 55, 197, 255, 127,
        133, 69, 35, 4, 181, 0, 226, 96, 5, 97, 130, 128, 151, 0, 0, 0, 231, 128, 32, 216, 65, 17,
        6, 228, 34, 224, 0, 8, 162, 96, 2, 100, 65, 1, 23, 3, 0, 0, 103, 0, 131, 0, 1, 17, 6, 236,
        34, 232, 38, 228, 0, 16, 193, 70, 99, 98, 214, 6, 187, 6, 160, 64, 19, 248, 118, 0, 179,
        15, 5, 1, 99, 125, 245, 1, 66, 135, 170, 134, 174, 135, 131, 200, 7, 0, 125, 23, 35, 128,
        22, 1, 133, 6, 133, 7, 109, 251, 194, 149, 179, 4, 6, 65, 19, 247, 132, 255, 19, 248, 117,
        0, 179, 134, 239, 0, 99, 22, 8, 4, 99, 250, 223, 0, 174, 135, 144, 99, 35, 176, 207, 0,
        161, 15, 161, 7, 227, 235, 223, 254, 186, 149, 19, 246, 116, 0, 51, 135, 198, 0, 99, 232,
        230, 0, 49, 168, 170, 134, 51, 7, 197, 0, 99, 122, 229, 0, 3, 199, 5, 0, 125, 22, 35, 128,
        230, 0, 133, 6, 133, 5, 109, 250, 226, 96, 66, 100, 162, 100, 5, 97, 130, 128, 129, 72, 33,
        70, 35, 48, 4, 254, 51, 3, 6, 65, 19, 6, 4, 254, 147, 119, 19, 0, 179, 98, 6, 1, 177, 235,
        19, 118, 35, 0, 57, 238, 19, 118, 67, 0, 37, 234, 131, 62, 4, 254, 147, 24, 56, 0, 19, 134,
        143, 0, 51, 143, 5, 65, 99, 127, 214, 6, 59, 6, 16, 65, 147, 115, 134, 3, 131, 50, 143, 0,
        19, 14, 143, 0, 51, 214, 30, 1, 19, 131, 143, 0, 179, 151, 114, 0, 93, 142, 147, 135, 15,
        1, 35, 176, 207, 0, 154, 143, 114, 143, 150, 142, 227, 238, 215, 252, 129, 168, 3, 198, 5,
        0, 35, 128, 194, 0, 133, 72, 19, 118, 35, 0, 93, 210, 51, 134, 21, 1, 3, 22, 6, 0, 179,
        135, 18, 1, 35, 144, 199, 0, 137, 8, 19, 118, 67, 0, 81, 218, 51, 134, 21, 1, 16, 66, 150,
        152, 35, 160, 200, 0, 131, 62, 4, 254, 147, 24, 56, 0, 19, 134, 143, 0, 51, 143, 5, 65,
        227, 101, 214, 248, 246, 130, 122, 142, 126, 131, 129, 71, 147, 3, 142, 0, 17, 70, 35, 48,
        4, 254, 99, 117, 200, 4, 19, 246, 37, 0, 41, 234, 19, 246, 21, 0, 9, 202, 190, 147, 3, 200,
        3, 0, 19, 6, 4, 254, 93, 142, 35, 0, 6, 1, 3, 56, 4, 254, 179, 215, 18, 1, 59, 6, 16, 65,
        19, 118, 134, 3, 51, 22, 200, 0, 93, 142, 35, 48, 195, 0, 186, 149, 19, 246, 116, 0, 51,
        135, 198, 0, 227, 238, 230, 236, 229, 181, 3, 166, 3, 0, 35, 32, 196, 254, 145, 71, 19,
        246, 37, 0, 77, 218, 51, 134, 243, 0, 3, 24, 6, 0, 19, 6, 4, 254, 93, 142, 35, 16, 6, 1,
        137, 7, 19, 246, 21, 0, 89, 254, 117, 183, 109, 117, 108, 100, 105, 118, 16, 0, 0, 0, 0, 0,
        0, 0, 1, 122, 82, 0, 1, 120, 1, 1, 27, 12, 2, 0, 28, 0, 0, 0, 24, 0, 0, 0, 94, 251, 255,
        255, 20, 0, 0, 0, 0, 66, 14, 16, 68, 129, 1, 136, 2, 66, 12, 8, 0, 0, 0, 0, 28, 0, 0, 0,
        56, 0, 0, 0, 82, 251, 255, 255, 16, 0, 0, 0, 0, 66, 14, 16, 68, 129, 1, 136, 2, 66, 12, 8,
        0, 0, 0, 0, 32, 0, 0, 0, 88, 0, 0, 0, 66, 251, 255, 255, 118, 2, 0, 0, 0, 66, 14, 32, 66,
        129, 1, 10, 3, 102, 2, 193, 66, 14, 0, 66, 11, 0, 0, 0, 40, 0, 0, 0, 124, 0, 0, 0, 148,
        253, 255, 255, 22, 0, 0, 0, 0, 66, 14, 16, 68, 129, 1, 136, 2, 66, 12, 8, 0, 12, 2, 16, 68,
        193, 200, 66, 14, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 168, 0, 0, 0, 126, 253, 255, 255, 208,
        1, 0, 0, 0, 66, 14, 32, 70, 129, 1, 136, 2, 137, 3, 66, 12, 8, 0, 10, 2, 128, 12, 2, 32,
        70, 193, 200, 201, 66, 14, 0, 66, 11, 0, 0, 0, 0, 0, 0, 114, 117, 115, 116, 99, 32, 118,
        101, 114, 115, 105, 111, 110, 32, 49, 46, 56, 56, 46, 48, 32, 40, 54, 98, 48, 48, 98, 99,
        51, 56, 56, 32, 50, 48, 50, 53, 45, 48, 54, 45, 50, 51, 41, 0, 76, 105, 110, 107, 101, 114,
        58, 32, 76, 76, 68, 32, 50, 48, 46, 49, 46, 53, 32, 40, 104, 116, 116, 112, 115, 58, 47,
        47, 103, 105, 116, 104, 117, 98, 46, 99, 111, 109, 47, 114, 117, 115, 116, 45, 108, 97,
        110, 103, 47, 108, 108, 118, 109, 45, 112, 114, 111, 106, 101, 99, 116, 46, 103, 105, 116,
        32, 99, 49, 49, 49, 56, 102, 100, 98, 98, 51, 48, 50, 52, 49, 53, 55, 100, 102, 55, 102,
        52, 99, 102, 101, 55, 54, 53, 102, 50, 98, 48, 98, 52, 51, 51, 57, 101, 56, 97, 50, 41, 0,
        0, 65, 70, 0, 0, 0, 114, 105, 115, 99, 118, 0, 1, 60, 0, 0, 0, 4, 16, 5, 114, 118, 54, 52,
        105, 50, 112, 49, 95, 109, 50, 112, 48, 95, 97, 50, 112, 49, 95, 99, 50, 112, 48, 95, 122,
        109, 109, 117, 108, 49, 112, 48, 95, 122, 97, 97, 109, 111, 49, 112, 48, 95, 122, 97, 108,
        114, 115, 99, 49, 112, 48, 0, 0, 46, 116, 101, 120, 116, 46, 98, 111, 111, 116, 0, 46, 116,
        101, 120, 116, 46, 117, 110, 108, 105, 107, 101, 108, 121, 46, 95, 90, 78, 52, 99, 111,
        114, 101, 57, 112, 97, 110, 105, 99, 107, 105, 110, 103, 57, 112, 97, 110, 105, 99, 95,
        102, 109, 116, 49, 55, 104, 54, 98, 51, 52, 51, 50, 57, 51, 57, 52, 97, 98, 52, 51, 53, 49,
        69, 0, 46, 116, 101, 120, 116, 46, 117, 110, 108, 105, 107, 101, 108, 121, 46, 95, 90, 78,
        52, 99, 111, 114, 101, 54, 114, 101, 115, 117, 108, 116, 49, 51, 117, 110, 119, 114, 97,
        112, 95, 102, 97, 105, 108, 101, 100, 49, 55, 104, 98, 54, 101, 99, 56, 99, 99, 48, 52,
        102, 51, 101, 49, 99, 51, 97, 69, 0, 46, 116, 101, 120, 116, 46, 109, 97, 105, 110, 0, 46,
        116, 101, 120, 116, 46, 109, 101, 109, 99, 112, 121, 0, 46, 116, 101, 120, 116, 46, 95, 90,
        78, 49, 55, 99, 111, 109, 112, 105, 108, 101, 114, 95, 98, 117, 105, 108, 116, 105, 110,
        115, 51, 109, 101, 109, 54, 109, 101, 109, 99, 112, 121, 49, 55, 104, 98, 57, 52, 55, 54,
        99, 102, 57, 98, 48, 102, 101, 57, 55, 57, 55, 69, 0, 46, 114, 111, 100, 97, 116, 97, 46,
        46, 76, 97, 110, 111, 110, 46, 57, 57, 57, 56, 48, 56, 99, 102, 57, 52, 57, 51, 100, 50,
        99, 99, 101, 49, 97, 97, 98, 54, 57, 55, 56, 49, 55, 53, 53, 52, 99, 102, 46, 49, 54, 0,
        46, 101, 104, 95, 102, 114, 97, 109, 101, 0, 46, 98, 115, 115, 0, 46, 99, 111, 109, 109,
        101, 110, 116, 0, 46, 114, 105, 115, 99, 118, 46, 97, 116, 116, 114, 105, 98, 117, 116,
        101, 115, 0, 46, 115, 104, 115, 116, 114, 116, 97, 98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 16, 0, 0, 0,
        0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 12, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 128, 0, 0, 0, 0,
        18, 16, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0,
        128, 0, 0, 0, 0, 38, 16, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 142, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0,
        0, 0, 54, 0, 0, 128, 0, 0, 0, 0, 54, 16, 0, 0, 0, 0, 0, 0, 118, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 153, 0, 0, 0, 1, 0, 0, 0,
        6, 0, 0, 0, 0, 0, 0, 0, 172, 2, 0, 128, 0, 0, 0, 0, 172, 18, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 166, 0,
        0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 194, 2, 0, 128, 0, 0, 0, 0, 194, 18, 0, 0, 0, 0,
        0, 0, 208, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 226, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 146, 4, 0, 128, 0, 0, 0, 0,
        146, 20, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 152, 4,
        0, 128, 0, 0, 0, 0, 152, 20, 0, 0, 0, 0, 0, 0, 216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 1, 0, 0, 8, 0, 0, 0, 3, 0, 0, 0,
        0, 0, 0, 0, 112, 5, 0, 128, 0, 0, 0, 0, 112, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 1, 0, 0, 1, 0,
        0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 21, 0, 0, 0, 0, 0, 0, 153, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        45, 1, 0, 0, 3, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 22, 0, 0, 0,
        0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 63, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80,
        22, 0, 0, 0, 0, 0, 0, 73, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const INPUTS: [u8; 6] = [0xbd, 0xaa, 0xde, 0x5, 0x11, 0x5c];

    #[test]
    /// Test that the trace function produces the expected number of cycles for a given ELF input.
    /// Test the checkpointing functionality by verifying the number of checkpoints created and
    /// if the traces from checkpoints match the overall execution trace.
    /// The test is based on the muldiv benchmark.
    fn test_checkpoints() {
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
            let ti: Vec<Cycle> = GeneralizedLazyTraceIter::new(checkpoint).collect();
            assert_eq!(trace_chunk[i], ti);
        }
    }

    #[test]
    fn test_trace_length() {
        let elf = ELF_CONTENTS.to_vec();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };

        let (_, execution_trace, _, _) = trace(&elf, None, &INPUTS, &[], &[], &memory_config);
        let mut emulator: Emulator = setup_emulator(&elf, &INPUTS, &[], &[], &memory_config);
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
