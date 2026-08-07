#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
extern crate core;

use itertools::Itertools;
use std::vec;
use tracing::{error, info};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use common::{self, jolt_device::MemoryConfig};
use emulator::{cpu, default_terminal::DefaultTerminal};
use instruction::{Cycle, Instruction};
use jolt_riscv::RV64IMAC_JOLT;

pub mod emulator;
pub mod execution_backend;
pub mod instruction;
mod jolt_cycle_adapter;
pub mod parallel;
pub mod trace_row;
pub mod utils;

pub use common::jolt_device::JoltDevice;
pub use cpu::{advice_tape_read, advice_tape_remaining, advice_tape_write, AdviceTape};
pub use execution_backend::TracerBackend;
pub use instruction::inline::{
    list_registered_inlines, InlineAdviceContext, InlineAdviceError, InlineRegistration,
    TracerInlineExpansionProvider,
};
pub use jolt_riscv::InlineExtension;
pub use trace_row::{build_trace_rows, cycle_to_trace_row, CycleConversionError};

use crate::emulator::{
    memory::{Memory, MemoryData},
    Emulator,
};

/// Initial trace capacity, in rows (`JOLT_TRACER_CAPACITY_ROWS` overrides —
/// the same knob the parallel path uses): the default covers the standard
/// 2^23-cycle proving scale without Vec regrowth (each doubling past the
/// hundreds of MB memcpys the whole trace). Reserved address space is only
/// faulted in as rows are pushed.
fn trace_capacity_reserve() -> usize {
    env_rows("JOLT_TRACER_CAPACITY_ROWS", parallel::DEFAULT_CAPACITY_ROWS)
}

/// Positive row-count env override; unset, unparsable, or zero falls back to
/// `default`.
fn env_rows(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&rows| rows > 0)
        .unwrap_or(default)
}

/// Executes a RISC-V program to completion and materializes its execution
/// trace.
///
/// Returns:
/// * `LazyTraceIterator` — an unexecuted iterator over the same program,
///   observing the pristine pre-execution state (the streaming-commitment
///   prover re-executes the program through it)
/// * `Vec<Cycle>` — the complete execution trace
/// * `Memory` — final guest memory state
/// * `JoltDevice` — final I/O device state
/// * `AdviceTape` — the populated advice tape
///
/// Tracing is serial by default. Setting `TRACER_PARALLEL=<workers>` opts
/// into two-pass parallel tracing (bit-identical output); see [`parallel`].
#[tracing::instrument(skip_all)]
pub fn trace(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    advice_tape: Option<cpu::AdviceTape>,
) -> (
    LazyTraceIterator,
    Vec<Cycle>,
    Memory,
    JoltDevice,
    cpu::AdviceTape,
) {
    let mut emulator = create_emulator(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        advice_tape,
    );
    // The returned iterator must observe the pristine pre-execution state:
    // the streaming-commitment prover re-executes the program from it.
    let lazy_trace_iter = LazyTraceIterator::new(CheckpointingTracer::new(emulator.clone()));

    let trace: Vec<Cycle> = match parallel_config_from_env() {
        Some(config) => {
            let (trace, finished) = parallel::run_two_pass(emulator, &config);
            emulator = finished;
            trace
        }
        None => {
            // Drive the emulator straight into the output vec, bypassing the
            // lazy iterator's per-cycle buffer/reverse/pop round-trip.
            // Termination matches the lazy path: stop on the first step that
            // emits no rows (PC stall or a trap that produced no trace).
            let mut trace: Vec<Cycle> = Vec::with_capacity(trace_capacity_reserve());
            let mut prev_pc: u64 = 0;
            loop {
                let rows_before = trace.len();
                step_emulator(&mut emulator, &mut prev_pc, Some(&mut trace));
                if trace.len() == rows_before {
                    break;
                }
            }
            trace
        }
    };

    let (advice_tape_result, final_memory_state, jolt_device) = finish_emulator(emulator);
    (
        lazy_trace_iter,
        trace,
        final_memory_state,
        jolt_device,
        advice_tape_result,
    )
}

/// Shared teardown for every execution path (eager [`trace`], execute-only
/// [`execute`], and the chunked fast pass): report a guest panic (log +
/// backtrace), then extract the advice tape, final memory, and device.
#[expect(clippy::expect_used)]
pub(crate) fn finish_emulator(mut emulator: Emulator) -> (cpu::AdviceTape, Memory, JoltDevice) {
    if emulator
        .get_cpu()
        .mmu
        .jolt_device
        .as_ref()
        .expect("JoltDevice was not initialized")
        .panic
    {
        error!(
            "Guest program terminated due to panic after {} cycles.",
            emulator.get_cpu().trace_len
        );
        utils::panic::display_panic_backtrace(&emulator);
    }

    let advice_tape = emulator.take_advice_tape();
    let cpu = emulator.get_mut_cpu();
    let final_memory = cpu.mmu.memory.memory.take_memory();
    let jolt_device = cpu
        .get_mut_mmu()
        .jolt_device
        .take()
        .expect("JoltDevice was not initialized");
    (advice_tape, final_memory, jolt_device)
}

/// Executes a RISC-V program to completion without materializing trace rows
/// (the emulator's execute-only path). Returns the trace row count (the
/// number of rows [`trace`] would have produced), the final `JoltDevice`,
/// and the populated advice tape.
///
/// This is the fast first-pass seam for two-pass parallel tracing: the CPU
/// state it produces is bit-identical to trace mode at every tick boundary
/// (instructions whose trace path expands a virtual sequence walk the same
/// cached sequence, just without row emission).
///
/// Termination is *almost* the same as [`trace`]: both stop on a PC stall,
/// but [`trace`] additionally stops on any step that emits zero rows (a
/// trap or WFI-sleep tick), which this loop does not check — the parallel
/// pass-1 driver does, via the `trace_len` delta. No valid Jolt guest hits
/// a zero-row step mid-program.
#[tracing::instrument(skip_all)]
pub fn execute(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    advice_tape: Option<cpu::AdviceTape>,
) -> (usize, JoltDevice, cpu::AdviceTape) {
    let mut emulator = create_emulator(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        advice_tape,
    );
    let mut prev_pc: u64 = 0;
    loop {
        let pc = emulator.get_cpu().read_pc();
        if pc == prev_pc {
            break;
        }
        emulator.tick(None);
        prev_pc = pc;
    }

    let executed = emulator.get_cpu().trace_len;
    let (advice_tape_result, _final_memory, jolt_device) = finish_emulator(emulator);
    (executed, jolt_device, advice_tape_result)
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
        None,
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
    advice_tape: Option<cpu::AdviceTape>,
) -> LazyTraceIterator {
    LazyTraceIterator::new(CheckpointingTracer::new(create_emulator(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        advice_tape,
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

/// Opt-in parallel tracing: `TRACER_PARALLEL=<workers>` (unset, 0, 1, or
/// unparsable = serial — a single worker would only re-trace what pass-1
/// already executed); `JOLT_TRACER_CHUNK_ROWS` overrides the default chunk
/// size and `JOLT_TRACER_CAPACITY_ROWS` the up-front output reservation.
fn parallel_config_from_env() -> Option<parallel::TwoPassConfig> {
    let workers: usize = std::env::var("TRACER_PARALLEL").ok()?.parse().ok()?;
    if workers <= 1 {
        return None;
    }
    Some(parallel::TwoPassConfig {
        workers,
        chunk_rows: env_rows("JOLT_TRACER_CHUNK_ROWS", parallel::DEFAULT_CHUNK_ROWS),
        capacity_rows: env_rows("JOLT_TRACER_CAPACITY_ROWS", parallel::DEFAULT_CAPACITY_ROWS),
    })
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
    create_emulator(
        elf_contents,
        None,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        None, // No advice tape by default
    )
}

#[tracing::instrument(skip_all)]
/// Sets up a ready-to-run emulator for a guest program, with access to the
/// elf-path for symbol loading and de-mangling.
///
/// Public seam for drivers that need to tick the emulator themselves
/// (mode-equivalence gates, two-pass parallel tracing).
pub fn create_emulator(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    advice_tape: Option<cpu::AdviceTape>,
) -> Emulator {
    let term = DefaultTerminal::default();
    let mut emulator = Emulator::new(Box::new(term));
    // Set the advice tape if provided
    if let Some(tape) = advice_tape {
        emulator.set_advice_tape(tape);
    }

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
    assert!(
        inputs.len() as u64 <= memory_config.max_input_size,
        "Input too long: got {} bytes, max is {} bytes (set by MemoryConfig.max_input_size).",
        inputs.len(),
        memory_config.max_input_size,
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
        let mmu = self.emulator_state.get_mut_cpu().get_mut_mmu();
        mmu.memory.memory.data.start_saving_checkpoints();
        // Replay needs every executed instruction's text bytes present in the
        // chunk's first-touch memory image, so each interval must start with
        // an empty decode cache (a cache hit skips the recorded fetch). The
        // cache is cleared again at every save_checkpoint; within an interval
        // it works normally.
        mmu.decode_cache.clear_entries();
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
        let mmu = self.emulator_state.get_mut_cpu().get_mut_mmu();
        let data = mmu.memory.memory.data.save_checkpoint();
        // The next interval's first-touch map must see each PC's first fetch
        // re-recorded (see start_saving_checkpoints).
        mmu.decode_cache.clear_entries();
        new_processor_state.set_memory_state(data, self.trace_steps_since_last_checkpoint);
        self.trace_steps_since_last_checkpoint = 0;

        new_processor_state
    }

    /// Take ownership of the advice tape from the emulator, replacing it with an empty one
    pub fn take_advice_tape(&mut self) -> cpu::AdviceTape {
        self.emulator_state.take_advice_tape()
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
            // When checkpoint saving is active, the caller still saves one
            // final checkpoint after termination, which snapshots the live
            // memory's capacity — clone instead of emptying it.
            let memory = &mut cpu.mmu.memory.memory;
            self.final_memory_state = Some(if memory.data.is_saving_checkpoints() {
                memory.clone()
            } else {
                memory.take_memory()
            });
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
pub fn decode(elf: &[u8]) -> (Vec<Instruction>, Vec<(u64, u8)>, u64, u64) {
    let obj = object::File::parse(elf).unwrap();
    if matches!(&obj, object::File::Elf32(_)) {
        panic!("tracer only supports RV64 ELF inputs");
    }

    let image = jolt_program::image::decode_elf(elf, RV64IMAC_JOLT)
        .expect("jolt-program ELF64 decoding failed");
    let instructions = image
        .instructions
        .into_iter()
        .map(|instruction| {
            Instruction::try_from_source_instruction(instruction)
                .expect("jolt-program image decoder produced an unknown tracer row")
        })
        .collect();
    (
        instructions,
        image.memory_init,
        image.program_end,
        image.entry_address,
    )
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
pub(crate) mod test_utils {
    /// Build the muldiv guest and return the ELF bytes.
    /// Mirrors the pattern used by `host::Program::build()` in jolt-prover-legacy.
    pub(crate) fn build_muldiv_guest() -> Vec<u8> {
        let guest = "muldiv-guest";
        let func = "muldiv";
        let target_dir = format!("/tmp/jolt-guest-targets/{guest}-{func}");

        let output = std::process::Command::new("jolt")
            .args([
                "build",
                "-p",
                guest,
                "--stack-size",
                &common::constants::DEFAULT_STACK_SIZE.to_string(),
                "--heap-size",
                "32768",
                "--",
                "--release",
                "--target-dir",
                &target_dir,
                "--features",
                "guest",
            ])
            .env("JOLT_FUNC_NAME", func)
            .output()
            .expect("failed to run jolt CLI — install with: cargo install --path .");

        if !output.status.success() {
            panic!(
                "failed to build muldiv guest:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let elf_path = format!("{target_dir}/riscv64imac-unknown-none-elf/release/{guest}");
        std::fs::read(&elf_path).unwrap_or_else(|e| panic!("failed to read ELF at {elf_path}: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::build_muldiv_guest;
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

    fn minimal_elf32() -> Vec<u8> {
        let mut elf = vec![0; 52];
        elf[0..4].copy_from_slice(b"\x7fELF");
        elf[4] = 1; // ELFCLASS32
        elf[5] = 1; // little endian
        elf[6] = 1; // current ELF version
        elf[16..18].copy_from_slice(&2u16.to_le_bytes()); // executable
        elf[18..20].copy_from_slice(&243u16.to_le_bytes()); // RISC-V
        elf[20..24].copy_from_slice(&1u32.to_le_bytes());
        elf[40..42].copy_from_slice(&52u16.to_le_bytes());
        elf[42..44].copy_from_slice(&32u16.to_le_bytes());
        elf[46..48].copy_from_slice(&40u16.to_le_bytes());
        elf
    }

    #[test]
    #[should_panic(expected = "tracer only supports RV64 ELF inputs")]
    fn decode_rejects_elf32() {
        decode(&minimal_elf32());
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
    #[should_panic(expected = "Input too long")]
    fn panics_when_input_exceeds_max() {
        let elf = minimal_elf();
        let memory_config = MemoryConfig {
            program_size: Some(1024),
            max_input_size: 64,
            ..Default::default()
        };
        let _ = setup_emulator(&elf, &[0u8; 128], &[], &[], &memory_config);
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

    const INPUTS: [u8; 6] = [0xbd, 0xaa, 0xde, 0x5, 0x11, 0x5c];

    #[test]
    /// Test that the trace function produces the expected number of cycles for a given ELF input.
    /// Test the checkpointing functionality by verifying the number of checkpoints created and
    /// if the traces from checkpoints match the overall execution trace.
    /// The test is based on the muldiv benchmark.
    fn test_checkpoints() {
        let elf = build_muldiv_guest();
        let n = 50;
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let (_, execution_trace, _, _, _) =
            trace(&elf, None, &INPUTS, &[], &[], &memory_config, None);
        let (checkpoints, _) = trace_checkpoints(&elf, &INPUTS, &[], &[], &memory_config, n);
        assert!(
            !execution_trace.is_empty(),
            "execution trace should not be empty"
        );
        assert_eq!(checkpoints.len(), execution_trace.len().div_ceil(n));

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
    /// Execute-mode CPU state must be bit-identical to trace-mode state at
    /// every tick boundary (foundation of two-pass parallel tracing: pass-1
    /// runs execute-mode and its checkpoints seed trace-mode chunk replays).
    fn test_execute_trace_state_lockstep() {
        let elf = build_muldiv_guest();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let mut em_trace = setup_emulator(&elf, &INPUTS, &[], &[], &memory_config);
        let mut em_exec = setup_emulator(&elf, &INPUTS, &[], &[], &memory_config);

        let mut rows: Vec<Cycle> = Vec::new();
        let mut prev_pc: u64 = 0;
        let mut tick_idx: usize = 0;
        loop {
            let pc = em_trace.get_cpu().read_pc();
            assert_eq!(pc, em_exec.get_cpu().read_pc(), "pc at tick {tick_idx}");
            if pc == prev_pc {
                break;
            }
            rows.clear();
            em_trace.tick(Some(&mut rows));
            em_exec.tick(None);
            assert!(!rows.is_empty(), "zero-row tick at {tick_idx}");
            if let Some(diff) = em_trace.get_cpu().arch_state_diff(em_exec.get_cpu()) {
                panic!("state diverged at tick {tick_idx}: {diff}");
            }
            prev_pc = pc;
            tick_idx += 1;
        }
        assert!(tick_idx > 0, "program did not execute");
        // trace_len is row-uniform across modes (execute mode counts
        // suppressed rows).
        assert_eq!(em_trace.get_cpu().trace_len, em_exec.get_cpu().trace_len);
        assert_eq!(
            em_trace
                .get_cpu()
                .mmu
                .memory
                .memory
                .materialized_nonzero_bytes(),
            em_exec
                .get_cpu()
                .mmu
                .memory
                .memory
                .materialized_nonzero_bytes(),
            "final memory diverged"
        );
    }

    /// Serial trace-mode reference: all rows plus per-tick row counts
    /// (termination identical to `trace()`).
    fn serial_reference(elf: &[u8], memory_config: &MemoryConfig) -> (Vec<Cycle>, Vec<usize>) {
        let mut emulator = setup_emulator(elf, &INPUTS, &[], &[], memory_config);
        let mut rows: Vec<Cycle> = Vec::new();
        let mut rows_per_tick: Vec<usize> = Vec::new();
        let mut prev_pc: u64 = 0;
        loop {
            let pc = emulator.get_cpu().read_pc();
            if pc == prev_pc {
                break;
            }
            let before = rows.len();
            emulator.tick(Some(&mut rows));
            if rows.len() == before {
                break;
            }
            rows_per_tick.push(rows.len() - before);
            prev_pc = pc;
        }
        (rows, rows_per_tick)
    }

    #[test]
    /// Capture a chunk checkpoint at tick N of an execute-mode pass, replay
    /// M ticks in a fresh worker, and require the emitted rows to be
    /// bit-exact against the serial trace's rows for ticks N..N+M.
    fn test_chunk_capture_replay() {
        use crate::parallel::{ChunkWorker, PassOne, SnapshotPool};

        let elf = build_muldiv_guest();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let (all_rows, rows_per_tick) = serial_reference(&elf, &memory_config);
        let total_ticks = rows_per_tick.len();
        assert!(total_ticks > 150, "muldiv should run a few hundred ticks");
        let row_offset = |tick: usize| rows_per_tick[..tick].iter().sum::<usize>();

        let cases = [
            (0usize, 40usize),
            (7, 64),
            (100, 100),
            (total_ticks - 25, 25),
        ];
        for (n, m) in cases {
            let mut pass1 = PassOne::new(setup_emulator(&elf, &INPUTS, &[], &[], &memory_config));
            for _ in 0..n {
                assert!(pass1.step(), "pass-1 ended before tick {n}");
            }
            let checkpoint = pass1.checkpoint();
            let mut pool = SnapshotPool::new();
            let image = pool.capture(&pass1.emulator().get_cpu().mmu.memory.memory);

            let mut worker = ChunkWorker::new(pass1.emulator());
            let _previous = worker.install_chunk(&checkpoint, image);
            let mut rows: Vec<Cycle> = Vec::new();
            worker.run_ticks(m, &mut rows);

            let expected = &all_rows[row_offset(n)..row_offset(n + m)];
            assert_eq!(
                rows.as_slice(),
                expected,
                "rows differ for chunk N={n} M={m}"
            );
        }
    }

    #[test]
    /// Cut the whole program into fixed-tick chunks from a single continuous
    /// execute-mode pass, replay every chunk through one resident worker,
    /// and require (a) each boundary state to match the next checkpoint
    /// exactly and (b) the concatenated rows to equal the serial trace.
    fn test_chunked_replay_reconstructs_full_trace() {
        use crate::parallel::{ChunkCheckpoint, ChunkWorker, PassOne, SnapshotPool};

        let elf = build_muldiv_guest();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let (all_rows, rows_per_tick) = serial_reference(&elf, &memory_config);
        let total_ticks = rows_per_tick.len();

        for chunk_ticks in [1usize, 29, 97, total_ticks + 5] {
            let mut pass1 = PassOne::new(setup_emulator(&elf, &INPUTS, &[], &[], &memory_config));
            let mut pool = SnapshotPool::new();
            let mut chunks: Vec<(ChunkCheckpoint, Vec<u64>, usize)> = Vec::new();
            loop {
                let checkpoint = pass1.checkpoint();
                let image = pool.capture(&pass1.emulator().get_cpu().mmu.memory.memory);
                let mut ticks = 0;
                while ticks < chunk_ticks && pass1.step() {
                    ticks += 1;
                }
                if ticks > 0 {
                    chunks.push((checkpoint, image, ticks));
                } else {
                    pool.put(image);
                }
                if pass1.is_done() {
                    break;
                }
            }
            assert_eq!(
                chunks.iter().map(|(_, _, t)| t).sum::<usize>(),
                total_ticks,
                "chunk tick counts must cover the program"
            );

            let mut worker = ChunkWorker::new(pass1.emulator());
            let mut replayed: Vec<Cycle> = Vec::new();
            for (idx, (checkpoint, image, ticks)) in chunks.iter().enumerate() {
                let previous = worker.install_chunk(checkpoint, image.clone());
                pool.put(previous);
                worker.run_ticks(*ticks, &mut replayed);
                // Boundary paranoia: worker end state must equal the next
                // chunk's captured start state.
                if let Some((next, _, _)) = chunks.get(idx + 1) {
                    if let Some(diff) = next.diff_vs_cpu(worker.cpu()) {
                        panic!("boundary mismatch after chunk {idx} (size {chunk_ticks}): {diff}");
                    }
                }
            }
            assert_eq!(
                replayed, all_rows,
                "replayed rows differ at chunk size {chunk_ticks}"
            );
        }
    }

    #[test]
    /// The threaded two-pass pipeline must reproduce the serial trace
    /// bit-exactly across chunk sizes (many tiny chunks → single-chunk
    /// degenerate) and worker counts.
    fn test_run_two_pass_matches_serial() {
        use crate::parallel::{run_two_pass, TwoPassConfig};

        let elf = build_muldiv_guest();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let (serial_rows, _) = serial_reference(&elf, &memory_config);

        // capacity_rows=100 forces the overflow fallback path on muldiv's
        // 473 rows (windowed prefix, then copy-assembled suffix).
        for (workers, chunk_rows, capacity_rows) in [
            (1usize, 64usize, 1usize << 24),
            (4, 64, 1 << 24),
            (4, 128, 1 << 24),
            (4, 1 << 21, 1 << 24),
            (4, 64, 100),
            (4, 64, 1),
        ] {
            let emulator = setup_emulator(&elf, &INPUTS, &[], &[], &memory_config);
            let (rows, finished) = run_two_pass(
                emulator,
                &TwoPassConfig {
                    workers,
                    chunk_rows,
                    capacity_rows,
                },
            );
            assert_eq!(
                rows, serial_rows,
                "two-pass rows differ (workers={workers}, chunk_rows={chunk_rows}, capacity_rows={capacity_rows})"
            );
            assert_eq!(finished.get_cpu().trace_len, serial_rows.len());
        }
    }

    #[test]
    /// A replay divergence must fail the trace call promptly and loudly.
    /// Regression test for the all-workers-dead hang: with a systematic
    /// row-count divergence (fault-injected), every worker panics on its
    /// first chunk; a blocking `send` on the full job queue would then hang
    /// pass-1 forever instead of propagating the panic.
    fn test_worker_divergence_panics_instead_of_hanging() {
        use crate::parallel::{run_two_pass, TwoPassConfig, TEST_CORRUPT_ROW_COUNTS};
        use std::sync::atomic::Ordering;
        use std::sync::mpsc;
        use std::time::Duration;

        let elf = build_muldiv_guest();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let emulator = setup_emulator(&elf, &INPUTS, &[], &[], &memory_config);
        TEST_CORRUPT_ROW_COUNTS.store(true, Ordering::Relaxed);

        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_two_pass(
                    emulator,
                    &TwoPassConfig {
                        workers: 1,
                        chunk_rows: 64,
                        capacity_rows: 1 << 24,
                    },
                )
            }));
            let _ = done_tx.send(result.is_err());
        });

        match done_rx.recv_timeout(Duration::from_secs(60)) {
            Ok(panicked) => assert!(panicked, "a corrupted chunk row count must panic the trace"),
            Err(_) => {
                panic!("two-pass trace hung on worker divergence instead of panicking")
            }
        }
    }

    /// A count-preserving replay divergence must trip the boundary-state
    /// verification. Fault injection corrupts one worker register after
    /// replay (row counts stay equal), so only the boundary check can catch
    /// it; without it the trace would be assembled silently.
    fn boundary_divergence_panics(workers: usize) {
        use crate::parallel::{run_two_pass, TwoPassConfig, TEST_CORRUPT_BOUNDARY_STATE};
        use std::sync::atomic::Ordering;
        use std::sync::mpsc;
        use std::time::Duration;

        let elf = build_muldiv_guest();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let emulator = setup_emulator(&elf, &INPUTS, &[], &[], &memory_config);
        TEST_CORRUPT_BOUNDARY_STATE.store(true, Ordering::Relaxed);

        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_two_pass(
                    emulator,
                    &TwoPassConfig {
                        workers,
                        chunk_rows: 64,
                        capacity_rows: 1 << 24,
                    },
                )
            }));
            let _ = done_tx.send(result.is_err());
        });

        match done_rx.recv_timeout(Duration::from_secs(60)) {
            Ok(panicked) => assert!(
                panicked,
                "a count-preserving state divergence must panic the trace"
            ),
            Err(_) => {
                panic!("two-pass trace hung on boundary divergence instead of panicking")
            }
        }
    }

    #[test]
    fn test_boundary_divergence_panics() {
        boundary_divergence_panics(1);
    }

    /// Multi-worker variant: the first tripped worker panics while its
    /// siblings may be blocked awaiting end boundaries pass-1 will never
    /// publish (its own boundary assert fires first) — the interleaving that
    /// hangs without the pass-1-side panic guard and the polling wait.
    #[test]
    fn test_boundary_divergence_panics_multiworker() {
        boundary_divergence_panics(4);
    }

    #[test]
    fn test_trace_length() {
        let elf = build_muldiv_guest();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };

        let (_, execution_trace, _, _, _) =
            trace(&elf, None, &INPUTS, &[], &[], &memory_config, None);
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
