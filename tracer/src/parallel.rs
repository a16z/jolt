//! Two-pass parallel tracing internals.
//!
//! Pass-1 drives the execute-only path (bit-identical CPU state to trace
//! mode at every tick boundary, see `Instruction::execute`) and cuts the
//! program into chunks at tick boundaries: a [`ChunkCheckpoint`] plus a
//! pooled full flat-memory image ([`SnapshotPool`]). Workers
//! ([`ChunkWorker`]) install a chunk and re-trace it in trace mode,
//! producing exactly the rows the serial tracer would have produced for
//! those ticks.

use crate::emulator::cpu::{ChunkCpuState, Cpu, HostIo};
use crate::emulator::decode_cache::DecodeCache;
use crate::emulator::memory::Memory;
use crate::emulator::mmu::ChunkMmuState;
use crate::emulator::terminal::DummyTerminal;
use crate::emulator::Emulator;
use crate::instruction::Cycle;

/// Everything needed to seed a bit-exact trace-mode replay from a tick
/// boundary, except the memory image (pooled separately — see
/// [`SnapshotPool`]).
#[derive(Clone, Debug)]
pub struct ChunkCheckpoint {
    cpu: ChunkCpuState,
    mmu: ChunkMmuState,
    /// JoltDevice outputs at the boundary: the guest can read outputs back,
    /// so they are live per-chunk state (inputs/advice regions are static).
    outputs: Vec<u8>,
    panic: bool,
    /// PC of the tick before the boundary (termination-heuristic seed; kept
    /// for asserts — workers replay by tick count, not by stall detection).
    pub prev_pc: u64,
}

impl ChunkCheckpoint {
    /// Capture at a tick boundary of a pass-1 emulator.
    #[expect(clippy::expect_used)]
    pub fn capture(emulator: &Emulator, prev_pc: u64) -> Self {
        let cpu = emulator.get_cpu();
        let device = cpu
            .mmu
            .jolt_device
            .as_ref()
            .expect("JoltDevice was not initialized");
        ChunkCheckpoint {
            cpu: cpu.capture_chunk_state(),
            mmu: cpu.mmu.capture_chunk_state(),
            outputs: device.outputs.clone(),
            panic: device.panic,
            prev_pc,
        }
    }

    /// Paranoia check: first difference between this captured boundary state
    /// and `cpu`'s current state (`None` = equal). A worker finishing chunk k
    /// must land exactly on checkpoint k+1's capture.
    pub fn diff_vs_cpu(&self, cpu: &Cpu) -> Option<String> {
        if let Some(diff) = self.cpu.diff_vs_cpu(cpu) {
            return Some(diff);
        }
        match cpu.mmu.jolt_device.as_ref() {
            Some(device) => {
                if self.outputs != device.outputs {
                    return Some(format!(
                        "jolt_device.outputs: {} bytes vs {} bytes (or contents differ)",
                        self.outputs.len(),
                        device.outputs.len()
                    ));
                }
                if self.panic != device.panic {
                    return Some(format!(
                        "jolt_device.panic: {} vs {}",
                        self.panic, device.panic
                    ));
                }
            }
            None => return Some("jolt_device: absent on cpu".to_string()),
        }
        None
    }
}

/// Pool of full flat-memory images. Capturing a snapshot memcpys the live
/// image into a pooled buffer; the buffer is handed to a worker as its
/// working memory (zero-copy install) and returns to the pool when the
/// worker installs its next chunk.
#[derive(Debug, Default)]
pub struct SnapshotPool {
    free: Vec<Vec<u64>>,
}

impl SnapshotPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot `memory`'s full flat image into a pooled buffer.
    pub fn capture(&mut self, memory: &Memory) -> Vec<u64> {
        let mut buf = self.free.pop().unwrap_or_default();
        memory.data.clone_flat_into(&mut buf);
        buf
    }

    /// Return a buffer for reuse.
    pub fn put(&mut self, buf: Vec<u64>) {
        self.free.push(buf);
    }
}

/// A resident chunk-replay worker: one CPU reused across chunks, running in
/// [`HostIo::Replay`] (no duplicate prints/markers/advice-appends, no call
/// tracking) for its whole lifetime.
pub struct ChunkWorker {
    cpu: Cpu,
    /// Empty-entry decode cache covering the program text, cloned per chunk:
    /// entries must not leak across out-of-order chunks (store invalidation
    /// only sees the chunks this worker happened to replay).
    decode_template: DecodeCache,
}

impl ChunkWorker {
    /// Build a worker from the pass-1 emulator's static parts (JoltDevice
    /// inputs/advice/layout, program text range). Independent of pass-1
    /// progress; outputs and panic are patched per chunk.
    pub fn new(emulator: &Emulator) -> Self {
        let src = emulator.get_cpu();
        let mut cpu = Cpu::new(Box::new(DummyTerminal {}));
        cpu.set_host_io(HostIo::Replay);
        cpu.mmu.jolt_device = src.mmu.jolt_device.clone();
        let decode_template = src.mmu.decode_cache.snapshot_with_empty_entries();
        cpu.mmu.decode_cache = decode_template.clone();
        ChunkWorker {
            cpu,
            decode_template,
        }
    }

    /// Install a chunk: seed CPU+MMU state, adopt `image` as the working
    /// memory (returns the previous buffer for the pool), patch device
    /// outputs/panic, and reset the decode cache.
    #[expect(clippy::expect_used)]
    pub fn install_chunk(&mut self, checkpoint: &ChunkCheckpoint, image: Vec<u64>) -> Vec<u64> {
        self.cpu.install_chunk_state(&checkpoint.cpu);
        self.cpu.mmu.install_chunk_state(&checkpoint.mmu);
        let previous = self.cpu.mmu.memory.memory.data.replace_flat(image);
        let device = self
            .cpu
            .mmu
            .jolt_device
            .as_mut()
            .expect("worker device installed at construction");
        device.outputs.clone_from(&checkpoint.outputs);
        device.panic = checkpoint.panic;
        self.cpu.mmu.decode_cache = self.decode_template.clone();
        previous
    }

    /// Re-trace `tick_count` ticks, appending rows to `out`. Tick counts are
    /// authoritative — workers never run the PC-stall heuristic.
    pub fn run_ticks(&mut self, tick_count: usize, out: &mut Vec<Cycle>) {
        for _ in 0..tick_count {
            self.cpu.tick(Some(out));
        }
    }

    pub fn cpu(&self) -> &Cpu {
        &self.cpu
    }
}

/// Pass-1 driver: executes the program tick by tick with trace-equivalent
/// state, exposing checkpoint capture at tick boundaries.
pub struct PassOne {
    emulator: Emulator,
    prev_pc: u64,
    done: bool,
}

impl PassOne {
    pub fn new(emulator: Emulator) -> Self {
        Self {
            emulator,
            prev_pc: 0,
            done: false,
        }
    }

    /// Execute one tick. Returns `false` (without ticking further) once the
    /// program has terminated. Termination matches `trace()` exactly: a PC
    /// stall, or a step that would emit zero rows (trap/WFI tick — observed
    /// here as a zero `trace_len` delta, which the execute path shares).
    pub fn step(&mut self) -> bool {
        if self.done {
            return false;
        }
        let pc = self.emulator.get_cpu().read_pc();
        if pc == self.prev_pc {
            self.done = true;
            return false;
        }
        let ticks_before = self.emulator.get_cpu().trace_len;
        self.emulator.tick(None);
        if self.emulator.get_cpu().trace_len == ticks_before {
            self.done = true;
            return false;
        }
        self.prev_pc = pc;
        true
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Capture a chunk checkpoint at the current tick boundary.
    pub fn checkpoint(&self) -> ChunkCheckpoint {
        ChunkCheckpoint::capture(&self.emulator, self.prev_pc)
    }

    pub fn emulator(&self) -> &Emulator {
        &self.emulator
    }

    pub fn into_emulator(self) -> Emulator {
        self.emulator
    }
}
