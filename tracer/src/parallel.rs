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
        Self::from_seed(
            src.mmu.jolt_device.clone(),
            src.mmu.decode_cache.snapshot_with_empty_entries(),
        )
    }

    /// Like [`ChunkWorker::new`], but from pre-cloned static parts — the
    /// pipeline takes the seed before pass-1 starts mutating the emulator
    /// and each worker thread builds its own CPU from it.
    pub fn from_seed(
        jolt_device: Option<common::jolt_device::JoltDevice>,
        decode_template: DecodeCache,
    ) -> Self {
        let mut cpu = Cpu::new(Box::new(DummyTerminal {}));
        cpu.set_host_io(HostIo::Replay);
        cpu.mmu.jolt_device = jolt_device;
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

    /// Trace rows produced so far (`trace_len` is row-uniform across modes:
    /// the execute path counts each suppressed row).
    pub fn rows(&self) -> usize {
        self.emulator.get_cpu().trace_len
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

/// Default chunk size in rows (~2M): large enough that snapshot memcpy and
/// dispatch stay ≪ chunk replay time, small enough that the final worker
/// wave doesn't dominate the wall clock.
pub const DEFAULT_CHUNK_ROWS: usize = 1 << 21;

#[derive(Clone, Copy, Debug)]
pub struct TwoPassConfig {
    pub workers: usize,
    /// Chunks close at the first tick boundary at or past this many rows.
    pub chunk_rows: usize,
}

struct ChunkJob {
    index: usize,
    checkpoint: ChunkCheckpoint,
    image: Vec<u64>,
    ticks: usize,
    rows: usize,
}

/// Two-pass parallel trace: pass-1 executes on the calling thread, cutting
/// row-bounded chunks into a bounded job queue; `workers` threads re-trace
/// chunks concurrently; rows are assembled in chunk order afterwards.
///
/// Returns the assembled trace (bit-identical to what `Cpu::tick(Some(..))`
/// over the whole program produces) and the finished pass-1 emulator, which
/// is the authoritative source for final memory, device and advice state.
///
/// In-flight memory is bounded by the job-queue depth: at most
/// `2*workers (queued) + workers (executing) + 1 (capturing)` full images.
#[expect(clippy::expect_used)]
pub fn run_two_pass(emulator: Emulator, config: &TwoPassConfig) -> (Vec<Cycle>, Emulator) {
    use std::sync::mpsc;
    use std::sync::Mutex;

    let workers = config.workers.max(1);
    let chunk_rows = config.chunk_rows.max(1);

    // Static worker seed, taken before pass-1 starts mutating the emulator.
    let seed_device = emulator.get_cpu().mmu.jolt_device.clone();
    let seed_decode = emulator
        .get_cpu()
        .mmu
        .decode_cache
        .snapshot_with_empty_entries();

    let (job_tx, job_rx) = mpsc::sync_channel::<ChunkJob>(2 * workers);
    let job_rx = Mutex::new(job_rx);
    let (buf_tx, buf_rx) = mpsc::channel::<Vec<u64>>();
    let (out_tx, out_rx) = mpsc::channel::<(usize, Vec<Cycle>)>();

    let mut pass1 = PassOne::new(emulator);
    let mut chunk_count = 0usize;

    std::thread::scope(|scope| {
        for _ in 0..workers {
            let job_rx = &job_rx;
            let buf_tx = buf_tx.clone();
            let out_tx = out_tx.clone();
            let seed_device = seed_device.clone();
            let seed_decode = seed_decode.clone();
            scope.spawn(move || {
                let mut worker = ChunkWorker::from_seed(seed_device, seed_decode);
                loop {
                    // Hold the lock only for the dequeue; idle workers block
                    // here, which is fine — there is no work for them anyway.
                    let job = {
                        let receiver = job_rx.lock().expect("job queue lock poisoned");
                        receiver.recv()
                    };
                    let Ok(job) = job else { break };
                    let previous = worker.install_chunk(&job.checkpoint, job.image);
                    // First install returns the (empty) construction-time
                    // backing; recycling it is harmless.
                    let _ = buf_tx.send(previous);
                    let mut rows: Vec<Cycle> = Vec::with_capacity(job.rows);
                    worker.run_ticks(job.ticks, &mut rows);
                    assert_eq!(
                        rows.len(),
                        job.rows,
                        "chunk {}: replay emitted a different row count than pass-1",
                        job.index
                    );
                    out_tx.send((job.index, rows)).expect("collector hung up");
                }
            });
        }
        // Workers hold clones; the originals must drop so the collector's
        // recv loop terminates once all workers exit.
        drop(out_tx);
        drop(buf_tx);

        // Pass-1 on the calling thread.
        let mut pool = SnapshotPool::new();
        loop {
            while let Ok(buffer) = buf_rx.try_recv() {
                pool.put(buffer);
            }
            let checkpoint = pass1.checkpoint();
            let image = pool.capture(&pass1.emulator().get_cpu().mmu.memory.memory);
            let rows_before = pass1.rows();
            let mut ticks = 0usize;
            while pass1.rows() - rows_before < chunk_rows && pass1.step() {
                ticks += 1;
            }
            let rows = pass1.rows() - rows_before;
            if ticks == 0 {
                pool.put(image);
                break;
            }
            job_tx
                .send(ChunkJob {
                    index: chunk_count,
                    checkpoint,
                    image,
                    ticks,
                    rows,
                })
                .expect("worker pool hung up");
            chunk_count += 1;
            if pass1.is_done() {
                break;
            }
        }
        drop(job_tx);
    });

    // All workers joined; collect and assemble in chunk order.
    let mut per_chunk: Vec<Option<Vec<Cycle>>> = (0..chunk_count).map(|_| None).collect();
    while let Ok((index, rows)) = out_rx.recv() {
        per_chunk[index] = Some(rows);
    }
    let total: usize = per_chunk
        .iter()
        .map(|chunk| chunk.as_ref().map_or(0, Vec::len))
        .sum();
    // Next-pow2 capacity so the prover's padding resize never reallocates;
    // untouched capacity is lazily committed by the OS.
    let mut trace: Vec<Cycle> = Vec::with_capacity(total.next_power_of_two());
    for chunk in per_chunk {
        trace.extend(chunk.expect("worker dropped a chunk"));
    }
    (trace, pass1.into_emulator())
}
