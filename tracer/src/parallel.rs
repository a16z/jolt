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

/// Pool of full-size flat-memory images. Capturing a snapshot memcpys only
/// the live image's touched prefix (its high-water mark) into a pooled
/// buffer; the buffer is handed to a worker as its working memory
/// (zero-copy install) and returns to the pool when the worker installs its
/// next chunk.
///
/// Beyond the copied prefix, pooled buffers are provably zero — matching the
/// untouched suffix of the live image: pass-1's high-water mark only grows,
/// and a worker's writes replay a chunk pass-1 already executed, so a buffer
/// returned from chunk k is dirty at most up to pass-1's high-water mark at
/// the end of chunk k, which is ≤ the mark at any later capture.
#[derive(Debug, Default)]
pub struct SnapshotPool {
    free: Vec<Vec<u64>>,
}

impl SnapshotPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot `memory`'s flat image into a pooled full-size buffer.
    pub fn capture(&mut self, memory: &Memory) -> Vec<u64> {
        let (image, touched) = memory.data.flat_parts();
        let mut buf = match self.free.pop() {
            Some(buf) if buf.len() == image.len() => buf,
            // First capture (or a size change): allocate the full zeroed
            // image once; it stays resident through pool reuse.
            _ => vec![0; image.len()],
        };
        buf[..touched].copy_from_slice(&image[..touched]);
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

    /// Like [`ChunkWorker::run_ticks`], but writes rows into `window`
    /// (exactly `window.len()` of them) through a small per-tick scratch
    /// buffer. The scratch stays cache-resident (a tick emits at most a few
    /// thousand rows), so the only cold writes are the window itself —
    /// avoiding a chunk-sized intermediate buffer entirely.
    pub fn run_ticks_into(
        &mut self,
        tick_count: usize,
        window: &mut [core::mem::MaybeUninit<Cycle>],
        scratch: &mut Vec<Cycle>,
    ) {
        let mut written = 0usize;
        for _ in 0..tick_count {
            scratch.clear();
            self.cpu.tick(Some(scratch));
            // Slice indexing bounds-checks a row-count overshoot (a real
            // divergence bug) before anything is written out of range.
            let destination = &mut window[written..written + scratch.len()];
            for (slot, row) in destination.iter_mut().zip(scratch.iter()) {
                slot.write(*row);
            }
            written += scratch.len();
        }
        assert_eq!(
            written,
            window.len(),
            "replay emitted a different row count than pass-1"
        );
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

/// Pin the current thread's macOS QoS class. Pass-1 is the pipeline's
/// critical path and gets `USER_INTERACTIVE` (P-core placement); workers get
/// `USER_INITIATED`, one step below, so they never displace pass-1.
#[cfg(target_os = "macos")]
fn set_thread_qos(qos_class: u32) {
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    // SAFETY: plain FFI call into libSystem (always linked on macOS); it only
    // adjusts the calling thread's scheduling class.
    unsafe {
        let _ = pthread_set_qos_class_self_np(qos_class, 0);
    }
}

#[cfg(target_os = "macos")]
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
#[cfg(target_os = "macos")]
const QOS_CLASS_USER_INITIATED: u32 = 0x19;

fn promote_pass1_thread() {
    #[cfg(target_os = "macos")]
    set_thread_qos(QOS_CLASS_USER_INTERACTIVE);
}

fn demote_worker_thread() {
    #[cfg(target_os = "macos")]
    set_thread_qos(QOS_CLASS_USER_INITIATED);
}

/// Default chunk size in rows (~1M): large enough that snapshot capture and
/// dispatch stay ≪ chunk replay time, small enough that the final worker
/// wave doesn't dominate the wall clock (measured optimum on ~10M-row
/// traces; at 100M+ rows both effects are negligible for any size here).
pub const DEFAULT_CHUNK_ROWS: usize = 1 << 20;

/// Default output capacity in rows. Chunks beyond this fall back to a
/// copy-assembled suffix (correct, just slower) — see `run_two_pass`.
pub const DEFAULT_CAPACITY_ROWS: usize = 1 << 24;

#[derive(Clone, Copy, Debug)]
pub struct TwoPassConfig {
    pub workers: usize,
    /// Chunks close at the first tick boundary at or past this many rows.
    pub chunk_rows: usize,
    /// Output vec capacity reserved up front. Workers write their rows
    /// directly into disjoint windows of it; if the trace outgrows it, the
    /// remaining chunks are assembled by copy instead.
    pub capacity_rows: usize,
}

impl Default for TwoPassConfig {
    fn default() -> Self {
        Self {
            workers: 1,
            chunk_rows: DEFAULT_CHUNK_ROWS,
            capacity_rows: DEFAULT_CAPACITY_ROWS,
        }
    }
}

struct ChunkJob<'trace> {
    index: usize,
    checkpoint: ChunkCheckpoint,
    image: Vec<u64>,
    ticks: usize,
    rows: usize,
    /// Destination window inside the output vec's spare capacity (disjoint
    /// across chunks, in chunk order). `None` = capacity exhausted; the
    /// worker ships its rows back for copy assembly instead.
    window: Option<&'trace mut [core::mem::MaybeUninit<Cycle>]>,
}

/// Two-pass parallel trace: pass-1 executes on the calling thread, cutting
/// row-bounded chunks into a bounded job queue; `workers` threads re-trace
/// chunks concurrently, writing rows straight into disjoint windows of the
/// pre-reserved output vec.
///
/// Returns the assembled trace (bit-identical to what `Cpu::tick(Some(..))`
/// over the whole program produces) and the finished pass-1 emulator, which
/// is the authoritative source for final memory, device and advice state.
///
/// In-flight memory is bounded by the job-queue depth: at most
/// `2*workers (queued) + workers (executing) + 1 (capturing)` full images.
#[expect(clippy::expect_used)]
pub fn run_two_pass(emulator: Emulator, config: &TwoPassConfig) -> (Vec<Cycle>, Emulator) {
    use core::mem::MaybeUninit;
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

    // Untouched capacity is lazily committed by the OS, so over-reserving is
    // cheap; the prover pads the trace to a power of two afterwards, which
    // this capacity absorbs without reallocation for typical trace sizes.
    let mut trace: Vec<Cycle> = Vec::with_capacity(config.capacity_rows.max(1));
    let mut spare: &mut [MaybeUninit<Cycle>] = trace.spare_capacity_mut();

    let (job_tx, job_rx) = mpsc::sync_channel::<ChunkJob<'_>>(2 * workers);
    let job_rx = Mutex::new(job_rx);
    let (buf_tx, buf_rx) = mpsc::channel::<Vec<u64>>();
    let (out_tx, out_rx) = mpsc::channel::<(usize, Vec<Cycle>)>();

    let mut pass1 = PassOne::new(emulator);

    let windowed_rows = std::thread::scope(|scope| {
        for _ in 0..workers {
            let job_rx = &job_rx;
            let buf_tx = buf_tx.clone();
            let out_tx = out_tx.clone();
            let seed_device = seed_device.clone();
            let seed_decode = seed_decode.clone();
            scope.spawn(move || {
                demote_worker_thread();
                let mut worker = ChunkWorker::from_seed(seed_device, seed_decode);
                // Per-tick row buffer, reused across ticks and chunks: it
                // grows to the largest single tick (a few thousand rows) and
                // stays cache-resident.
                let mut scratch: Vec<Cycle> = Vec::new();
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
                    match job.window {
                        Some(window) => {
                            debug_assert_eq!(window.len(), job.rows);
                            worker.run_ticks_into(job.ticks, window, &mut scratch);
                        }
                        None => {
                            // Overflow fallback: assemble by copy afterwards.
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
                    }
                }
            });
        }
        // Workers hold clones; the originals must drop so the collector's
        // recv loop terminates once all workers exit.
        drop(out_tx);
        drop(buf_tx);

        // Pass-1 on the calling thread.
        promote_pass1_thread();
        let timing = std::env::var("JOLT_TRACER_TIMING").is_ok();
        let started = std::time::Instant::now();
        let mut capture_time = std::time::Duration::ZERO;
        let mut exec_time = std::time::Duration::ZERO;
        let mut send_time = std::time::Duration::ZERO;
        let mut pool = SnapshotPool::new();
        let mut chunk_index = 0usize;
        let mut windowed_rows = 0usize;
        let mut overflowed = false;
        loop {
            while let Ok(buffer) = buf_rx.try_recv() {
                pool.put(buffer);
            }
            let t0 = std::time::Instant::now();
            let checkpoint = pass1.checkpoint();
            let image = pool.capture(&pass1.emulator().get_cpu().mmu.memory.memory);
            let t1 = std::time::Instant::now();
            capture_time += t1 - t0;
            let rows_before = pass1.rows();
            let mut ticks = 0usize;
            while pass1.rows() - rows_before < chunk_rows && pass1.step() {
                ticks += 1;
            }
            let t2 = std::time::Instant::now();
            exec_time += t2 - t1;
            let rows = pass1.rows() - rows_before;
            if ticks == 0 {
                pool.put(image);
                break;
            }
            // Once a chunk misses the reserved capacity, all later chunks
            // fall back too — the windowed region must stay a contiguous,
            // in-order prefix for the final set_len.
            overflowed |= rows > spare.len();
            let window = if overflowed {
                None
            } else {
                let (window, rest) = core::mem::take(&mut spare).split_at_mut(rows);
                spare = rest;
                windowed_rows += rows;
                Some(window)
            };
            job_tx
                .send(ChunkJob {
                    index: chunk_index,
                    checkpoint,
                    image,
                    ticks,
                    rows,
                    window,
                })
                .expect("worker pool hung up");
            send_time += t2.elapsed();
            chunk_index += 1;
            if pass1.is_done() {
                break;
            }
        }
        drop(job_tx);
        let pass1_done = started.elapsed();
        if timing {
            eprintln!(
                "two-pass timing: pass-1 {pass1_done:?} (exec {exec_time:?}, capture {capture_time:?}, send-block {send_time:?}, {chunk_index} chunks)"
            );
        }
        (windowed_rows, started, timing)
    });
    let (windowed_rows, started, timing) = windowed_rows;
    if timing {
        eprintln!("two-pass timing: workers joined at {:?}", started.elapsed());
    }
    // The job channel's type carries the window lifetime (a borrow of
    // `trace`); it must drop before `trace` can be touched again.
    drop(job_rx);

    // SAFETY: every window handed to a worker was split off `spare` in chunk
    // order, so windows tile `trace[0..windowed_rows]` contiguously and
    // disjointly; the thread scope has joined all workers, and each worker
    // asserted it wrote exactly its window's length. All `windowed_rows`
    // elements are therefore initialized.
    unsafe { trace.set_len(windowed_rows) };

    // Rare overflow suffix: chunks past the reserved capacity, in order.
    let mut overflow: Vec<(usize, Vec<Cycle>)> = out_rx.iter().collect();
    overflow.sort_unstable_by_key(|(index, _)| *index);
    debug_assert!(overflow
        .iter()
        .zip(overflow.iter().skip(1))
        .all(|((a, _), (b, _))| a + 1 == *b));
    let total_overflow: usize = overflow.iter().map(|(_, rows)| rows.len()).sum();
    trace.reserve(total_overflow);
    for (_, rows) in overflow {
        trace.extend(rows);
    }
    (trace, pass1.into_emulator())
}
