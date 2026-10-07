use std::{collections::HashMap, mem::size_of, time::Instant};

use metal::{
    foreign_types::ForeignType,
    objc::{
        runtime::{Object, Sel},
        Message,
    },
    Buffer, CommandBufferRef, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Heap,
    HeapDescriptor, MTLCommandBufferStatus, MTLHazardTrackingMode, MTLHeapType, MTLResourceOptions,
    MTLSize, MTLStorageMode, NSRange,
};

use super::super::{
    residency::new_residency_set,
    runtime::{command_buffer_timestamp, validate_completed_command},
    set_inline_bytes, Fp128, MetalError, SolinasMetal,
};
use super::F;

pub(super) const TREE_LEAVES: &str = "byte_link_tree_leaves";
pub(super) const TREE_UPPER: &str = "byte_link_tree_upper";
pub(super) const SUM_PARTIALS: &str = "byte_link_sum_partials";
pub(super) const ROUND_EVAL: &str = "byte_link_round_eval";
pub(super) const ROUND_BIND_EVAL: &str = "byte_link_round_bind_eval";
pub(super) const ROUND_STREAM: &str = "byte_link_round_stream";
pub(super) const BOTTOM_STREAM: &str = "byte_link_bottom_stream";
pub(super) const BOTTOM_BIND_EVAL: &str = "byte_link_bottom_bind_eval";
pub(super) const PRODUCT2_EVAL: &str = "byte_link_product2_eval";
pub(super) const PRODUCT2_BIND_EVAL: &str = "byte_link_product2_bind_eval";
pub(super) const PRODUCT4_EVAL: &str = "byte_link_product4_eval";
pub(super) const PRODUCT4_BIND_EVAL: &str = "byte_link_product4_bind_eval";
pub(super) const QUERY_EVAL: &str = "byte_link_query_eval";
pub(super) const QUERY_BIND: &str = "byte_link_query_bind";
pub(super) const SOURCE_EVAL: &str = "byte_link_source_eval";
pub(super) const SOURCE_BIND: &str = "byte_link_source_bind";
pub(super) const SORT_COUNT: &str = "byte_link_sort_count";
pub(super) const SORT_SCAN: &str = "byte_link_sort_scan";
pub(super) const SORT_SCATTER: &str = "byte_link_sort_scatter";
pub(super) const SORT_BUCKETS: &str = "byte_link_sort_buckets";
pub(super) const WEIGHTS_PREPARE: &str = "byte_link_weights_prepare";
pub(super) const WEIGHTS_RUNS: &str = "byte_link_weights_runs";
pub(super) const WEIGHTS_FINISH: &str = "byte_link_weights_finish";
pub(super) const COLUMN_MLE: &str = "byte_link_column_mle";
pub(super) const SUM_COLUMNS: &str = "byte_link_sum_columns";

const KERNELS: [&str; 25] = [
    TREE_LEAVES,
    TREE_UPPER,
    SUM_PARTIALS,
    ROUND_EVAL,
    ROUND_BIND_EVAL,
    ROUND_STREAM,
    BOTTOM_STREAM,
    BOTTOM_BIND_EVAL,
    PRODUCT2_EVAL,
    PRODUCT2_BIND_EVAL,
    PRODUCT4_EVAL,
    PRODUCT4_BIND_EVAL,
    QUERY_EVAL,
    QUERY_BIND,
    SOURCE_EVAL,
    SOURCE_BIND,
    SORT_COUNT,
    SORT_SCAN,
    SORT_SCATTER,
    SORT_BUCKETS,
    WEIGHTS_PREPARE,
    WEIGHTS_RUNS,
    WEIGHTS_FINISH,
    COLUMN_MLE,
    SUM_COLUMNS,
];

/// Threads per threadgroup of the round, product and column kernels (`LINK_THREADS`).
pub(super) const THREADS: usize = 256;
/// Partial sums per threadgroup and per pack of a round (`LINK_SUMS`).
pub(super) const SUMS: usize = 2;

/// Wall and GPU seconds and command buffers of one named phase of a proof.
#[derive(Clone, Debug, Default)]
pub(super) struct Phase {
    pub name: &'static str,
    pub wall: f64,
    pub gpu: f64,
    pub commands: usize,
}

pub(super) enum Grid {
    Threads(usize, usize),
    Groups(usize, usize),
}

pub(super) struct Recorder<'a> {
    pipelines: &'a HashMap<&'static str, ComputePipelineState>,
    command: &'a CommandBufferRef,
}

impl Recorder<'_> {
    pub fn dispatch(
        &mut self,
        kernel: &'static str,
        grid: Grid,
        bind: impl FnOnce(&ComputeCommandEncoderRef),
    ) {
        let encoder = self.command.new_compute_command_encoder();
        encoder.set_label(kernel);
        encoder.set_compute_pipeline_state(&self.pipelines[kernel]);
        bind(encoder);
        match grid {
            Grid::Threads(threads, width) => encoder.dispatch_threads(
                MTLSize::new(threads as u64, 1, 1),
                MTLSize::new(width as u64, 1, 1),
            ),
            Grid::Groups(groups, width) => encoder.dispatch_thread_groups(
                MTLSize::new(groups as u64, 1, 1),
                MTLSize::new(width as u64, 1, 1),
            ),
        }
        encoder.end_encoding();
    }

    pub fn zero(&mut self, buffer: &Buffer) {
        let blit = self.command.new_blit_command_encoder();
        blit.fill_buffer(buffer, NSRange::new(0, buffer.length()), 0);
        blit.end_encoding();
    }
}

pub(super) fn bind(encoder: &ComputeCommandEncoderRef, index: u64, buffer: &Buffer, offset: usize) {
    encoder.set_buffer(index, Some(buffer), offset as u64);
}

pub(super) fn bytes<T>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
    set_inline_bytes(encoder, index, value);
}

pub(super) fn limbs(value: F) -> Fp128 {
    Fp128::from_jolt_field(&value)
}

pub(super) fn field(value: Fp128) -> F {
    value.into_jolt_field()
}

pub(super) fn upload(values: &[F]) -> Vec<Fp128> {
    values.iter().map(|&v| limbs(v)).collect()
}

/// `len` values of `T` from `start` in shared storage no command is writing.
pub(super) fn view<T>(buffer: &Buffer, start: usize, len: usize) -> &[T] {
    assert!(((start + len) * size_of::<T>()) as u64 <= buffer.length());
    // SAFETY: in bounds of the shared storage, plain-old-data `T`; every command writing it has
    // completed (each command buffer is waited on before the next host step).
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>().add(start), len) }
}

pub(super) fn write<T: Copy>(buffer: &Buffer, values: &[T]) {
    assert!(size_of_val(values) as u64 <= buffer.length());
    // SAFETY: shared storage of at least `values.len()` plain-old-data `T`; no command reading it
    // is in flight.
    unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<T>(), values.len()) }
        .copy_from_slice(values);
}

/// A residency set attached to the link's queue: every link command keeps the set's allocations
/// resident, so a phase binding the arena or `W` does not first wait for the driver to wire them
/// (20–60 ms per phase at 2^26 without it, `/private/tmp/pika-scratch/akita-p0/proto/notes.md`
/// §Levers).
struct QueueResidency {
    set: *mut Object,
    queue: CommandQueue,
}

impl QueueResidency {
    /// `None` before macOS 15, where commands wire their allocations themselves.
    fn attach(queue: &CommandQueue) -> Option<Self> {
        let set = new_residency_set(queue.device())?;
        // SAFETY: MTLCommandQueue `addResidencySet:` takes the live +1 set from
        // new_residency_set; a set the queue did not take is released here.
        unsafe {
            let queue_object = queue.as_ptr().cast::<Object>();
            if (*queue_object)
                .send_message::<_, ()>(Sel::register("addResidencySet:"), (set,))
                .is_err()
            {
                let _ = (*set).send_message::<_, ()>(Sel::register("release"), ());
                return None;
            }
        }
        Some(Self {
            set,
            queue: queue.clone(),
        })
    }

    /// Adds a buffer or heap and requests its residency.
    fn add(&self, allocation: *mut Object) {
        self.send("addAllocation:", Some(allocation));
        self.send("commit", None);
        self.send("requestResidency", None);
    }

    fn remove(&self, allocation: *mut Object) {
        self.send("removeAllocation:", Some(allocation));
        self.send("commit", None);
    }

    fn send(&self, selector: &str, allocation: Option<*mut Object>) {
        // SAFETY: `addAllocation:`/`removeAllocation:` take an MTLAllocation (a buffer or heap)
        // that `Gpu` keeps alive while it is in the set; `commit` and `requestResidency` take
        // nothing; `self.set` is live until drop.
        unsafe {
            let _ = match allocation {
                Some(allocation) => {
                    (*self.set).send_message::<_, ()>(Sel::register(selector), (allocation,))
                }
                None => (*self.set).send_message::<_, ()>(Sel::register(selector), ()),
            };
        }
    }
}

impl Drop for QueueResidency {
    fn drop(&mut self) {
        // SAFETY: detaches the set from the queue it was attached to, then releases the +1 set.
        unsafe {
            let queue_object = self.queue.as_ptr().cast::<Object>();
            let _ = (*queue_object)
                .send_message::<_, ()>(Sel::register("removeResidencySet:"), (self.set,));
            let _ = (*self.set).send_message::<_, ()>(Sel::register("release"), ());
        }
    }
}

/// The Metal side of the byte link: its pipelines, its own queue with a residency set, the
/// per-proof arena every phase allocates from, the latest `W` held resident beside it, and the
/// phase clock. Field order is drop order: the set goes before what it holds.
pub(super) struct Gpu {
    metal: SolinasMetal,
    pipelines: HashMap<&'static str, ComputePipelineState>,
    queue: CommandQueue,
    residency: Option<QueueResidency>,
    arena: Option<Heap>,
    histograms: Option<Buffer>,
    phases: Vec<Phase>,
    phase_start: Option<Instant>,
}

impl Gpu {
    pub fn new(metal: &SolinasMetal) -> Result<Self, MetalError> {
        let pipelines = KERNELS
            .iter()
            .map(|&name| Ok((name, metal.compile_named_pipeline(name)?)))
            .collect::<Result<_, MetalError>>()?;
        let queue = metal.device.new_command_queue();
        Ok(Self {
            metal: metal.clone(),
            pipelines,
            residency: QueueResidency::attach(&queue),
            queue,
            arena: None,
            histograms: None,
            phases: Vec::new(),
            phase_start: None,
        })
    }

    fn arena_options() -> MTLResourceOptions {
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked
    }

    /// Heap bytes of one arena buffer of `bytes`.
    pub fn arena_size(&self, bytes: usize) -> usize {
        let size = self
            .metal
            .device
            .heap_buffer_size_and_align(bytes.max(16) as u64, Self::arena_options());
        size.size.next_multiple_of(size.align) as usize
    }

    /// Keeps an arena of at least `bytes`; a smaller one is replaced.
    pub fn reserve_arena(&mut self, bytes: usize) -> Result<(), MetalError> {
        if self
            .arena
            .as_ref()
            .is_some_and(|heap| heap.size() >= bytes as u64)
        {
            return Ok(());
        }
        if let (Some(set), Some(old)) = (&self.residency, self.arena.take()) {
            set.remove(old.as_ptr().cast());
        }
        self.metal.validate_buffer_length(bytes as u64)?;
        let descriptor = HeapDescriptor::new();
        descriptor.set_heap_type(MTLHeapType::Automatic);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_hazard_tracking_mode(MTLHazardTrackingMode::Tracked);
        descriptor.set_size(bytes as u64);
        let heap = self.metal.device.new_heap(&descriptor);
        if let Some(set) = &self.residency {
            set.add(heap.as_ptr().cast());
        }
        self.arena = Some(heap);
        Ok(())
    }

    pub fn arena(&self, bytes: usize) -> Result<Buffer, MetalError> {
        self.arena
            .as_ref()
            .and_then(|heap| heap.new_buffer(bytes.max(16) as u64, Self::arena_options()))
            .ok_or(MetalError::ByteLinkArena {
                bytes: bytes as u64,
            })
    }

    /// A shared buffer for `W`, outside the arena and resident for every link command until the
    /// next proof's `W` replaces it.
    pub fn histogram_buffer(&mut self, bytes: usize) -> Buffer {
        let buffer = self.scratch(bytes);
        if let Some(set) = &self.residency {
            if let Some(old) = self.histograms.take() {
                set.remove(old.as_ptr().cast());
            }
            set.add(buffer.as_ptr().cast());
            self.histograms = Some(buffer.clone());
        }
        buffer
    }

    pub fn scratch(&self, bytes: usize) -> Buffer {
        self.metal
            .device
            .new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
    }

    pub fn buffer<T: Copy>(&self, values: &[T]) -> Buffer {
        let buffer = self.scratch(size_of_val(values));
        write(&buffer, values);
        buffer
    }

    pub fn fields(&self, values: &[F]) -> Buffer {
        self.buffer(&upload(values))
    }

    /// Ends the running phase and opens `name`.
    pub fn phase(&mut self, name: &'static str) {
        self.end_phase();
        self.phases.push(Phase {
            name,
            ..Phase::default()
        });
        self.phase_start = Some(Instant::now());
    }

    pub fn end_phase(&mut self) {
        let (Some(start), Some(last)) = (self.phase_start.take(), self.phases.last_mut()) else {
            return;
        };
        last.wall = start.elapsed().as_secs_f64();
        tracing::debug!(
            phase = last.name,
            wall_s = last.wall,
            gpu_s = last.gpu,
            commands = last.commands,
            "byte link phase"
        );
    }

    #[cfg(test)]
    pub fn take_phases(&mut self) -> Vec<Phase> {
        self.end_phase();
        std::mem::take(&mut self.phases)
    }

    #[cfg(test)]
    pub fn arena_bytes(&self) -> u64 {
        self.arena.as_ref().map_or(0, |heap| heap.size())
    }

    /// One command buffer: record, commit, wait, account to the running phase.
    pub fn run(
        &mut self,
        label: &'static str,
        record: impl FnOnce(&mut Recorder),
    ) -> Result<(), MetalError> {
        let command = self.queue.new_command_buffer().to_owned();
        command.set_label(label);
        record(&mut Recorder {
            pipelines: &self.pipelines,
            command: &command,
        });
        command.commit();
        // Polling, not waitUntilCompleted: a proof waits on ~400 short round commands in sequence
        // and the blocking wake-up cost ~10 ms per proof at 2^26
        // (`/private/tmp/pika-scratch/akita-p0/proto/time-m26-{spin,block}.log`).
        while !matches!(
            command.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        ) {
            std::hint::spin_loop();
        }
        validate_completed_command(&command)?;
        let gpu = command_buffer_timestamp(&command, "GPUEndTime")?
            - command_buffer_timestamp(&command, "GPUStartTime")?;
        if let Some(phase) = self.phases.last_mut() {
            phase.gpu += gpu;
            phase.commands += 1;
        }
        Ok(())
    }
}
