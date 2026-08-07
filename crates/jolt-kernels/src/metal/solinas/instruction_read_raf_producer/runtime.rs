use std::{
    mem::size_of,
    time::{Duration, Instant},
};

use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CompileOptions,
    ComputePipelineState, Library, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use super::super::{
    buffer_from_slice, command_buffer_timestamp, MetalError, PipelineLimits, SolinasMetal,
};
use super::binding::{BoundScatter, ResidentProducerBufferOwner};
use super::{
    PlaneRole, ProducerGeometry, ProducerLayoutError, ProducerShardPlan, ScatterDispatchPlan,
    ScatterLayout, ScatterParams, MAX_BUFFER_BYTES, METAL_SOURCE, PRODUCER_THREADS_PER_GROUP,
    SCATTER_BUFFER_ROLES,
};

const SCATTER_PIPELINE: &str = "instruction_read_raf_producer_scatter_4096";
const SCATTER_PARAMS_SLOT: u64 = 9;

#[derive(Debug, thiserror::Error)]
pub enum ProducerRuntimeError {
    #[error(transparent)]
    Layout(#[from] ProducerLayoutError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("producer source generation must be nonzero")]
    MissingSourceGeneration,
    #[error("producer source completion serial must be nonzero")]
    MissingSourceCompletionSerial,
    #[error("producer has {got} shard inputs, expected {expected}")]
    ShardCount { expected: usize, got: usize },
    #[error("producer shard input {index} does not match the geometry shard")]
    ShardOrder { index: usize },
    #[error("producer shard {shard} source generation is {got}, expected {expected}")]
    SourceGeneration {
        shard: usize,
        expected: u64,
        got: u64,
    },
    #[error("producer shard {shard} source completion serial is {got}, expected {expected}")]
    SourceCompletionSerial {
        shard: usize,
        expected: u64,
        got: u64,
    },
    #[error("producer shard {shard} {role:?} belongs to Metal device {got}, expected {expected}")]
    BufferDevice {
        shard: usize,
        role: PlaneRole,
        expected: u64,
        got: u64,
    },
    #[error("producer shard {shard} {role:?} has {got} bytes, expected {expected}")]
    BufferLength {
        shard: usize,
        role: PlaneRole,
        expected: u64,
        got: u64,
    },
    #[error(
        "producer shard {shard} {role:?} has {bytes} bytes, exceeding the device maximum {maximum}"
    )]
    DeviceBufferLimit {
        shard: usize,
        role: PlaneRole,
        bytes: u64,
        maximum: u64,
    },
    #[error("producer shard {shard} {role:?} allocation identity changed")]
    BufferIdentity { shard: usize, role: PlaneRole },
    #[error(
        "producer buffers alias: shard {first_shard} {first_role:?} and shard {second_shard} {second_role:?} use allocation {identity:#x}"
    )]
    AliasedBuffers {
        first_shard: usize,
        first_role: PlaneRole,
        second_shard: usize,
        second_role: PlaneRole,
        identity: usize,
    },
    #[error("producer pipeline needs {requested} threads per threadgroup, maximum is {maximum}")]
    ThreadLimit { requested: usize, maximum: usize },
    #[error("producer pipeline uses {requested} threadgroup bytes, device maximum is {maximum}")]
    ThreadgroupMemory { requested: u64, maximum: u64 },
    #[error(
        "producer shader reported {nonzero_shards} nonzero shard statuses; first is shard {first_shard} status {first_status:#010x}, combined status {combined_status:#010x}"
    )]
    ShaderStatus {
        nonzero_shards: usize,
        first_shard: usize,
        first_status: u32,
        combined_status: u32,
    },
    #[error("producer dispatch serial overflowed")]
    DispatchSerialOverflow,
    #[error("invalid producer runtime state: {0}")]
    InvalidState(&'static str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProducerPlaneInitialization {
    /// Upstream producer completion serial supplied when the source was attached.
    Source {
        completion_serial: u64,
    },
    FrozenLayout,
    /// Reuse serial local to this scatter owner.
    ScatterDispatch {
        serial: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerPlaneReceipt {
    shard: ProducerShardPlan,
    role: PlaneRole,
    bytes: u64,
    device_registry_id: u64,
    allocation_identity: usize,
    source_generation: u64,
    source_completion_serial: u64,
    initialization: ProducerPlaneInitialization,
}

impl ProducerPlaneReceipt {
    pub const fn shard(self) -> ProducerShardPlan {
        self.shard
    }

    pub const fn role(self) -> PlaneRole {
        self.role
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    pub const fn device_registry_id(self) -> u64 {
        self.device_registry_id
    }

    pub const fn allocation_identity(self) -> usize {
        self.allocation_identity
    }

    pub const fn source_generation(self) -> u64 {
        self.source_generation
    }

    pub const fn source_completion_serial(self) -> u64 {
        self.source_completion_serial
    }

    pub const fn initialization(self) -> ProducerPlaneInitialization {
        self.initialization
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerSourceReceipt {
    shard: ProducerShardPlan,
    device_registry_id: u64,
    source_generation: u64,
    source_completion_serial: u64,
    allocation_identities: [usize; 3],
}

impl ProducerSourceReceipt {
    pub const fn shard(self) -> ProducerShardPlan {
        self.shard
    }

    pub const fn device_registry_id(self) -> u64 {
        self.device_registry_id
    }

    pub const fn source_generation(self) -> u64 {
        self.source_generation
    }

    pub const fn source_completion_serial(self) -> u64 {
        self.source_completion_serial
    }

    pub const fn allocation_identities(self) -> [usize; 3] {
        self.allocation_identities
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerCompletionReceipt {
    total_rows: usize,
    shard_count: usize,
    device_registry_id: u64,
    source_generation: u64,
    source_completion_serial: u64,
    dispatch_serial: u64,
}

impl ProducerCompletionReceipt {
    pub const fn total_rows(self) -> usize {
        self.total_rows
    }

    pub const fn shard_count(self) -> usize {
        self.shard_count
    }

    pub const fn device_registry_id(self) -> u64 {
        self.device_registry_id
    }

    pub const fn source_generation(self) -> u64 {
        self.source_generation
    }

    pub const fn source_completion_serial(self) -> u64 {
        self.source_completion_serial
    }

    pub const fn dispatch_serial(self) -> u64 {
        self.dispatch_serial
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerPreparationTiming {
    total: Duration,
    pipeline_compile: Duration,
    allocation_and_layout_upload: Duration,
}

impl ProducerPreparationTiming {
    pub const fn total_wall(self) -> Duration {
        self.total
    }

    pub const fn pipeline_compile_wall(self) -> Duration {
        self.pipeline_compile
    }

    pub const fn allocation_and_layout_upload_wall(self) -> Duration {
        self.allocation_and_layout_upload
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerExecutionTiming {
    resident_wall: Duration,
    gpu_active: Duration,
}

impl ProducerExecutionTiming {
    pub const fn resident_wall(self) -> Duration {
        self.resident_wall
    }

    pub const fn gpu_active(self) -> Duration {
        self.gpu_active
    }
}

pub struct ResidentProducerPlane<'a> {
    buffer: &'a Buffer,
    receipt: ProducerPlaneReceipt,
}

impl ResidentProducerPlane<'_> {
    pub const fn buffer(&self) -> &Buffer {
        self.buffer
    }

    pub const fn receipt(&self) -> ProducerPlaneReceipt {
        self.receipt
    }
}

pub struct ResidentProducerSourceShard {
    shard: ProducerShardPlan,
    cycle_lookup_lo: Buffer,
    cycle_lookup_hi: Buffer,
    cycle_claims: Buffer,
    receipt: ProducerSourceReceipt,
}

impl ResidentProducerSourceShard {
    pub const fn shard(&self) -> ProducerShardPlan {
        self.shard
    }

    pub const fn receipt(&self) -> ProducerSourceReceipt {
        self.receipt
    }

    pub fn plane(&self, role: PlaneRole) -> Option<ResidentProducerPlane<'_>> {
        let buffer = self.buffer(role)?;
        Some(ResidentProducerPlane {
            buffer,
            receipt: ProducerPlaneReceipt {
                shard: self.shard,
                role,
                bytes: buffer.length(),
                device_registry_id: self.receipt.device_registry_id,
                allocation_identity: buffer_identity(buffer),
                source_generation: self.receipt.source_generation,
                source_completion_serial: self.receipt.source_completion_serial,
                initialization: ProducerPlaneInitialization::Source {
                    completion_serial: self.receipt.source_completion_serial,
                },
            },
        })
    }

    fn buffer(&self, role: PlaneRole) -> Option<&Buffer> {
        match role {
            PlaneRole::CycleLookupLo => Some(&self.cycle_lookup_lo),
            PlaneRole::CycleLookupHi => Some(&self.cycle_lookup_hi),
            PlaneRole::CycleClaims => Some(&self.cycle_claims),
            _ => None,
        }
    }

    fn validate_for(
        &self,
        expected_shard: ProducerShardPlan,
        expected_device: u64,
        expected_generation: u64,
        expected_completion_serial: u64,
        maximum_buffer_length: u64,
    ) -> Result<(), ProducerRuntimeError> {
        if self.shard != expected_shard {
            return Err(ProducerRuntimeError::ShardOrder {
                index: expected_shard.shard_index(),
            });
        }
        if self.receipt.source_generation != expected_generation {
            return Err(ProducerRuntimeError::SourceGeneration {
                shard: self.shard.shard_index(),
                expected: expected_generation,
                got: self.receipt.source_generation,
            });
        }
        if self.receipt.source_completion_serial != expected_completion_serial {
            return Err(ProducerRuntimeError::SourceCompletionSerial {
                shard: self.shard.shard_index(),
                expected: expected_completion_serial,
                got: self.receipt.source_completion_serial,
            });
        }
        if self.receipt.device_registry_id != expected_device {
            return Err(ProducerRuntimeError::BufferDevice {
                shard: self.shard.shard_index(),
                role: PlaneRole::CycleLookupLo,
                expected: expected_device,
                got: self.receipt.device_registry_id,
            });
        }
        for role in source_roles() {
            let buffer = self.buffer(role).ok_or(ProducerRuntimeError::InvalidState(
                "source receipt is missing a cycle plane",
            ))?;
            let expected_bytes = shape_bytes(self.shard, role)?;
            let identity = self.receipt.allocation_identities[role.metal_buffer_slot()];
            validate_recorded_binding(
                BindingRecord {
                    shard: self.shard.shard_index(),
                    role,
                    bytes: expected_bytes,
                    device_registry_id: expected_device,
                    allocation_identity: identity,
                    source_generation: expected_generation,
                },
                binding_facts(buffer),
                expected_generation,
                maximum_buffer_length,
            )?;
        }
        Ok(())
    }
}

pub struct ProducerShardInput {
    source: ResidentProducerSourceShard,
    layout: ScatterLayout,
}

impl ProducerShardInput {
    pub fn new(
        source: ResidentProducerSourceShard,
        layout: ScatterLayout,
    ) -> Result<Self, ProducerRuntimeError> {
        if source.shard != layout.shard() {
            return Err(ProducerLayoutError::ShardMismatch.into());
        }
        Ok(Self { source, layout })
    }

    pub const fn shard(&self) -> ProducerShardPlan {
        self.source.shard
    }
}

struct ScatterBuffers {
    source: ResidentProducerSourceShard,
    chunk_segment_bases: Buffer,
    segment_offsets: Buffer,
    grouped_lookup_lo: Buffer,
    grouped_lookup_hi: Buffer,
    cycle_to_grouped_local: Buffer,
    status: Buffer,
}

impl ScatterBuffers {
    fn buffer(&self, role: PlaneRole) -> &Buffer {
        match role {
            PlaneRole::CycleLookupLo => &self.source.cycle_lookup_lo,
            PlaneRole::CycleLookupHi => &self.source.cycle_lookup_hi,
            PlaneRole::CycleClaims => &self.source.cycle_claims,
            PlaneRole::ChunkSegmentBases => &self.chunk_segment_bases,
            PlaneRole::SegmentOffsets => &self.segment_offsets,
            PlaneRole::GroupedLookupLo => &self.grouped_lookup_lo,
            PlaneRole::GroupedLookupHi => &self.grouped_lookup_hi,
            PlaneRole::CycleToGroupedLocal => &self.cycle_to_grouped_local,
            PlaneRole::Status => &self.status,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct BindingRecord {
    pub(super) shard: usize,
    pub(super) role: PlaneRole,
    pub(super) bytes: u64,
    pub(super) device_registry_id: u64,
    pub(super) allocation_identity: usize,
    pub(super) source_generation: u64,
}

#[derive(Clone, Copy)]
pub(super) struct BindingFacts {
    pub(super) bytes: u64,
    pub(super) device_registry_id: u64,
    pub(super) allocation_identity: usize,
}

struct ResidentProducerShard {
    plan: ScatterDispatchPlan,
    buffers: ScatterBuffers,
    records: [BindingRecord; 9],
    maximum_buffer_length: u64,
    source_generation: u64,
    source_completion_serial: u64,
}

impl ResidentProducerShard {
    fn new(
        device: &metal::Device,
        plan: ScatterDispatchPlan,
        source: ResidentProducerSourceShard,
        device_registry_id: u64,
        source_generation: u64,
        source_completion_serial: u64,
        maximum_buffer_length: u64,
    ) -> Result<Self, ProducerRuntimeError> {
        let shard = plan.shard();
        let grouped_lookup_lo = allocate_private(device, shard, PlaneRole::GroupedLookupLo)?;
        let grouped_lookup_hi = allocate_private(device, shard, PlaneRole::GroupedLookupHi)?;
        let cycle_to_grouped_local =
            allocate_private(device, shard, PlaneRole::CycleToGroupedLocal)?;
        let chunk_segment_bases = buffer_from_slice(device, plan.layout().chunk_segment_bases());
        let segment_offsets = buffer_from_slice(device, plan.layout().segment_offsets());
        let status = buffer_from_slice(device, &[0u32]);
        let buffers = ScatterBuffers {
            source,
            chunk_segment_bases,
            segment_offsets,
            grouped_lookup_lo,
            grouped_lookup_hi,
            cycle_to_grouped_local,
            status,
        };
        let records = std::array::from_fn(|slot| {
            let role = SCATTER_BUFFER_ROLES[slot];
            let buffer = buffers.buffer(role);
            BindingRecord {
                shard: shard.shard_index(),
                role,
                bytes: plan.required_buffers()[slot].bytes() as u64,
                device_registry_id,
                allocation_identity: buffer_identity(buffer),
                source_generation,
            }
        });
        let resident = Self {
            plan,
            buffers,
            records,
            maximum_buffer_length,
            source_generation,
            source_completion_serial,
        };
        let _ = resident.bind_checked(&resident.plan)?;
        Ok(resident)
    }

    fn record(&self, role: PlaneRole) -> BindingRecord {
        self.records[role.metal_buffer_slot()]
    }

    fn clear_status(&self) {
        // SAFETY: the status allocation contains one u32 in shared storage,
        // and no command uses it while execute holds the owner mutably.
        unsafe {
            self.buffers.status.contents().cast::<u32>().write(0);
        }
    }

    fn read_status(&self) -> u32 {
        // SAFETY: command completion synchronizes the one-word shared buffer.
        unsafe { self.buffers.status.contents().cast::<u32>().read() }
    }
}

impl ResidentProducerBufferOwner for ResidentProducerShard {
    type Error = ProducerRuntimeError;

    fn bind_checked<'a>(
        &'a self,
        plan: &ScatterDispatchPlan,
    ) -> Result<BoundScatter<'a>, Self::Error> {
        if plan != &self.plan {
            return Err(ProducerRuntimeError::InvalidState(
                "dispatch plan differs from the frozen shard layout",
            ));
        }
        if self.buffers.source.receipt.source_completion_serial != self.source_completion_serial {
            return Err(ProducerRuntimeError::SourceCompletionSerial {
                shard: self.plan.shard().shard_index(),
                expected: self.source_completion_serial,
                got: self.buffers.source.receipt.source_completion_serial,
            });
        }
        let buffers = std::array::from_fn(|slot| self.buffers.buffer(SCATTER_BUFFER_ROLES[slot]));
        for role in SCATTER_BUFFER_ROLES {
            validate_recorded_binding(
                self.record(role),
                binding_facts(self.buffers.buffer(role)),
                self.source_generation,
                self.maximum_buffer_length,
            )?;
        }
        validate_aliases(&self.records)?;
        Ok(BoundScatter::new(buffers))
    }
}

pub struct ResidentInstructionReadRafProducer {
    context: SolinasMetal,
    _library: Library,
    pipeline: ComputePipelineState,
    limits: PipelineLimits,
    geometry: ProducerGeometry,
    shards: Vec<ResidentProducerShard>,
    source_generation: u64,
    source_completion_serial: u64,
    preparation_timing: ProducerPreparationTiming,
    source_resident_bytes: u64,
    owned_bytes: u64,
    layout_upload_bytes: u64,
    dispatch_serial: u64,
}

impl SolinasMetal {
    /// Attaches the three cycle-order planes without copying them.
    ///
    /// `source_completion_serial` must come from the completed upstream
    /// producer; this owner preserves and checks that receipt but cannot infer
    /// command completion from an untyped `MTLBuffer`.
    pub fn attach_instruction_read_raf_producer_source_shard(
        &self,
        shard: ProducerShardPlan,
        source_generation: u64,
        source_completion_serial: u64,
        cycle_lookup_lo: Buffer,
        cycle_lookup_hi: Buffer,
        cycle_claims: Buffer,
    ) -> Result<ResidentProducerSourceShard, ProducerRuntimeError> {
        if source_generation == 0 {
            return Err(ProducerRuntimeError::MissingSourceGeneration);
        }
        if source_completion_serial == 0 {
            return Err(ProducerRuntimeError::MissingSourceCompletionSerial);
        }
        let device_registry_id = self.device_registry_id();
        let maximum_buffer_length = self.device.max_buffer_length();
        let buffers = [&cycle_lookup_lo, &cycle_lookup_hi, &cycle_claims];
        for (role, buffer) in source_roles().into_iter().zip(buffers) {
            let expected = shape_bytes(shard, role)?;
            let facts = binding_facts(buffer);
            validate_binding_shape(
                shard.shard_index(),
                role,
                expected,
                facts,
                device_registry_id,
                maximum_buffer_length,
            )?;
        }
        let allocation_identities = [
            buffer_identity(&cycle_lookup_lo),
            buffer_identity(&cycle_lookup_hi),
            buffer_identity(&cycle_claims),
        ];
        validate_source_aliases(shard.shard_index(), allocation_identities)?;
        Ok(ResidentProducerSourceShard {
            shard,
            cycle_lookup_lo,
            cycle_lookup_hi,
            cycle_claims,
            receipt: ProducerSourceReceipt {
                shard,
                device_registry_id,
                source_generation,
                source_completion_serial,
                allocation_identities,
            },
        })
    }

    pub fn prepare_instruction_read_raf_producer(
        &self,
        geometry: ProducerGeometry,
        inputs: Vec<ProducerShardInput>,
    ) -> Result<ResidentInstructionReadRafProducer, ProducerRuntimeError> {
        let total_started = Instant::now();
        if inputs.len() != geometry.shard_count() {
            return Err(ProducerRuntimeError::ShardCount {
                expected: geometry.shard_count(),
                got: inputs.len(),
            });
        }
        let first_source = inputs
            .first()
            .ok_or(ProducerRuntimeError::InvalidState(
                "validated geometry has no shards",
            ))?
            .source
            .receipt;
        let source_generation = first_source.source_generation;
        let source_completion_serial = first_source.source_completion_serial;
        let device_registry_id = self.device_registry_id();
        let maximum_buffer_length = self.device.max_buffer_length();
        let mut plans = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            let shard = geometry.shard(index)?;
            input.source.validate_for(
                shard,
                device_registry_id,
                source_generation,
                source_completion_serial,
                maximum_buffer_length,
            )?;
            if input.layout.shard() != shard {
                return Err(ProducerRuntimeError::ShardOrder { index });
            }
            plans.push(ScatterDispatchPlan::new(shard, &input.layout)?);
        }

        let compile_started = Instant::now();
        let options = CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(METAL_SOURCE, &options)
            .map_err(MetalError::LibraryCompilation)?;
        let function = library
            .get_function(SCATTER_PIPELINE, None)
            .map_err(|message| MetalError::FunctionLookup {
                name: SCATTER_PIPELINE,
                message,
            })?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|message| MetalError::PipelineCompilation {
                name: SCATTER_PIPELINE,
                message,
            })?;
        let pipeline_compile_wall = compile_started.elapsed();
        let limits = SolinasMetal::limits(&pipeline);
        validate_pipeline(limits, self.device.max_threadgroup_memory_length())?;

        let mut owned_bytes = 0u64;
        let mut layout_upload_bytes = 0u64;
        let mut source_resident_bytes = 0u64;
        for plan in &plans {
            for role in source_roles() {
                source_resident_bytes = source_resident_bytes
                    .checked_add(shape_bytes(plan.shard(), role)?)
                    .ok_or(ProducerLayoutError::SizeOverflow(
                        "resident producer source bytes",
                    ))?;
            }
            for role in owned_roles() {
                let bytes = shape_bytes(plan.shard(), role)?;
                if bytes > maximum_buffer_length {
                    return Err(ProducerRuntimeError::DeviceBufferLimit {
                        shard: plan.shard().shard_index(),
                        role,
                        bytes,
                        maximum: maximum_buffer_length,
                    });
                }
                owned_bytes = owned_bytes
                    .checked_add(bytes)
                    .ok_or(ProducerLayoutError::SizeOverflow("resident producer bytes"))?;
                if matches!(
                    role,
                    PlaneRole::ChunkSegmentBases | PlaneRole::SegmentOffsets
                ) {
                    layout_upload_bytes = layout_upload_bytes.checked_add(bytes).ok_or(
                        ProducerLayoutError::SizeOverflow("producer layout upload bytes"),
                    )?;
                }
            }
        }
        self.validate_additional_working_set(owned_bytes)?;

        let allocation_started = Instant::now();
        let shards = inputs
            .into_iter()
            .zip(plans)
            .map(|(input, plan)| {
                ResidentProducerShard::new(
                    &self.device,
                    plan,
                    input.source,
                    device_registry_id,
                    source_generation,
                    source_completion_serial,
                    maximum_buffer_length,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        validate_owner_aliases(&shards)?;
        let allocation_and_layout_upload_wall = allocation_started.elapsed();
        let preparation_timing = ProducerPreparationTiming {
            total: total_started.elapsed(),
            pipeline_compile: pipeline_compile_wall,
            allocation_and_layout_upload: allocation_and_layout_upload_wall,
        };

        Ok(ResidentInstructionReadRafProducer {
            context: self.clone(),
            _library: library,
            pipeline,
            limits,
            geometry,
            shards,
            source_generation,
            source_completion_serial,
            preparation_timing,
            source_resident_bytes,
            owned_bytes,
            layout_upload_bytes,
            dispatch_serial: 0,
        })
    }
}

impl ResidentInstructionReadRafProducer {
    pub const fn geometry(&self) -> ProducerGeometry {
        self.geometry
    }

    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.limits
    }

    pub const fn preparation_timing(&self) -> ProducerPreparationTiming {
        self.preparation_timing
    }

    pub const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub const fn source_resident_bytes(&self) -> u64 {
        self.source_resident_bytes
    }

    pub const fn layout_upload_bytes(&self) -> u64 {
        self.layout_upload_bytes
    }

    /// Source planes are attached by buffer handle, so owner preparation copies no source rows.
    pub const fn source_copy_bytes(&self) -> u64 {
        0
    }

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    /// Dispatches every shard and retains all bound resources until completion.
    /// The returned borrow prevents output reuse while its receipts are live.
    pub fn execute(
        &mut self,
    ) -> Result<CompletedInstructionReadRafProducer<'_>, ProducerRuntimeError> {
        let next_serial = self
            .dispatch_serial
            .checked_add(1)
            .ok_or(ProducerRuntimeError::DispatchSerialOverflow)?;
        let resident_started = Instant::now();
        for shard in &self.shards {
            let _ = shard.bind_checked(&shard.plan)?;
        }
        validate_owner_aliases(&self.shards)?;
        for shard in &self.shards {
            shard.clear_status();
        }

        let gpu_active = autoreleasepool(|| -> Result<Duration, ProducerRuntimeError> {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            for shard in &self.shards {
                let bound = shard.bind_checked(&shard.plan)?;
                for role in SCATTER_BUFFER_ROLES {
                    encoder.set_buffer(
                        role.metal_buffer_slot() as u64,
                        Some(bound.buffer(role)),
                        0,
                    );
                }
                set_inline_bytes(encoder, SCATTER_PARAMS_SLOT, &shard.plan.params());
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: shard.plan.threadgroups() as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: shard.plan.threads_per_group() as u64,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(status).into());
            }

            let mut nonzero_shards = 0usize;
            let mut first_shard = 0usize;
            let mut first_status = 0u32;
            let mut combined_status = 0u32;
            for shard in &self.shards {
                let status = shard.read_status();
                combined_status |= status;
                if status != 0 {
                    if nonzero_shards == 0 {
                        first_shard = shard.plan.shard().shard_index();
                        first_status = status;
                    }
                    nonzero_shards += 1;
                }
            }
            if nonzero_shards != 0 {
                return Err(ProducerRuntimeError::ShaderStatus {
                    nonzero_shards,
                    first_shard,
                    first_status,
                    combined_status,
                });
            }

            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end }.into());
            }
            Ok(Duration::from_secs_f64(end - start))
        })?;
        self.dispatch_serial = next_serial;
        let receipt = ProducerCompletionReceipt {
            total_rows: self.geometry.total_rows(),
            shard_count: self.geometry.shard_count(),
            device_registry_id: self.context.device_registry_id(),
            source_generation: self.source_generation,
            source_completion_serial: self.source_completion_serial,
            dispatch_serial: next_serial,
        };
        Ok(CompletedInstructionReadRafProducer {
            owner: self,
            receipt,
            timing: ProducerExecutionTiming {
                resident_wall: resident_started.elapsed(),
                gpu_active,
            },
        })
    }
}

pub struct CompletedInstructionReadRafProducer<'a> {
    owner: &'a ResidentInstructionReadRafProducer,
    receipt: ProducerCompletionReceipt,
    timing: ProducerExecutionTiming,
}

impl CompletedInstructionReadRafProducer<'_> {
    pub const fn receipt(&self) -> ProducerCompletionReceipt {
        self.receipt
    }

    pub const fn timing(&self) -> ProducerExecutionTiming {
        self.timing
    }

    pub fn shard(&self, index: usize) -> Option<CompletedProducerShard<'_>> {
        let shard = self.owner.shards.get(index)?;
        Some(CompletedProducerShard {
            shard,
            receipt: self.receipt,
        })
    }
}

pub struct CompletedProducerShard<'a> {
    shard: &'a ResidentProducerShard,
    receipt: ProducerCompletionReceipt,
}

impl CompletedProducerShard<'_> {
    pub const fn shard_plan(&self) -> ProducerShardPlan {
        self.shard.plan.shard()
    }

    pub fn layout(&self) -> &ScatterLayout {
        self.shard.plan.layout()
    }

    pub fn plane(&self, role: PlaneRole) -> Option<ResidentProducerPlane<'_>> {
        if role == PlaneRole::Status {
            return None;
        }
        let initialization = match role {
            PlaneRole::CycleLookupLo | PlaneRole::CycleLookupHi | PlaneRole::CycleClaims => {
                ProducerPlaneInitialization::Source {
                    completion_serial: self.receipt.source_completion_serial,
                }
            }
            PlaneRole::ChunkSegmentBases | PlaneRole::SegmentOffsets => {
                ProducerPlaneInitialization::FrozenLayout
            }
            PlaneRole::GroupedLookupLo
            | PlaneRole::GroupedLookupHi
            | PlaneRole::CycleToGroupedLocal => ProducerPlaneInitialization::ScatterDispatch {
                serial: self.receipt.dispatch_serial,
            },
            PlaneRole::Status => return None,
        };
        Some(self.resident_plane(role, initialization))
    }

    pub fn cycle_lookup_lo(&self) -> ResidentProducerPlane<'_> {
        self.resident_plane(
            PlaneRole::CycleLookupLo,
            ProducerPlaneInitialization::Source {
                completion_serial: self.receipt.source_completion_serial,
            },
        )
    }

    pub fn cycle_lookup_hi(&self) -> ResidentProducerPlane<'_> {
        self.resident_plane(
            PlaneRole::CycleLookupHi,
            ProducerPlaneInitialization::Source {
                completion_serial: self.receipt.source_completion_serial,
            },
        )
    }

    pub fn cycle_claims(&self) -> ResidentProducerPlane<'_> {
        self.resident_plane(
            PlaneRole::CycleClaims,
            ProducerPlaneInitialization::Source {
                completion_serial: self.receipt.source_completion_serial,
            },
        )
    }

    pub fn grouped_lookup_lo(&self) -> ResidentProducerPlane<'_> {
        self.resident_plane(
            PlaneRole::GroupedLookupLo,
            ProducerPlaneInitialization::ScatterDispatch {
                serial: self.receipt.dispatch_serial,
            },
        )
    }

    pub fn grouped_lookup_hi(&self) -> ResidentProducerPlane<'_> {
        self.resident_plane(
            PlaneRole::GroupedLookupHi,
            ProducerPlaneInitialization::ScatterDispatch {
                serial: self.receipt.dispatch_serial,
            },
        )
    }

    pub fn cycle_to_grouped_local(&self) -> ResidentProducerPlane<'_> {
        self.resident_plane(
            PlaneRole::CycleToGroupedLocal,
            ProducerPlaneInitialization::ScatterDispatch {
                serial: self.receipt.dispatch_serial,
            },
        )
    }

    pub fn segment_offsets(&self) -> ResidentProducerPlane<'_> {
        self.resident_plane(
            PlaneRole::SegmentOffsets,
            ProducerPlaneInitialization::FrozenLayout,
        )
    }

    pub fn chunk_segment_bases(&self) -> ResidentProducerPlane<'_> {
        self.resident_plane(
            PlaneRole::ChunkSegmentBases,
            ProducerPlaneInitialization::FrozenLayout,
        )
    }

    fn resident_plane(
        &self,
        role: PlaneRole,
        initialization: ProducerPlaneInitialization,
    ) -> ResidentProducerPlane<'_> {
        let buffer = self.shard.buffers.buffer(role);
        ResidentProducerPlane {
            buffer,
            receipt: ProducerPlaneReceipt {
                shard: self.shard.plan.shard(),
                role,
                bytes: buffer.length(),
                device_registry_id: self.receipt.device_registry_id,
                allocation_identity: buffer_identity(buffer),
                source_generation: self.receipt.source_generation,
                source_completion_serial: self.receipt.source_completion_serial,
                initialization,
            },
        }
    }
}

pub(super) fn validate_recorded_binding(
    record: BindingRecord,
    facts: BindingFacts,
    expected_generation: u64,
    maximum_buffer_length: u64,
) -> Result<(), ProducerRuntimeError> {
    validate_binding_shape(
        record.shard,
        record.role,
        record.bytes,
        facts,
        record.device_registry_id,
        maximum_buffer_length,
    )?;
    if record.allocation_identity != facts.allocation_identity {
        return Err(ProducerRuntimeError::BufferIdentity {
            shard: record.shard,
            role: record.role,
        });
    }
    if record.source_generation != expected_generation {
        return Err(ProducerRuntimeError::SourceGeneration {
            shard: record.shard,
            expected: expected_generation,
            got: record.source_generation,
        });
    }
    Ok(())
}

fn validate_binding_shape(
    shard: usize,
    role: PlaneRole,
    expected_bytes: u64,
    facts: BindingFacts,
    expected_device: u64,
    maximum_buffer_length: u64,
) -> Result<(), ProducerRuntimeError> {
    if facts.bytes > maximum_buffer_length {
        return Err(ProducerRuntimeError::DeviceBufferLimit {
            shard,
            role,
            bytes: facts.bytes,
            maximum: maximum_buffer_length,
        });
    }
    if facts.bytes != expected_bytes {
        return Err(ProducerRuntimeError::BufferLength {
            shard,
            role,
            expected: expected_bytes,
            got: facts.bytes,
        });
    }
    if facts.device_registry_id != expected_device {
        return Err(ProducerRuntimeError::BufferDevice {
            shard,
            role,
            expected: expected_device,
            got: facts.device_registry_id,
        });
    }
    Ok(())
}

fn validate_aliases(records: &[BindingRecord; 9]) -> Result<(), ProducerRuntimeError> {
    for left in 0..records.len() {
        for right in left + 1..records.len() {
            if records[left].allocation_identity == records[right].allocation_identity {
                return Err(ProducerRuntimeError::AliasedBuffers {
                    first_shard: records[left].shard,
                    first_role: records[left].role,
                    second_shard: records[right].shard,
                    second_role: records[right].role,
                    identity: records[left].allocation_identity,
                });
            }
        }
    }
    Ok(())
}

pub(super) fn validate_source_aliases(
    shard: usize,
    identities: [usize; 3],
) -> Result<(), ProducerRuntimeError> {
    let roles = source_roles();
    for left in 0..identities.len() {
        for right in left + 1..identities.len() {
            if identities[left] == identities[right] {
                return Err(ProducerRuntimeError::AliasedBuffers {
                    first_shard: shard,
                    first_role: roles[left],
                    second_shard: shard,
                    second_role: roles[right],
                    identity: identities[left],
                });
            }
        }
    }
    Ok(())
}

fn validate_owner_aliases(shards: &[ResidentProducerShard]) -> Result<(), ProducerRuntimeError> {
    for (first_shard, first_resident) in shards.iter().enumerate() {
        for first_slot in 0..SCATTER_BUFFER_ROLES.len() {
            let first = first_resident.records[first_slot];
            for (second_shard, second_resident) in shards.iter().enumerate().skip(first_shard) {
                let start_slot = if second_shard == first_shard {
                    first_slot + 1
                } else {
                    0
                };
                for second_slot in start_slot..SCATTER_BUFFER_ROLES.len() {
                    let second = second_resident.records[second_slot];
                    if first.allocation_identity == second.allocation_identity {
                        return Err(ProducerRuntimeError::AliasedBuffers {
                            first_shard: first.shard,
                            first_role: first.role,
                            second_shard: second.shard,
                            second_role: second.role,
                            identity: first.allocation_identity,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_pipeline(
    limits: PipelineLimits,
    maximum_threadgroup_memory: u64,
) -> Result<(), ProducerRuntimeError> {
    if PRODUCER_THREADS_PER_GROUP > limits.max_total_threads_per_threadgroup {
        return Err(ProducerRuntimeError::ThreadLimit {
            requested: PRODUCER_THREADS_PER_GROUP,
            maximum: limits.max_total_threads_per_threadgroup,
        });
    }
    if limits.static_threadgroup_memory_length > maximum_threadgroup_memory {
        return Err(ProducerRuntimeError::ThreadgroupMemory {
            requested: limits.static_threadgroup_memory_length,
            maximum: maximum_threadgroup_memory,
        });
    }
    Ok(())
}

fn allocate_private(
    device: &metal::Device,
    shard: ProducerShardPlan,
    role: PlaneRole,
) -> Result<Buffer, ProducerRuntimeError> {
    Ok(device.new_buffer(
        shape_bytes(shard, role)?,
        MTLResourceOptions::StorageModePrivate,
    ))
}

fn shape_bytes(shard: ProducerShardPlan, role: PlaneRole) -> Result<u64, ProducerLayoutError> {
    u64::try_from(shard.buffer_shape(role)?.bytes())
        .map_err(|_| ProducerLayoutError::SizeOverflow("Metal buffer bytes"))
}

fn binding_facts(buffer: &Buffer) -> BindingFacts {
    BindingFacts {
        bytes: buffer.length(),
        device_registry_id: buffer.device().registry_id(),
        allocation_identity: buffer_identity(buffer),
    }
}

fn buffer_identity(buffer: &Buffer) -> usize {
    buffer.as_ptr() as usize
}

const fn source_roles() -> [PlaneRole; 3] {
    [
        PlaneRole::CycleLookupLo,
        PlaneRole::CycleLookupHi,
        PlaneRole::CycleClaims,
    ]
}

const fn owned_roles() -> [PlaneRole; 6] {
    [
        PlaneRole::ChunkSegmentBases,
        PlaneRole::SegmentOffsets,
        PlaneRole::GroupedLookupLo,
        PlaneRole::GroupedLookupHi,
        PlaneRole::CycleToGroupedLocal,
        PlaneRole::Status,
    ]
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, slot: u64, value: &T) {
    encoder.set_bytes(
        slot,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

const _: () = assert!(SCATTER_PARAMS_SLOT == 9);
const _: () = assert!(size_of::<ScatterParams>() == 64);
const _: () = assert!(MAX_BUFFER_BYTES == 2 * 1024 * 1024 * 1024);
