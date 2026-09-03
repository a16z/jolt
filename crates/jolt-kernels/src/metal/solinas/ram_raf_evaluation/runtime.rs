use std::{mem::size_of, slice, time::Duration};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBuffer,
    ComputePipelineState, MTLResourceOptions, MTLSize, NSRange,
};

use super::super::{
    buffer_from_slice, completed_command_gpu_time, set_inline_bytes, Fp128, MetalError,
    RamRafSegmentedAddressPlane, SolinasMetal,
};
use super::{
    split_equality, RamRafAddress, RamRafConfig, RamRafCounters, RamRafDeviceLimits, RamRafError,
    RamRafFoldParams, RamRafSegmentedParams, RamRafShape, RamRafStoragePlan,
    ValidatedRamRafAddressPlane, RAM_RAF_ADDRESS_DOMAIN, RAM_RAF_FINALIZE_PIPELINE,
    RAM_RAF_FOLD_PIPELINE, RAM_RAF_INNER_LENGTH, RAM_RAF_SEGMENTED_BOUNDED_PIPELINE,
    RAM_RAF_SEGMENTED_COLD_PIPELINE, RAM_RAF_SEGMENTED_HOT_CHUNK_PIPELINE,
    RAM_RAF_SEGMENTED_HOT_FINALIZE_PIPELINE, RAM_RAF_SEGMENTED_THREADGROUP_BYTES,
    RAM_RAF_SEGMENTED_THREADS, RAM_RAF_SIMD_WIDTH,
};
use crate::metal::ram_records::ValidatedRamAccessAddresses;

#[derive(Clone)]
pub struct RamRafAddressPlane {
    buffer: Buffer,
    shape: RamRafShape,
    metadata: ValidatedRamRafAddressPlane,
}

impl RamRafAddressPlane {
    pub const fn rows(&self) -> usize {
        self.shape.rows()
    }

    pub const fn address_domain(&self) -> usize {
        self.shape.addresses()
    }

    pub const fn resident_bytes(&self) -> usize {
        self.metadata.byte_length()
    }

    pub const fn storage_id(&self) -> usize {
        self.metadata.storage_id()
    }

    pub const fn device_registry_id(&self) -> u64 {
        self.metadata.device_registry_id()
    }

    ref_field_getters! { pub(crate), { buffer: Buffer }}
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamRafAddressPlane {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_addresses"),
            self.resident_bytes(),
        );
        visitor.exit();
    }
}

struct RamRafBuffers {
    e_lo: Buffer,
    e_hi: Buffer,
    deferred: Buffer,
    output: Buffer,
    counters: Buffer,
}

pub struct RamRafSequence {
    context: SolinasMetal,
    fold_pipeline: ComputePipelineState,
    finalize_pipeline: ComputePipelineState,
    fold_threads: usize,
    finalize_threads: usize,
    addresses: RamRafAddressPlane,
    buffers: RamRafBuffers,
    params: RamRafFoldParams,
    storage: RamRafStoragePlan,
}

struct RamRafSegmentedBuffers {
    e_lo: Buffer,
    e_hi: Buffer,
    hot_partials: Buffer,
    output: Buffer,
    counters: Buffer,
}

struct RamRafSegmentedPipelines {
    cold: ComputePipelineState,
    bounded: ComputePipelineState,
    hot_chunk: ComputePipelineState,
    hot_finalize: ComputePipelineState,
}

pub(crate) struct RamRafSegmentedSequence {
    context: SolinasMetal,
    pipelines: RamRafSegmentedPipelines,
    source: RamRafSegmentedAddressPlane,
    buffers: RamRafSegmentedBuffers,
    params: RamRafSegmentedParams,
    threads: usize,
    #[cfg(feature = "allocative")]
    owned_bytes: usize,
}

struct RamRafCommand {
    command_buffer: CommandBuffer,
    resource_identities: [usize; 6],
}

struct RamRafSegmentedCommand {
    command_buffer: CommandBuffer,
    resource_identities: [usize; 10],
}

enum PendingRamRafKind {
    Dense {
        sequence: RamRafSequence,
        command: RamRafCommand,
    },
    Segmented {
        sequence: RamRafSegmentedSequence,
        command: RamRafSegmentedCommand,
    },
}

#[must_use = "a submitted RAM RAF sequence must be joined before its output is used"]
pub struct PendingRamRafSequence {
    kind: Option<PendingRamRafKind>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingRamRafSequence {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(kind) = &self.kind {
            let resident_bytes = match kind {
                PendingRamRafKind::Dense { sequence, .. } => sequence.resident_bytes(),
                PendingRamRafKind::Segmented { sequence, .. } => sequence.resident_bytes(),
            };
            visitor.visit_simple(allocative::Key::new("device_buffers"), resident_bytes);
        }
        visitor.exit();
    }
}

impl Drop for PendingRamRafSequence {
    fn drop(&mut self) {
        match &self.kind {
            Some(PendingRamRafKind::Dense { command, .. }) => {
                command.command_buffer.wait_until_completed();
            }
            Some(PendingRamRafKind::Segmented { command, .. }) => {
                command.command_buffer.wait_until_completed();
            }
            None => {}
        }
    }
}

impl PendingRamRafSequence {
    pub fn rows(&self) -> Option<usize> {
        match &self.kind {
            Some(PendingRamRafKind::Dense { sequence, .. }) => Some(sequence.addresses.rows()),
            Some(PendingRamRafKind::Segmented { sequence, .. }) => Some(sequence.source.rows()),
            None => None,
        }
    }

    pub fn address_domain(&self) -> Option<usize> {
        match &self.kind {
            Some(PendingRamRafKind::Dense { sequence, .. }) => {
                Some(sequence.addresses.address_domain())
            }
            Some(PendingRamRafKind::Segmented { sequence, .. }) => {
                Some(sequence.source.addresses())
            }
            None => None,
        }
    }

    pub fn address_storage_id(&self) -> Option<usize> {
        match &self.kind {
            Some(PendingRamRafKind::Dense { sequence, .. }) => Some(sequence.address_storage_id()),
            Some(PendingRamRafKind::Segmented { sequence, .. }) => {
                Some(sequence.source.storage_id())
            }
            None => None,
        }
    }

    pub fn is_segmented(&self) -> bool {
        matches!(&self.kind, Some(PendingRamRafKind::Segmented { .. }))
    }

    pub fn join(mut self) -> Result<RamRafObservation, MetalError> {
        match self.kind.take().ok_or(MetalError::NotExecuted)? {
            PendingRamRafKind::Dense { sequence, command } => sequence.complete(command),
            PendingRamRafKind::Segmented { sequence, command } => sequence.complete(command),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamRafObservation {
    pub masses: Vec<AkitaField>,
    pub counters: RamRafCounters,
    pub gpu_active: Duration,
}

impl SolinasMetal {
    pub fn prepare_ram_raf_addresses(
        &self,
        addresses: &[u32],
        config: RamRafConfig,
    ) -> Result<RamRafAddressPlane, MetalError> {
        let shape = config.validate_metal(addresses.len(), RAM_RAF_ADDRESS_DOMAIN)?;
        for &address in addresses {
            let _ = RamRafAddress::try_from(address)?;
        }
        self.prepare_ram_raf_address_buffer(addresses, shape)
    }

    pub(crate) fn prepare_ram_raf_certified_addresses(
        &self,
        addresses: ValidatedRamAccessAddresses<'_>,
        config: RamRafConfig,
    ) -> Result<RamRafAddressPlane, MetalError> {
        let addresses = addresses.as_slice();
        let shape = config.validate_metal(addresses.len(), RAM_RAF_ADDRESS_DOMAIN)?;
        self.prepare_ram_raf_address_buffer(addresses, shape)
    }

    fn prepare_ram_raf_address_buffer(
        &self,
        addresses: &[u32],
        shape: RamRafShape,
    ) -> Result<RamRafAddressPlane, MetalError> {
        let byte_length = addresses
            .len()
            .checked_mul(size_of::<u32>())
            .ok_or(MetalError::InputTooLong(addresses.len()))?;
        let bytes =
            u64::try_from(byte_length).map_err(|_| MetalError::InputTooLong(addresses.len()))?;
        self.validate_buffer_length(bytes)?;
        self.validate_additional_working_set(bytes)?;
        let buffer = buffer_from_slice(&self.device, addresses);
        let metadata = ValidatedRamRafAddressPlane::new_after_content_validation(
            shape,
            byte_length,
            self.device_registry_id(),
            buffer.as_ptr() as usize,
        )?;
        Ok(RamRafAddressPlane {
            buffer,
            shape,
            metadata,
        })
    }

    pub fn prepare_ram_raf_sequence(
        &self,
        addresses: RamRafAddressPlane,
        cycle_point: &[AkitaField],
        config: RamRafConfig,
    ) -> Result<RamRafSequence, MetalError> {
        addresses
            .metadata
            .validate_consumer(addresses.shape, self.device_registry_id())
            .map_err(|error| match error {
                RamRafError::ResidentDeviceMismatch => MetalError::RamRafRowsDevice {
                    expected: self.device_registry_id(),
                    got: addresses.metadata_device_registry_id(),
                },
                other => other.into(),
            })?;
        let expected_point = addresses.shape.rows().ilog2() as usize;
        if cycle_point.len() != expected_point {
            return Err(RamRafError::Length {
                label: "cycle point",
                expected: expected_point,
                got: cycle_point.len(),
            }
            .into());
        }
        let (e_lo, e_hi) = split_equality(cycle_point)?;
        if e_lo.len() != addresses.shape.inner_length()
            || e_hi.len() != addresses.shape.outer_length()
        {
            return Err(RamRafError::ShapeConfigMismatch.into());
        }
        let e_lo = encode_fields(self, "RAM RAF E_lo", &e_lo)?;
        let e_hi = encode_fields(self, "RAM RAF E_hi", &e_hi)?;

        let fold_pipeline = self.compile_named_pipeline(RAM_RAF_FOLD_PIPELINE)?;
        let finalize_pipeline = self.compile_named_pipeline(RAM_RAF_FINALIZE_PIPELINE)?;
        let fold_limits = Self::limits(&fold_pipeline);
        let finalize_limits = Self::limits(&finalize_pipeline);
        let fold_threads = Self::resolve_threadgroup_width(Some(config.threads), fold_limits)?;
        let finalize_threads = Self::resolve_threadgroup_width(None, finalize_limits)?;
        let storage = RamRafStoragePlan::new(addresses.shape)?;
        let device_info = self.device_info();
        let device_limits = RamRafDeviceLimits {
            thread_execution_width: fold_limits.thread_execution_width,
            max_threads_per_threadgroup: fold_limits.max_total_threads_per_threadgroup,
            max_threadgroup_memory_bytes: usize::try_from(
                device_info.max_threadgroup_memory_length,
            )
            .map_err(|_| MetalError::InputTooLong(storage.dynamic_threadgroup_bytes))?,
            pipeline_static_threadgroup_bytes: usize::try_from(
                fold_limits.static_threadgroup_memory_length,
            )
            .map_err(|_| MetalError::InputTooLong(storage.dynamic_threadgroup_bytes))?,
            max_buffer_bytes: usize::try_from(device_info.max_buffer_length)
                .map_err(|_| MetalError::InputTooLong(addresses.shape.rows()))?,
            recommended_working_set_bytes: usize::try_from(
                device_info.recommended_max_working_set_size,
            )
            .map_err(|_| MetalError::InputTooLong(storage.sequence_owned_bytes))?,
        };
        let _ = device_limits.validate(config, addresses.shape)?;
        if finalize_limits.thread_execution_width != RAM_RAF_SIMD_WIDTH {
            return Err(RamRafError::ThreadExecutionWidth {
                got: finalize_limits.thread_execution_width,
            }
            .into());
        }
        let workspace_bytes = u64::try_from(storage.sequence_owned_bytes)
            .map_err(|_| MetalError::InputTooLong(storage.sequence_owned_bytes))?;
        self.validate_additional_working_set(workspace_bytes)?;
        let params = config.fold_params(addresses.shape)?;

        Ok(RamRafSequence {
            context: self.clone(),
            fold_pipeline,
            finalize_pipeline,
            fold_threads,
            finalize_threads,
            addresses,
            buffers: RamRafBuffers {
                e_lo: buffer_from_slice(&self.device, &e_lo),
                e_hi: buffer_from_slice(&self.device, &e_hi),
                deferred: self.device.new_buffer(
                    u64::try_from(storage.deferred_bytes)
                        .map_err(|_| MetalError::InputTooLong(storage.deferred_bytes))?,
                    MTLResourceOptions::StorageModeShared,
                ),
                output: self.device.new_buffer(
                    u64::try_from(storage.canonical_bytes)
                        .map_err(|_| MetalError::InputTooLong(storage.canonical_bytes))?,
                    MTLResourceOptions::StorageModeShared,
                ),
                counters: self.device.new_buffer(
                    size_of::<RamRafCounters>() as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
            },
            params,
            storage,
        })
    }

    pub(crate) fn prepare_ram_raf_segmented_sequence(
        &self,
        source: RamRafSegmentedAddressPlane,
        cycle_point: &[AkitaField],
    ) -> Result<RamRafSegmentedSequence, MetalError> {
        if source.device_registry_id() != self.device_registry_id() {
            return Err(MetalError::RamRafRowsDevice {
                expected: self.device_registry_id(),
                got: source.device_registry_id(),
            });
        }
        if source.rows() < RAM_RAF_INNER_LENGTH
            || !source.rows().is_power_of_two()
            || !source.rows().is_multiple_of(RAM_RAF_INNER_LENGTH)
            || source.addresses() == 0
            || !source.addresses().is_power_of_two()
            || source.accesses() == 0
            || source.cold_segment_threshold() == 0
            || source.cold_segment_threshold() > source.hot_message_chunk_size()
            || source.hot_message_chunk_size() != 1 << 12
        {
            return Err(MetalError::InvalidRamRafState(
                "segmented source has inconsistent geometry",
            ));
        }
        let expected_point = source.rows().ilog2() as usize;
        if cycle_point.len() != expected_point {
            return Err(RamRafError::Length {
                label: "cycle point",
                expected: expected_point,
                got: cycle_point.len(),
            }
            .into());
        }
        let (e_lo, e_hi) = split_equality(cycle_point)?;
        let outer_length = source.rows() / RAM_RAF_INNER_LENGTH;
        if e_lo.len() != RAM_RAF_INNER_LENGTH || e_hi.len() != outer_length {
            return Err(MetalError::InvalidRamRafState(
                "segmented equality split has the wrong shape",
            ));
        }
        let e_lo = encode_fields(self, "segmented RAM RAF E_lo", &e_lo)?;
        let e_hi = encode_fields(self, "segmented RAM RAF E_hi", &e_hi)?;

        let pipelines = RamRafSegmentedPipelines {
            cold: self.compile_named_pipeline(RAM_RAF_SEGMENTED_COLD_PIPELINE)?,
            bounded: self.compile_named_pipeline(RAM_RAF_SEGMENTED_BOUNDED_PIPELINE)?,
            hot_chunk: self.compile_named_pipeline(RAM_RAF_SEGMENTED_HOT_CHUNK_PIPELINE)?,
            hot_finalize: self.compile_named_pipeline(RAM_RAF_SEGMENTED_HOT_FINALIZE_PIPELINE)?,
        };
        let limits = [
            Self::limits(&pipelines.cold),
            Self::limits(&pipelines.bounded),
            Self::limits(&pipelines.hot_chunk),
            Self::limits(&pipelines.hot_finalize),
        ];
        for limit in limits {
            if limit.thread_execution_width != RAM_RAF_SIMD_WIDTH
                || limit.max_total_threads_per_threadgroup < RAM_RAF_SEGMENTED_THREADS
            {
                return Err(MetalError::InvalidRamRafState(
                    "segmented pipeline has unsupported execution limits",
                ));
            }
        }
        let bounded_threadgroup_bytes = limits[1]
            .static_threadgroup_memory_length
            .saturating_add(RAM_RAF_SEGMENTED_THREADGROUP_BYTES as u64);
        let hot_threadgroup_bytes = limits[2]
            .static_threadgroup_memory_length
            .saturating_add(RAM_RAF_SEGMENTED_THREADGROUP_BYTES as u64);
        if bounded_threadgroup_bytes > self.device.max_threadgroup_memory_length()
            || hot_threadgroup_bytes > self.device.max_threadgroup_memory_length()
        {
            return Err(MetalError::InvalidRamRafState(
                "segmented reduction exceeds threadgroup memory",
            ));
        }
        let threads = Self::resolve_threadgroup_width(
            Some(RAM_RAF_SEGMENTED_THREADS),
            Self::limits(&pipelines.bounded),
        )?;
        if threads != RAM_RAF_SEGMENTED_THREADS {
            return Err(MetalError::InvalidRamRafState(
                "segmented reduction resolved the wrong thread width",
            ));
        }

        let e_lo_bytes = e_lo
            .len()
            .checked_mul(size_of::<Fp128>())
            .ok_or(MetalError::InputTooLong(e_lo.len()))?;
        let e_hi_bytes = e_hi
            .len()
            .checked_mul(size_of::<Fp128>())
            .ok_or(MetalError::InputTooLong(e_hi.len()))?;
        let hot_partial_elements = source.hot_message_chunk_count().max(1);
        let hot_partial_bytes = hot_partial_elements
            .checked_mul(size_of::<Fp128>())
            .ok_or(MetalError::InputTooLong(hot_partial_elements))?;
        let output_bytes = source
            .addresses()
            .checked_mul(size_of::<Fp128>())
            .ok_or(MetalError::InputTooLong(source.addresses()))?;
        let counters_bytes = size_of::<RamRafCounters>();
        let owned_bytes = [
            e_lo_bytes,
            e_hi_bytes,
            hot_partial_bytes,
            output_bytes,
            counters_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .ok_or(MetalError::InputTooLong(source.addresses()))?;
        self.validate_additional_working_set(
            u64::try_from(owned_bytes).map_err(|_| MetalError::InputTooLong(owned_bytes))?,
        )?;
        for bytes in [hot_partial_bytes, output_bytes, counters_bytes] {
            self.validate_buffer_length(
                u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(bytes))?,
            )?;
        }

        let params = RamRafSegmentedParams {
            rows: segmented_count("rows", source.rows())?,
            addresses: segmented_count("addresses", source.addresses())?,
            accesses: segmented_count("accesses", source.accesses())?,
            inner_length: RAM_RAF_INNER_LENGTH as u32,
            outer_length: segmented_count("outer length", outer_length)?,
            cold_segment_threshold: segmented_count(
                "cold segment threshold",
                source.cold_segment_threshold(),
            )?,
            hot_message_chunk_size: segmented_count(
                "hot message chunk size",
                source.hot_message_chunk_size(),
            )?,
            bounded_address_count: segmented_count(
                "bounded address count",
                source.bounded_address_count(),
            )?,
            hot_address_count: segmented_count("hot address count", source.hot_address_count())?,
            hot_message_chunk_count: segmented_count(
                "hot message chunk count",
                source.hot_message_chunk_count(),
            )?,
        };
        Ok(RamRafSegmentedSequence {
            context: self.clone(),
            pipelines,
            source,
            buffers: RamRafSegmentedBuffers {
                e_lo: buffer_from_slice(&self.device, &e_lo),
                e_hi: buffer_from_slice(&self.device, &e_hi),
                hot_partials: self.device.new_buffer(
                    hot_partial_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
                output: self
                    .device
                    .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared),
                counters: self
                    .device
                    .new_buffer(counters_bytes as u64, MTLResourceOptions::StorageModeShared),
            },
            params,
            threads,
            #[cfg(feature = "allocative")]
            owned_bytes,
        })
    }
}

impl RamRafAddressPlane {
    fn metadata_device_registry_id(&self) -> u64 {
        self.metadata.device_registry_id()
    }
}

impl RamRafSequence {
    copy_field_getters! { pub, {
        fold_threads: usize,
        finalize_threads: usize,
    }}

    pub const fn address_storage_id(&self) -> usize {
        self.addresses.storage_id()
    }

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    pub const fn resident_bytes(&self) -> usize {
        self.addresses.resident_bytes() + self.storage.sequence_owned_bytes
    }

    pub fn execute_timed(&self) -> Result<RamRafObservation, MetalError> {
        let command = self.submit_command();
        self.complete(command)
    }

    pub fn submit(self) -> PendingRamRafSequence {
        let command = self.submit_command();
        PendingRamRafSequence {
            kind: Some(PendingRamRafKind::Dense {
                sequence: self,
                command,
            }),
        }
    }

    fn resource_identities(&self) -> [usize; 6] {
        [
            self.addresses.storage_id(),
            self.buffers.e_lo.as_ptr() as usize,
            self.buffers.e_hi.as_ptr() as usize,
            self.buffers.deferred.as_ptr() as usize,
            self.buffers.output.as_ptr() as usize,
            self.buffers.counters.as_ptr() as usize,
        ]
    }

    fn submit_command(&self) -> RamRafCommand {
        let command_buffer = self.context.queue.new_command_buffer().to_owned();
        autoreleasepool(|| {
            let blit = command_buffer.new_blit_command_encoder();
            blit.fill_buffer(
                &self.buffers.deferred,
                NSRange::new(0, self.buffers.deferred.length()),
                0,
            );
            blit.fill_buffer(
                &self.buffers.counters,
                NSRange::new(0, self.buffers.counters.length()),
                0,
            );
            blit.end_encoding();

            let fold = command_buffer.new_compute_command_encoder();
            fold.set_compute_pipeline_state(&self.fold_pipeline);
            fold.set_buffer(0, Some(self.addresses.buffer()), 0);
            fold.set_buffer(1, Some(&self.buffers.e_lo), 0);
            fold.set_buffer(2, Some(&self.buffers.e_hi), 0);
            fold.set_buffer(3, Some(&self.buffers.deferred), 0);
            fold.set_buffer(4, Some(&self.buffers.counters), 0);
            set_inline_bytes(fold, 5, &self.params);
            fold.set_threadgroup_memory_length(0, self.storage.dynamic_threadgroup_bytes as u64);
            fold.dispatch_thread_groups(
                MTLSize {
                    width: self.params.outer_length as u64,
                    height: self.params.tiles as u64,
                    depth: 1,
                },
                MTLSize {
                    width: self.fold_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            fold.end_encoding();

            let finalize = command_buffer.new_compute_command_encoder();
            finalize.set_compute_pipeline_state(&self.finalize_pipeline);
            finalize.set_buffer(0, Some(&self.buffers.deferred), 0);
            finalize.set_buffer(1, Some(&self.buffers.output), 0);
            set_inline_bytes(finalize, 2, &self.params);
            finalize.dispatch_thread_groups(
                MTLSize {
                    width: (self.params.addresses as usize).div_ceil(self.finalize_threads) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.finalize_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            finalize.end_encoding();

            command_buffer.commit();
        });
        RamRafCommand {
            command_buffer,
            resource_identities: self.resource_identities(),
        }
    }

    fn complete(&self, command: RamRafCommand) -> Result<RamRafObservation, MetalError> {
        if command.resource_identities != self.resource_identities() {
            return Err(MetalError::NotExecuted);
        }
        command.command_buffer.wait_until_completed();
        let gpu_active = completed_command_gpu_time(&command.command_buffer)?;
        let counters = self.read_counters();
        if counters.invalid_rows != 0 || counters.unsupported_dispatches != 0 {
            return Err(MetalError::RamRafDispatch {
                invalid_rows: counters.invalid_rows,
                unsupported_dispatches: counters.unsupported_dispatches,
            });
        }
        Ok(RamRafObservation {
            masses: self.read_masses()?,
            counters,
            gpu_active,
        })
    }

    fn read_counters(&self) -> RamRafCounters {
        // SAFETY: the shared buffer has exactly one `RamRafCounters` value and
        // the command completed before this host read.
        unsafe { *self.buffers.counters.contents().cast::<RamRafCounters>() }
    }

    fn read_masses(&self) -> Result<Vec<AkitaField>, MetalError> {
        // SAFETY: the shared output buffer has exactly `addresses` fields and
        // the command completed before this host read.
        let fields = unsafe {
            slice::from_raw_parts(
                self.buffers.output.contents().cast::<Fp128>(),
                self.params.addresses as usize,
            )
        };
        let mut masses = Vec::with_capacity(fields.len());
        for (index, &field) in fields.iter().enumerate() {
            if !field.is_canonical(self.context.offset) {
                return Err(MetalError::NonCanonicalOutput {
                    index,
                    offset: self.context.offset,
                });
            }
            masses.push(field.into_jolt_field());
        }
        Ok(masses)
    }
}

impl RamRafSegmentedSequence {
    #[cfg(feature = "allocative")]
    pub(crate) fn resident_bytes(&self) -> usize {
        self.source.borrowed_bytes() + self.owned_bytes
    }

    pub(crate) fn address_storage_id(&self) -> usize {
        self.source.storage_id()
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fn execute_timed(&self) -> Result<RamRafObservation, MetalError> {
        let command = self.submit_command();
        self.complete(command)
    }

    pub(crate) fn submit(self) -> PendingRamRafSequence {
        let command = self.submit_command();
        PendingRamRafSequence {
            kind: Some(PendingRamRafKind::Segmented {
                sequence: self,
                command,
            }),
        }
    }

    fn resource_identities(&self) -> [usize; 10] {
        let source = self.source.resource_identities();
        [
            source[0],
            source[1],
            source[2],
            source[3],
            source[4],
            self.buffers.e_lo.as_ptr() as usize,
            self.buffers.e_hi.as_ptr() as usize,
            self.buffers.hot_partials.as_ptr() as usize,
            self.buffers.output.as_ptr() as usize,
            self.buffers.counters.as_ptr() as usize,
        ]
    }

    fn submit_command(&self) -> RamRafSegmentedCommand {
        let command_buffer = self.context.queue.new_command_buffer().to_owned();
        autoreleasepool(|| {
            let blit = command_buffer.new_blit_command_encoder();
            blit.fill_buffer(
                &self.buffers.counters,
                NSRange::new(0, self.buffers.counters.length()),
                0,
            );
            blit.end_encoding();

            let cold = command_buffer.new_compute_command_encoder();
            cold.set_compute_pipeline_state(&self.pipelines.cold);
            cold.set_buffer(0, Some(self.source.segments()), 0);
            cold.set_buffer(1, Some(self.source.blocks()), 0);
            cold.set_buffer(2, Some(&self.buffers.e_lo), 0);
            cold.set_buffer(3, Some(&self.buffers.e_hi), 0);
            cold.set_buffer(4, Some(&self.buffers.output), 0);
            cold.set_buffer(5, Some(&self.buffers.counters), 0);
            set_inline_bytes(cold, 6, &self.params);
            cold.dispatch_thread_groups(
                MTLSize {
                    width: (self.params.addresses as usize).div_ceil(self.threads) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            cold.end_encoding();

            if self.params.bounded_address_count != 0 {
                let bounded = command_buffer.new_compute_command_encoder();
                bounded.set_compute_pipeline_state(&self.pipelines.bounded);
                bounded.set_buffer(0, Some(self.source.segments()), 0);
                bounded.set_buffer(1, Some(self.source.blocks()), 0);
                bounded.set_buffer(2, Some(self.source.bounded_segments()), 0);
                bounded.set_buffer(3, Some(&self.buffers.e_lo), 0);
                bounded.set_buffer(4, Some(&self.buffers.e_hi), 0);
                bounded.set_buffer(5, Some(&self.buffers.output), 0);
                bounded.set_buffer(6, Some(&self.buffers.counters), 0);
                set_inline_bytes(bounded, 7, &self.params);
                bounded
                    .set_threadgroup_memory_length(0, RAM_RAF_SEGMENTED_THREADGROUP_BYTES as u64);
                bounded.dispatch_thread_groups(
                    MTLSize {
                        width: self.params.bounded_address_count as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.threads as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                bounded.end_encoding();
            }

            if self.params.hot_message_chunk_count != 0 {
                let hot_chunk = command_buffer.new_compute_command_encoder();
                hot_chunk.set_compute_pipeline_state(&self.pipelines.hot_chunk);
                hot_chunk.set_buffer(0, Some(self.source.segments()), 0);
                hot_chunk.set_buffer(1, Some(self.source.blocks()), 0);
                hot_chunk.set_buffer(2, Some(self.source.hot_segments()), 0);
                hot_chunk.set_buffer(3, Some(self.source.hot_message_chunks()), 0);
                hot_chunk.set_buffer(4, Some(&self.buffers.e_lo), 0);
                hot_chunk.set_buffer(5, Some(&self.buffers.e_hi), 0);
                hot_chunk.set_buffer(6, Some(&self.buffers.hot_partials), 0);
                hot_chunk.set_buffer(7, Some(&self.buffers.counters), 0);
                set_inline_bytes(hot_chunk, 8, &self.params);
                hot_chunk
                    .set_threadgroup_memory_length(0, RAM_RAF_SEGMENTED_THREADGROUP_BYTES as u64);
                hot_chunk.dispatch_thread_groups(
                    MTLSize {
                        width: self.params.hot_message_chunk_count as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.threads as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                hot_chunk.end_encoding();
            }

            if self.params.hot_address_count != 0 {
                let hot_finalize = command_buffer.new_compute_command_encoder();
                hot_finalize.set_compute_pipeline_state(&self.pipelines.hot_finalize);
                hot_finalize.set_buffer(0, Some(self.source.hot_segments()), 0);
                hot_finalize.set_buffer(1, Some(&self.buffers.hot_partials), 0);
                hot_finalize.set_buffer(2, Some(&self.buffers.output), 0);
                hot_finalize.set_buffer(3, Some(&self.buffers.counters), 0);
                set_inline_bytes(hot_finalize, 4, &self.params);
                hot_finalize.dispatch_thread_groups(
                    MTLSize {
                        width: (self.params.hot_address_count as usize).div_ceil(self.threads)
                            as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.threads as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                hot_finalize.end_encoding();
            }

            command_buffer.commit();
        });
        RamRafSegmentedCommand {
            command_buffer,
            resource_identities: self.resource_identities(),
        }
    }

    fn complete(&self, command: RamRafSegmentedCommand) -> Result<RamRafObservation, MetalError> {
        if command.resource_identities != self.resource_identities() {
            return Err(MetalError::NotExecuted);
        }
        command.command_buffer.wait_until_completed();
        let gpu_active = completed_command_gpu_time(&command.command_buffer)?;
        let mut counters = self.read_counters();
        counters.accessed_rows = self.params.accesses;
        if counters.invalid_rows != 0 || counters.unsupported_dispatches != 0 {
            return Err(MetalError::RamRafDispatch {
                invalid_rows: counters.invalid_rows,
                unsupported_dispatches: counters.unsupported_dispatches,
            });
        }
        Ok(RamRafObservation {
            masses: read_canonical_masses(
                &self.context,
                &self.buffers.output,
                self.params.addresses as usize,
            )?,
            counters,
            gpu_active,
        })
    }

    fn read_counters(&self) -> RamRafCounters {
        // SAFETY: the shared buffer has exactly one counter value and the
        // command completed before this host read.
        unsafe { *self.buffers.counters.contents().cast::<RamRafCounters>() }
    }
}

fn read_canonical_masses(
    context: &SolinasMetal,
    output: &Buffer,
    addresses: usize,
) -> Result<Vec<AkitaField>, MetalError> {
    // SAFETY: the completed shared output contains exactly `addresses` fields.
    let fields = unsafe { slice::from_raw_parts(output.contents().cast::<Fp128>(), addresses) };
    let mut masses = Vec::with_capacity(fields.len());
    for (index, &field) in fields.iter().enumerate() {
        if !field.is_canonical(context.offset) {
            return Err(MetalError::NonCanonicalOutput {
                index,
                offset: context.offset,
            });
        }
        masses.push(field.into_jolt_field());
    }
    Ok(masses)
}

fn encode_fields(
    context: &SolinasMetal,
    name: &'static str,
    fields: &[AkitaField],
) -> Result<Vec<Fp128>, MetalError> {
    let encoded = fields
        .iter()
        .map(Fp128::from_jolt_field)
        .collect::<Vec<_>>();
    context.validate_inputs(name, &encoded)?;
    Ok(encoded)
}

fn segmented_count(label: &'static str, value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| {
        let _ = label;
        MetalError::InputTooLong(value)
    })
}
