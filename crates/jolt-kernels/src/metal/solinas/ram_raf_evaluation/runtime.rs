use std::{
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBuffer,
    ComputePipelineState, MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSRange,
};

use super::super::{
    buffer_from_slice, completed_command_gpu_time, set_inline_bytes, Fp128, MetalError,
    SolinasMetal,
};
use super::{
    split_equality, RamRafAddress, RamRafConfig, RamRafCounters, RamRafDeviceLimits, RamRafError,
    RamRafFoldParams, RamRafShape, RamRafStoragePlan, ValidatedRamRafAddressPlane,
    RAM_RAF_ADDRESS_DOMAIN, RAM_RAF_FINALIZE_PIPELINE, RAM_RAF_FOLD_PIPELINE, RAM_RAF_SIMD_WIDTH,
};
use crate::optimized::ram_trace::ValidatedRamAccessAddresses;

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

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
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

struct RamRafCommand {
    command_buffer: CommandBuffer,
    submitted_at: Instant,
    submit_wall: Duration,
    resource_identities: [usize; 6],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRafSubmissionStats {
    pub submit_wall: Duration,
    pub overlap_wall: Duration,
    pub join_wall: Duration,
    pub lifecycle_wall: Duration,
    pub gpu_active: Duration,
    pub completed_before_join: bool,
}

#[must_use = "a submitted RAM RAF sequence must be joined before its output is used"]
pub struct PendingRamRafSequence {
    sequence: Option<RamRafSequence>,
    command: Option<RamRafCommand>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingRamRafSequence {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(sequence) = &self.sequence {
            visitor.visit_simple(
                allocative::Key::new("device_buffers"),
                sequence.resident_bytes(),
            );
        }
        visitor.exit();
    }
}

impl Drop for PendingRamRafSequence {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.command_buffer.wait_until_completed();
        }
    }
}

impl PendingRamRafSequence {
    pub const fn rows(&self) -> Option<usize> {
        match &self.sequence {
            Some(sequence) => Some(sequence.addresses.rows()),
            None => None,
        }
    }

    pub const fn address_domain(&self) -> Option<usize> {
        match &self.sequence {
            Some(sequence) => Some(sequence.addresses.address_domain()),
            None => None,
        }
    }

    pub const fn address_storage_id(&self) -> Option<usize> {
        match &self.sequence {
            Some(sequence) => Some(sequence.address_storage_id()),
            None => None,
        }
    }

    pub fn join(mut self) -> Result<(RamRafObservation, RamRafSubmissionStats), MetalError> {
        let sequence = self.sequence.take().ok_or(MetalError::NotExecuted)?;
        let command = self.command.take().ok_or(MetalError::NotExecuted)?;
        sequence.complete(command)
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
}

impl RamRafAddressPlane {
    fn metadata_device_registry_id(&self) -> u64 {
        self.metadata.device_registry_id()
    }
}

impl RamRafSequence {
    pub const fn fold_threads(&self) -> usize {
        self.fold_threads
    }

    pub const fn finalize_threads(&self) -> usize {
        self.finalize_threads
    }

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
        self.complete(command).map(|(observation, _)| observation)
    }

    pub fn submit(self) -> PendingRamRafSequence {
        let command = self.submit_command();
        PendingRamRafSequence {
            sequence: Some(self),
            command: Some(command),
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
        let submitted_at = Instant::now();
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
            submitted_at,
            submit_wall: submitted_at.elapsed(),
            resource_identities: self.resource_identities(),
        }
    }

    fn complete(
        &self,
        command: RamRafCommand,
    ) -> Result<(RamRafObservation, RamRafSubmissionStats), MetalError> {
        if command.resource_identities != self.resource_identities() {
            return Err(MetalError::NotExecuted);
        }
        let completed_before_join =
            command.command_buffer.status() == MTLCommandBufferStatus::Completed;
        let join_started = Instant::now();
        let overlap_wall = join_started
            .saturating_duration_since(command.submitted_at)
            .saturating_sub(command.submit_wall);
        command.command_buffer.wait_until_completed();
        let gpu_active = completed_command_gpu_time(&command.command_buffer)?;
        let counters = self.read_counters();
        if counters.invalid_rows != 0 || counters.unsupported_dispatches != 0 {
            return Err(MetalError::RamRafDispatch {
                invalid_rows: counters.invalid_rows,
                unsupported_dispatches: counters.unsupported_dispatches,
            });
        }
        let observation = RamRafObservation {
            masses: self.read_masses()?,
            counters,
            gpu_active,
        };
        let stats = RamRafSubmissionStats {
            submit_wall: command.submit_wall,
            overlap_wall,
            join_wall: join_started.elapsed(),
            lifecycle_wall: command.submitted_at.elapsed(),
            gpu_active,
            completed_before_join,
        };
        Ok((observation, stats))
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
