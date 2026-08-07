use std::{
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSRange,
};
use thiserror::Error;

use super::super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};
use super::{
    RamRaClaimAddress, RamRaClaimConfig, RamRaClaimCounters, RamRaClaimError, RamRaClaimExecution,
    RamRaClaimFallback, RamRaClaimProjection, RamRaClaimQAccumulator, RamRaClaimQPlan,
    RamRaClaimShape, ValidatedRamRaClaimAddressPlane, Q_BUILD_ADDRESSES_SLOT,
    Q_BUILD_COUNTERS_SLOT, Q_BUILD_EQ_ADDRESS_SLOT, Q_BUILD_EQ_HI_SLOT, Q_BUILD_PARAMS_SLOT,
    Q_BUILD_PARTIALS_SLOT, Q_COMPACT_BUILD_COUNTERS_SLOT, Q_COMPACT_BUILD_ENTRIES_SLOT,
    Q_COMPACT_BUILD_EQ_ADDRESS_SLOT, Q_COMPACT_BUILD_EQ_HI_SLOT, Q_COMPACT_BUILD_OFFSETS_SLOT,
    Q_COMPACT_BUILD_PARAMS_SLOT, Q_COMPACT_BUILD_PARTIALS_SLOT, Q_REDUCE_COUNTERS_SLOT,
    Q_REDUCE_OUTPUT_SLOT, Q_REDUCE_PARAMS_SLOT, Q_REDUCE_PARTIALS_SLOT,
    RAM_RA_CLAIM_ADDRESS_DOMAIN, RAM_RA_CLAIM_AKITA_OFFSET, RAM_RA_CLAIM_SIMD_WIDTH,
    RAM_RA_CLAIM_TERMS, REDUCE_Q_PIPELINE,
};

#[derive(Debug, Error)]
pub enum RamRaClaimQRuntimeError {
    #[error(transparent)]
    Contract(#[from] RamRaClaimError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("RAM RA Q requires Akita offset {expected:#x}, got {got:#x}")]
    UnsupportedOffset { expected: u32, got: u32 },
    #[error("RAM RA Q execution was rejected by the checked policy: {0:?}")]
    ExecutionRejected(RamRaClaimFallback),
    #[error("{name} buffer belongs to Metal device {got}, expected {expected}")]
    BufferDevice {
        name: &'static str,
        expected: u64,
        got: u64,
    },
    #[error("{name} buffer has {actual} bytes, expected {expected}")]
    BufferLength {
        name: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error("RAM RA Q buffers alias across read and write bindings")]
    AliasedBuffers,
    #[error("RAM RA Q compact layout is unavailable for this resident address plane")]
    MissingCompactLayout,
    #[error("{pipeline} has execution width {got}, expected {expected}")]
    UnsupportedExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("invalid RAM RA Q state: {0}")]
    InvalidState(&'static str),
}

/// Immutable resident cycle-address plane with validated content metadata.
#[derive(Clone)]
pub struct RamRaClaimResidentAddresses {
    buffer: Buffer,
    compact: Option<RamRaClaimCompactAddresses>,
    shape: RamRaClaimShape,
    metadata: ValidatedRamRaClaimAddressPlane,
}

#[derive(Clone)]
struct RamRaClaimCompactAddresses {
    entries: Buffer,
    offsets: Buffer,
    entry_count: usize,
    entries_bytes: usize,
    offsets_bytes: usize,
    entry_identity: usize,
    offset_identity: usize,
}

impl RamRaClaimResidentAddresses {
    pub const fn rows(&self) -> usize {
        self.shape.rows()
    }

    pub const fn shape(&self) -> RamRaClaimShape {
        self.shape
    }

    pub const fn accessed_rows(&self) -> usize {
        self.metadata.accessed_rows()
    }

    pub const fn resident_bytes(&self) -> usize {
        self.metadata.byte_length()
    }

    pub const fn device_registry_id(&self) -> u64 {
        self.metadata.device_registry_id()
    }

    pub fn allocation_identity(&self) -> usize {
        self.buffer.as_ptr() as usize
    }

    pub fn compact_resident_bytes(&self) -> usize {
        self.compact
            .as_ref()
            .map_or(0, |compact| compact.entries_bytes + compact.offsets_bytes)
    }

    fn validate_for(
        &self,
        context: &SolinasMetal,
        shape: RamRaClaimShape,
    ) -> Result<(), RamRaClaimQRuntimeError> {
        if self.shape != shape {
            return Err(RamRaClaimQRuntimeError::InvalidState(
                "resident shape differs from the invocation plan",
            ));
        }
        let expected_device = context.device_registry_id();
        let identity = self.allocation_identity();
        self.metadata.validate_consumer(
            shape,
            usize::try_from(self.buffer.length()).unwrap_or(usize::MAX),
            expected_device,
            identity,
        )?;
        validate_buffer_binding(
            &self.buffer,
            "cycle addresses",
            to_u64(self.metadata.byte_length())?,
            expected_device,
            self.metadata.storage_id(),
        )?;
        if let Some(compact) = &self.compact {
            validate_buffer_binding(
                &compact.entries,
                "compact RAM entries",
                to_u64(compact.entries_bytes)?,
                expected_device,
                compact.entry_identity,
            )?;
            validate_buffer_binding(
                &compact.offsets,
                "compact RAM offsets",
                to_u64(compact.offsets_bytes)?,
                expected_device,
                compact.offset_identity,
            )?;
            if compact.entry_count != self.metadata.accessed_rows() {
                return Err(RamRaClaimQRuntimeError::InvalidState(
                    "compact entry count differs from validated access count",
                ));
            }
        }
        Ok(())
    }
}

struct RamRaClaimQBuffers {
    eq_address: Buffer,
    eq_hi: Buffer,
    q_partials: Buffer,
    q: Buffer,
    counters: Buffer,
}

/// Prepared Q producer/reducer invocation over a resident address plane.
pub struct RamRaClaimQInvocation {
    context: SolinasMetal,
    addresses: RamRaClaimResidentAddresses,
    producer_pipeline: ComputePipelineState,
    reducer_pipeline: ComputePipelineState,
    producer_limits: PipelineLimits,
    reducer_limits: PipelineLimits,
    buffers: RamRaClaimQBuffers,
    buffer_identities: [usize; 5],
    plan: RamRaClaimQPlan,
    projection: RamRaClaimProjection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamRaClaimQObservation {
    pub q: [Vec<AkitaField>; RAM_RA_CLAIM_TERMS],
    pub counters: RamRaClaimCounters,
    pub checksum: u64,
    pub useful_full_products: u64,
    pub perfect_cache_bytes: u64,
    pub shader_requested_bytes: u64,
    pub producer_threadgroups: usize,
    pub reducer_threadgroups: usize,
    pub gpu_active: Duration,
    pub resident_wall: Duration,
}

impl SolinasMetal {
    pub fn prepare_ram_ra_claim_addresses(
        &self,
        addresses: &[u32],
    ) -> Result<RamRaClaimResidentAddresses, RamRaClaimQRuntimeError> {
        let shape = RamRaClaimShape::new(addresses.len(), RAM_RA_CLAIM_ADDRESS_DOMAIN)?;
        let mut accessed_rows = 0usize;
        for &raw in addresses {
            if RamRaClaimAddress::try_from(raw)?.is_access() {
                accessed_rows += 1;
            }
        }
        let (compact_entries, compact_offsets) = compact_addresses_by_low(addresses, shape)?;
        let byte_length =
            shape
                .rows()
                .checked_mul(size_of::<u32>())
                .ok_or(RamRaClaimError::SizeOverflow {
                    label: "resident address bytes",
                })?;
        let bytes = to_u64(byte_length)?;
        self.validate_buffer_length(bytes)?;
        self.validate_additional_working_set(bytes)?;
        let buffer = buffer_from_slice(&self.device, addresses);
        let compact_entry_buffer = buffer_from_slice(&self.device, &compact_entries);
        let compact_offset_buffer = buffer_from_slice(&self.device, &compact_offsets);
        let metadata = ValidatedRamRaClaimAddressPlane::new_after_content_validation(
            shape,
            byte_length,
            accessed_rows,
            self.device_registry_id(),
            buffer.as_ptr() as usize,
        )?;
        Ok(RamRaClaimResidentAddresses {
            buffer,
            compact: Some(RamRaClaimCompactAddresses {
                entry_identity: compact_entry_buffer.as_ptr() as usize,
                offset_identity: compact_offset_buffer.as_ptr() as usize,
                entries_bytes: compact_entries.len() * size_of::<u32>(),
                offsets_bytes: compact_offsets.len() * size_of::<u32>(),
                entry_count: accessed_rows,
                entries: compact_entry_buffer,
                offsets: compact_offset_buffer,
            }),
            shape,
            metadata,
        })
    }

    pub fn attach_ram_ra_claim_addresses(
        &self,
        buffer: Buffer,
        metadata: ValidatedRamRaClaimAddressPlane,
    ) -> Result<RamRaClaimResidentAddresses, RamRaClaimQRuntimeError> {
        let shape = RamRaClaimShape::new(metadata.rows(), RAM_RA_CLAIM_ADDRESS_DOMAIN)?;
        let expected_device = self.device_registry_id();
        let identity = buffer.as_ptr() as usize;
        self.validate_buffer_length(buffer.length())?;
        metadata.validate_consumer(
            shape,
            usize::try_from(buffer.length()).unwrap_or(usize::MAX),
            expected_device,
            identity,
        )?;
        validate_buffer_binding(
            &buffer,
            "cycle addresses",
            to_u64(metadata.byte_length())?,
            expected_device,
            metadata.storage_id(),
        )?;
        Ok(RamRaClaimResidentAddresses {
            buffer,
            compact: None,
            shape,
            metadata,
        })
    }

    pub fn prepare_ram_ra_claim_q(
        &self,
        addresses: &RamRaClaimResidentAddresses,
        r_address: &[AkitaField],
        cycle_points: [&[AkitaField]; RAM_RA_CLAIM_TERMS],
        config: RamRaClaimConfig,
    ) -> Result<RamRaClaimQInvocation, RamRaClaimQRuntimeError> {
        if self.offset != RAM_RA_CLAIM_AKITA_OFFSET {
            return Err(RamRaClaimQRuntimeError::UnsupportedOffset {
                expected: RAM_RA_CLAIM_AKITA_OFFSET,
                got: self.offset,
            });
        }
        let plan = RamRaClaimQPlan::new(config, addresses.shape)?;
        addresses.validate_for(self, plan.shape)?;
        if plan.config.q_accumulator == RamRaClaimQAccumulator::Compact
            && addresses.compact.is_none()
        {
            return Err(RamRaClaimQRuntimeError::MissingCompactLayout);
        }
        match config.execution_for_validated_plane(
            plan.shape,
            addresses.metadata,
            addresses.resident_bytes(),
            addresses.device_registry_id(),
            addresses.allocation_identity(),
        )? {
            RamRaClaimExecution::MetalHybrid => {}
            RamRaClaimExecution::OptimizedCpu(reason) => {
                return Err(RamRaClaimQRuntimeError::ExecutionRejected(reason));
            }
        }
        let address_bits = RAM_RA_CLAIM_ADDRESS_DOMAIN.ilog2() as usize;
        if r_address.len() != address_bits {
            return Err(RamRaClaimError::PointLength {
                point: "address point",
                expected: address_bits,
                got: r_address.len(),
            }
            .into());
        }
        for point in &cycle_points {
            if point.len() != plan.shape.log_t() {
                return Err(RamRaClaimError::PointLength {
                    point: "cycle point",
                    expected: plan.shape.log_t(),
                    got: point.len(),
                }
                .into());
            }
        }

        let eq_address = encode_fields(&EqPolynomial::<AkitaField>::evals(r_address, None));
        let eq_hi = cycle_points.map(|point| {
            EqPolynomial::<AkitaField>::evals(&point[..plan.shape.suffix_bits()], None)
        });
        let eq_hi = encode_term_tables(&eq_hi);
        self.validate_inputs("RAM RA Q eq_address", &eq_address)?;
        self.validate_inputs("RAM RA Q eq_hi", &eq_hi)?;

        validate_q_allocations(self, plan)?;
        let producer_pipeline_name = plan.config.q_accumulator.pipeline();
        let producer_pipeline = self.compile_named_pipeline(producer_pipeline_name)?;
        let reducer_pipeline = self.compile_named_pipeline(REDUCE_Q_PIPELINE)?;
        let producer_limits = Self::limits(&producer_pipeline);
        let reducer_limits = Self::limits(&reducer_pipeline);
        validate_pipeline(producer_pipeline_name, producer_limits)?;
        validate_pipeline(REDUCE_Q_PIPELINE, reducer_limits)?;
        let producer_threads =
            Self::resolve_threadgroup_width(Some(RAM_RA_CLAIM_SIMD_WIDTH), producer_limits)?;
        let reducer_threads =
            Self::resolve_threadgroup_width(Some(RAM_RA_CLAIM_SIMD_WIDTH), reducer_limits)?;
        if producer_threads != plan.producer_dispatch.threads_per_threadgroup
            || reducer_threads != plan.reducer_dispatch.threads_per_threadgroup
        {
            return Err(RamRaClaimQRuntimeError::InvalidState(
                "resolved widths differ from the checked dispatch plan",
            ));
        }

        let buffers = RamRaClaimQBuffers {
            eq_address: buffer_from_slice(&self.device, &eq_address),
            eq_hi: buffer_from_slice(&self.device, &eq_hi),
            q_partials: self.device.new_buffer(
                to_u64(plan.storage.q_partial_bytes)?,
                MTLResourceOptions::StorageModePrivate,
            ),
            q: self.device.new_buffer(
                to_u64(plan.storage.q_bytes)?,
                MTLResourceOptions::StorageModeShared,
            ),
            counters: self.device.new_buffer(
                to_u64(plan.storage.counter_bytes)?,
                MTLResourceOptions::StorageModeShared,
            ),
        };
        let buffer_identities = [
            buffers.eq_address.as_ptr() as usize,
            buffers.eq_hi.as_ptr() as usize,
            buffers.q_partials.as_ptr() as usize,
            buffers.q.as_ptr() as usize,
            buffers.counters.as_ptr() as usize,
        ];
        validate_aliases(addresses, buffer_identities)?;
        let projection = RamRaClaimProjection::new(plan.shape.rows(), addresses.accessed_rows())?;

        Ok(RamRaClaimQInvocation {
            context: self.clone(),
            addresses: addresses.clone(),
            producer_pipeline,
            reducer_pipeline,
            producer_limits,
            reducer_limits,
            buffers,
            buffer_identities,
            plan,
            projection,
        })
    }
}

impl RamRaClaimQInvocation {
    pub fn execute(
        &self,
    ) -> Result<[Vec<AkitaField>; RAM_RA_CLAIM_TERMS], RamRaClaimQRuntimeError> {
        self.execute_timed().map(|observation| observation.q)
    }

    /// Times one prepared resident Q scan and its product-free reduction.
    ///
    /// `resident_wall` includes validation, counter clearing, both dispatches,
    /// synchronization, counter audit, canonical output conversion, and the
    /// output checksum. It excludes producer upload, equality generation,
    /// allocation, and pipeline compilation.
    pub fn execute_timed(&self) -> Result<RamRaClaimQObservation, RamRaClaimQRuntimeError> {
        let wall_started = Instant::now();
        self.validate_state()?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let blit = command_buffer.new_blit_command_encoder();
            blit.fill_buffer(
                &self.buffers.counters,
                NSRange::new(0, self.buffers.counters.length()),
                0,
            );
            blit.end_encoding();

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.producer_pipeline);
            if self.plan.config.q_accumulator == RamRaClaimQAccumulator::Compact {
                let compact = self
                    .addresses
                    .compact
                    .as_ref()
                    .ok_or(RamRaClaimQRuntimeError::MissingCompactLayout)?;
                encoder.set_buffer(Q_COMPACT_BUILD_ENTRIES_SLOT, Some(&compact.entries), 0);
                encoder.set_buffer(Q_COMPACT_BUILD_OFFSETS_SLOT, Some(&compact.offsets), 0);
                encoder.set_buffer(
                    Q_COMPACT_BUILD_EQ_ADDRESS_SLOT,
                    Some(&self.buffers.eq_address),
                    0,
                );
                encoder.set_buffer(Q_COMPACT_BUILD_EQ_HI_SLOT, Some(&self.buffers.eq_hi), 0);
                encoder.set_buffer(
                    Q_COMPACT_BUILD_PARTIALS_SLOT,
                    Some(&self.buffers.q_partials),
                    0,
                );
                encoder.set_buffer(
                    Q_COMPACT_BUILD_COUNTERS_SLOT,
                    Some(&self.buffers.counters),
                    0,
                );
                set_inline_bytes(encoder, Q_COMPACT_BUILD_PARAMS_SLOT, &self.plan.params);
            } else {
                encoder.set_buffer(Q_BUILD_ADDRESSES_SLOT, Some(&self.addresses.buffer), 0);
                encoder.set_buffer(Q_BUILD_EQ_ADDRESS_SLOT, Some(&self.buffers.eq_address), 0);
                encoder.set_buffer(Q_BUILD_EQ_HI_SLOT, Some(&self.buffers.eq_hi), 0);
                encoder.set_buffer(Q_BUILD_PARTIALS_SLOT, Some(&self.buffers.q_partials), 0);
                encoder.set_buffer(Q_BUILD_COUNTERS_SLOT, Some(&self.buffers.counters), 0);
                set_inline_bytes(encoder, Q_BUILD_PARAMS_SLOT, &self.plan.params);
            }
            dispatch(encoder, self.plan.producer_dispatch);

            encoder.set_compute_pipeline_state(&self.reducer_pipeline);
            encoder.set_buffer(Q_REDUCE_PARTIALS_SLOT, Some(&self.buffers.q_partials), 0);
            encoder.set_buffer(Q_REDUCE_OUTPUT_SLOT, Some(&self.buffers.q), 0);
            encoder.set_buffer(Q_REDUCE_COUNTERS_SLOT, Some(&self.buffers.counters), 0);
            set_inline_bytes(encoder, Q_REDUCE_PARAMS_SLOT, &self.plan.params);
            dispatch(encoder, self.plan.reducer_dispatch);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(command_buffer.status()).into());
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end }.into());
            }
            let counters = self.read_counters();
            self.addresses
                .metadata
                .validate_completed_dispatches(counters)?;
            let q = self.read_q()?;
            Ok(RamRaClaimQObservation {
                checksum: ram_ra_claim_q_checksum(&q),
                q,
                counters,
                useful_full_products: self.projection.q_full_width_products,
                perfect_cache_bytes: self.projection.q_perfect_cache_bytes,
                shader_requested_bytes: self.projection.q_shader_logical_bytes,
                producer_threadgroups: self.plan.producer_dispatch.threadgroups,
                reducer_threadgroups: self.plan.reducer_dispatch.threadgroups,
                gpu_active: Duration::from_secs_f64(end - start),
                resident_wall: wall_started.elapsed(),
            })
        })
    }

    pub const fn plan(&self) -> RamRaClaimQPlan {
        self.plan
    }

    pub const fn projection(&self) -> RamRaClaimProjection {
        self.projection
    }

    pub const fn producer_pipeline_limits(&self) -> PipelineLimits {
        self.producer_limits
    }

    pub const fn reducer_pipeline_limits(&self) -> PipelineLimits {
        self.reducer_limits
    }

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn compact_resident_bytes(&self) -> usize {
        self.addresses.compact_resident_bytes()
    }

    pub fn source_allocation_identity(&self) -> usize {
        self.addresses.allocation_identity()
    }

    pub fn output_allocation_identity(&self) -> usize {
        self.buffers.q.as_ptr() as usize
    }

    fn validate_state(&self) -> Result<(), RamRaClaimQRuntimeError> {
        if self.context.offset != RAM_RA_CLAIM_AKITA_OFFSET {
            return Err(RamRaClaimQRuntimeError::UnsupportedOffset {
                expected: RAM_RA_CLAIM_AKITA_OFFSET,
                got: self.context.offset,
            });
        }
        self.addresses
            .validate_for(&self.context, self.plan.shape)?;
        if self.plan.params != self.plan.shape.params(self.plan.config)? {
            return Err(RamRaClaimQRuntimeError::InvalidState(
                "shader parameters differ from checked geometry",
            ));
        }
        if self.plan.config.execution_for_validated_plane(
            self.plan.shape,
            self.addresses.metadata,
            self.addresses.resident_bytes(),
            self.addresses.device_registry_id(),
            self.addresses.allocation_identity(),
        )? != RamRaClaimExecution::MetalHybrid
        {
            return Err(RamRaClaimQRuntimeError::InvalidState(
                "resident plane no longer satisfies the checked execution policy",
            ));
        }
        validate_pipeline(
            self.plan.config.q_accumulator.pipeline(),
            self.producer_limits,
        )?;
        validate_pipeline(REDUCE_Q_PIPELINE, self.reducer_limits)?;

        let expected_device = self.context.device_registry_id();
        for (name, buffer, expected_bytes, expected_identity) in [
            (
                "eq_address",
                &self.buffers.eq_address,
                self.plan.storage.eq_address_bytes,
                self.buffer_identities[0],
            ),
            (
                "eq_hi",
                &self.buffers.eq_hi,
                self.plan.storage.eq_hi_bytes,
                self.buffer_identities[1],
            ),
            (
                "Q partials",
                &self.buffers.q_partials,
                self.plan.storage.q_partial_bytes,
                self.buffer_identities[2],
            ),
            (
                "Q output",
                &self.buffers.q,
                self.plan.storage.q_bytes,
                self.buffer_identities[3],
            ),
            (
                "Q counters",
                &self.buffers.counters,
                self.plan.storage.counter_bytes,
                self.buffer_identities[4],
            ),
        ] {
            validate_buffer_binding(
                buffer,
                name,
                to_u64(expected_bytes)?,
                expected_device,
                expected_identity,
            )?;
        }
        validate_aliases(&self.addresses, self.buffer_identities)
    }

    fn read_counters(&self) -> RamRaClaimCounters {
        // SAFETY: the shared counter buffer contains exactly one initialized
        // counter record, and command completion precedes this host read.
        unsafe {
            *self
                .buffers
                .counters
                .contents()
                .cast::<RamRaClaimCounters>()
        }
    }

    fn read_q(&self) -> Result<[Vec<AkitaField>; RAM_RA_CLAIM_TERMS], RamRaClaimQRuntimeError> {
        let elements = RAM_RA_CLAIM_TERMS * self.plan.shape.prefix_length();
        // SAFETY: the shared Q buffer owns `elements` fields, and command
        // completion precedes this host read.
        let fields =
            unsafe { slice::from_raw_parts(self.buffers.q.contents().cast::<Fp128>(), elements) };
        self.context.validate_inputs("RAM RA Q output", fields)?;
        Ok(core::array::from_fn(|term| {
            fields[term * self.plan.shape.prefix_length()
                ..(term + 1) * self.plan.shape.prefix_length()]
                .iter()
                .map(|&value| value.into_jolt_field())
                .collect()
        }))
    }
}

pub fn ram_ra_claim_q_checksum(q: &[Vec<AkitaField>; RAM_RA_CLAIM_TERMS]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    q.iter()
        .flatten()
        .fold(OFFSET_BASIS, |mut checksum, value| {
            for byte in Fp128::from_jolt_field(value).to_u128().to_le_bytes() {
                checksum ^= u64::from(byte);
                checksum = checksum.wrapping_mul(PRIME);
            }
            checksum
        })
}

fn validate_q_allocations(
    context: &SolinasMetal,
    plan: RamRaClaimQPlan,
) -> Result<(), RamRaClaimQRuntimeError> {
    context.validate_additional_working_set(to_u64(plan.storage.sequence_owned_bytes)?)?;
    for bytes in [
        plan.storage.eq_address_bytes,
        plan.storage.eq_hi_bytes,
        plan.storage.q_partial_bytes,
        plan.storage.q_bytes,
        plan.storage.counter_bytes,
    ] {
        context.validate_buffer_length(to_u64(bytes)?)?;
    }
    Ok(())
}

fn validate_pipeline(
    pipeline: &'static str,
    limits: PipelineLimits,
) -> Result<(), RamRaClaimQRuntimeError> {
    if limits.thread_execution_width != RAM_RA_CLAIM_SIMD_WIDTH {
        return Err(RamRaClaimQRuntimeError::UnsupportedExecutionWidth {
            pipeline,
            expected: RAM_RA_CLAIM_SIMD_WIDTH,
            got: limits.thread_execution_width,
        });
    }
    if limits.max_total_threads_per_threadgroup < RAM_RA_CLAIM_SIMD_WIDTH {
        return Err(RamRaClaimQRuntimeError::InvalidState(
            "pipeline cannot admit one complete SIMDgroup",
        ));
    }
    Ok(())
}

fn validate_aliases(
    addresses: &RamRaClaimResidentAddresses,
    buffer_identities: [usize; 5],
) -> Result<(), RamRaClaimQRuntimeError> {
    let mut identities = Vec::with_capacity(8);
    identities.push(addresses.allocation_identity());
    if let Some(compact) = &addresses.compact {
        identities.push(compact.entry_identity);
        identities.push(compact.offset_identity);
    }
    identities.extend_from_slice(&buffer_identities);
    for left in 0..identities.len() {
        for right in left + 1..identities.len() {
            if identities[left] == identities[right] {
                return Err(RamRaClaimQRuntimeError::AliasedBuffers);
            }
        }
    }
    Ok(())
}

fn compact_addresses_by_low(
    addresses: &[u32],
    shape: RamRaClaimShape,
) -> Result<(Vec<u32>, Vec<u32>), RamRaClaimQRuntimeError> {
    let prefix_length = shape.prefix_length();
    let mut counts = vec![0usize; prefix_length];
    for (row, &address) in addresses.iter().enumerate() {
        if RamRaClaimAddress::try_from(address)?.is_access() {
            counts[row & (prefix_length - 1)] += 1;
        }
    }
    let mut offsets = Vec::with_capacity(prefix_length + 1);
    offsets.push(0u32);
    for count in counts {
        let next = offsets
            .last()
            .copied()
            .map(|offset| offset as usize + count)
            .ok_or(RamRaClaimError::SizeOverflow {
                label: "compact RAM offsets",
            })?;
        offsets.push(
            u32::try_from(next).map_err(|_| RamRaClaimError::SizeOverflow {
                label: "compact RAM offsets",
            })?,
        );
    }
    let entry_count = offsets.last().copied().unwrap_or(0) as usize;
    let physical_count = entry_count.max(1);
    let mut entries = vec![0u32; physical_count];
    let mut cursors: Vec<usize> = offsets[..prefix_length]
        .iter()
        .map(|&offset| offset as usize)
        .collect();
    for (row, &address) in addresses.iter().enumerate() {
        if address == super::RAM_RA_CLAIM_NO_ACCESS {
            continue;
        }
        let lo = row & (prefix_length - 1);
        let hi = row >> shape.prefix_bits();
        let packed = u32::try_from(hi)
            .ok()
            .and_then(|hi| hi.checked_shl(RAM_RA_CLAIM_ADDRESS_DOMAIN.ilog2()))
            .and_then(|hi| hi.checked_add(address))
            .ok_or(RamRaClaimError::SizeOverflow {
                label: "compact RAM entry",
            })?;
        entries[cursors[lo]] = packed;
        cursors[lo] += 1;
    }
    Ok((entries, offsets))
}

fn validate_buffer_binding(
    buffer: &Buffer,
    name: &'static str,
    expected_bytes: u64,
    expected_device: u64,
    expected_identity: usize,
) -> Result<(), RamRaClaimQRuntimeError> {
    let got_device = buffer.device().registry_id();
    if got_device != expected_device {
        return Err(RamRaClaimQRuntimeError::BufferDevice {
            name,
            expected: expected_device,
            got: got_device,
        });
    }
    if buffer.length() != expected_bytes {
        return Err(RamRaClaimQRuntimeError::BufferLength {
            name,
            expected: expected_bytes,
            actual: buffer.length(),
        });
    }
    if buffer.as_ptr() as usize != expected_identity {
        return Err(RamRaClaimQRuntimeError::InvalidState(
            "buffer allocation identity changed",
        ));
    }
    Ok(())
}

fn dispatch(encoder: &metal::ComputeCommandEncoderRef, plan: super::RamRaClaimDispatch) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: plan.threadgroups as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: plan.threads_per_threadgroup as u64,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_fields(values: &[AkitaField]) -> Vec<Fp128> {
    values.iter().map(Fp128::from_jolt_field).collect()
}

fn encode_term_tables(values: &[Vec<AkitaField>; RAM_RA_CLAIM_TERMS]) -> Vec<Fp128> {
    values
        .iter()
        .flat_map(|table| table.iter().map(Fp128::from_jolt_field))
        .collect()
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

fn to_u64(value: usize) -> Result<u64, MetalError> {
    u64::try_from(value).map_err(|_| MetalError::InputTooLong(value))
}
