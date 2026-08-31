use std::{
    ffi::c_void,
    mem::size_of,
    slice,
    sync::Arc,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use metal::{objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};

use super::{
    buffer_from_slice, completed_command_gpu_time, set_inline_bytes, Fp128, MetalError,
    PipelineLimits, SolinasMetal,
};
use crate::optimized::ram_trace::{RamAccessColumns, RamRaCompactRecord, RamRaQRecord, NO_ACCESS};

pub const SOURCE: &str = include_str!("shader.metal");

pub const BUILD_Q_PIPELINE: &str = "solinas_ram_ra_claim_build_q";
pub const BUILD_Q_SPARSE_PIPELINE: &str = "solinas_ram_ra_claim_build_q_sparse";
pub const REDUCE_Q_PIPELINE: &str = "solinas_ram_ra_claim_reduce_q";
pub const GATHER_H_PIPELINE: &str = "solinas_ram_ra_claim_gather_h";
pub const GATHER_H_SPARSE_PIPELINE: &str = "solinas_ram_ra_claim_gather_h_sparse";
pub const TERMS: usize = 3;
pub const SIMD_WIDTH: usize = 32;
pub const Q_THREADS: usize = 256;
pub const GATHER_THREADS: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RamRaClaimReductionParams {
    prefix_elements: u32,
    suffix_elements: u32,
    active_high_elements: u32,
    no_access: u32,
    q_slices: u32,
    active_q_slices: u32,
}

const _: [(); 24] = [(); size_of::<RamRaClaimReductionParams>()];

enum SourceBuffers {
    Dense {
        addresses: Buffer,
    },
    Sparse {
        q_offsets: Buffer,
        q_records: Buffer,
        h_offsets: Buffer,
        h_records: Buffer,
    },
}

struct Buffers {
    source: SourceBuffers,
    eq_address: Buffer,
    eq_hi: Buffer,
    q_partials: Option<Buffer>,
    q: Buffer,
    h_prime: Buffer,
}

pub(crate) struct RamRaClaimReductionSequence {
    context: SolinasMetal,
    _columns: Arc<RamAccessColumns>,
    build_q_pipeline: ComputePipelineState,
    reduce_q_pipeline: ComputePipelineState,
    gather_h_pipeline: ComputePipelineState,
    buffers: Buffers,
    params: RamRaClaimReductionParams,
    prefix_elements: usize,
    suffix_elements: usize,
    q_threads: usize,
    q_threadgroups: usize,
    gather_threads: usize,
    gather_threadgroup_bytes: usize,
    address_alias_reused: bool,
    compact_source: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RamRaClaimQObservation {
    pub q: [Vec<AkitaField>; TERMS],
    pub gpu_active: Duration,
    pub wait_wall: Duration,
    pub readback_wall: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RamRaClaimHObservation {
    pub h_prime: Vec<AkitaField>,
    pub gpu_active: Duration,
}

impl SolinasMetal {
    pub(crate) fn prepare_ram_ra_claim_reduction(
        &self,
        columns: Arc<RamAccessColumns>,
        address_count: usize,
        prefix_bits: usize,
        eq_address: &[AkitaField],
        eq_hi: &[Vec<AkitaField>; TERMS],
        q_slices: usize,
    ) -> Result<RamRaClaimReductionSequence, MetalError> {
        let rows = columns.addresses.len();
        if rows < 16 || !rows.is_power_of_two() || address_count == 0 {
            return Err(MetalError::InvalidRamRaState(
                "RAM RA claim-reduction source has invalid geometry",
            ));
        }
        let prefix_elements = 1usize
            .checked_shl(
                u32::try_from(prefix_bits).map_err(|_| MetalError::InputTooLong(prefix_bits))?,
            )
            .ok_or(MetalError::InputTooLong(prefix_bits))?;
        if !rows.is_multiple_of(prefix_elements) {
            return Err(MetalError::InvalidRamRaState(
                "RAM RA claim-reduction split does not divide the source",
            ));
        }
        let suffix_elements = rows / prefix_elements;
        let active_high_elements = columns
            .active_cycle_bound()
            .div_ceil(prefix_elements)
            .min(suffix_elements);
        if q_slices == 0 || !suffix_elements.is_multiple_of(q_slices) {
            return Err(MetalError::InvalidRamRaState(
                "RAM RA claim-reduction Q slices do not divide the suffix",
            ));
        }
        let high_per_slice = suffix_elements / q_slices;
        let active_q_slices = active_high_elements
            .div_ceil(high_per_slice)
            .clamp(1, q_slices);
        if eq_address.len() != address_count
            || eq_hi.iter().any(|table| table.len() != suffix_elements)
        {
            return Err(MetalError::InvalidRamRaState(
                "RAM RA claim-reduction equality tables have the wrong shape",
            ));
        }
        let params = RamRaClaimReductionParams {
            prefix_elements: abi_count(prefix_elements)?,
            suffix_elements: abi_count(suffix_elements)?,
            active_high_elements: abi_count(active_high_elements)?,
            no_access: NO_ACCESS,
            q_slices: abi_count(q_slices)?,
            active_q_slices: abi_count(active_q_slices)?,
        };
        let encoded_address = encode_fields(eq_address);
        let encoded_hi = eq_hi
            .iter()
            .flat_map(|table| table.iter().map(Fp128::from_jolt_field))
            .collect::<Vec<_>>();
        self.validate_inputs("RAM RA claim address equality", &encoded_address)?;
        self.validate_inputs("RAM RA claim high equality", &encoded_hi)?;

        let compact_source = columns.ram_ra_sparse_layout().is_some();
        let build_pipeline_name = if compact_source {
            BUILD_Q_SPARSE_PIPELINE
        } else {
            BUILD_Q_PIPELINE
        };
        let build_q_pipeline = self.compile_named_pipeline(build_pipeline_name)?;
        let reduce_q_pipeline = self.compile_named_pipeline(REDUCE_Q_PIPELINE)?;
        let gather_pipeline_name = if compact_source {
            GATHER_H_SPARSE_PIPELINE
        } else {
            GATHER_H_PIPELINE
        };
        let gather_h_pipeline = self.compile_named_pipeline(gather_pipeline_name)?;
        let build_limits = Self::limits(&build_q_pipeline);
        let reduce_limits = Self::limits(&reduce_q_pipeline);
        let gather_limits = Self::limits(&gather_h_pipeline);
        let q_threads = Q_THREADS;
        validate_pipeline(build_pipeline_name, build_limits, q_threads)?;
        validate_pipeline(REDUCE_Q_PIPELINE, reduce_limits, Q_THREADS)?;
        validate_pipeline(gather_pipeline_name, gather_limits, GATHER_THREADS)?;
        let q_threads = Self::resolve_threadgroup_width(Some(q_threads), build_limits)?;
        let q_threadgroups = if compact_source {
            prefix_elements.div_ceil(q_threads)
        } else {
            (prefix_elements * active_q_slices).div_ceil(q_threads)
        };
        let gather_threads = Self::resolve_threadgroup_width(Some(GATHER_THREADS), gather_limits)?;
        let gather_threadgroup_bytes = gather_threads / SIMD_WIDTH * size_of::<Fp128>();
        let threadgroup_bytes = u64::try_from(gather_threadgroup_bytes)
            .ok()
            .and_then(|dynamic| dynamic.checked_add(gather_limits.static_threadgroup_memory_length))
            .ok_or(MetalError::InputTooLong(gather_threadgroup_bytes))?;
        if build_limits.static_threadgroup_memory_length
            > self.device.max_threadgroup_memory_length()
            || threadgroup_bytes > self.device.max_threadgroup_memory_length()
        {
            return Err(MetalError::InvalidRamRaState(
                "RAM RA claim-reduction exceeds threadgroup memory",
            ));
        }

        let source_bytes = if let Some(layout) = columns.ram_ra_sparse_layout() {
            [
                byte_length::<u32>(layout.q_offsets().len())?,
                byte_length::<RamRaQRecord>(layout.q_records().len())?,
                byte_length::<u32>(layout.h_offsets().len())?,
                byte_length::<RamRaCompactRecord>(layout.h_records().len())?,
            ]
            .into_iter()
            .try_fold(0u64, |total, bytes| total.checked_add(bytes))
            .ok_or(MetalError::InputTooLong(rows))?
        } else {
            byte_length::<u32>(rows)?
        };
        let address_eq_bytes = byte_length::<Fp128>(encoded_address.len())?;
        let eq_hi_bytes = byte_length::<Fp128>(encoded_hi.len())?;
        let q_bytes = byte_length::<Fp128>(TERMS * prefix_elements)?;
        let q_partial_bytes = if compact_source || active_q_slices == 1 {
            0
        } else {
            byte_length::<Fp128>(TERMS * prefix_elements * active_q_slices)?
        };
        let h_prime_bytes = byte_length::<Fp128>(active_high_elements)?;
        for bytes in [
            source_bytes,
            address_eq_bytes,
            eq_hi_bytes,
            q_partial_bytes,
            q_bytes,
            h_prime_bytes,
        ] {
            self.validate_buffer_length(bytes)?;
        }
        let resident_bytes = [
            source_bytes,
            address_eq_bytes,
            eq_hi_bytes,
            q_partial_bytes,
            q_bytes,
            h_prime_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or(MetalError::InputTooLong(rows))?;
        self.validate_additional_working_set(resident_bytes)?;

        let (source, address_alias_reused) = if let Some(layout) = columns.ram_ra_sparse_layout() {
            let q_offsets_bytes = byte_length::<u32>(layout.q_offsets().len())?;
            let q_records_bytes = byte_length::<RamRaQRecord>(layout.q_records().len())?;
            let h_offsets_bytes = byte_length::<u32>(layout.h_offsets().len())?;
            let h_records_bytes = byte_length::<RamRaCompactRecord>(layout.h_records().len())?;
            let (q_offsets, q_offsets_reused) = self.shared_no_copy_buffer(
                Arc::clone(&columns),
                layout.q_offsets().as_ptr().cast_mut().cast::<c_void>(),
                q_offsets_bytes,
            )?;
            let (q_records, q_records_reused) = self.shared_no_copy_buffer(
                Arc::clone(&columns),
                layout.q_records().as_ptr().cast_mut().cast::<c_void>(),
                q_records_bytes,
            )?;
            let (h_offsets, h_offsets_reused) = self.shared_no_copy_buffer(
                Arc::clone(&columns),
                layout.h_offsets().as_ptr().cast_mut().cast::<c_void>(),
                h_offsets_bytes,
            )?;
            let (h_records, h_records_reused) = self.shared_no_copy_buffer(
                Arc::clone(&columns),
                layout.h_records().as_ptr().cast_mut().cast::<c_void>(),
                h_records_bytes,
            )?;
            (
                SourceBuffers::Sparse {
                    q_offsets,
                    q_records,
                    h_offsets,
                    h_records,
                },
                q_offsets_reused && q_records_reused && h_offsets_reused && h_records_reused,
            )
        } else {
            let (addresses, reused) = self.shared_no_copy_buffer(
                Arc::clone(&columns),
                columns.addresses.as_ptr().cast_mut().cast::<c_void>(),
                source_bytes,
            )?;
            (SourceBuffers::Dense { addresses }, reused)
        };
        Ok(RamRaClaimReductionSequence {
            context: self.clone(),
            _columns: columns,
            build_q_pipeline,
            reduce_q_pipeline,
            gather_h_pipeline,
            buffers: Buffers {
                source,
                eq_address: buffer_from_slice(&self.device, &encoded_address),
                eq_hi: buffer_from_slice(&self.device, &encoded_hi),
                q_partials: (!compact_source && active_q_slices > 1).then(|| {
                    self.device
                        .new_buffer(q_partial_bytes, MTLResourceOptions::StorageModePrivate)
                }),
                q: self
                    .device
                    .new_buffer(q_bytes, MTLResourceOptions::StorageModeShared),
                h_prime: self
                    .device
                    .new_buffer(h_prime_bytes, MTLResourceOptions::StorageModeShared),
            },
            params,
            prefix_elements,
            suffix_elements,
            q_threads,
            q_threadgroups,
            gather_threads,
            gather_threadgroup_bytes,
            address_alias_reused,
            compact_source,
        })
    }
}

impl RamRaClaimReductionSequence {
    pub(crate) fn build_q(&self) -> Result<RamRaClaimQObservation, MetalError> {
        let command = self.context.queue.new_command_buffer().to_owned();
        autoreleasepool(|| {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.build_q_pipeline);
            match &self.buffers.source {
                SourceBuffers::Dense { addresses } => {
                    encoder.set_buffer(0, Some(addresses), 0);
                    encoder.set_buffer(1, Some(&self.buffers.eq_address), 0);
                    encoder.set_buffer(2, Some(&self.buffers.eq_hi), 0);
                    encoder.set_buffer(
                        3,
                        Some(self.buffers.q_partials.as_ref().unwrap_or(&self.buffers.q)),
                        0,
                    );
                    set_inline_bytes(encoder, 4, &self.params);
                }
                SourceBuffers::Sparse {
                    q_offsets,
                    q_records,
                    ..
                } => {
                    encoder.set_buffer(0, Some(q_offsets), 0);
                    encoder.set_buffer(1, Some(q_records), 0);
                    encoder.set_buffer(2, Some(&self.buffers.eq_address), 0);
                    encoder.set_buffer(3, Some(&self.buffers.eq_hi), 0);
                    encoder.set_buffer(4, Some(&self.buffers.q), 0);
                    set_inline_bytes(encoder, 5, &self.params);
                }
            }
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.q_threadgroups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.q_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            if let Some(partials) = &self.buffers.q_partials {
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.reduce_q_pipeline);
                encoder.set_buffer(0, Some(partials), 0);
                encoder.set_buffer(1, Some(&self.buffers.q), 0);
                set_inline_bytes(encoder, 2, &self.params);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: self.prefix_elements.div_ceil(self.q_threads) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.q_threads as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.end_encoding();
            }
            command.commit();
        });
        let wait_started = Instant::now();
        command.wait_until_completed();
        let wait_wall = wait_started.elapsed();
        let gpu_active = completed_command_gpu_time(&command)?;
        let readback_started = Instant::now();
        let output = read_fields(
            &self.context,
            &self.buffers.q,
            TERMS * self.prefix_elements,
            "RAM RA claim Q",
        )?;
        let q = std::array::from_fn(|term| {
            output[term * self.prefix_elements..(term + 1) * self.prefix_elements].to_vec()
        });
        let readback_wall = readback_started.elapsed();
        Ok(RamRaClaimQObservation {
            q,
            gpu_active,
            wait_wall,
            readback_wall,
        })
    }

    pub(crate) fn gather_h(
        &self,
        eq_prefix: &[AkitaField],
    ) -> Result<RamRaClaimHObservation, MetalError> {
        if eq_prefix.len() != self.prefix_elements {
            return Err(MetalError::InvalidRamRaState(
                "RAM RA claim bound-prefix equality has the wrong shape",
            ));
        }
        let encoded_prefix = encode_fields(eq_prefix);
        self.context
            .validate_inputs("RAM RA claim bound-prefix equality", &encoded_prefix)?;
        let eq_prefix = buffer_from_slice(&self.context.device, &encoded_prefix);
        let command = self.context.queue.new_command_buffer().to_owned();
        autoreleasepool(|| {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.gather_h_pipeline);
            match &self.buffers.source {
                SourceBuffers::Dense { addresses } => {
                    encoder.set_buffer(0, Some(addresses), 0);
                    encoder.set_buffer(1, Some(&self.buffers.eq_address), 0);
                    encoder.set_buffer(2, Some(&eq_prefix), 0);
                    encoder.set_buffer(3, Some(&self.buffers.h_prime), 0);
                    set_inline_bytes(encoder, 4, &self.params);
                }
                SourceBuffers::Sparse {
                    h_offsets,
                    h_records,
                    ..
                } => {
                    encoder.set_buffer(0, Some(h_offsets), 0);
                    encoder.set_buffer(1, Some(h_records), 0);
                    encoder.set_buffer(2, Some(&self.buffers.eq_address), 0);
                    encoder.set_buffer(3, Some(&eq_prefix), 0);
                    encoder.set_buffer(4, Some(&self.buffers.h_prime), 0);
                    set_inline_bytes(encoder, 5, &self.params);
                }
            }
            encoder.set_threadgroup_memory_length(0, self.gather_threadgroup_bytes as u64);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: u64::from(self.params.active_high_elements),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.gather_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command.commit();
        });
        command.wait_until_completed();
        let gpu_active = completed_command_gpu_time(&command)?;
        let mut h_prime = read_fields(
            &self.context,
            &self.buffers.h_prime,
            self.params.active_high_elements as usize,
            "RAM RA claim H-prime",
        )?;
        h_prime.resize(self.suffix_elements, AkitaField::zero());
        Ok(RamRaClaimHObservation {
            h_prime,
            gpu_active,
        })
    }

    pub(crate) const fn source_copy_bytes() -> usize {
        0
    }

    pub(crate) const fn address_alias_reused(&self) -> bool {
        self.address_alias_reused
    }

    pub(crate) const fn active_high_elements(&self) -> usize {
        self.params.active_high_elements as usize
    }

    pub(crate) const fn active_q_slices(&self) -> usize {
        self.params.active_q_slices as usize
    }

    pub(crate) const fn compact_source(&self) -> bool {
        self.compact_source
    }

    pub(crate) const fn readback_bytes(&self) -> usize {
        (TERMS * self.prefix_elements + self.params.active_high_elements as usize)
            * size_of::<Fp128>()
    }
}

fn validate_pipeline(
    pipeline: &'static str,
    limits: PipelineLimits,
    threads: usize,
) -> Result<(), MetalError> {
    if limits.thread_execution_width != SIMD_WIDTH {
        return Err(MetalError::UnsupportedRamRaExecutionWidth {
            pipeline,
            expected: SIMD_WIDTH,
            got: limits.thread_execution_width,
        });
    }
    if limits.max_total_threads_per_threadgroup < threads {
        return Err(MetalError::InvalidRamRaState(
            "RAM RA claim-reduction pipeline admits too few threads",
        ));
    }
    Ok(())
}

fn encode_fields(values: &[AkitaField]) -> Vec<Fp128> {
    values.iter().map(Fp128::from_jolt_field).collect()
}

fn read_fields(
    context: &SolinasMetal,
    buffer: &Buffer,
    elements: usize,
    side: &'static str,
) -> Result<Vec<AkitaField>, MetalError> {
    // SAFETY: each output buffer owns exactly `elements` fields and its command
    // has completed before this immutable host view.
    let output = unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), elements) };
    context.validate_inputs(side, output)?;
    Ok(output
        .iter()
        .map(|&value| value.into_jolt_field())
        .collect())
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

fn abi_count(value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::InputTooLong(value))
}
