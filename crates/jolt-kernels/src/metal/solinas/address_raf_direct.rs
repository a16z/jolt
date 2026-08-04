use std::{cell::Cell, mem::size_of, slice, time::Duration};

use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::{
    address_raf::{AddressRafScanConfig, AddressRafScanRow, AddressRafSums},
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

const ADDRESS_RAF_BINS: usize = 256;
const ADDRESS_RAF_KEYS: usize = 2 * ADDRESS_RAF_BINS;
const ADDRESS_RAF_LANES: usize = 6;
const ADDRESS_RAF_PARTIAL_LANES: usize = 3;
const ADDRESS_RAF_FIELDS: usize = ADDRESS_RAF_KEYS * ADDRESS_RAF_PARTIAL_LANES;
const ADDRESS_RAF_ACCUMULATOR_WORDS: usize = 5;
const ADDRESS_RAF_SIMD_WIDTH: usize = 32;
const TILE_PIPELINE: &str = "solinas_address_raf_direct_tile";
const FINALIZE_PIPELINE: &str = "solinas_address_raf_direct_finalize";

#[repr(C)]
#[derive(Clone, Copy)]
struct AddressRafDirectParams {
    rows: u32,
    suffix_len: u32,
    rows_per_threadgroup: u32,
    threadgroup_count: u32,
    condense: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AddressRafDirectLookup {
    limbs: [u64; 2],
}

struct AddressRafDirectBuffers {
    raf_flags: Buffer,
    lookups: Buffer,
    weights: Buffer,
    previous_phase_table: Buffer,
    partials: Buffer,
    output: Buffer,
    params: Buffer,
}

pub struct AddressRafDirectInvocation<'a> {
    context: &'a SolinasMetal,
    tile_pipeline: ComputePipelineState,
    finalize_pipeline: ComputePipelineState,
    tile_limits: PipelineLimits,
    buffers: AddressRafDirectBuffers,
    threadgroup_count: usize,
    threads_per_threadgroup: usize,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub fn prepare_direct_address_raf_scan(
        &self,
        rows: &[AddressRafScanRow],
        weights: &[Fp128],
        config: AddressRafScanConfig,
    ) -> Result<AddressRafDirectInvocation<'_>, MetalError> {
        self.prepare_direct_address_raf_scan_inner(rows, weights, None, config)
    }

    pub fn prepare_direct_condensed_address_raf_scan(
        &self,
        rows: &[AddressRafScanRow],
        weights: &[Fp128],
        previous_phase_table: &[Fp128; ADDRESS_RAF_BINS],
        config: AddressRafScanConfig,
    ) -> Result<AddressRafDirectInvocation<'_>, MetalError> {
        self.prepare_direct_address_raf_scan_inner(
            rows,
            weights,
            Some(previous_phase_table),
            config,
        )
    }

    fn prepare_direct_address_raf_scan_inner(
        &self,
        rows: &[AddressRafScanRow],
        weights: &[Fp128],
        previous_phase_table: Option<&[Fp128; ADDRESS_RAF_BINS]>,
        config: AddressRafScanConfig,
    ) -> Result<AddressRafDirectInvocation<'_>, MetalError> {
        if rows.is_empty() {
            return Err(MetalError::EmptyInput);
        }
        if rows.len() != weights.len() {
            return Err(MetalError::AddressRafLengthMismatch {
                rows: rows.len(),
                weights: weights.len(),
            });
        }
        if config.suffix_len > 120 || !config.suffix_len.is_multiple_of(8) {
            return Err(MetalError::InvalidAddressRafSuffixLength(config.suffix_len));
        }
        if previous_phase_table.is_some() && config.suffix_len > 112 {
            return Err(MetalError::InvalidAddressRafCondensationSuffixLength(
                config.suffix_len,
            ));
        }
        // The fifth accumulator word counts 2^128 wraps. This bound keeps
        // `wraps * SOLINAS_OFFSET` within the shader's u64 correction.
        if config.rows_per_threadgroup == 0 || config.rows_per_threadgroup > 1 << 16 {
            return Err(MetalError::InvalidAddressRafDirectRowsPerThreadgroup(
                config.rows_per_threadgroup,
            ));
        }
        self.validate_inputs("direct address RAF weights", weights)?;
        if let Some(table) = previous_phase_table {
            self.validate_inputs("direct address RAF condensation table", table)?;
        }

        let threadgroup_count = rows.len().div_ceil(config.rows_per_threadgroup);
        let params = AddressRafDirectParams {
            rows: u32::try_from(rows.len()).map_err(|_| MetalError::InputTooLong(rows.len()))?,
            suffix_len: config.suffix_len,
            rows_per_threadgroup: u32::try_from(config.rows_per_threadgroup)
                .map_err(|_| MetalError::InputTooLong(config.rows_per_threadgroup))?,
            threadgroup_count: u32::try_from(threadgroup_count)
                .map_err(|_| MetalError::InputTooLong(threadgroup_count))?,
            condense: u32::from(previous_phase_table.is_some()),
        };

        let tile_pipeline = self.compile_named_pipeline(TILE_PIPELINE)?;
        let finalize_pipeline = self.compile_named_pipeline(FINALIZE_PIPELINE)?;
        let tile_limits = Self::limits(&tile_pipeline);
        let finalize_limits = Self::limits(&finalize_pipeline);
        for (pipeline, limits) in [
            (TILE_PIPELINE, tile_limits),
            (FINALIZE_PIPELINE, finalize_limits),
        ] {
            if limits.thread_execution_width != ADDRESS_RAF_SIMD_WIDTH {
                return Err(MetalError::UnsupportedAddressRafExecutionWidth {
                    pipeline,
                    expected: ADDRESS_RAF_SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, tile_limits)?;
        if threads_per_threadgroup > finalize_limits.max_total_threads_per_threadgroup {
            return Err(MetalError::InvalidThreadgroupWidth {
                requested: threads_per_threadgroup,
                execution_width: finalize_limits.thread_execution_width,
                maximum: finalize_limits.max_total_threads_per_threadgroup,
            });
        }
        let accumulator_bytes =
            ADDRESS_RAF_FIELDS * ADDRESS_RAF_ACCUMULATOR_WORDS * size_of::<u32>();
        if accumulator_bytes as u64 > self.device.max_threadgroup_memory_length() {
            return Err(MetalError::AddressRafDirectThreadgroupMemory {
                requested: accumulator_bytes as u64,
                maximum: self.device.max_threadgroup_memory_length(),
            });
        }

        let partial_elements = threadgroup_count
            .checked_mul(ADDRESS_RAF_FIELDS)
            .ok_or(MetalError::InputTooLong(threadgroup_count))?;
        let partial_bytes = byte_length::<Fp128>(partial_elements)?;
        let flag_bytes = byte_length::<u8>(rows.len())?;
        let lookup_bytes = byte_length::<AddressRafDirectLookup>(rows.len())?;
        let output_bytes = byte_length::<Fp128>(ADDRESS_RAF_LANES * ADDRESS_RAF_BINS)?;
        for requested in [
            byte_length::<Fp128>(weights.len())?,
            flag_bytes,
            lookup_bytes,
            partial_bytes,
            output_bytes,
        ] {
            let maximum = self.device.max_buffer_length();
            if requested > maximum {
                return Err(MetalError::BufferTooLong { requested, maximum });
            }
        }

        let raf_flags: Vec<u8> = rows.iter().map(|row| u8::from(row.raf_flag())).collect();
        let lookups: Vec<AddressRafDirectLookup> = rows
            .iter()
            .map(|row| AddressRafDirectLookup {
                limbs: [row.lookup_index() as u64, (row.lookup_index() >> 64) as u64],
            })
            .collect();
        let identity_table = [Fp128::ONE; ADDRESS_RAF_BINS];
        let previous_phase_table = previous_phase_table.unwrap_or(&identity_table);

        Ok(AddressRafDirectInvocation {
            context: self,
            tile_pipeline,
            finalize_pipeline,
            tile_limits,
            buffers: AddressRafDirectBuffers {
                raf_flags: buffer_from_slice(&self.device, &raf_flags),
                lookups: buffer_from_slice(&self.device, &lookups),
                weights: buffer_from_slice(&self.device, weights),
                previous_phase_table: buffer_from_slice(&self.device, previous_phase_table),
                partials: self
                    .device
                    .new_buffer(partial_bytes, MTLResourceOptions::StorageModeShared),
                output: self
                    .device
                    .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared),
                params: buffer_from_slice(&self.device, slice::from_ref(&params)),
            },
            threadgroup_count,
            threads_per_threadgroup,
            completed: Cell::new(false),
        })
    }
}

impl AddressRafDirectInvocation<'_> {
    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.tile_limits
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn threadgroup_count(&self) -> usize {
        self.threadgroup_count
    }

    pub const fn intermediate_partial_bytes(&self) -> u64 {
        self.threadgroup_count as u64 * ADDRESS_RAF_FIELDS as u64 * size_of::<Fp128>() as u64
    }

    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let tile = command_buffer.new_compute_command_encoder();
            tile.set_compute_pipeline_state(&self.tile_pipeline);
            tile.set_buffer(0, Some(&self.buffers.raf_flags), 0);
            tile.set_buffer(1, Some(&self.buffers.lookups), 0);
            tile.set_buffer(2, Some(&self.buffers.weights), 0);
            tile.set_buffer(3, Some(&self.buffers.previous_phase_table), 0);
            tile.set_buffer(4, Some(&self.buffers.partials), 0);
            tile.set_buffer(5, Some(&self.buffers.params), 0);
            tile.set_threadgroup_memory_length(
                0,
                (ADDRESS_RAF_FIELDS * ADDRESS_RAF_ACCUMULATOR_WORDS * size_of::<u32>()) as u64,
            );
            tile.dispatch_thread_groups(
                MTLSize {
                    width: self.threadgroup_count as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            tile.end_encoding();

            let finalize = command_buffer.new_compute_command_encoder();
            finalize.set_compute_pipeline_state(&self.finalize_pipeline);
            finalize.set_buffer(0, Some(&self.buffers.partials), 0);
            finalize.set_buffer(1, Some(&self.buffers.output), 0);
            finalize.set_buffer(2, Some(&self.buffers.params), 0);
            let simdgroups = self.threads_per_threadgroup / ADDRESS_RAF_SIMD_WIDTH;
            finalize.set_threadgroup_memory_length(
                0,
                (ADDRESS_RAF_PARTIAL_LANES * simdgroups * size_of::<Fp128>()) as u64,
            );
            finalize.dispatch_thread_groups(
                MTLSize {
                    width: ADDRESS_RAF_KEYS as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            finalize.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(status));
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end });
            }
            self.completed.set(true);
            Ok(Duration::from_secs_f64(end - start))
        })
    }

    pub fn read_output(&self) -> Result<AddressRafSums, MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        let mut values = vec![Fp128::ZERO; ADDRESS_RAF_LANES * ADDRESS_RAF_BINS];
        self.read_output_into(&mut values)?;
        Ok(AddressRafSums::from_values(values))
    }

    pub fn read_output_into(&self, output: &mut [Fp128]) -> Result<(), MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        let elements = ADDRESS_RAF_LANES * ADDRESS_RAF_BINS;
        if output.len() != elements {
            return Err(MetalError::LengthMismatch {
                lhs: output.len(),
                rhs: elements,
            });
        }
        // SAFETY: the shared output buffer contains `elements` field values and
        // the command buffer has completed.
        let values = unsafe {
            slice::from_raw_parts(self.buffers.output.contents().cast::<Fp128>(), elements)
        };
        self.context
            .validate_inputs("direct address RAF output", values)?;
        output.copy_from_slice(values);
        Ok(())
    }
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

const _: () = assert!(size_of::<AddressRafDirectParams>() == 20);
const _: () = assert!(size_of::<AddressRafDirectLookup>() == 16);
