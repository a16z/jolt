use std::{mem::size_of, slice, time::Duration};

use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::{
    address_raf::{AddressRafScanRow, AddressRafSums},
    address_suffix_full::AddressSuffixFullSums,
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, SolinasMetal, ADDRESS_RAF_BINS,
    ADDRESS_RAF_LANES, ADDRESS_SUFFIX_BINS, ADDRESS_SUFFIX_TABLES,
};

const RAF_KEYS: usize = 2 * ADDRESS_RAF_BINS;
const RAF_PARTIAL_LANES: usize = 3;
const RAF_FIELDS: usize = RAF_KEYS * RAF_PARTIAL_LANES;
const SUFFIX_MAX_SUFFIXES: usize = 4;
const SUFFIX_FIELDS: usize = SUFFIX_MAX_SUFFIXES * ADDRESS_SUFFIX_BINS;
const ACCUMULATOR_WORDS: usize = 5;
const SIMD_WIDTH: usize = 32;
const RAF_TILE_PIPELINE: &str = "solinas_address_raf_direct_tile";
const RAF_FINALIZE_PIPELINE: &str = "solinas_address_raf_direct_finalize";
const SUFFIX_TILE_PIPELINE: &str = "solinas_address_suffix_full_tile";
const SUFFIX_FINALIZE_PIPELINE: &str = "solinas_address_suffix_full_finalize";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressPhaseSequenceConfig {
    pub rows_per_threadgroup: usize,
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for AddressPhaseSequenceConfig {
    fn default() -> Self {
        Self {
            rows_per_threadgroup: 1 << 16,
            threads_per_threadgroup: Some(1024),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RafParams {
    rows: u32,
    suffix_len: u32,
    rows_per_threadgroup: u32,
    threadgroup_count: u32,
    condense: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SuffixParams {
    suffix_len: u32,
    job_count: u32,
    output_elements: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SuffixJob {
    start: u32,
    end: u32,
    table: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SuffixTable {
    job_start: u32,
    job_end: u32,
    output_start: u32,
    suffix_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AddressLookup {
    limbs: [u64; 2],
}

struct AddressPhaseBuffers {
    raf_flags: Buffer,
    lookups: Buffer,
    weights: Buffer,
    previous_phase_table: Buffer,
    raf_partials: Buffer,
    raf_output: Buffer,
    raf_params: Buffer,
    suffix_jobs: Buffer,
    suffix_tables: Buffer,
    suffix_kinds: Buffer,
    suffix_counts: Buffer,
    suffix_partials: Buffer,
    suffix_output: Buffer,
    suffix_params: Buffer,
}

pub struct AddressPhaseSequence {
    context: SolinasMetal,
    raf_tile_pipeline: ComputePipelineState,
    raf_finalize_pipeline: ComputePipelineState,
    suffix_tile_pipeline: ComputePipelineState,
    suffix_finalize_pipeline: ComputePipelineState,
    buffers: AddressPhaseBuffers,
    rows: usize,
    raf_threadgroups: usize,
    suffix_jobs: usize,
    suffix_slots: usize,
    table_offsets: Vec<usize>,
    rows_per_threadgroup: usize,
    threads_per_threadgroup: usize,
    phases_executed: usize,
    gpu_active_time: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressPhaseSums {
    raf: AddressRafSums,
    suffix: AddressSuffixFullSums,
    gpu_active_time: Duration,
}

impl AddressPhaseSums {
    pub const fn raf(&self) -> &AddressRafSums {
        &self.raf
    }

    pub const fn suffix(&self) -> &AddressSuffixFullSums {
        &self.suffix
    }

    pub const fn gpu_active_time(&self) -> Duration {
        self.gpu_active_time
    }
}

impl SolinasMetal {
    pub fn prepare_address_phase_sequence(
        &self,
        rows: &[AddressRafScanRow],
        weights: &[Fp128],
        config: AddressPhaseSequenceConfig,
    ) -> Result<AddressPhaseSequence, MetalError> {
        if rows.len() != weights.len() {
            return Err(MetalError::AddressRafLengthMismatch {
                rows: rows.len(),
                weights: weights.len(),
            });
        }
        let mut buckets = vec![Vec::new(); ADDRESS_SUFFIX_TABLES];
        for (index, row) in rows.iter().enumerate() {
            if let Some(table) = row.table_index() {
                buckets
                    .get_mut(table)
                    .ok_or(MetalError::InvalidAddressSuffixTable(table))?
                    .push(u32::try_from(index).map_err(|_| MetalError::InputTooLong(index))?);
            }
        }
        self.prepare_address_phase_sequence_from_buckets(rows.len(), &buckets, config, |index| {
            (rows[index], weights[index])
        })
    }

    pub(crate) fn prepare_address_phase_sequence_from_buckets(
        &self,
        rows: usize,
        buckets: &[Vec<u32>],
        config: AddressPhaseSequenceConfig,
        source: impl Fn(usize) -> (AddressRafScanRow, Fp128),
    ) -> Result<AddressPhaseSequence, MetalError> {
        if rows == 0 {
            return Err(MetalError::EmptyInput);
        }
        if buckets.len() != ADDRESS_SUFFIX_TABLES {
            return Err(MetalError::AddressPhaseBucketCount {
                expected: ADDRESS_SUFFIX_TABLES,
                got: buckets.len(),
            });
        }
        if config.rows_per_threadgroup == 0 || config.rows_per_threadgroup > 1 << 16 {
            return Err(MetalError::InvalidAddressRafDirectRowsPerThreadgroup(
                config.rows_per_threadgroup,
            ));
        }

        let lookup_tables: Vec<_> = LookupTableKind::<RISCV_XLEN>::iter().collect();
        let mut suffix_kinds = vec![0u8; ADDRESS_SUFFIX_TABLES * SUFFIX_MAX_SUFFIXES];
        let mut suffix_counts = Vec::with_capacity(ADDRESS_SUFFIX_TABLES);
        let mut table_offsets = Vec::with_capacity(ADDRESS_SUFFIX_TABLES + 1);
        table_offsets.push(0usize);
        for table in &lookup_tables {
            let table_index = table.index();
            let suffixes = table.suffixes();
            if suffixes.len() > SUFFIX_MAX_SUFFIXES {
                return Err(MetalError::InvalidAddressSuffixCount {
                    table: table_index,
                    count: suffixes.len(),
                    maximum: SUFFIX_MAX_SUFFIXES,
                });
            }
            for (suffix, kind) in suffixes.iter().enumerate() {
                suffix_kinds[table_index * SUFFIX_MAX_SUFFIXES + suffix] = *kind as u8;
            }
            suffix_counts.push(suffixes.len() as u8);
            table_offsets.push(table_offsets.last().copied().unwrap_or(0) + suffixes.len());
        }

        let mut lookups = Vec::with_capacity(rows);
        let mut raf_flags = Vec::with_capacity(rows);
        let mut table_major_weights = Vec::with_capacity(rows);
        let mut suffix_jobs = Vec::new();
        let mut suffix_tables = Vec::with_capacity(ADDRESS_SUFFIX_TABLES);
        for (table, bucket) in buckets.iter().enumerate() {
            let job_start = u32::try_from(suffix_jobs.len())
                .map_err(|_| MetalError::InputTooLong(suffix_jobs.len()))?;
            let bucket_start = lookups.len();
            for &row in bucket {
                let row = row as usize;
                if row >= rows {
                    return Err(MetalError::InputTooLong(row));
                }
                let (source_row, weight) = source(row);
                if source_row.table_index() != Some(table) {
                    return Err(MetalError::InvalidAddressPhaseBucket {
                        bucket: table,
                        row,
                        actual: source_row.table_index(),
                    });
                }
                push_source(
                    source_row,
                    weight,
                    &mut lookups,
                    &mut raf_flags,
                    &mut table_major_weights,
                );
            }
            for local_start in (0..bucket.len()).step_by(config.rows_per_threadgroup) {
                let local_end = (local_start + config.rows_per_threadgroup).min(bucket.len());
                suffix_jobs.push(SuffixJob {
                    start: u32::try_from(bucket_start + local_start)
                        .map_err(|_| MetalError::InputTooLong(bucket_start + local_start))?,
                    end: u32::try_from(bucket_start + local_end)
                        .map_err(|_| MetalError::InputTooLong(bucket_start + local_end))?,
                    table: table as u32,
                    reserved: 0,
                });
            }
            suffix_tables.push(SuffixTable {
                job_start,
                job_end: u32::try_from(suffix_jobs.len())
                    .map_err(|_| MetalError::InputTooLong(suffix_jobs.len()))?,
                output_start: u32::try_from(table_offsets[table])
                    .map_err(|_| MetalError::InputTooLong(table_offsets[table]))?,
                suffix_count: u32::from(suffix_counts[table]),
            });
        }
        if suffix_jobs.is_empty() {
            return Err(MetalError::EmptyAddressSuffixBuckets);
        }
        for index in 0..rows {
            let (row, weight) = source(index);
            if row.table_index().is_none() {
                push_source(
                    row,
                    weight,
                    &mut lookups,
                    &mut raf_flags,
                    &mut table_major_weights,
                );
            }
        }
        if lookups.len() != rows {
            return Err(MetalError::AddressPhaseLayoutLength {
                expected: rows,
                got: lookups.len(),
            });
        }
        self.validate_inputs("resident address weights", &table_major_weights)?;

        let raf_tile_pipeline = self.compile_named_pipeline(RAF_TILE_PIPELINE)?;
        let raf_finalize_pipeline = self.compile_named_pipeline(RAF_FINALIZE_PIPELINE)?;
        let suffix_tile_pipeline = self.compile_named_pipeline(SUFFIX_TILE_PIPELINE)?;
        let suffix_finalize_pipeline = self.compile_named_pipeline(SUFFIX_FINALIZE_PIPELINE)?;
        let limits = [
            (RAF_TILE_PIPELINE, Self::limits(&raf_tile_pipeline)),
            (RAF_FINALIZE_PIPELINE, Self::limits(&raf_finalize_pipeline)),
            (SUFFIX_TILE_PIPELINE, Self::limits(&suffix_tile_pipeline)),
            (
                SUFFIX_FINALIZE_PIPELINE,
                Self::limits(&suffix_finalize_pipeline),
            ),
        ];
        for (pipeline, limits) in limits {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedAddressRafExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let tile_limits = Self::limits(&raf_tile_pipeline);
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, tile_limits)?;
        for limits in [
            Self::limits(&raf_finalize_pipeline),
            Self::limits(&suffix_tile_pipeline),
        ] {
            if threads_per_threadgroup > limits.max_total_threads_per_threadgroup {
                return Err(MetalError::InvalidThreadgroupWidth {
                    requested: threads_per_threadgroup,
                    execution_width: limits.thread_execution_width,
                    maximum: limits.max_total_threads_per_threadgroup,
                });
            }
        }
        let suffix_finalize_limits = Self::limits(&suffix_finalize_pipeline);
        if suffix_finalize_limits.max_total_threads_per_threadgroup < SUFFIX_FIELDS {
            return Err(MetalError::InvalidThreadgroupWidth {
                requested: SUFFIX_FIELDS,
                execution_width: suffix_finalize_limits.thread_execution_width,
                maximum: suffix_finalize_limits.max_total_threads_per_threadgroup,
            });
        }
        for (requested, suffix) in [
            (RAF_FIELDS * ACCUMULATOR_WORDS * size_of::<u32>(), false),
            (SUFFIX_FIELDS * ACCUMULATOR_WORDS * size_of::<u32>(), true),
        ] {
            let maximum = self.device.max_threadgroup_memory_length();
            if requested as u64 > maximum {
                return Err(if suffix {
                    MetalError::AddressSuffixThreadgroupMemory {
                        requested: requested as u64,
                        maximum,
                    }
                } else {
                    MetalError::AddressRafDirectThreadgroupMemory {
                        requested: requested as u64,
                        maximum,
                    }
                });
            }
        }

        let raf_threadgroups = rows.div_ceil(config.rows_per_threadgroup);
        let suffix_slots = *table_offsets.last().unwrap_or(&0);
        let suffix_output_elements = suffix_slots
            .checked_mul(ADDRESS_SUFFIX_BINS)
            .ok_or(MetalError::InputTooLong(suffix_slots))?;
        let raf_partial_elements = raf_threadgroups
            .checked_mul(RAF_FIELDS)
            .ok_or(MetalError::InputTooLong(raf_threadgroups))?;
        let suffix_partial_elements = suffix_jobs
            .len()
            .checked_mul(SUFFIX_FIELDS)
            .ok_or(MetalError::InputTooLong(suffix_jobs.len()))?;
        let buffer_lengths = [
            byte_length::<u8>(raf_flags.len())?,
            byte_length::<AddressLookup>(lookups.len())?,
            byte_length::<Fp128>(table_major_weights.len())?,
            byte_length::<Fp128>(ADDRESS_RAF_BINS)?,
            byte_length::<Fp128>(raf_partial_elements)?,
            byte_length::<Fp128>(ADDRESS_RAF_LANES * ADDRESS_RAF_BINS)?,
            byte_length::<SuffixJob>(suffix_jobs.len())?,
            byte_length::<SuffixTable>(suffix_tables.len())?,
            byte_length::<u8>(suffix_kinds.len())?,
            byte_length::<u8>(suffix_counts.len())?,
            byte_length::<Fp128>(suffix_partial_elements)?,
            byte_length::<Fp128>(suffix_output_elements)?,
        ];
        for requested in buffer_lengths {
            let maximum = self.device.max_buffer_length();
            if requested > maximum {
                return Err(MetalError::BufferTooLong { requested, maximum });
            }
        }

        let raf_params = RafParams {
            rows: u32::try_from(rows).map_err(|_| MetalError::InputTooLong(rows))?,
            suffix_len: 120,
            rows_per_threadgroup: u32::try_from(config.rows_per_threadgroup)
                .map_err(|_| MetalError::InputTooLong(config.rows_per_threadgroup))?,
            threadgroup_count: u32::try_from(raf_threadgroups)
                .map_err(|_| MetalError::InputTooLong(raf_threadgroups))?,
            condense: 0,
        };
        let suffix_params = SuffixParams {
            suffix_len: 120,
            job_count: u32::try_from(suffix_jobs.len())
                .map_err(|_| MetalError::InputTooLong(suffix_jobs.len()))?,
            output_elements: u32::try_from(suffix_output_elements)
                .map_err(|_| MetalError::InputTooLong(suffix_output_elements))?,
            reserved: 0,
        };
        let identity = [Fp128::ONE; ADDRESS_RAF_BINS];

        Ok(AddressPhaseSequence {
            context: self.clone(),
            raf_tile_pipeline,
            raf_finalize_pipeline,
            suffix_tile_pipeline,
            suffix_finalize_pipeline,
            buffers: AddressPhaseBuffers {
                raf_flags: buffer_from_slice(&self.device, &raf_flags),
                lookups: buffer_from_slice(&self.device, &lookups),
                weights: buffer_from_slice(&self.device, &table_major_weights),
                previous_phase_table: buffer_from_slice(&self.device, &identity),
                raf_partials: self.device.new_buffer(
                    byte_length::<Fp128>(raf_partial_elements)?,
                    MTLResourceOptions::StorageModeShared,
                ),
                raf_output: self.device.new_buffer(
                    byte_length::<Fp128>(ADDRESS_RAF_LANES * ADDRESS_RAF_BINS)?,
                    MTLResourceOptions::StorageModeShared,
                ),
                raf_params: buffer_from_slice(&self.device, slice::from_ref(&raf_params)),
                suffix_jobs: buffer_from_slice(&self.device, &suffix_jobs),
                suffix_tables: buffer_from_slice(&self.device, &suffix_tables),
                suffix_kinds: buffer_from_slice(&self.device, &suffix_kinds),
                suffix_counts: buffer_from_slice(&self.device, &suffix_counts),
                suffix_partials: self.device.new_buffer(
                    byte_length::<Fp128>(suffix_partial_elements)?,
                    MTLResourceOptions::StorageModeShared,
                ),
                suffix_output: self.device.new_buffer(
                    byte_length::<Fp128>(suffix_output_elements)?,
                    MTLResourceOptions::StorageModeShared,
                ),
                suffix_params: buffer_from_slice(&self.device, slice::from_ref(&suffix_params)),
            },
            rows,
            raf_threadgroups,
            suffix_jobs: suffix_jobs.len(),
            suffix_slots,
            table_offsets,
            rows_per_threadgroup: config.rows_per_threadgroup,
            threads_per_threadgroup,
            phases_executed: 0,
            gpu_active_time: Duration::ZERO,
        })
    }
}

impl AddressPhaseSequence {
    pub fn phase(
        &mut self,
        suffix_len: u32,
        previous_phase_table: Option<&[Fp128; ADDRESS_RAF_BINS]>,
    ) -> Result<AddressPhaseSums, MetalError> {
        if suffix_len > 120 || !suffix_len.is_multiple_of(8) {
            return Err(MetalError::InvalidAddressRafSuffixLength(suffix_len));
        }
        if previous_phase_table.is_some() && suffix_len > 112 {
            return Err(MetalError::InvalidAddressRafCondensationSuffixLength(
                suffix_len,
            ));
        }
        if let Some(table) = previous_phase_table {
            self.context
                .validate_inputs("resident address condensation table", table)?;
            write_buffer(&self.buffers.previous_phase_table, table);
        }
        write_value(
            &self.buffers.raf_params,
            RafParams {
                rows: self.rows as u32,
                suffix_len,
                rows_per_threadgroup: self.rows_per_threadgroup as u32,
                threadgroup_count: self.raf_threadgroups as u32,
                condense: u32::from(previous_phase_table.is_some()),
            },
        );
        write_value(
            &self.buffers.suffix_params,
            SuffixParams {
                suffix_len,
                job_count: self.suffix_jobs as u32,
                output_elements: (self.suffix_slots * ADDRESS_SUFFIX_BINS) as u32,
                reserved: 0,
            },
        );

        let command_buffer = self.context.queue.new_command_buffer();
        autoreleasepool(|| {
            let raf_tile = command_buffer.new_compute_command_encoder();
            raf_tile.set_compute_pipeline_state(&self.raf_tile_pipeline);
            raf_tile.set_buffer(0, Some(&self.buffers.raf_flags), 0);
            raf_tile.set_buffer(1, Some(&self.buffers.lookups), 0);
            raf_tile.set_buffer(2, Some(&self.buffers.weights), 0);
            raf_tile.set_buffer(3, Some(&self.buffers.previous_phase_table), 0);
            raf_tile.set_buffer(4, Some(&self.buffers.raf_partials), 0);
            raf_tile.set_buffer(5, Some(&self.buffers.raf_params), 0);
            raf_tile.set_threadgroup_memory_length(
                0,
                (RAF_FIELDS * ACCUMULATOR_WORDS * size_of::<u32>()) as u64,
            );
            raf_tile.dispatch_thread_groups(
                MTLSize {
                    width: self.raf_threadgroups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            raf_tile.end_encoding();

            let raf_finalize = command_buffer.new_compute_command_encoder();
            raf_finalize.set_compute_pipeline_state(&self.raf_finalize_pipeline);
            raf_finalize.set_buffer(0, Some(&self.buffers.raf_partials), 0);
            raf_finalize.set_buffer(1, Some(&self.buffers.raf_output), 0);
            raf_finalize.set_buffer(2, Some(&self.buffers.raf_params), 0);
            let simdgroups = self.threads_per_threadgroup / SIMD_WIDTH;
            raf_finalize.set_threadgroup_memory_length(
                0,
                (RAF_PARTIAL_LANES * simdgroups * size_of::<Fp128>()) as u64,
            );
            raf_finalize.dispatch_thread_groups(
                MTLSize {
                    width: RAF_KEYS as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            raf_finalize.end_encoding();

            let suffix_tile = command_buffer.new_compute_command_encoder();
            suffix_tile.set_compute_pipeline_state(&self.suffix_tile_pipeline);
            suffix_tile.set_buffer(0, Some(&self.buffers.lookups), 0);
            suffix_tile.set_buffer(1, Some(&self.buffers.weights), 0);
            suffix_tile.set_buffer(2, Some(&self.buffers.suffix_jobs), 0);
            suffix_tile.set_buffer(3, Some(&self.buffers.suffix_kinds), 0);
            suffix_tile.set_buffer(4, Some(&self.buffers.suffix_counts), 0);
            suffix_tile.set_buffer(5, Some(&self.buffers.suffix_partials), 0);
            suffix_tile.set_buffer(6, Some(&self.buffers.suffix_params), 0);
            suffix_tile.set_threadgroup_memory_length(
                0,
                (SUFFIX_FIELDS * ACCUMULATOR_WORDS * size_of::<u32>()) as u64,
            );
            suffix_tile.dispatch_thread_groups(
                MTLSize {
                    width: self.suffix_jobs as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            suffix_tile.end_encoding();

            let suffix_finalize = command_buffer.new_compute_command_encoder();
            suffix_finalize.set_compute_pipeline_state(&self.suffix_finalize_pipeline);
            suffix_finalize.set_buffer(0, Some(&self.buffers.suffix_partials), 0);
            suffix_finalize.set_buffer(1, Some(&self.buffers.suffix_tables), 0);
            suffix_finalize.set_buffer(2, Some(&self.buffers.suffix_output), 0);
            suffix_finalize.dispatch_thread_groups(
                MTLSize {
                    width: ADDRESS_SUFFIX_TABLES as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: SUFFIX_FIELDS as u64,
                    height: 1,
                    depth: 1,
                },
            );
            suffix_finalize.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        let status = command_buffer.status();
        if status != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(status));
        }
        let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        let gpu_active_time = Duration::from_secs_f64(end - start);
        self.gpu_active_time += gpu_active_time;
        self.phases_executed += 1;

        let raf_elements = ADDRESS_RAF_LANES * ADDRESS_RAF_BINS;
        let suffix_elements = self.suffix_slots * ADDRESS_SUFFIX_BINS;
        // SAFETY: both shared output buffers have the stated capacities and
        // the command buffer completed before these reads.
        let raf_values = unsafe {
            slice::from_raw_parts(
                self.buffers.raf_output.contents().cast::<Fp128>(),
                raf_elements,
            )
        };
        // SAFETY: see the preceding output-buffer synchronization argument.
        let suffix_values = unsafe {
            slice::from_raw_parts(
                self.buffers.suffix_output.contents().cast::<Fp128>(),
                suffix_elements,
            )
        };
        self.context
            .validate_inputs("resident address RAF output", raf_values)?;
        self.context
            .validate_inputs("resident address suffix output", suffix_values)?;
        Ok(AddressPhaseSums {
            raf: AddressRafSums::from_values(raf_values.to_vec()),
            suffix: AddressSuffixFullSums::from_values(
                suffix_values.to_vec(),
                self.table_offsets.clone(),
            ),
            gpu_active_time,
        })
    }

    pub const fn phases_executed(&self) -> usize {
        self.phases_executed
    }

    pub const fn gpu_active_time(&self) -> Duration {
        self.gpu_active_time
    }

    pub const fn resident_buffer_count(&self) -> usize {
        14
    }

    pub const fn phase_device_buffer_allocations(&self) -> usize {
        0
    }
}

fn push_source(
    row: AddressRafScanRow,
    weight: Fp128,
    lookups: &mut Vec<AddressLookup>,
    raf_flags: &mut Vec<u8>,
    weights: &mut Vec<Fp128>,
) {
    let lookup = row.lookup_index();
    lookups.push(AddressLookup {
        limbs: [lookup as u64, (lookup >> 64) as u64],
    });
    raf_flags.push(u8::from(row.raf_flag()));
    weights.push(weight);
}

fn write_buffer<T: Copy>(buffer: &Buffer, values: &[T]) {
    // SAFETY: callers allocate the shared buffer for this exact fixed-size
    // value array and wait for the previous command before overwriting it.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<T>(), values.len()) };
    output.copy_from_slice(values);
}

fn write_value<T: Copy>(buffer: &Buffer, value: T) {
    write_buffer(buffer, slice::from_ref(&value));
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

const _: () = assert!(size_of::<RafParams>() == 20);
const _: () = assert!(size_of::<SuffixParams>() == 16);
const _: () = assert!(size_of::<SuffixJob>() == 16);
const _: () = assert!(size_of::<SuffixTable>() == 16);
const _: () = assert!(size_of::<AddressLookup>() == 16);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::AkitaField;

    use super::{
        AddressPhaseSequenceConfig, AddressRafScanRow, Fp128, SolinasMetal, ADDRESS_RAF_BINS,
        ADDRESS_SUFFIX_TABLES,
    };
    use crate::metal::solinas::AddressRafScanConfig;

    #[test]
    fn resident_phases_match_independent_invocations() {
        let mut state = 0xa11c_e5e0_1234_5678;
        let rows: Vec<_> = (0..4099)
            .map(|index| {
                let lookup =
                    (u128::from(splitmix(&mut state)) << 64) | u128::from(splitmix(&mut state));
                let table = (index % 17 != 0).then_some(index % ADDRESS_SUFFIX_TABLES);
                AddressRafScanRow::new_with_table(lookup, table, index % 3 == 0)
            })
            .collect();
        let weights: Vec<_> = (0..rows.len())
            .map(|_| {
                Fp128::from_u128(
                    u128::from(splitmix(&mut state))
                        | (u128::from(splitmix(&mut state) & 0x7fff_ffff_ffff_ffff) << 64),
                )
            })
            .collect();
        let previous: [Fp128; ADDRESS_RAF_BINS] = std::array::from_fn(|_| {
            Fp128::from_u128(
                u128::from(splitmix(&mut state))
                    | (u128::from(splitmix(&mut state) & 0x7fff_ffff_ffff_ffff) << 64),
            )
        });
        let context = SolinasMetal::for_akita().unwrap();
        let sequence_config = AddressPhaseSequenceConfig {
            rows_per_threadgroup: 64,
            threads_per_threadgroup: Some(128),
        };
        let scan_config = |suffix_len| AddressRafScanConfig {
            suffix_len,
            rows_per_threadgroup: 64,
            threads_per_threadgroup: Some(128),
        };
        let mut sequence = context
            .prepare_address_phase_sequence(&rows, &weights, sequence_config)
            .unwrap();

        let phase_0 = sequence.phase(120, None).unwrap();
        let raf_0 = context
            .prepare_direct_address_raf_scan(&rows, &weights, scan_config(120))
            .unwrap();
        raf_0.execute().unwrap();
        let suffix_0 = context
            .prepare_address_suffix_full(&rows, &weights, scan_config(120))
            .unwrap();
        suffix_0.execute().unwrap();
        assert_eq!(phase_0.raf(), &raf_0.read_output().unwrap());
        assert_eq!(phase_0.suffix(), &suffix_0.read_output().unwrap());

        let phase_1 = sequence.phase(112, Some(&previous)).unwrap();
        let raf_1 = context
            .prepare_direct_condensed_address_raf_scan(&rows, &weights, &previous, scan_config(112))
            .unwrap();
        raf_1.execute().unwrap();
        let condensed: Vec<_> = rows
            .iter()
            .zip(&weights)
            .map(|(row, weight)| {
                let chunk = ((row.lookup_index() >> 120) as usize) & (ADDRESS_RAF_BINS - 1);
                Fp128::from_jolt_field(
                    &(weight.into_jolt_field::<AkitaField>()
                        * previous[chunk].into_jolt_field::<AkitaField>()),
                )
            })
            .collect();
        let suffix_1 = context
            .prepare_address_suffix_full(&rows, &condensed, scan_config(112))
            .unwrap();
        suffix_1.execute().unwrap();
        assert_eq!(phase_1.raf(), &raf_1.read_output().unwrap());
        assert_eq!(phase_1.suffix(), &suffix_1.read_output().unwrap());
        assert_eq!(sequence.phases_executed(), 2);
        assert_eq!(sequence.phase_device_buffer_allocations(), 0);
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}
