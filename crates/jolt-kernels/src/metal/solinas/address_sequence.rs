use std::{mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    address_raf::{AddressRafScanRow, AddressRafSums},
    address_suffix_full::AddressSuffixFullSums,
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits,
    Product5Sequence, Product5SequenceConfig, SolinasMetal, ADDRESS_RAF_BINS, ADDRESS_RAF_LANES,
    ADDRESS_SUFFIX_BINS, ADDRESS_SUFFIX_TABLES, PRODUCT5_FACTORS,
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
const CYCLE_MESSAGE_PIPELINE: &str = "solinas_address_cycle_message";
const CYCLE_BIND_PIPELINE: &str = "solinas_address_cycle_bind";
const CYCLE_TRANSITION_PIPELINE: &str = "solinas_address_cycle_fused_transition";
const PRODUCT_REDUCE_PIPELINE: &str = "solinas_product5_reduce";
const CYCLE_PHASES: usize = 16;
const CYCLE_PHASE_ELEMENTS: usize = CYCLE_PHASES * ADDRESS_RAF_BINS;
const CYCLE_THREADS_PER_THREADGROUP: usize = 128;
const CYCLE_BIND_THREADS_PER_THREADGROUP: usize = 256;

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
    packed_rows: u32,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct CycleParams {
    rows: u32,
    e_in_length: u32,
    e_out_length: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CycleReductionParams {
    input_count: u32,
    output_count: u32,
    reserved: [u32; 2],
}

struct AddressPhaseBuffers {
    packed_rows: Buffer,
    lookups: Buffer,
    cycle_to_table_major: Buffer,
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
    cycle_phase_tables: Buffer,
    cycle_table_values: Buffer,
    cycle_e_in: Buffer,
    cycle_e_out: Buffer,
    cycle_partial_a: Buffer,
    cycle_partial_b: Buffer,
}

pub struct AddressPhaseSequence {
    context: SolinasMetal,
    raf_tile_pipeline: ComputePipelineState,
    raf_finalize_pipeline: ComputePipelineState,
    suffix_tile_pipeline: ComputePipelineState,
    suffix_finalize_pipeline: ComputePipelineState,
    cycle_message_pipeline: ComputePipelineState,
    cycle_bind_pipeline: ComputePipelineState,
    cycle_transition_pipeline: ComputePipelineState,
    cycle_reduce_pipeline: ComputePipelineState,
    cycle_reduce_limits: PipelineLimits,
    buffers: AddressPhaseBuffers,
    rows: usize,
    raf_threadgroups: usize,
    suffix_jobs: usize,
    suffix_slots: usize,
    table_offsets: Vec<usize>,
    rows_per_threadgroup: usize,
    threads_per_threadgroup: usize,
    cycle_threads_per_threadgroup: usize,
    cycle_bind_threads_per_threadgroup: usize,
    cycle_e_in_capacity: usize,
    cycle_e_out_capacity: usize,
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
        source: impl Fn(usize) -> (AddressRafScanRow, Fp128) + Sync,
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

        let inverse_bytes = byte_length::<u32>(rows)?;
        let maximum = self.device.max_buffer_length();
        if inverse_bytes > maximum {
            return Err(MetalError::BufferTooLong {
                requested: inverse_bytes,
                maximum,
            });
        }
        let cycle_to_table_major_buffer = self
            .device
            .new_buffer(inverse_bytes, MTLResourceOptions::StorageModeShared);
        // SAFETY: this fresh shared buffer has exactly `rows` u32 slots and
        // remains CPU-exclusive until preparation returns.
        let cycle_to_table_major = unsafe {
            slice::from_raw_parts_mut(cycle_to_table_major_buffer.contents().cast::<u32>(), rows)
        };
        let mut table_selected = vec![false; rows];
        let mut cycle_order = Vec::with_capacity(rows);
        let mut suffix_jobs = Vec::new();
        let mut suffix_tables = Vec::with_capacity(ADDRESS_SUFFIX_TABLES);
        let mut table_row_ranges = Vec::with_capacity(ADDRESS_SUFFIX_TABLES);
        for (table, bucket) in buckets.iter().enumerate() {
            let job_start = u32::try_from(suffix_jobs.len())
                .map_err(|_| MetalError::InputTooLong(suffix_jobs.len()))?;
            let bucket_start = cycle_order.len();
            for &row in bucket {
                let row = row as usize;
                if row >= rows {
                    return Err(MetalError::InputTooLong(row));
                }
                if table_selected[row] {
                    return Err(MetalError::AddressPhaseLayoutLength {
                        expected: rows,
                        got: cycle_order.len(),
                    });
                }
                table_selected[row] = true;
                cycle_to_table_major[row] = u32::try_from(cycle_order.len())
                    .map_err(|_| MetalError::InputTooLong(cycle_order.len()))?;
                cycle_order.push(row as u32);
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
            table_row_ranges.push(bucket_start..cycle_order.len());
        }
        if suffix_jobs.is_empty() {
            return Err(MetalError::EmptyAddressSuffixBuckets);
        }
        let no_table_start = cycle_order.len();
        for (cycle, &selected) in table_selected.iter().enumerate() {
            if !selected {
                cycle_to_table_major[cycle] = u32::try_from(cycle_order.len())
                    .map_err(|_| MetalError::InputTooLong(cycle_order.len()))?;
                cycle_order.push(cycle as u32);
            }
        }
        drop(table_selected);
        if cycle_order.len() != rows {
            return Err(MetalError::AddressPhaseLayoutLength {
                expected: rows,
                got: cycle_order.len(),
            });
        }

        let row_buffer_lengths = [
            byte_length::<u8>(rows)?,
            byte_length::<AddressLookup>(rows)?,
            byte_length::<Fp128>(rows)?,
        ];
        for requested in row_buffer_lengths {
            let maximum = self.device.max_buffer_length();
            if requested > maximum {
                return Err(MetalError::BufferTooLong { requested, maximum });
            }
        }
        let packed_rows_buffer = self.device.new_buffer(
            byte_length::<u8>(rows)?,
            MTLResourceOptions::StorageModeShared,
        );
        let lookups_buffer = self.device.new_buffer(
            byte_length::<AddressLookup>(rows)?,
            MTLResourceOptions::StorageModeShared,
        );
        let weights_buffer = self.device.new_buffer(
            byte_length::<Fp128>(rows)?,
            MTLResourceOptions::StorageModeShared,
        );
        // SAFETY: the three shared buffers have exactly `rows` elements, are
        // distinct allocations, and are not visible to a command buffer yet.
        let lookups = unsafe {
            slice::from_raw_parts_mut(lookups_buffer.contents().cast::<AddressLookup>(), rows)
        };
        // SAFETY: see the allocation and exclusivity argument above.
        let packed_rows =
            unsafe { slice::from_raw_parts_mut(packed_rows_buffer.contents().cast::<u8>(), rows) };
        // SAFETY: see the allocation and exclusivity argument above.
        let table_major_weights =
            unsafe { slice::from_raw_parts_mut(weights_buffer.contents().cast::<Fp128>(), rows) };
        #[cfg(feature = "parallel")]
        lookups
            .par_iter_mut()
            .zip(packed_rows.par_iter_mut())
            .zip(table_major_weights.par_iter_mut())
            .zip(cycle_order.par_iter())
            .for_each(|(((lookup, packed), weight), &cycle)| {
                (*lookup, *packed, *weight) = packed_source(source(cycle as usize));
            });
        #[cfg(not(feature = "parallel"))]
        for (((lookup, packed), weight), &cycle) in lookups
            .iter_mut()
            .zip(&mut packed_rows)
            .zip(&mut table_major_weights)
            .zip(&cycle_order)
        {
            (*lookup, *packed, *weight) = packed_source(source(cycle as usize));
        }
        for (table, range) in table_row_ranges.iter().enumerate() {
            if let Some(position) = packed_rows[range.clone()]
                .iter()
                .position(|packed| usize::from(*packed & 0x7f) != table + 1)
            {
                let row = cycle_order[range.start + position] as usize;
                let packed = packed_rows[range.start + position];
                return Err(MetalError::InvalidAddressPhaseBucket {
                    bucket: table,
                    row,
                    actual: packed_table(packed),
                });
            }
        }
        if let Some(position) = packed_rows[no_table_start..]
            .iter()
            .position(|packed| packed & 0x7f != 0)
        {
            let row = cycle_order[no_table_start + position] as usize;
            let packed = packed_rows[no_table_start + position];
            return Err(MetalError::InvalidAddressPhaseBucket {
                bucket: ADDRESS_SUFFIX_TABLES,
                row,
                actual: packed_table(packed),
            });
        }
        self.validate_inputs("resident address weights", table_major_weights)?;
        drop(cycle_order);

        let raf_tile_pipeline = self.compile_named_pipeline(RAF_TILE_PIPELINE)?;
        let raf_finalize_pipeline = self.compile_named_pipeline(RAF_FINALIZE_PIPELINE)?;
        let suffix_tile_pipeline = self.compile_named_pipeline(SUFFIX_TILE_PIPELINE)?;
        let suffix_finalize_pipeline = self.compile_named_pipeline(SUFFIX_FINALIZE_PIPELINE)?;
        let cycle_message_pipeline = self.compile_named_pipeline(CYCLE_MESSAGE_PIPELINE)?;
        let cycle_bind_pipeline = self.compile_named_pipeline(CYCLE_BIND_PIPELINE)?;
        let cycle_transition_pipeline = self.compile_named_pipeline(CYCLE_TRANSITION_PIPELINE)?;
        let cycle_reduce_pipeline = self.compile_named_pipeline(PRODUCT_REDUCE_PIPELINE)?;
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
        let cycle_reduce_limits = Self::limits(&cycle_reduce_pipeline);
        for (pipeline, limits) in [
            (
                CYCLE_MESSAGE_PIPELINE,
                Self::limits(&cycle_message_pipeline),
            ),
            (CYCLE_BIND_PIPELINE, Self::limits(&cycle_bind_pipeline)),
            (
                CYCLE_TRANSITION_PIPELINE,
                Self::limits(&cycle_transition_pipeline),
            ),
            (PRODUCT_REDUCE_PIPELINE, cycle_reduce_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedAddressCycleExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let tile_limits = Self::limits(&raf_tile_pipeline);
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, tile_limits)?;
        let cycle_threads_per_threadgroup = Self::resolve_threadgroup_width(
            Some(CYCLE_THREADS_PER_THREADGROUP),
            Self::limits(&cycle_message_pipeline),
        )?;
        let cycle_bind_threads_per_threadgroup = Self::resolve_threadgroup_width(
            Some(CYCLE_BIND_THREADS_PER_THREADGROUP),
            Self::limits(&cycle_bind_pipeline),
        )?;
        if cycle_threads_per_threadgroup
            > Self::limits(&cycle_transition_pipeline).max_total_threads_per_threadgroup
        {
            return Err(MetalError::InvalidThreadgroupWidth {
                requested: cycle_threads_per_threadgroup,
                execution_width: SIMD_WIDTH,
                maximum: Self::limits(&cycle_transition_pipeline).max_total_threads_per_threadgroup,
            });
        }
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
        let cycle_pairs = (rows / 2).max(1);
        let cycle_e_out_capacity = 1usize << ((rows.ilog2() as usize) / 2);
        let cycle_e_in_capacity = cycle_pairs.div_ceil(cycle_e_out_capacity).max(1);
        let cycle_partial_elements = PRODUCT5_FACTORS
            .checked_mul(cycle_e_out_capacity)
            .ok_or(MetalError::InputTooLong(cycle_e_out_capacity))?;
        let buffer_lengths = [
            byte_length::<u8>(rows)?,
            byte_length::<AddressLookup>(rows)?,
            byte_length::<u32>(rows)?,
            byte_length::<Fp128>(rows)?,
            byte_length::<Fp128>(ADDRESS_RAF_BINS)?,
            byte_length::<Fp128>(raf_partial_elements)?,
            byte_length::<Fp128>(ADDRESS_RAF_LANES * ADDRESS_RAF_BINS)?,
            byte_length::<SuffixJob>(suffix_jobs.len())?,
            byte_length::<SuffixTable>(suffix_tables.len())?,
            byte_length::<u8>(suffix_kinds.len())?,
            byte_length::<u8>(suffix_counts.len())?,
            byte_length::<Fp128>(suffix_partial_elements)?,
            byte_length::<Fp128>(suffix_output_elements)?,
            byte_length::<Fp128>(CYCLE_PHASE_ELEMENTS)?,
            byte_length::<Fp128>(ADDRESS_SUFFIX_TABLES)?,
            byte_length::<Fp128>(cycle_e_in_capacity)?,
            byte_length::<Fp128>(cycle_e_out_capacity)?,
            byte_length::<Fp128>(cycle_partial_elements)?,
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
            packed_rows: 1,
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
            cycle_message_pipeline,
            cycle_bind_pipeline,
            cycle_transition_pipeline,
            cycle_reduce_pipeline,
            cycle_reduce_limits,
            buffers: AddressPhaseBuffers {
                packed_rows: packed_rows_buffer,
                lookups: lookups_buffer,
                cycle_to_table_major: cycle_to_table_major_buffer,
                weights: weights_buffer,
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
                cycle_phase_tables: self.device.new_buffer(
                    byte_length::<Fp128>(CYCLE_PHASE_ELEMENTS)?,
                    MTLResourceOptions::StorageModeShared,
                ),
                cycle_table_values: self.device.new_buffer(
                    byte_length::<Fp128>(ADDRESS_SUFFIX_TABLES)?,
                    MTLResourceOptions::StorageModeShared,
                ),
                cycle_e_in: self.device.new_buffer(
                    byte_length::<Fp128>(cycle_e_in_capacity)?,
                    MTLResourceOptions::StorageModeShared,
                ),
                cycle_e_out: self.device.new_buffer(
                    byte_length::<Fp128>(cycle_e_out_capacity)?,
                    MTLResourceOptions::StorageModeShared,
                ),
                cycle_partial_a: self.device.new_buffer(
                    byte_length::<Fp128>(cycle_partial_elements)?,
                    MTLResourceOptions::StorageModeShared,
                ),
                cycle_partial_b: self.device.new_buffer(
                    byte_length::<Fp128>(cycle_partial_elements)?,
                    MTLResourceOptions::StorageModeShared,
                ),
            },
            rows,
            raf_threadgroups,
            suffix_jobs: suffix_jobs.len(),
            suffix_slots,
            table_offsets,
            rows_per_threadgroup: config.rows_per_threadgroup,
            threads_per_threadgroup,
            cycle_threads_per_threadgroup,
            cycle_bind_threads_per_threadgroup,
            cycle_e_in_capacity,
            cycle_e_out_capacity,
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
                packed_rows: 1,
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
            raf_tile.set_buffer(0, Some(&self.buffers.packed_rows), 0);
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

    pub(crate) fn cycle_message(
        &mut self,
        phase_tables: &[Vec<AkitaField>],
        table_values: &[AkitaField],
        raf_interleaved: AkitaField,
        raf_identity: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; PRODUCT5_FACTORS], MetalError> {
        self.execute_cycle(
            phase_tables,
            table_values,
            raf_interleaved,
            raf_identity,
            e_in,
            e_out,
            None,
        )
        .map(|(message, _)| message)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "cycle derivation consumes the pending relation constants"
    )]
    pub(crate) fn fused_cycle_transition(
        mut self,
        phase_tables: &[Vec<AkitaField>],
        table_values: &[AkitaField],
        raf_interleaved: AkitaField,
        raf_identity: AkitaField,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: Product5SequenceConfig,
    ) -> Result<(Product5Sequence, [AkitaField; PRODUCT5_FACTORS]), MetalError> {
        let elements = self.rows / 2;
        let mut sequence = self.context.prepare_product5_sequence_storage(
            elements,
            e_in.len(),
            e_out.len(),
            config,
        )?;
        let active_time = self.bind_cycle_tables(
            phase_tables,
            table_values,
            raf_interleaved,
            raf_identity,
            challenge,
            sequence.initial_table_buffer(),
        )?;
        sequence.record_gpu_active_time(active_time);
        let message = sequence.message(e_in, e_out)?;
        Ok((sequence, message))
    }

    fn bind_cycle_tables(
        &mut self,
        phase_tables: &[Vec<AkitaField>],
        table_values: &[AkitaField],
        raf_interleaved: AkitaField,
        raf_identity: AkitaField,
        challenge: AkitaField,
        bound: &Buffer,
    ) -> Result<Duration, MetalError> {
        if self.rows < 4 || !self.rows.is_power_of_two() {
            return Err(MetalError::InvalidProduct5TableLength {
                minimum: 4,
                got: self.rows,
            });
        }
        if phase_tables.len() != CYCLE_PHASES
            || phase_tables
                .iter()
                .any(|table| table.len() != ADDRESS_RAF_BINS)
        {
            let got = phase_tables.iter().map(Vec::len).sum();
            return Err(MetalError::AddressCyclePhaseTableShape {
                expected: CYCLE_PHASE_ELEMENTS,
                got,
            });
        }
        if table_values.len() != ADDRESS_SUFFIX_TABLES {
            return Err(MetalError::AddressCycleTableValueCount {
                expected: ADDRESS_SUFFIX_TABLES,
                got: table_values.len(),
            });
        }
        write_phase_tables(&self.buffers.cycle_phase_tables, phase_tables);
        write_akita_fields(&self.buffers.cycle_table_values, table_values);
        let raf_interleaved = Fp128::from_jolt_field(&raf_interleaved);
        let raf_identity = Fp128::from_jolt_field(&raf_identity);
        let challenge = Fp128::from_jolt_field(&challenge);
        let params = CycleParams {
            rows: self.rows as u32,
            e_in_length: 0,
            e_out_length: 0,
            reserved: 0,
        };

        let command_buffer = self.context.queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.cycle_bind_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.packed_rows), 0);
            encoder.set_buffer(1, Some(&self.buffers.lookups), 0);
            encoder.set_buffer(2, Some(&self.buffers.cycle_to_table_major), 0);
            encoder.set_buffer(3, Some(&self.buffers.cycle_phase_tables), 0);
            encoder.set_buffer(4, Some(&self.buffers.cycle_table_values), 0);
            encoder.set_buffer(5, Some(bound), 0);
            set_inline_bytes(encoder, 6, &raf_interleaved);
            set_inline_bytes(encoder, 7, &raf_identity);
            set_inline_bytes(encoder, 8, &challenge);
            set_inline_bytes(encoder, 9, &params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (self.rows / 2).div_ceil(self.cycle_bind_threads_per_threadgroup) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.cycle_bind_threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(command_buffer.status()));
        }
        let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        let active_time = Duration::from_secs_f64(end - start);
        self.gpu_active_time += active_time;
        Ok(active_time)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the shader consumes compact rows, relation constants, and split weights"
    )]
    fn execute_cycle(
        &mut self,
        phase_tables: &[Vec<AkitaField>],
        table_values: &[AkitaField],
        raf_interleaved: AkitaField,
        raf_identity: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        transition: Option<(AkitaField, &Buffer)>,
    ) -> Result<([AkitaField; PRODUCT5_FACTORS], Duration), MetalError> {
        if self.rows < 4 || !self.rows.is_power_of_two() {
            return Err(MetalError::InvalidProduct5TableLength {
                minimum: 4,
                got: self.rows,
            });
        }
        if phase_tables.len() != CYCLE_PHASES
            || phase_tables
                .iter()
                .any(|table| table.len() != ADDRESS_RAF_BINS)
        {
            let got = phase_tables.iter().map(Vec::len).sum();
            return Err(MetalError::AddressCyclePhaseTableShape {
                expected: CYCLE_PHASE_ELEMENTS,
                got,
            });
        }
        if table_values.len() != ADDRESS_SUFFIX_TABLES {
            return Err(MetalError::AddressCycleTableValueCount {
                expected: ADDRESS_SUFFIX_TABLES,
                got: table_values.len(),
            });
        }
        let expected_pairs = if transition.is_some() {
            self.rows / 4
        } else {
            self.rows / 2
        };
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(self.rows))?;
        if e_in.is_empty()
            || e_out.is_empty()
            || e_in.len() > self.cycle_e_in_capacity
            || e_out.len() > self.cycle_e_out_capacity
            || covered != expected_pairs
        {
            return Err(MetalError::Product5WeightShape {
                expected: expected_pairs,
                covered,
            });
        }

        write_phase_tables(&self.buffers.cycle_phase_tables, phase_tables);
        write_akita_fields(&self.buffers.cycle_table_values, table_values);
        write_akita_fields(&self.buffers.cycle_e_in, e_in);
        write_akita_fields(&self.buffers.cycle_e_out, e_out);
        let raf_interleaved = Fp128::from_jolt_field(&raf_interleaved);
        let raf_identity = Fp128::from_jolt_field(&raf_identity);
        let params = CycleParams {
            rows: self.rows as u32,
            e_in_length: e_in.len() as u32,
            e_out_length: e_out.len() as u32,
            reserved: 0,
        };

        let command_buffer = self.context.queue.new_command_buffer();
        let mut final_in_a = true;
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            if let Some((challenge, bound)) = transition {
                encoder.set_compute_pipeline_state(&self.cycle_transition_pipeline);
                encoder.set_buffer(0, Some(&self.buffers.packed_rows), 0);
                encoder.set_buffer(1, Some(&self.buffers.lookups), 0);
                encoder.set_buffer(2, Some(&self.buffers.cycle_to_table_major), 0);
                encoder.set_buffer(3, Some(&self.buffers.cycle_phase_tables), 0);
                encoder.set_buffer(4, Some(&self.buffers.cycle_table_values), 0);
                encoder.set_buffer(5, Some(bound), 0);
                encoder.set_buffer(6, Some(&self.buffers.cycle_e_in), 0);
                encoder.set_buffer(7, Some(&self.buffers.cycle_e_out), 0);
                encoder.set_buffer(8, Some(&self.buffers.cycle_partial_a), 0);
                set_inline_bytes(encoder, 9, &raf_interleaved);
                set_inline_bytes(encoder, 10, &raf_identity);
                set_inline_bytes(encoder, 11, &Fp128::from_jolt_field(&challenge));
                set_inline_bytes(encoder, 12, &params);
            } else {
                encoder.set_compute_pipeline_state(&self.cycle_message_pipeline);
                encoder.set_buffer(0, Some(&self.buffers.packed_rows), 0);
                encoder.set_buffer(1, Some(&self.buffers.lookups), 0);
                encoder.set_buffer(2, Some(&self.buffers.cycle_to_table_major), 0);
                encoder.set_buffer(3, Some(&self.buffers.cycle_phase_tables), 0);
                encoder.set_buffer(4, Some(&self.buffers.cycle_table_values), 0);
                encoder.set_buffer(5, Some(&self.buffers.cycle_e_in), 0);
                encoder.set_buffer(6, Some(&self.buffers.cycle_e_out), 0);
                encoder.set_buffer(7, Some(&self.buffers.cycle_partial_a), 0);
                set_inline_bytes(encoder, 8, &raf_interleaved);
                set_inline_bytes(encoder, 9, &raf_identity);
                set_inline_bytes(encoder, 10, &params);
            }
            encoder.set_threadgroup_memory_length(
                0,
                (PRODUCT5_FACTORS
                    * (self.cycle_threads_per_threadgroup / SIMD_WIDTH)
                    * size_of::<Fp128>()) as u64,
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: e_out.len() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.cycle_threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );

            let mut input_count = e_out.len();
            while input_count > 1 {
                let output_count =
                    input_count.div_ceil(self.cycle_reduce_limits.thread_execution_width);
                let reduction_params = CycleReductionParams {
                    input_count: input_count as u32,
                    output_count: output_count as u32,
                    reserved: [0; 2],
                };
                encoder.set_compute_pipeline_state(&self.cycle_reduce_pipeline);
                let (input, output) = if final_in_a {
                    (&self.buffers.cycle_partial_a, &self.buffers.cycle_partial_b)
                } else {
                    (&self.buffers.cycle_partial_b, &self.buffers.cycle_partial_a)
                };
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                set_inline_bytes(encoder, 2, &reduction_params);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: output_count as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.cycle_reduce_limits.thread_execution_width as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                input_count = output_count;
                final_in_a = !final_in_a;
            }
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(command_buffer.status()));
        }
        let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        let active_time = Duration::from_secs_f64(end - start);
        self.gpu_active_time += active_time;
        let output = if final_in_a {
            &self.buffers.cycle_partial_a
        } else {
            &self.buffers.cycle_partial_b
        };
        // SAFETY: recursive reduction leaves five canonical fields at the
        // front of the selected shared buffer before the command completes.
        let values =
            unsafe { slice::from_raw_parts(output.contents().cast::<Fp128>(), PRODUCT5_FACTORS) };
        self.context
            .validate_inputs("resident address cycle message", values)?;
        Ok((
            std::array::from_fn(|index| values[index].into_jolt_field()),
            active_time,
        ))
    }

    pub const fn phases_executed(&self) -> usize {
        self.phases_executed
    }

    pub const fn gpu_active_time(&self) -> Duration {
        self.gpu_active_time
    }

    pub const fn resident_buffer_count(&self) -> usize {
        21
    }

    pub const fn phase_device_buffer_allocations(&self) -> usize {
        0
    }
}

fn packed_source(row_and_weight: (AddressRafScanRow, Fp128)) -> (AddressLookup, u8, Fp128) {
    let (row, weight) = row_and_weight;
    let lookup = row.lookup_index();
    let table = row.table_index().map_or(0, |table| table + 1) as u8;
    (
        AddressLookup {
            limbs: [lookup as u64, (lookup >> 64) as u64],
        },
        table | (u8::from(row.raf_flag()) << 7),
        weight,
    )
}

fn packed_table(packed: u8) -> Option<usize> {
    let table = usize::from(packed & 0x7f);
    (table != 0).then_some(table - 1)
}

fn write_phase_tables(buffer: &Buffer, tables: &[Vec<AkitaField>]) {
    // SAFETY: validation fixes the flattened table length to the buffer's
    // allocation, and no command buffer is using it during this write.
    let output = unsafe {
        slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), CYCLE_PHASE_ELEMENTS)
    };
    for (output, value) in output.iter_mut().zip(tables.iter().flatten()) {
        *output = Fp128::from_jolt_field(value);
    }
}

fn write_akita_fields(buffer: &Buffer, values: &[AkitaField]) {
    // SAFETY: each caller checks its logical capacity before writing shared
    // storage, and dispatch begins only after the write completes.
    let output =
        unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), values.len()) };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
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

const _: () = assert!(size_of::<RafParams>() == 24);
const _: () = assert!(size_of::<SuffixParams>() == 16);
const _: () = assert!(size_of::<SuffixJob>() == 16);
const _: () = assert!(size_of::<SuffixTable>() == 16);
const _: () = assert!(size_of::<AddressLookup>() == 16);
const _: () = assert!(size_of::<CycleParams>() == 16);
const _: () = assert!(size_of::<CycleReductionParams>() == 16);

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
