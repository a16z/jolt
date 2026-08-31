use std::{mem::size_of, slice};

use jolt_field::AkitaField;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLResourceOptions, MTLSize,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    address_raf::{AddressRafScanRow, AddressRafSums},
    address_suffix_full::AddressSuffixFullSums,
    buffer_from_slice, set_inline_bytes, validate_completed_command, Fp128,
    InstructionReadRafCountOrder, InstructionReadRafDenseGroupedPlanes, MetalError, PipelineLimits,
    Product5Sequence, Product5SequenceConfig, SolinasMetal, ADDRESS_RAF_BINS, ADDRESS_RAF_LANES,
    ADDRESS_SUFFIX_BINS, ADDRESS_SUFFIX_TABLES, INSTRUCTION_READ_RAF_SEGMENTS, PRODUCT5_FACTORS,
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

#[derive(Clone)]
pub(crate) struct ResidentLookupIndexPlane {
    lookups: Buffer,
    cycle_to_table_major: Buffer,
    rows: usize,
    device_registry_id: u64,
}

impl ResidentLookupIndexPlane {
    pub(super) fn from_buffers(
        lookups: Buffer,
        cycle_to_table_major: Buffer,
        rows: usize,
        device_registry_id: u64,
    ) -> Self {
        Self {
            lookups,
            cycle_to_table_major,
            rows,
            device_registry_id,
        }
    }

    copy_field_getters! { pub(crate), {
        len => rows: usize,
        device_registry_id: u64,
    }}
    ref_field_getters! { pub(crate), {
        lookups: Buffer,
        cycle_to_table_major: Buffer,
    }}
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for ResidentLookupIndexPlane {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_lookup_indices"),
            self.rows * size_of::<AddressLookup>(),
        );
        visitor.visit_simple(
            allocative::Key::new("device_cycle_to_table_major"),
            self.rows * size_of::<u32>(),
        );
        visitor.exit();
    }
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

struct PreparedAddressPhasePlanes {
    packed_rows: Buffer,
    lookups: Buffer,
    cycle_to_table_major: Buffer,
    weights: Buffer,
    segment_ranges: [std::ops::Range<usize>; INSTRUCTION_READ_RAF_SEGMENTS],
    rows: usize,
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressPhaseSums {
    raf: AddressRafSums,
    suffix: AddressSuffixFullSums,
}

impl AddressPhaseSums {
    pub const fn raf(&self) -> &AddressRafSums {
        &self.raf
    }

    pub const fn suffix(&self) -> &AddressSuffixFullSums {
        &self.suffix
    }
}

impl SolinasMetal {
    pub(crate) fn prepare_address_phase_sequence_from_buckets(
        &self,
        rows: usize,
        buckets: &[Vec<u32>],
        config: AddressPhaseSequenceConfig,
        source: impl Fn(usize) -> (AddressRafScanRow, Fp128) + Sync,
    ) -> Result<AddressPhaseSequence, MetalError> {
        self.prepare_address_phase_sequence_inner(rows, buckets, config, None, source)
    }

    pub(crate) fn prepare_address_phase_sequence_from_resident_grouped(
        &self,
        planes: InstructionReadRafDenseGroupedPlanes,
        config: AddressPhaseSequenceConfig,
    ) -> Result<AddressPhaseSequence, MetalError> {
        let prepared = validate_resident_grouped_planes(self, planes)?;
        let rows = prepared.rows;
        let empty_buckets = vec![Vec::new(); ADDRESS_SUFFIX_TABLES];
        self.prepare_address_phase_sequence_inner(
            rows,
            &empty_buckets,
            config,
            Some(prepared),
            |_| unreachable!("resident grouped preparation does not inspect host rows"),
        )
    }

    fn prepare_address_phase_sequence_inner(
        &self,
        rows: usize,
        buckets: &[Vec<u32>],
        config: AddressPhaseSequenceConfig,
        prepared: Option<PreparedAddressPhasePlanes>,
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

        let (
            packed_rows_buffer,
            lookups_buffer,
            cycle_to_table_major_buffer,
            weights_buffer,
            suffix_jobs,
            suffix_tables,
        ) = if let Some(prepared) = prepared {
            let (suffix_jobs, suffix_tables) = resident_suffix_schedule(
                rows,
                &prepared.segment_ranges,
                config.rows_per_threadgroup,
                &suffix_counts,
                &table_offsets,
            )?;
            (
                prepared.packed_rows,
                prepared.lookups,
                prepared.cycle_to_table_major,
                prepared.weights,
                suffix_jobs,
                suffix_tables,
            )
        } else {
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
                slice::from_raw_parts_mut(
                    cycle_to_table_major_buffer.contents().cast::<u32>(),
                    rows,
                )
            };
            let mut table_selected = vec![false; rows];
            let mut table_major_len = 0usize;
            let mut suffix_jobs = Vec::new();
            let mut suffix_tables = Vec::with_capacity(ADDRESS_SUFFIX_TABLES);
            let mut table_row_ranges = Vec::with_capacity(ADDRESS_SUFFIX_TABLES);
            for (table, bucket) in buckets.iter().enumerate() {
                let job_start = u32::try_from(suffix_jobs.len())
                    .map_err(|_| MetalError::InputTooLong(suffix_jobs.len()))?;
                let bucket_start = table_major_len;
                for &row in bucket {
                    let row = row as usize;
                    if row >= rows {
                        return Err(MetalError::InputTooLong(row));
                    }
                    if table_selected[row] {
                        return Err(MetalError::AddressPhaseLayoutLength {
                            expected: rows,
                            got: table_major_len,
                        });
                    }
                    table_selected[row] = true;
                }
                for &row in bucket {
                    let row = row as usize;
                    cycle_to_table_major[row] = u32::try_from(table_major_len)
                        .map_err(|_| MetalError::InputTooLong(table_major_len))?;
                    table_major_len += 1;
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
                table_row_ranges.push(bucket_start..table_major_len);
            }
            if suffix_jobs.is_empty() {
                return Err(MetalError::EmptyAddressSuffixBuckets);
            }
            let no_table_start = table_major_len;
            for (cycle, &selected) in table_selected.iter().enumerate() {
                if !selected {
                    cycle_to_table_major[cycle] = u32::try_from(table_major_len)
                        .map_err(|_| MetalError::InputTooLong(table_major_len))?;
                    table_major_len += 1;
                }
            }
            drop(table_selected);
            if table_major_len != rows {
                return Err(MetalError::AddressPhaseLayoutLength {
                    expected: rows,
                    got: table_major_len,
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
            let packed_rows = unsafe {
                slice::from_raw_parts_mut(packed_rows_buffer.contents().cast::<u8>(), rows)
            };
            // SAFETY: see the allocation and exclusivity argument above.
            let table_major_weights = unsafe {
                slice::from_raw_parts_mut(weights_buffer.contents().cast::<Fp128>(), rows)
            };
            #[cfg(feature = "parallel")]
            {
                let lookups_address = lookups.as_mut_ptr() as usize;
                let packed_address = packed_rows.as_mut_ptr() as usize;
                let weights_address = table_major_weights.as_mut_ptr() as usize;
                cycle_to_table_major
                    .par_iter()
                    .enumerate()
                    .for_each(|(cycle, &table_major)| {
                        let (lookup, packed, weight) = packed_source(source(cycle));
                        let table_major = table_major as usize;
                        // SAFETY: the validated inverse is a permutation, so
                        // parallel iterations write disjoint initialized slots.
                        unsafe {
                            (lookups_address as *mut AddressLookup)
                                .add(table_major)
                                .write(lookup);
                            (packed_address as *mut u8).add(table_major).write(packed);
                            (weights_address as *mut Fp128)
                                .add(table_major)
                                .write(weight);
                        }
                    });
            }
            #[cfg(not(feature = "parallel"))]
            for (cycle, &table_major) in cycle_to_table_major.iter().enumerate() {
                let table_major = table_major as usize;
                (
                    lookups[table_major],
                    packed_rows[table_major],
                    table_major_weights[table_major],
                ) = packed_source(source(cycle));
            }
            let original_cycle = |table_major: usize| {
                cycle_to_table_major
                    .iter()
                    .position(|&mapped| mapped as usize == table_major)
                    .unwrap_or(rows)
            };
            for (table, range) in table_row_ranges.iter().enumerate() {
                if let Some(position) = packed_rows[range.clone()]
                    .iter()
                    .position(|packed| usize::from(*packed & 0x7f) != table + 1)
                {
                    let table_major = range.start + position;
                    let packed = packed_rows[table_major];
                    return Err(MetalError::InvalidAddressPhaseBucket {
                        bucket: table,
                        row: original_cycle(table_major),
                        actual: packed_table(packed),
                    });
                }
            }
            if let Some(position) = packed_rows[no_table_start..]
                .iter()
                .position(|packed| packed & 0x7f != 0)
            {
                let table_major = no_table_start + position;
                let packed = packed_rows[table_major];
                return Err(MetalError::InvalidAddressPhaseBucket {
                    bucket: ADDRESS_SUFFIX_TABLES,
                    row: original_cycle(table_major),
                    actual: packed_table(packed),
                });
            }
            self.validate_inputs("resident address weights", table_major_weights)?;
            (
                packed_rows_buffer,
                lookups_buffer,
                cycle_to_table_major_buffer,
                weights_buffer,
                suffix_jobs,
                suffix_tables,
            )
        };

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
        })
    }
}

impl AddressPhaseSequence {
    pub(crate) fn resident_lookup_index_plane(&self) -> ResidentLookupIndexPlane {
        ResidentLookupIndexPlane::from_buffers(
            self.buffers.lookups.clone(),
            self.buffers.cycle_to_table_major.clone(),
            self.rows,
            self.context.device.registry_id(),
        )
    }

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
        validate_completed_command(command_buffer)?;
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
        self.bind_cycle_tables(
            phase_tables,
            table_values,
            raf_interleaved,
            raf_identity,
            challenge,
            sequence.initial_table_buffer(),
        )?;
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
    ) -> Result<(), MetalError> {
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
        validate_completed_command(command_buffer)?;
        Ok(())
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
    ) -> Result<[AkitaField; PRODUCT5_FACTORS], MetalError> {
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
        validate_completed_command(command_buffer)?;
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
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }

    copy_field_getters! { pub, { phases_executed: usize }}

    pub const fn resident_buffer_count(&self) -> usize {
        21
    }
}

fn validate_resident_grouped_planes(
    context: &SolinasMetal,
    planes: InstructionReadRafDenseGroupedPlanes,
) -> Result<PreparedAddressPhasePlanes, MetalError> {
    let parts = planes.into_parts();
    let receipt = &parts.receipt;
    let rows = receipt.rows();
    if rows == 0 || !rows.is_power_of_two() {
        return Err(resident_grouped_error(
            "resident grouped row domain is not a nonzero power of two",
        ));
    }
    if receipt.device_registry_id() != context.device.registry_id()
        || receipt.source().device_registry_id() != context.device.registry_id()
    {
        return Err(resident_grouped_error(
            "resident grouped planes belong to another Metal device",
        ));
    }
    if receipt.source().count_order() != InstructionReadRafCountOrder::TableMajorThenNoneV1
        || receipt.source().source_generation() == 0
        || receipt.source().completion_serial() == 0
        || receipt.completion_serial() == 0
    {
        return Err(resident_grouped_error(
            "resident grouped provenance receipt is incomplete",
        ));
    }
    if !receipt.complete_overwrite()
        || receipt.dispatches() != 1
        || receipt.source_copy_bytes() != 0
        || receipt.full_plane_readback_bytes() != 0
    {
        return Err(resident_grouped_error(
            "resident grouped scatter did not publish a complete zero-copy result",
        ));
    }
    if receipt.e_in_length().checked_mul(receipt.e_out_length()) != Some(rows)
        || !receipt.e_in_length().is_power_of_two()
        || !receipt.e_out_length().is_power_of_two()
    {
        return Err(resident_grouped_error(
            "resident grouped equality split disagrees with the row domain",
        ));
    }

    let expected_bytes = [
        byte_length::<u8>(rows)?,
        byte_length::<AddressLookup>(rows)?,
        byte_length::<u32>(rows)?,
        byte_length::<Fp128>(rows)?,
    ];
    let actual_bytes = [
        parts.packed_rows.length(),
        parts.lookups.length(),
        parts.inverse.length(),
        parts.weights.length(),
    ];
    let receipt_bytes = [
        receipt.packed_rows_bytes(),
        receipt.lookups_bytes(),
        receipt.inverse_bytes(),
        receipt.weights_bytes(),
    ];
    if actual_bytes != expected_bytes || receipt_bytes != expected_bytes {
        return Err(resident_grouped_error(
            "resident grouped plane lengths disagree with the typed layout",
        ));
    }
    let actual_identities = [
        parts.packed_rows.as_ptr() as usize,
        parts.lookups.as_ptr() as usize,
        parts.inverse.as_ptr() as usize,
        parts.weights.as_ptr() as usize,
    ];
    if actual_identities != receipt.allocation_identities()
        || actual_identities.contains(&0)
        || actual_identities
            .iter()
            .enumerate()
            .any(|(index, identity)| actual_identities[..index].contains(identity))
    {
        return Err(resident_grouped_error(
            "resident grouped output allocations are stale or alias",
        ));
    }
    let segment_ranges = receipt.segment_ranges().clone();
    validate_resident_segment_ranges(rows, &segment_ranges)?;

    Ok(PreparedAddressPhasePlanes {
        packed_rows: parts.packed_rows,
        lookups: parts.lookups,
        cycle_to_table_major: parts.inverse,
        weights: parts.weights,
        segment_ranges,
        rows,
    })
}

fn validate_resident_segment_ranges(
    rows: usize,
    ranges: &[std::ops::Range<usize>; INSTRUCTION_READ_RAF_SEGMENTS],
) -> Result<(), MetalError> {
    let mut cursor = 0usize;
    for physical in 0..INSTRUCTION_READ_RAF_SEGMENTS {
        let logical = if physical < 2 * ADDRESS_SUFFIX_TABLES {
            physical + 2
        } else {
            physical - 2 * ADDRESS_SUFFIX_TABLES
        };
        let range = &ranges[logical];
        if range.start != cursor || range.end < range.start || range.end > rows {
            return Err(resident_grouped_error(
                "resident grouped segment ranges are not the canonical physical partition",
            ));
        }
        cursor = range.end;
    }
    if cursor != rows {
        return Err(resident_grouped_error(
            "resident grouped segment ranges do not cover the row domain",
        ));
    }
    Ok(())
}

fn resident_suffix_schedule(
    rows: usize,
    ranges: &[std::ops::Range<usize>; INSTRUCTION_READ_RAF_SEGMENTS],
    rows_per_threadgroup: usize,
    suffix_counts: &[u8],
    table_offsets: &[usize],
) -> Result<(Vec<SuffixJob>, Vec<SuffixTable>), MetalError> {
    let mut jobs = Vec::new();
    let mut tables = Vec::with_capacity(ADDRESS_SUFFIX_TABLES);
    for table in 0..ADDRESS_SUFFIX_TABLES {
        let false_range = &ranges[2 * (table + 1)];
        let true_range = &ranges[2 * (table + 1) + 1];
        if false_range.end != true_range.start {
            return Err(resident_grouped_error(
                "resident grouped table flag ranges are not adjacent",
            ));
        }
        let job_start =
            u32::try_from(jobs.len()).map_err(|_| MetalError::InputTooLong(jobs.len()))?;
        for start in (false_range.start..true_range.end).step_by(rows_per_threadgroup) {
            let end = (start + rows_per_threadgroup).min(true_range.end);
            jobs.push(SuffixJob {
                start: u32::try_from(start).map_err(|_| MetalError::InputTooLong(start))?,
                end: u32::try_from(end).map_err(|_| MetalError::InputTooLong(end))?,
                table: table as u32,
                reserved: 0,
            });
        }
        tables.push(SuffixTable {
            job_start,
            job_end: u32::try_from(jobs.len()).map_err(|_| MetalError::InputTooLong(jobs.len()))?,
            output_start: u32::try_from(table_offsets[table])
                .map_err(|_| MetalError::InputTooLong(table_offsets[table]))?,
            suffix_count: u32::from(suffix_counts[table]),
        });
    }
    if jobs.is_empty() {
        return Err(MetalError::EmptyAddressSuffixBuckets);
    }
    if ranges.iter().any(|range| range.end > rows) {
        return Err(resident_grouped_error(
            "resident grouped suffix schedule exceeds the row domain",
        ));
    }
    Ok((jobs, tables))
}

fn resident_grouped_error(message: &'static str) -> MetalError {
    MetalError::InvalidInstructionReadRafGrouped(message.to_owned())
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
