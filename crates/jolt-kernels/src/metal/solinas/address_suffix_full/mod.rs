use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::{
    address_raf::{AddressRafScanConfig, AddressRafScanRow},
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
    ADDRESS_SUFFIX_BINS, ADDRESS_SUFFIX_TABLES,
};

const ADDRESS_SUFFIX_MAX_SUFFIXES: usize = 4;
const ADDRESS_SUFFIX_FIELDS: usize = ADDRESS_SUFFIX_MAX_SUFFIXES * ADDRESS_SUFFIX_BINS;
const ADDRESS_SUFFIX_ACCUMULATOR_WORDS: usize = 5;
const ADDRESS_SUFFIX_SIMD_WIDTH: usize = 32;
const TILE_PIPELINE: &str = "solinas_address_suffix_full_tile";
const FINALIZE_PIPELINE: &str = "solinas_address_suffix_full_finalize";

#[repr(C)]
#[derive(Clone, Copy)]
struct AddressSuffixFullParams {
    suffix_len: u32,
    job_count: u32,
    output_elements: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AddressSuffixFullJob {
    start: u32,
    end: u32,
    table: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AddressSuffixFullTable {
    job_start: u32,
    job_end: u32,
    output_start: u32,
    suffix_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AddressSuffixFullLookup {
    limbs: [u64; 2],
}

struct AddressSuffixFullBuffers {
    lookups: Buffer,
    weights: Buffer,
    jobs: Buffer,
    tables: Buffer,
    suffix_kinds: Buffer,
    suffix_counts: Buffer,
    partials: Buffer,
    output: Buffer,
    params: Buffer,
}

pub struct AddressSuffixFullInvocation<'a> {
    context: &'a SolinasMetal,
    tile_pipeline: ComputePipelineState,
    finalize_pipeline: ComputePipelineState,
    tile_limits: PipelineLimits,
    buffers: AddressSuffixFullBuffers,
    jobs: usize,
    suffix_slots: usize,
    table_offsets: Vec<usize>,
    threads_per_threadgroup: usize,
    completed: Cell<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressSuffixFullSums {
    values: Vec<Fp128>,
    table_offsets: Vec<usize>,
}

impl AddressSuffixFullSums {
    pub(super) fn from_values(values: Vec<Fp128>, table_offsets: Vec<usize>) -> Self {
        Self {
            values,
            table_offsets,
        }
    }

    pub fn as_flat_slice(&self) -> &[Fp128] {
        &self.values
    }

    pub fn table(&self, table: usize) -> Option<&[Fp128]> {
        let start = *self.table_offsets.get(table)? * ADDRESS_SUFFIX_BINS;
        let end = *self.table_offsets.get(table + 1)? * ADDRESS_SUFFIX_BINS;
        Some(&self.values[start..end])
    }

    pub fn suffix(&self, table: usize, suffix: usize) -> Option<&[Fp128]> {
        let table_start = *self.table_offsets.get(table)?;
        let table_end = *self.table_offsets.get(table + 1)?;
        (table_start + suffix < table_end).then(|| {
            let start = (table_start + suffix) * ADDRESS_SUFFIX_BINS;
            &self.values[start..start + ADDRESS_SUFFIX_BINS]
        })
    }
}

impl SolinasMetal {
    pub fn prepare_address_suffix_full(
        &self,
        rows: &[AddressRafScanRow],
        weights: &[Fp128],
        config: AddressRafScanConfig,
    ) -> Result<AddressSuffixFullInvocation<'_>, MetalError> {
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
        if config.rows_per_threadgroup == 0 || config.rows_per_threadgroup > 1 << 16 {
            return Err(MetalError::InvalidAddressRafDirectRowsPerThreadgroup(
                config.rows_per_threadgroup,
            ));
        }
        self.validate_inputs("address suffix weights", weights)?;

        let lookup_tables: Vec<_> = LookupTableKind::<RISCV_XLEN>::iter().collect();
        let mut suffix_kinds = vec![0u8; ADDRESS_SUFFIX_TABLES * ADDRESS_SUFFIX_MAX_SUFFIXES];
        let mut suffix_counts = Vec::with_capacity(ADDRESS_SUFFIX_TABLES);
        let mut table_offsets = Vec::with_capacity(ADDRESS_SUFFIX_TABLES + 1);
        table_offsets.push(0usize);
        for table in &lookup_tables {
            let table_index = table.index();
            let suffixes = table.suffixes();
            if suffixes.len() > ADDRESS_SUFFIX_MAX_SUFFIXES {
                return Err(MetalError::InvalidAddressSuffixCount {
                    table: table_index,
                    count: suffixes.len(),
                    maximum: ADDRESS_SUFFIX_MAX_SUFFIXES,
                });
            }
            for (suffix, kind) in suffixes.iter().enumerate() {
                suffix_kinds[table_index * ADDRESS_SUFFIX_MAX_SUFFIXES + suffix] = *kind as u8;
            }
            suffix_counts.push(suffixes.len() as u8);
            table_offsets.push(table_offsets.last().copied().unwrap_or(0) + suffixes.len());
        }

        let mut buckets = vec![Vec::<u32>::new(); ADDRESS_SUFFIX_TABLES];
        for (row_index, row) in rows.iter().enumerate() {
            if let Some(table) = row.table_index() {
                let bucket = buckets
                    .get_mut(table)
                    .ok_or(MetalError::InvalidAddressSuffixTable(table))?;
                bucket.push(
                    u32::try_from(row_index).map_err(|_| MetalError::InputTooLong(row_index))?,
                );
            }
        }

        let selected_rows: usize = buckets.iter().map(Vec::len).sum();
        let mut table_major_lookups = Vec::with_capacity(selected_rows);
        let mut table_major_weights = Vec::with_capacity(selected_rows);
        let mut jobs = Vec::new();
        let mut tables = Vec::with_capacity(ADDRESS_SUFFIX_TABLES);
        for (table, bucket) in buckets.iter().enumerate() {
            let job_start =
                u32::try_from(jobs.len()).map_err(|_| MetalError::InputTooLong(jobs.len()))?;
            let bucket_start = table_major_lookups.len();
            for &row in bucket {
                let row = row as usize;
                let lookup = rows[row].lookup_index();
                table_major_lookups.push(AddressSuffixFullLookup {
                    limbs: [lookup as u64, (lookup >> 64) as u64],
                });
                table_major_weights.push(weights[row]);
            }
            for local_start in (0..bucket.len()).step_by(config.rows_per_threadgroup) {
                let local_end = (local_start + config.rows_per_threadgroup).min(bucket.len());
                jobs.push(AddressSuffixFullJob {
                    start: u32::try_from(bucket_start + local_start)
                        .map_err(|_| MetalError::InputTooLong(bucket_start + local_start))?,
                    end: u32::try_from(bucket_start + local_end)
                        .map_err(|_| MetalError::InputTooLong(bucket_start + local_end))?,
                    table: table as u32,
                    reserved: 0,
                });
            }
            tables.push(AddressSuffixFullTable {
                job_start,
                job_end: u32::try_from(jobs.len())
                    .map_err(|_| MetalError::InputTooLong(jobs.len()))?,
                output_start: u32::try_from(table_offsets[table])
                    .map_err(|_| MetalError::InputTooLong(table_offsets[table]))?,
                suffix_count: u32::from(suffix_counts[table]),
            });
        }
        if jobs.is_empty() {
            return Err(MetalError::EmptyAddressSuffixBuckets);
        }

        let suffix_slots = *table_offsets.last().unwrap_or(&0);
        let output_elements = suffix_slots
            .checked_mul(ADDRESS_SUFFIX_BINS)
            .ok_or(MetalError::InputTooLong(suffix_slots))?;
        let params = AddressSuffixFullParams {
            suffix_len: config.suffix_len,
            job_count: u32::try_from(jobs.len())
                .map_err(|_| MetalError::InputTooLong(jobs.len()))?,
            output_elements: u32::try_from(output_elements)
                .map_err(|_| MetalError::InputTooLong(output_elements))?,
            reserved: 0,
        };

        let tile_pipeline = self.compile_named_pipeline(TILE_PIPELINE)?;
        let finalize_pipeline = self.compile_named_pipeline(FINALIZE_PIPELINE)?;
        let tile_limits = Self::limits(&tile_pipeline);
        let finalize_limits = Self::limits(&finalize_pipeline);
        for (pipeline, limits) in [
            (TILE_PIPELINE, tile_limits),
            (FINALIZE_PIPELINE, finalize_limits),
        ] {
            if limits.thread_execution_width != ADDRESS_SUFFIX_SIMD_WIDTH {
                return Err(MetalError::UnsupportedAddressRafExecutionWidth {
                    pipeline,
                    expected: ADDRESS_SUFFIX_SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, tile_limits)?;
        if finalize_limits.max_total_threads_per_threadgroup < ADDRESS_SUFFIX_FIELDS {
            return Err(MetalError::InvalidThreadgroupWidth {
                requested: ADDRESS_SUFFIX_FIELDS,
                execution_width: finalize_limits.thread_execution_width,
                maximum: finalize_limits.max_total_threads_per_threadgroup,
            });
        }
        let accumulator_bytes =
            ADDRESS_SUFFIX_FIELDS * ADDRESS_SUFFIX_ACCUMULATOR_WORDS * size_of::<u32>();
        if accumulator_bytes as u64 > self.device.max_threadgroup_memory_length() {
            return Err(MetalError::AddressSuffixThreadgroupMemory {
                requested: accumulator_bytes as u64,
                maximum: self.device.max_threadgroup_memory_length(),
            });
        }

        let partial_elements = jobs
            .len()
            .checked_mul(ADDRESS_SUFFIX_FIELDS)
            .ok_or(MetalError::InputTooLong(jobs.len()))?;
        let partial_bytes = byte_length::<Fp128>(partial_elements)?;
        let output_bytes = byte_length::<Fp128>(output_elements)?;
        for requested in [
            byte_length::<AddressSuffixFullLookup>(table_major_lookups.len())?,
            byte_length::<Fp128>(table_major_weights.len())?,
            byte_length::<AddressSuffixFullJob>(jobs.len())?,
            byte_length::<AddressSuffixFullTable>(tables.len())?,
            byte_length::<u8>(suffix_kinds.len())?,
            byte_length::<u8>(suffix_counts.len())?,
            partial_bytes,
            output_bytes,
        ] {
            let maximum = self.device.max_buffer_length();
            if requested > maximum {
                return Err(MetalError::BufferTooLong { requested, maximum });
            }
        }

        Ok(AddressSuffixFullInvocation {
            context: self,
            tile_pipeline,
            finalize_pipeline,
            tile_limits,
            buffers: AddressSuffixFullBuffers {
                lookups: buffer_from_slice(&self.device, &table_major_lookups),
                weights: buffer_from_slice(&self.device, &table_major_weights),
                jobs: buffer_from_slice(&self.device, &jobs),
                tables: buffer_from_slice(&self.device, &tables),
                suffix_kinds: buffer_from_slice(&self.device, &suffix_kinds),
                suffix_counts: buffer_from_slice(&self.device, &suffix_counts),
                partials: self
                    .device
                    .new_buffer(partial_bytes, MTLResourceOptions::StorageModeShared),
                output: self
                    .device
                    .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared),
                params: buffer_from_slice(&self.device, slice::from_ref(&params)),
            },
            jobs: jobs.len(),
            suffix_slots,
            table_offsets,
            threads_per_threadgroup,
            completed: Cell::new(false),
        })
    }
}

impl AddressSuffixFullInvocation<'_> {
    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.tile_limits
    }

    pub const fn job_count(&self) -> usize {
        self.jobs
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn intermediate_partial_bytes(&self) -> u64 {
        self.jobs as u64 * ADDRESS_SUFFIX_FIELDS as u64 * size_of::<Fp128>() as u64
    }

    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let tile = command_buffer.new_compute_command_encoder();
            tile.set_compute_pipeline_state(&self.tile_pipeline);
            tile.set_buffer(0, Some(&self.buffers.lookups), 0);
            tile.set_buffer(1, Some(&self.buffers.weights), 0);
            tile.set_buffer(2, Some(&self.buffers.jobs), 0);
            tile.set_buffer(3, Some(&self.buffers.suffix_kinds), 0);
            tile.set_buffer(4, Some(&self.buffers.suffix_counts), 0);
            tile.set_buffer(5, Some(&self.buffers.partials), 0);
            tile.set_buffer(6, Some(&self.buffers.params), 0);
            tile.set_threadgroup_memory_length(
                0,
                (ADDRESS_SUFFIX_FIELDS * ADDRESS_SUFFIX_ACCUMULATOR_WORDS * size_of::<u32>())
                    as u64,
            );
            tile.dispatch_thread_groups(
                MTLSize {
                    width: self.jobs as u64,
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
            finalize.set_buffer(1, Some(&self.buffers.tables), 0);
            finalize.set_buffer(2, Some(&self.buffers.output), 0);
            finalize.dispatch_thread_groups(
                MTLSize {
                    width: ADDRESS_SUFFIX_TABLES as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: ADDRESS_SUFFIX_FIELDS as u64,
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

    pub fn read_output(&self) -> Result<AddressSuffixFullSums, MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        let elements = self.suffix_slots * ADDRESS_SUFFIX_BINS;
        // SAFETY: the shared output buffer contains `elements` field values and
        // the command buffer has completed.
        let values = unsafe {
            slice::from_raw_parts(self.buffers.output.contents().cast::<Fp128>(), elements)
        };
        self.context
            .validate_inputs("full address suffix output", values)?;
        Ok(AddressSuffixFullSums::from_values(
            values.to_vec(),
            self.table_offsets.clone(),
        ))
    }
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

const _: () = assert!(size_of::<AddressSuffixFullParams>() == 16);
const _: () = assert!(size_of::<AddressSuffixFullJob>() == 16);
const _: () = assert!(size_of::<AddressSuffixFullTable>() == 16);
const _: () = assert!(size_of::<AddressSuffixFullLookup>() == 16);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::{AkitaField, MulPrimitiveInt};
    use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};

    use super::{
        AddressRafScanConfig, AddressRafScanRow, Fp128, SolinasMetal, ADDRESS_SUFFIX_BINS,
        ADDRESS_SUFFIX_TABLES,
    };

    #[test]
    fn every_address_suffix_matches_jolt_field() {
        let mut state = 0x87bd_86f1_2345_6789;
        let rows: Vec<_> = (0..4099)
            .map(|index| {
                let lookup_index =
                    (u128::from(splitmix(&mut state)) << 64) | u128::from(splitmix(&mut state));
                AddressRafScanRow::new_with_table(
                    lookup_index,
                    Some(index % ADDRESS_SUFFIX_TABLES),
                    index % 3 == 0,
                )
            })
            .collect();
        let weights: Vec<_> = (0..rows.len())
            .map(|_| {
                let value = u128::from(splitmix(&mut state))
                    | (u128::from(splitmix(&mut state) & 0x7fff_ffff_ffff_ffff) << 64);
                Fp128::from_u128(value)
            })
            .collect();
        let context = SolinasMetal::for_akita().unwrap();

        for suffix_len in [0, 8, 32, 56, 64, 112, 120] {
            let invocation = context
                .prepare_address_suffix_full(
                    &rows,
                    &weights,
                    AddressRafScanConfig {
                        suffix_len,
                        rows_per_threadgroup: 64,
                        threads_per_threadgroup: Some(128),
                    },
                )
                .unwrap();
            invocation.execute().unwrap();
            assert_eq!(
                invocation.read_output().unwrap().as_flat_slice(),
                oracle(&rows, &weights, suffix_len)
            );
        }
    }

    fn oracle(rows: &[AddressRafScanRow], weights: &[Fp128], suffix_len: u32) -> Vec<Fp128> {
        let tables: Vec<_> = LookupTableKind::<RISCV_XLEN>::iter().collect();
        let mut offsets = Vec::with_capacity(ADDRESS_SUFFIX_TABLES + 1);
        offsets.push(0usize);
        for table in &tables {
            offsets.push(offsets.last().copied().unwrap_or(0) + table.suffixes().len());
        }
        let mut output = vec![AkitaField::zero(); offsets[ADDRESS_SUFFIX_TABLES] * 256];
        for (&row, &weight) in rows.iter().zip(weights) {
            let table = row.table_index().unwrap();
            let lookup = row.lookup_index();
            let suffix_bits = LookupBits::new(lookup, suffix_len as usize);
            let chunk = ((lookup >> suffix_len) as usize) & (ADDRESS_SUFFIX_BINS - 1);
            let weight = weight.into_jolt_field::<AkitaField>();
            for (suffix_index, suffix) in tables[table].suffixes().iter().enumerate() {
                let scalar = suffix.suffix_mle(suffix_bits);
                output[(offsets[table] + suffix_index) * ADDRESS_SUFFIX_BINS + chunk] +=
                    weight.mul_u64(scalar);
            }
        }
        output.iter().map(Fp128::from_jolt_field).collect()
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}
