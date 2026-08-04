use std::{cell::Cell, mem::size_of, slice, time::Duration};

use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

pub const ADDRESS_RAF_LANES: usize = 6;
pub const ADDRESS_RAF_BINS: usize = 256;
const ADDRESS_RAF_KEYS: usize = 2 * ADDRESS_RAF_BINS;
const ADDRESS_RAF_OUTPUTS: usize = ADDRESS_RAF_LANES * ADDRESS_RAF_BINS;
const ADDRESS_RAF_PARTIAL_LANES: usize = 3;
const ADDRESS_RAF_SIMD_WIDTH: usize = 32;
const HISTOGRAM_PIPELINE: &str = "solinas_address_raf_histogram";
const OFFSETS_PIPELINE: &str = "solinas_address_raf_offsets";
const SCATTER_PIPELINE: &str = "solinas_address_raf_scatter";
const REDUCE_PIPELINE: &str = "solinas_address_raf_reduce";
const RAF_FLAG_SHIFT: u32 = 62;

/// The 40-byte row ABI consumed by the address RAF scan probe.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AddressRafScanRow {
    words: [u64; 5],
}

impl AddressRafScanRow {
    pub const fn new(lookup_index: u128, raf_flag: bool) -> Self {
        Self {
            words: [
                lookup_index as u64,
                (lookup_index >> 64) as u64,
                0,
                0,
                (raf_flag as u64) << RAF_FLAG_SHIFT,
            ],
        }
    }

    pub const fn lookup_index(self) -> u128 {
        self.words[0] as u128 | ((self.words[1] as u128) << 64)
    }

    pub const fn raf_flag(self) -> bool {
        self.words[4] & (1 << RAF_FLAG_SHIFT) != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressRafScanConfig {
    pub suffix_len: u32,
    pub rows_per_threadgroup: usize,
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for AddressRafScanConfig {
    fn default() -> Self {
        Self {
            suffix_len: 120,
            rows_per_threadgroup: 1 << 16,
            threads_per_threadgroup: Some(1024),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressRafSums {
    values: Vec<Fp128>,
}

impl AddressRafSums {
    pub fn as_flat_slice(&self) -> &[Fp128] {
        &self.values
    }

    pub fn shift_half(&self) -> &[Fp128] {
        self.lane(0)
    }

    pub fn left(&self) -> &[Fp128] {
        self.lane(1)
    }

    pub fn right(&self) -> &[Fp128] {
        self.lane(2)
    }

    pub fn shift_full(&self) -> &[Fp128] {
        self.lane(3)
    }

    pub fn identity(&self) -> &[Fp128] {
        self.lane(4)
    }

    pub fn upper_all_ones(&self) -> &[Fp128] {
        self.lane(5)
    }

    fn lane(&self, lane: usize) -> &[Fp128] {
        &self.values[lane * ADDRESS_RAF_BINS..(lane + 1) * ADDRESS_RAF_BINS]
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AddressRafParams {
    rows: u32,
    suffix_len: u32,
    rows_per_threadgroup: u32,
    threadgroup_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AddressRafLookup {
    limbs: [u64; 2],
}

#[repr(C)]
struct AddressRafContribution {
    weight: Fp128,
    scalars: [u64; 2],
}

struct AddressRafBuffers {
    keys: Buffer,
    lookups: Buffer,
    weights: Buffer,
    group_counts: Buffer,
    group_offsets: Buffer,
    bin_offsets: Buffer,
    bucketed_contributions: Buffer,
    output: Buffer,
    params: Buffer,
}

pub struct AddressRafScanInvocation<'a> {
    context: &'a SolinasMetal,
    histogram_pipeline: ComputePipelineState,
    offsets_pipeline: ComputePipelineState,
    scatter_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    histogram_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: AddressRafBuffers,
    rows: usize,
    threadgroup_count: usize,
    threads_per_threadgroup: usize,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub fn prepare_address_raf_scan(
        &self,
        rows: &[AddressRafScanRow],
        weights: &[Fp128],
        config: AddressRafScanConfig,
    ) -> Result<AddressRafScanInvocation<'_>, MetalError> {
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
        if config.rows_per_threadgroup == 0 {
            return Err(MetalError::InvalidAddressRafRowsPerThreadgroup(
                config.rows_per_threadgroup,
            ));
        }
        self.validate_inputs("address RAF weights", weights)?;

        let row_count =
            u32::try_from(rows.len()).map_err(|_| MetalError::InputTooLong(rows.len()))?;
        let rows_per_threadgroup = u32::try_from(config.rows_per_threadgroup)
            .map_err(|_| MetalError::InputTooLong(config.rows_per_threadgroup))?;
        let threadgroup_count = rows.len().div_ceil(config.rows_per_threadgroup);
        let params = AddressRafParams {
            rows: row_count,
            suffix_len: config.suffix_len,
            rows_per_threadgroup,
            threadgroup_count: u32::try_from(threadgroup_count)
                .map_err(|_| MetalError::InputTooLong(threadgroup_count))?,
        };

        let histogram_pipeline = self.compile_named_pipeline(HISTOGRAM_PIPELINE)?;
        let offsets_pipeline = self.compile_named_pipeline(OFFSETS_PIPELINE)?;
        let scatter_pipeline = self.compile_named_pipeline(SCATTER_PIPELINE)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let histogram_limits = Self::limits(&histogram_pipeline);
        let offsets_limits = Self::limits(&offsets_pipeline);
        let scatter_limits = Self::limits(&scatter_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (HISTOGRAM_PIPELINE, histogram_limits),
            (OFFSETS_PIPELINE, offsets_limits),
            (SCATTER_PIPELINE, scatter_limits),
            (REDUCE_PIPELINE, reduction_limits),
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
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, histogram_limits)?;
        for limits in [scatter_limits, reduction_limits] {
            if threads_per_threadgroup > limits.max_total_threads_per_threadgroup {
                return Err(MetalError::InvalidThreadgroupWidth {
                    requested: threads_per_threadgroup,
                    execution_width: limits.thread_execution_width,
                    maximum: limits.max_total_threads_per_threadgroup,
                });
            }
        }
        if offsets_limits.max_total_threads_per_threadgroup < ADDRESS_RAF_KEYS {
            return Err(MetalError::InvalidThreadgroupWidth {
                requested: ADDRESS_RAF_KEYS,
                execution_width: offsets_limits.thread_execution_width,
                maximum: offsets_limits.max_total_threads_per_threadgroup,
            });
        }

        let group_entries = threadgroup_count
            .checked_mul(ADDRESS_RAF_KEYS)
            .ok_or(MetalError::InputTooLong(threadgroup_count))?;
        let group_bytes = byte_length::<u32>(group_entries)?;
        let bin_offset_bytes = byte_length::<u32>(ADDRESS_RAF_KEYS + 1)?;
        let contribution_bytes = byte_length::<AddressRafContribution>(rows.len())?;
        let key_bytes = byte_length::<u16>(rows.len())?;
        let lookup_bytes = byte_length::<AddressRafLookup>(rows.len())?;
        let output_bytes = byte_length::<Fp128>(ADDRESS_RAF_OUTPUTS)?;
        for requested in [
            size_of_val_u64(weights)?,
            key_bytes,
            lookup_bytes,
            group_bytes,
            bin_offset_bytes,
            contribution_bytes,
            output_bytes,
        ] {
            let maximum = self.device.max_buffer_length();
            if requested > maximum {
                return Err(MetalError::BufferTooLong { requested, maximum });
            }
        }
        let keys: Vec<u16> = rows
            .iter()
            .map(|row| {
                let chunk = ((row.lookup_index() >> config.suffix_len) & 0xff) as u16;
                chunk | (u16::from(row.raf_flag()) << 8)
            })
            .collect();
        let lookups: Vec<AddressRafLookup> = rows
            .iter()
            .map(|row| AddressRafLookup {
                limbs: [row.lookup_index() as u64, (row.lookup_index() >> 64) as u64],
            })
            .collect();

        Ok(AddressRafScanInvocation {
            context: self,
            histogram_pipeline,
            offsets_pipeline,
            scatter_pipeline,
            reduction_pipeline,
            histogram_limits,
            reduction_limits,
            buffers: AddressRafBuffers {
                keys: buffer_from_slice(&self.device, &keys),
                lookups: buffer_from_slice(&self.device, &lookups),
                weights: buffer_from_slice(&self.device, weights),
                group_counts: self
                    .device
                    .new_buffer(group_bytes, MTLResourceOptions::StorageModeShared),
                group_offsets: self
                    .device
                    .new_buffer(group_bytes, MTLResourceOptions::StorageModeShared),
                bin_offsets: self
                    .device
                    .new_buffer(bin_offset_bytes, MTLResourceOptions::StorageModeShared),
                bucketed_contributions: self
                    .device
                    .new_buffer(contribution_bytes, MTLResourceOptions::StorageModeShared),
                output: self
                    .device
                    .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared),
                params: buffer_from_slice(&self.device, slice::from_ref(&params)),
            },
            rows: rows.len(),
            threadgroup_count,
            threads_per_threadgroup,
            completed: Cell::new(false),
        })
    }
}

impl AddressRafScanInvocation<'_> {
    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.histogram_limits
    }

    pub const fn reduction_pipeline_limits(&self) -> PipelineLimits {
        self.reduction_limits
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn threadgroup_count(&self) -> usize {
        self.threadgroup_count
    }

    pub const fn intermediate_contribution_bytes(&self) -> u64 {
        self.rows as u64 * size_of::<AddressRafContribution>() as u64
    }

    pub const fn input_bytes(&self) -> u64 {
        self.rows as u64
            * (size_of::<AddressRafLookup>() + size_of::<Fp128>() + size_of::<u16>()) as u64
    }

    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let histogram = command_buffer.new_compute_command_encoder();
            histogram.set_compute_pipeline_state(&self.histogram_pipeline);
            histogram.set_buffer(0, Some(&self.buffers.keys), 0);
            histogram.set_buffer(1, Some(&self.buffers.group_counts), 0);
            histogram.set_buffer(2, Some(&self.buffers.params), 0);
            histogram
                .set_threadgroup_memory_length(0, (ADDRESS_RAF_KEYS * size_of::<u32>()) as u64);
            histogram.dispatch_thread_groups(
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
            histogram.end_encoding();

            let offsets = command_buffer.new_compute_command_encoder();
            offsets.set_compute_pipeline_state(&self.offsets_pipeline);
            offsets.set_buffer(0, Some(&self.buffers.group_counts), 0);
            offsets.set_buffer(1, Some(&self.buffers.group_offsets), 0);
            offsets.set_buffer(2, Some(&self.buffers.bin_offsets), 0);
            offsets.set_buffer(3, Some(&self.buffers.params), 0);
            offsets.set_threadgroup_memory_length(0, (ADDRESS_RAF_KEYS * size_of::<u32>()) as u64);
            offsets.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: ADDRESS_RAF_KEYS as u64,
                    height: 1,
                    depth: 1,
                },
            );
            offsets.end_encoding();

            let scatter = command_buffer.new_compute_command_encoder();
            scatter.set_compute_pipeline_state(&self.scatter_pipeline);
            scatter.set_buffer(0, Some(&self.buffers.keys), 0);
            scatter.set_buffer(1, Some(&self.buffers.lookups), 0);
            scatter.set_buffer(2, Some(&self.buffers.weights), 0);
            scatter.set_buffer(3, Some(&self.buffers.group_offsets), 0);
            scatter.set_buffer(4, Some(&self.buffers.bucketed_contributions), 0);
            scatter.set_buffer(5, Some(&self.buffers.params), 0);
            scatter.set_threadgroup_memory_length(0, (ADDRESS_RAF_KEYS * size_of::<u32>()) as u64);
            scatter.dispatch_thread_groups(
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
            scatter.end_encoding();

            let reduce = command_buffer.new_compute_command_encoder();
            reduce.set_compute_pipeline_state(&self.reduction_pipeline);
            reduce.set_buffer(0, Some(&self.buffers.bucketed_contributions), 0);
            reduce.set_buffer(1, Some(&self.buffers.bin_offsets), 0);
            reduce.set_buffer(2, Some(&self.buffers.output), 0);
            reduce.set_buffer(3, Some(&self.buffers.params), 0);
            let simdgroups = self.threads_per_threadgroup / ADDRESS_RAF_SIMD_WIDTH;
            reduce.set_threadgroup_memory_length(
                0,
                (ADDRESS_RAF_PARTIAL_LANES * simdgroups * size_of::<Fp128>()) as u64,
            );
            reduce.dispatch_thread_groups(
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
            reduce.end_encoding();

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
        let mut values = vec![Fp128::ZERO; ADDRESS_RAF_OUTPUTS];
        self.read_output_into(&mut values)?;
        Ok(AddressRafSums { values })
    }

    pub fn read_output_into(&self, output: &mut [Fp128]) -> Result<(), MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        if output.len() != ADDRESS_RAF_OUTPUTS {
            return Err(MetalError::LengthMismatch {
                lhs: output.len(),
                rhs: ADDRESS_RAF_OUTPUTS,
            });
        }
        // SAFETY: the shared output buffer contains exactly `ADDRESS_RAF_OUTPUTS`
        // field values and the command buffer has completed.
        let values = unsafe {
            slice::from_raw_parts(
                self.buffers.output.contents().cast::<Fp128>(),
                ADDRESS_RAF_OUTPUTS,
            )
        };
        self.context.validate_inputs("address RAF output", values)?;
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

fn size_of_val_u64<T>(values: &[T]) -> Result<u64, MetalError> {
    u64::try_from(std::mem::size_of_val(values)).map_err(|_| MetalError::InputTooLong(values.len()))
}

const _: () = assert!(size_of::<AddressRafScanRow>() == 40);
const _: () = assert!(size_of::<AddressRafLookup>() == 16);
const _: () = assert!(size_of::<AddressRafContribution>() == 32);
const _: () = assert!(size_of::<AddressRafParams>() == 16);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::{AkitaField, FromPrimitiveInt};
    use jolt_lookup_tables::uninterleave_bits;

    use super::{
        AddressRafScanConfig, AddressRafScanRow, Fp128, SolinasMetal, ADDRESS_RAF_BINS,
        ADDRESS_RAF_OUTPUTS,
    };

    #[test]
    fn mixed_rows_match_jolt_field_at_every_phase_shape() {
        let mut state = 0xa11a_5eed_0123_4567;
        let mut rows = Vec::with_capacity(4099);
        let mut weights = Vec::with_capacity(4099);
        for index in 0..4099 {
            let lookup_index = match index {
                0 => 0,
                1 => u128::MAX,
                2 => (u64::MAX as u128) << 64,
                _ => (u128::from(splitmix(&mut state)) << 64) | u128::from(splitmix(&mut state)),
            };
            rows.push(AddressRafScanRow::new(lookup_index, index % 3 == 0));
            let value = u128::from(splitmix(&mut state))
                | (u128::from(splitmix(&mut state) & 0x7fff_ffff_ffff_ffff) << 64);
            weights.push(Fp128::from_u128(value));
        }

        let context = SolinasMetal::for_akita().unwrap();
        for suffix_len in [0, 8, 56, 64, 120] {
            let invocation = context
                .prepare_address_raf_scan(
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
            let actual = invocation.read_output().unwrap();
            let expected = oracle(&rows, &weights, suffix_len);
            let difference = actual
                .as_flat_slice()
                .iter()
                .zip(&expected)
                .position(|(actual, expected)| actual != expected);
            let values = difference.map(|index| (actual.as_flat_slice()[index], expected[index]));
            assert_eq!(
                difference, None,
                "suffix_len={suffix_len}, first difference={difference:?}, values={values:?}"
            );
        }
    }

    fn oracle(rows: &[AddressRafScanRow], weights: &[Fp128], suffix_len: u32) -> Vec<Fp128> {
        let mut sums = vec![AkitaField::zero(); ADDRESS_RAF_OUTPUTS];
        let suffix_mask = if suffix_len == 0 {
            0
        } else {
            (1u128 << suffix_len) - 1
        };
        let upper_bits = suffix_len.saturating_sub(64);
        for (&row, &weight) in rows.iter().zip(weights) {
            let lookup_index = row.lookup_index();
            let chunk = ((lookup_index >> suffix_len) as usize) & (ADDRESS_RAF_BINS - 1);
            let suffix = lookup_index & suffix_mask;
            let weight: AkitaField = weight.into_jolt_field();
            if row.raf_flag() {
                sums[3 * ADDRESS_RAF_BINS + chunk] += weight;
                sums[4 * ADDRESS_RAF_BINS + chunk] += weight * AkitaField::from_u128(suffix);
                let upper_mask = if upper_bits == 0 {
                    0
                } else {
                    (1u128 << upper_bits) - 1
                };
                if upper_bits == 0 || suffix >> 64 == upper_mask {
                    sums[5 * ADDRESS_RAF_BINS + chunk] += weight;
                }
            } else {
                let (left, right) = uninterleave_bits(suffix);
                sums[chunk] += weight;
                sums[ADDRESS_RAF_BINS + chunk] += weight * AkitaField::from_u64(left);
                sums[2 * ADDRESS_RAF_BINS + chunk] += weight * AkitaField::from_u64(right);
            }
        }
        sums.iter().map(Fp128::from_jolt_field).collect()
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}
