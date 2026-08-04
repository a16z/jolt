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
const ADDRESS_RAF_OUTPUTS: usize = ADDRESS_RAF_LANES * ADDRESS_RAF_BINS;
const ADDRESS_RAF_SIMD_WIDTH: usize = 32;
const SCAN_PIPELINE: &str = "solinas_address_raf_scan";
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
    pub rows_per_simdgroup: usize,
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for AddressRafScanConfig {
    fn default() -> Self {
        Self {
            suffix_len: 120,
            rows_per_simdgroup: 1 << 16,
            threads_per_threadgroup: Some(128),
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
    rows_per_simdgroup: u32,
    simdgroup_count: u32,
}

struct AddressRafBuffers {
    rows: Buffer,
    weights: Buffer,
    partials: Buffer,
    output: Buffer,
    params: Buffer,
}

pub struct AddressRafScanInvocation<'a> {
    context: &'a SolinasMetal,
    scan_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    scan_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: AddressRafBuffers,
    rows: usize,
    simdgroup_count: usize,
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
        if config.rows_per_simdgroup == 0
            || !config
                .rows_per_simdgroup
                .is_multiple_of(ADDRESS_RAF_SIMD_WIDTH)
        {
            return Err(MetalError::InvalidAddressRafRowsPerSimdgroup(
                config.rows_per_simdgroup,
            ));
        }
        self.validate_inputs("address RAF weights", weights)?;

        let row_count =
            u32::try_from(rows.len()).map_err(|_| MetalError::InputTooLong(rows.len()))?;
        let rows_per_simdgroup = u32::try_from(config.rows_per_simdgroup)
            .map_err(|_| MetalError::InputTooLong(config.rows_per_simdgroup))?;
        let simdgroup_count = rows.len().div_ceil(config.rows_per_simdgroup);
        let params = AddressRafParams {
            rows: row_count,
            suffix_len: config.suffix_len,
            rows_per_simdgroup,
            simdgroup_count: u32::try_from(simdgroup_count)
                .map_err(|_| MetalError::InputTooLong(simdgroup_count))?,
        };

        let scan_pipeline = self.compile_named_pipeline(SCAN_PIPELINE)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let scan_limits = Self::limits(&scan_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (SCAN_PIPELINE, scan_limits),
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
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, scan_limits)?;

        let partial_elements = simdgroup_count
            .checked_mul(ADDRESS_RAF_OUTPUTS)
            .ok_or(MetalError::InputTooLong(simdgroup_count))?;
        let partial_bytes = byte_length::<Fp128>(partial_elements)?;
        let output_bytes = byte_length::<Fp128>(ADDRESS_RAF_OUTPUTS)?;
        for requested in [
            size_of_val_u64(rows)?,
            size_of_val_u64(weights)?,
            partial_bytes,
            output_bytes,
        ] {
            let maximum = self.device.max_buffer_length();
            if requested > maximum {
                return Err(MetalError::BufferTooLong { requested, maximum });
            }
        }

        Ok(AddressRafScanInvocation {
            context: self,
            scan_pipeline,
            reduction_pipeline,
            scan_limits,
            reduction_limits,
            buffers: AddressRafBuffers {
                rows: buffer_from_slice(&self.device, rows),
                weights: buffer_from_slice(&self.device, weights),
                partials: self
                    .device
                    .new_buffer(partial_bytes, MTLResourceOptions::StorageModeShared),
                output: self
                    .device
                    .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared),
                params: buffer_from_slice(&self.device, slice::from_ref(&params)),
            },
            rows: rows.len(),
            simdgroup_count,
            threads_per_threadgroup,
            completed: Cell::new(false),
        })
    }
}

impl AddressRafScanInvocation<'_> {
    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.scan_limits
    }

    pub const fn reduction_pipeline_limits(&self) -> PipelineLimits {
        self.reduction_limits
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn simdgroup_count(&self) -> usize {
        self.simdgroup_count
    }

    pub const fn input_bytes(&self) -> u64 {
        self.rows as u64 * (size_of::<AddressRafScanRow>() + size_of::<Fp128>()) as u64
    }

    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let scan = command_buffer.new_compute_command_encoder();
            scan.set_compute_pipeline_state(&self.scan_pipeline);
            scan.set_buffer(0, Some(&self.buffers.rows), 0);
            scan.set_buffer(1, Some(&self.buffers.weights), 0);
            scan.set_buffer(2, Some(&self.buffers.partials), 0);
            scan.set_buffer(3, Some(&self.buffers.params), 0);
            let simdgroups_per_threadgroup = self.threads_per_threadgroup / ADDRESS_RAF_SIMD_WIDTH;
            scan.dispatch_thread_groups(
                MTLSize {
                    width: self.simdgroup_count.div_ceil(simdgroups_per_threadgroup) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            scan.end_encoding();

            let reduce = command_buffer.new_compute_command_encoder();
            reduce.set_compute_pipeline_state(&self.reduction_pipeline);
            reduce.set_buffer(0, Some(&self.buffers.partials), 0);
            reduce.set_buffer(1, Some(&self.buffers.output), 0);
            reduce.set_buffer(2, Some(&self.buffers.params), 0);
            let reduction_width = (self.reduction_limits.thread_execution_width * 8)
                .min(self.reduction_limits.max_total_threads_per_threadgroup);
            reduce.dispatch_thread_groups(
                MTLSize {
                    width: ADDRESS_RAF_OUTPUTS.div_ceil(reduction_width) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: reduction_width as u64,
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
                        rows_per_simdgroup: 64,
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
