use std::{
    ffi::c_void,
    marker::PhantomData,
    mem::{size_of, size_of_val},
    slice,
    time::{Duration, Instant},
};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use metal::{objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};

use super::{
    REGISTERS_READ_WRITE_FIRST_MESSAGE_PIPELINE, REGISTERS_READ_WRITE_PAIRS_PER_GROUP,
    REGISTERS_READ_WRITE_REDUCTION_PIPELINE, REGISTERS_READ_WRITE_SIMD_WIDTH,
    REGISTERS_READ_WRITE_THREADGROUP_BYTES_MAX, REGISTERS_READ_WRITE_THREADS,
};
use crate::metal::solinas::{
    buffer_from_slice, completed_command_gpu_time, encode_column_reductions, set_inline_bytes,
    Fp128, MetalError, PipelineLimits, SolinasMetal,
};
use crate::optimized::registers_read_write::{
    PackedRegisterCycleRow, PackedRegisterRowsDeviceView, PACKED_REGISTER_ROWS_ALIGNMENT,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct FirstMessageParams {
    row_count: u32,
    pair_count: u32,
    output_stride: u32,
    e_in_length: u32,
}

const _: [(); 16] = [(); size_of::<FirstMessageParams>()];

struct FirstMessageBuffers {
    rows: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
    geometry_counts: Buffer,
    geometry_offsets: Buffer,
    geometry_masks: Buffer,
}

pub(crate) struct RegistersReadWriteFirstMessageInvocation<'a> {
    context: SolinasMetal,
    first_message_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    buffers: FirstMessageBuffers,
    params: FirstMessageParams,
    gamma: Fp128,
    gamma_sq: Fp128,
    threads: usize,
    limits: PipelineLimits,
    resident_bytes: usize,
    source_marker: PhantomData<&'a [PackedRegisterCycleRow]>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RegistersReadWriteFirstMessageObservation {
    pub(crate) quadratic: [AkitaField; 2],
    pub(crate) wall: Duration,
    pub(crate) gpu_active: Duration,
    pub(crate) limits: PipelineLimits,
    pub(crate) threads: usize,
    pub(crate) resident_bytes: usize,
    pub(crate) source_zero_copy: bool,
}

impl SolinasMetal {
    pub(crate) fn prepare_registers_read_write_first_message<'a>(
        &self,
        rows: PackedRegisterRowsDeviceView<'a>,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        gamma: AkitaField,
    ) -> Result<RegistersReadWriteFirstMessageInvocation<'a>, MetalError> {
        if rows.rows() == 0 || e_in.is_empty() || e_out.is_empty() {
            return Err(MetalError::EmptyInput);
        }
        if rows.active_registers() > 64 {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write Metal state has more than 64 active registers",
            ));
        }
        if !rows
            .as_ptr()
            .addr()
            .is_multiple_of(PACKED_REGISTER_ROWS_ALIGNMENT)
            || !rows
                .allocation_bytes()
                .is_multiple_of(PACKED_REGISTER_ROWS_ALIGNMENT)
            || rows.allocation_bytes()
                < rows
                    .rows()
                    .checked_mul(size_of::<PackedRegisterCycleRow>())
                    .ok_or(MetalError::InputTooLong(rows.rows()))?
        {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "packed source does not satisfy the no-copy allocation contract",
            ));
        }
        let pair_count = rows.rows().div_ceil(2);
        let partial_count = pair_count.div_ceil(REGISTERS_READ_WRITE_PAIRS_PER_GROUP);
        let row_count =
            u32::try_from(rows.rows()).map_err(|_| MetalError::InputTooLong(rows.rows()))?;
        let pair_count_u32 =
            u32::try_from(pair_count).map_err(|_| MetalError::InputTooLong(pair_count))?;
        let output_stride =
            u32::try_from(partial_count).map_err(|_| MetalError::InputTooLong(partial_count))?;
        let e_in_length =
            u32::try_from(e_in.len()).map_err(|_| MetalError::InputTooLong(e_in.len()))?;
        if !e_in.len().is_power_of_two() || !e_out.len().is_power_of_two() {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "split-equality tables must have power-of-two lengths",
            ));
        }

        let e_in = encode_fields(e_in);
        let e_out = encode_fields(e_out);
        self.validate_inputs("registers read-write e_in", &e_in)?;
        self.validate_inputs("registers read-write e_out", &e_out)?;
        let row_bytes = rows.allocation_bytes();
        let e_in_bytes = size_of_val(e_in.as_slice());
        let e_out_bytes = size_of_val(e_out.as_slice());
        let partial_bytes = partial_count
            .checked_mul(2)
            .and_then(|elements| elements.checked_mul(size_of::<Fp128>()))
            .ok_or(MetalError::InputTooLong(partial_count))?;
        let geometry_count_bytes = partial_count
            .checked_mul(size_of::<u32>())
            .ok_or(MetalError::InputTooLong(partial_count))?;
        let geometry_offset_bytes = pair_count
            .checked_mul(size_of::<u16>())
            .ok_or(MetalError::InputTooLong(pair_count))?;
        let geometry_mask_bytes = pair_count
            .checked_mul(size_of::<u64>())
            .ok_or(MetalError::InputTooLong(pair_count))?;
        let resident_bytes = row_bytes
            .checked_add(e_in_bytes)
            .and_then(|bytes| bytes.checked_add(e_out_bytes))
            .and_then(|bytes| bytes.checked_add(2 * partial_bytes))
            .and_then(|bytes| bytes.checked_add(geometry_count_bytes))
            .and_then(|bytes| bytes.checked_add(geometry_offset_bytes))
            .and_then(|bytes| bytes.checked_add(geometry_mask_bytes))
            .ok_or(MetalError::InputTooLong(rows.rows()))?;
        for bytes in [
            row_bytes,
            e_in_bytes,
            e_out_bytes,
            partial_bytes,
            geometry_count_bytes,
            geometry_offset_bytes,
            geometry_mask_bytes,
        ] {
            self.validate_buffer_length(
                u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(bytes))?,
            )?;
        }
        self.validate_additional_working_set(
            u64::try_from(resident_bytes).map_err(|_| MetalError::InputTooLong(resident_bytes))?,
        )?;

        let first_message_pipeline = self.compile_registers_read_write_source_pipeline(
            REGISTERS_READ_WRITE_FIRST_MESSAGE_PIPELINE,
            rows.remaps_registers(),
            false,
        )?;
        let reduction_pipeline =
            self.compile_named_pipeline(REGISTERS_READ_WRITE_REDUCTION_PIPELINE)?;
        let limits = Self::limits(&first_message_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        if limits.thread_execution_width != REGISTERS_READ_WRITE_SIMD_WIDTH
            || reduction_limits.thread_execution_width != REGISTERS_READ_WRITE_SIMD_WIDTH
        {
            return Err(MetalError::UnsupportedRegistersReadWriteExecutionWidth {
                expected: REGISTERS_READ_WRITE_SIMD_WIDTH,
                got: limits
                    .thread_execution_width
                    .min(reduction_limits.thread_execution_width),
            });
        }
        if limits.static_threadgroup_memory_length > REGISTERS_READ_WRITE_THREADGROUP_BYTES_MAX {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "first-message static threadgroup memory exceeds the registered limit",
            ));
        }
        let threads = Self::resolve_threadgroup_width(Some(REGISTERS_READ_WRITE_THREADS), limits)?;
        let partial_options = MTLResourceOptions::StorageModeShared;
        let source = self.device.new_buffer_with_bytes_no_copy(
            rows.as_ptr().cast_mut().cast::<c_void>(),
            row_bytes as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        let buffers = FirstMessageBuffers {
            rows: source,
            e_in: buffer_from_slice(&self.device, &e_in),
            e_out: buffer_from_slice(&self.device, &e_out),
            partial_a: self
                .device
                .new_buffer(partial_bytes as u64, partial_options),
            partial_b: self
                .device
                .new_buffer(partial_bytes as u64, partial_options),
            geometry_counts: self
                .device
                .new_buffer(geometry_count_bytes as u64, partial_options),
            geometry_offsets: self.device.new_buffer(
                geometry_offset_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            ),
            geometry_masks: self.device.new_buffer(
                geometry_mask_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            ),
        };
        Ok(RegistersReadWriteFirstMessageInvocation {
            context: self.clone(),
            first_message_pipeline,
            reduction_pipeline,
            buffers,
            params: FirstMessageParams {
                row_count,
                pair_count: pair_count_u32,
                output_stride,
                e_in_length,
            },
            gamma: Fp128::from_jolt_field(&gamma),
            gamma_sq: Fp128::from_jolt_field(&(gamma * gamma)),
            threads,
            limits,
            resident_bytes,
            source_marker: PhantomData,
        })
    }
}

impl RegistersReadWriteFirstMessageInvocation<'_> {
    pub(crate) fn execute(&self) -> Result<RegistersReadWriteFirstMessageObservation, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.first_message_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.rows), 0);
            encoder.set_buffer(1, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(3, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 4, &self.params);
            set_inline_bytes(encoder, 5, &self.gamma);
            set_inline_bytes(encoder, 6, &self.gamma_sq);
            encoder.set_buffer(7, Some(&self.buffers.geometry_counts), 0);
            encoder.set_buffer(8, Some(&self.buffers.geometry_offsets), 0);
            encoder.set_buffer(9, Some(&self.buffers.geometry_masks), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: u64::from(self.params.output_stride),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            let final_in_a = encode_column_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                self.params.output_stride as usize,
                2,
                REGISTERS_READ_WRITE_SIMD_WIDTH,
            )?;
            encoder.end_encoding();
            let started = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let wall = started.elapsed();
            let gpu_active = completed_command_gpu_time(command_buffer)?;
            let output = if final_in_a {
                &self.buffers.partial_a
            } else {
                &self.buffers.partial_b
            };
            // SAFETY: the completed two-column reduction writes one field per
            // column at the front of the selected shared buffer.
            let values = unsafe { slice::from_raw_parts(output.contents().cast::<Fp128>(), 2) };
            self.context
                .validate_inputs("registers read-write first message", values)?;
            Ok(RegistersReadWriteFirstMessageObservation {
                quadratic: [values[0].into_jolt_field(), values[1].into_jolt_field()],
                wall,
                gpu_active,
                limits: self.limits,
                threads: self.threads,
                resident_bytes: self.resident_bytes,
                source_zero_copy: true,
            })
        })
    }
}

fn encode_fields(values: &[AkitaField]) -> Vec<Fp128> {
    values.iter().map(Fp128::from_jolt_field).collect()
}
