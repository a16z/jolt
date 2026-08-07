use std::{
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use metal::{
    objc::rc::autoreleasepool, Buffer, CommandBuffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize, NSRange,
};

use super::super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits,
    RamRafAddressPlane, SolinasMetal,
};
use super::abi::{MESSAGE_COLUMNS, REDUCE_PIPELINE, SIMD_WIDTH, SPARSE_FIRST_MESSAGE_PIPELINE};
use super::{RamValActivePair, RamValReductionParams, RamValSparseFirstMessageParams};

struct RamValSparseBuffers {
    active_pairs: Buffer,
    eq_address: Buffer,
    lt_low: Buffer,
    lt_high: Buffer,
    eq_high: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
    status: Buffer,
}

pub struct RamValSparseFirstMessage {
    context: SolinasMetal,
    first_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    first_limits: PipelineLimits,
    reduce_limits: PipelineLimits,
    addresses: RamRafAddressPlane,
    buffers: RamValSparseBuffers,
    params: RamValSparseFirstMessageParams,
    groups: usize,
}

struct RamValSparseCommand {
    command: CommandBuffer,
    submitted_at: Instant,
    submit_wall: Duration,
    final_in_a: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamValSparseFirstMessageStats {
    pub submit_wall: Duration,
    pub overlap_wall: Duration,
    pub join_wall: Duration,
    pub lifecycle_wall: Duration,
    pub gpu_active: Duration,
    pub completed_before_join: bool,
    pub active_pairs: usize,
    pub address_storage_id: usize,
}

#[must_use = "a submitted RAM value-check shadow must be joined"]
pub struct PendingRamValSparseFirstMessage {
    invocation: Option<RamValSparseFirstMessage>,
    command: Option<RamValSparseCommand>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingRamValSparseFirstMessage {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(invocation) = &self.invocation {
            visitor.visit_simple(
                allocative::Key::new("device_scratch"),
                invocation.resident_bytes(),
            );
        }
        visitor.exit();
    }
}

impl Drop for PendingRamValSparseFirstMessage {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.command.wait_until_completed();
        }
    }
}

impl SolinasMetal {
    pub fn prepare_ram_val_sparse_first_message(
        &self,
        active_pairs: &[RamValActivePair],
        addresses: RamRafAddressPlane,
        eq_address: &[AkitaField],
        lt_low: &[AkitaField],
        lt_high: &[AkitaField],
        eq_high: &[AkitaField],
    ) -> Result<RamValSparseFirstMessage, MetalError> {
        let rows = addresses.rows();
        let params = RamValSparseFirstMessageParams::new(
            active_pairs.len(),
            rows,
            lt_low.len(),
            addresses.address_domain(),
        )
        .map_err(|_| MetalError::InvalidRamValCheckState("invalid sparse first-message shape"))?;
        if eq_address.len() != addresses.address_domain()
            || lt_high.len() != rows / lt_low.len()
            || eq_high.len() != lt_high.len()
            || active_pairs
                .iter()
                .any(|pair| pair.pair_index() >= rows / 2)
        {
            return Err(MetalError::InvalidRamValCheckState(
                "sparse first-message inputs have inconsistent lengths",
            ));
        }
        let groups = active_pairs.len().div_ceil(SIMD_WIDTH);
        let partial_fields = MESSAGE_COLUMNS
            .checked_mul(groups)
            .ok_or(MetalError::InputTooLong(groups))?;
        let scratch_bytes = active_pairs
            .len()
            .checked_mul(size_of::<RamValActivePair>())
            .and_then(|bytes| {
                bytes.checked_add(
                    (eq_address.len() + lt_low.len() + lt_high.len() + eq_high.len())
                        * size_of::<Fp128>(),
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(2 * partial_fields * size_of::<Fp128>() + size_of::<u32>())
            })
            .ok_or(MetalError::InputTooLong(partial_fields))?;
        self.validate_additional_working_set(
            u64::try_from(scratch_bytes).map_err(|_| MetalError::InputTooLong(scratch_bytes))?,
        )?;

        let first_pipeline = self.compile_named_pipeline(SPARSE_FIRST_MESSAGE_PIPELINE)?;
        let reduce_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let first_limits = Self::limits(&first_pipeline);
        let reduce_limits = Self::limits(&reduce_pipeline);
        if first_limits.thread_execution_width != SIMD_WIDTH
            || reduce_limits.thread_execution_width != SIMD_WIDTH
        {
            return Err(MetalError::UnsupportedRamValCheckExecutionWidth {
                pipeline: SPARSE_FIRST_MESSAGE_PIPELINE,
                expected: SIMD_WIDTH,
                got: first_limits.thread_execution_width,
            });
        }

        let encoded_eq_address = encode_fields(self, "RAM value sparse address eq", eq_address)?;
        let encoded_lt_low = encode_fields(self, "RAM value sparse LT low", lt_low)?;
        let encoded_lt_high = encode_fields(self, "RAM value sparse LT high", lt_high)?;
        let encoded_eq_high = encode_fields(self, "RAM value sparse EQ high", eq_high)?;
        let partial_bytes = u64::try_from(partial_fields * size_of::<Fp128>())
            .map_err(|_| MetalError::InputTooLong(partial_fields))?;

        Ok(RamValSparseFirstMessage {
            context: self.clone(),
            first_pipeline,
            reduce_pipeline,
            first_limits,
            reduce_limits,
            addresses,
            buffers: RamValSparseBuffers {
                active_pairs: buffer_from_slice(&self.device, active_pairs),
                eq_address: buffer_from_slice(&self.device, &encoded_eq_address),
                lt_low: buffer_from_slice(&self.device, &encoded_lt_low),
                lt_high: buffer_from_slice(&self.device, &encoded_lt_high),
                eq_high: buffer_from_slice(&self.device, &encoded_eq_high),
                partial_a: self
                    .device
                    .new_buffer(partial_bytes, MTLResourceOptions::StorageModeShared),
                partial_b: self
                    .device
                    .new_buffer(partial_bytes, MTLResourceOptions::StorageModeShared),
                status: self.device.new_buffer(
                    size_of::<u32>() as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
            },
            params,
            groups,
        })
    }
}

impl RamValSparseFirstMessage {
    pub const fn first_pipeline_limits(&self) -> PipelineLimits {
        self.first_limits
    }

    pub const fn reduce_pipeline_limits(&self) -> PipelineLimits {
        self.reduce_limits
    }

    pub const fn active_pairs(&self) -> usize {
        self.params.active_pairs as usize
    }

    pub const fn address_storage_id(&self) -> usize {
        self.addresses.storage_id()
    }

    pub fn resident_bytes(&self) -> usize {
        self.buffers.active_pairs.length() as usize
            + self.buffers.eq_address.length() as usize
            + self.buffers.lt_low.length() as usize
            + self.buffers.lt_high.length() as usize
            + self.buffers.eq_high.length() as usize
            + self.buffers.partial_a.length() as usize
            + self.buffers.partial_b.length() as usize
            + self.buffers.status.length() as usize
    }

    pub fn submit(self) -> PendingRamValSparseFirstMessage {
        let command = self.encode_and_submit();
        PendingRamValSparseFirstMessage {
            invocation: Some(self),
            command: Some(command),
        }
    }

    fn encode_and_submit(&self) -> RamValSparseCommand {
        let submitted_at = Instant::now();
        let command = self.context.queue.new_command_buffer().to_owned();
        let final_in_a = autoreleasepool(|| {
            let blit = command.new_blit_command_encoder();
            blit.fill_buffer(
                &self.buffers.status,
                NSRange::new(0, self.buffers.status.length()),
                0,
            );
            blit.end_encoding();

            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.first_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.active_pairs), 0);
            encoder.set_buffer(1, Some(self.addresses.buffer()), 0);
            encoder.set_buffer(2, Some(&self.buffers.eq_address), 0);
            encoder.set_buffer(3, Some(&self.buffers.lt_low), 0);
            encoder.set_buffer(4, Some(&self.buffers.lt_high), 0);
            encoder.set_buffer(5, Some(&self.buffers.eq_high), 0);
            encoder.set_buffer(6, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 7, &self.params);
            encoder.set_buffer(8, Some(&self.buffers.status), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: SIMD_WIDTH as u64,
                    height: 1,
                    depth: 1,
                },
            );
            let final_in_a = encode_reductions(
                encoder,
                &self.reduce_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                &self.buffers.status,
                self.groups,
            );
            encoder.end_encoding();
            final_in_a
        });
        let submit_started = Instant::now();
        command.commit();
        RamValSparseCommand {
            command,
            submitted_at,
            submit_wall: submit_started.elapsed(),
            final_in_a,
        }
    }
}

impl PendingRamValSparseFirstMessage {
    pub fn join(
        mut self,
    ) -> Result<([AkitaField; MESSAGE_COLUMNS], RamValSparseFirstMessageStats), MetalError> {
        let invocation = self.invocation.take().ok_or(MetalError::NotExecuted)?;
        let command = self.command.take().ok_or(MetalError::NotExecuted)?;
        let join_started = Instant::now();
        let completed_before_join = command.command.status() == MTLCommandBufferStatus::Completed;
        let overlap_wall = join_started.saturating_duration_since(command.submitted_at);
        command.command.wait_until_completed();
        let join_wall = join_started.elapsed();
        if command.command.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(command.command.status()));
        }
        // SAFETY: `status` is a shared four-byte buffer retained through completion.
        let status = unsafe { *invocation.buffers.status.contents().cast::<u32>() };
        if status != 0 {
            return Err(MetalError::InvalidRamValCheckState(
                "sparse first-message shader rejected its inputs",
            ));
        }
        let start = command_buffer_timestamp(&command.command, "GPUStartTime")?;
        let end = command_buffer_timestamp(&command.command, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        let output = if command.final_in_a {
            &invocation.buffers.partial_a
        } else {
            &invocation.buffers.partial_b
        };
        // SAFETY: both scratch buffers contain at least three Fp128 values and
        // remain alive until this slice is consumed.
        let values =
            unsafe { slice::from_raw_parts(output.contents().cast::<Fp128>(), MESSAGE_COLUMNS) };
        invocation
            .context
            .validate_inputs("RAM value sparse first message", values)?;
        let message = std::array::from_fn(|index| values[index].into_jolt_field());
        Ok((
            message,
            RamValSparseFirstMessageStats {
                submit_wall: command.submit_wall,
                overlap_wall,
                join_wall,
                lifecycle_wall: command.submitted_at.elapsed(),
                gpu_active: Duration::from_secs_f64(end - start),
                completed_before_join,
                active_pairs: invocation.active_pairs(),
                address_storage_id: invocation.address_storage_id(),
            },
        ))
    }
}

fn encode_reductions(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    partial_a: &Buffer,
    partial_b: &Buffer,
    status: &Buffer,
    mut input_count: usize,
) -> bool {
    let mut input_a = true;
    while input_count > 1 {
        let output_count = input_count.div_ceil(SIMD_WIDTH);
        let params = RamValReductionParams {
            input_count: input_count as u32,
            output_count: output_count as u32,
            columns: MESSAGE_COLUMNS as u32,
            reserved: 0,
        };
        encoder.set_compute_pipeline_state(pipeline);
        let (input, output) = if input_a {
            (partial_a, partial_b)
        } else {
            (partial_b, partial_a)
        };
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        set_inline_bytes(encoder, 2, &params);
        encoder.set_buffer(3, Some(status), 0);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: output_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: SIMD_WIDTH as u64,
                height: 1,
                depth: 1,
            },
        );
        input_count = output_count;
        input_a = !input_a;
    }
    input_a
}

fn encode_fields(
    context: &SolinasMetal,
    label: &'static str,
    fields: &[AkitaField],
) -> Result<Vec<Fp128>, MetalError> {
    let encoded = fields
        .iter()
        .map(Fp128::from_jolt_field)
        .collect::<Vec<_>>();
    context.validate_inputs(label, &encoded)?;
    Ok(encoded)
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast(),
    );
}
