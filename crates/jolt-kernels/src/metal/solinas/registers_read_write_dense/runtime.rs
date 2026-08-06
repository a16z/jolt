use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use super::super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};
use super::{
    RegistersRwDenseAbiError, RegistersRwDensePhaseParams, RegistersRwDenseReductionParams,
    RegistersRwDenseStateWords, DENSE_BIND_MESSAGE_PIPELINE, REDUCE_PIPELINE,
    REGISTERS_RW_DENSE_COLUMNS, REGISTERS_RW_DENSE_SIMD_WIDTH, REGISTERS_RW_DENSE_THREADS,
};

const MESSAGE_FIELDS: usize = 2;

struct RegistersRwDenseBuffers {
    source_state: Buffer,
    source_inc: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    destination_state: Buffer,
    destination_inc: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersRwDenseRoundStorage {
    pub source_state_bytes: usize,
    pub source_inc_bytes: usize,
    pub e_in_bytes: usize,
    pub e_out_bytes: usize,
    pub destination_state_bytes: usize,
    pub destination_inc_bytes: usize,
    pub partial_a_bytes: usize,
    pub partial_b_bytes: usize,
    pub total_bytes: usize,
}

pub struct RegistersRwDenseRoundInvocation {
    context: SolinasMetal,
    message_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    message_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: RegistersRwDenseBuffers,
    params: RegistersRwDensePhaseParams,
    reduction_steps: Vec<RegistersRwDenseReductionParams>,
    challenge: Fp128,
    threads: usize,
    dynamic_threadgroup_bytes: usize,
    final_in_a: bool,
    storage: RegistersRwDenseRoundStorage,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub fn prepare_registers_rw_dense_round(
        &self,
        source_state: &[RegistersRwDenseStateWords],
        source_inc: &[AkitaField],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        challenge: AkitaField,
    ) -> Result<RegistersRwDenseRoundInvocation, MetalError> {
        let params = RegistersRwDensePhaseParams::new(source_inc.len(), e_in.len(), e_out.len())?;
        let expected_state = source_inc
            .len()
            .checked_mul(REGISTERS_RW_DENSE_COLUMNS)
            .ok_or(MetalError::InputTooLong(source_inc.len()))?;
        if source_state.len() != expected_state {
            return Err(RegistersRwDenseAbiError::StateLength {
                expected: expected_state,
                got: source_state.len(),
            }
            .into());
        }

        let source_fields = dense_state_fields(source_state)?;
        self.validate_inputs("registers read/write dense source", source_fields)?;
        let source_inc = encode_fields(self, "registers read/write dense increment", source_inc)?;
        let e_in = encode_fields(self, "registers read/write dense e_in", e_in)?;
        let e_out = encode_fields(self, "registers read/write dense e_out", e_out)?;
        let challenge = Fp128::from_jolt_field(&challenge);
        self.validate_inputs(
            "registers read/write dense challenge",
            slice::from_ref(&challenge),
        )?;

        let message_pipeline = self.compile_named_pipeline(DENSE_BIND_MESSAGE_PIPELINE)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let message_limits = Self::limits(&message_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (DENSE_BIND_MESSAGE_PIPELINE, message_limits),
            (REDUCE_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != REGISTERS_RW_DENSE_SIMD_WIDTH {
                return Err(MetalError::UnsupportedRegistersReadWriteExecutionWidth {
                    pipeline,
                    expected: REGISTERS_RW_DENSE_SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let threads =
            Self::resolve_threadgroup_width(Some(REGISTERS_RW_DENSE_THREADS), message_limits)?;
        let dynamic_threadgroup_bytes = RegistersRwDensePhaseParams::threadgroup_bytes(threads)?;
        let total_threadgroup_bytes = u64::try_from(dynamic_threadgroup_bytes)
            .ok()
            .and_then(|dynamic| {
                dynamic.checked_add(message_limits.static_threadgroup_memory_length)
            })
            .ok_or(MetalError::InputTooLong(dynamic_threadgroup_bytes))?;
        if total_threadgroup_bytes > self.device_info().max_threadgroup_memory_length {
            return Err(MetalError::RegistersReadWriteThreadgroupMemory {
                requested: total_threadgroup_bytes,
                maximum: self.device_info().max_threadgroup_memory_length,
            });
        }

        let (reduction_steps, final_in_a) = reduction_steps(e_out.len())?;
        let storage = round_storage(&params)?;
        validate_storage(self, storage)?;
        let buffers = RegistersRwDenseBuffers {
            source_state: buffer_from_slice(&self.device, source_state),
            source_inc: buffer_from_slice(&self.device, &source_inc),
            e_in: buffer_from_slice(&self.device, &e_in),
            e_out: buffer_from_slice(&self.device, &e_out),
            destination_state: new_buffer(self, storage.destination_state_bytes)?,
            destination_inc: new_buffer(self, storage.destination_inc_bytes)?,
            partial_a: new_buffer(self, storage.partial_a_bytes)?,
            partial_b: new_buffer(self, storage.partial_b_bytes)?,
        };
        if buffers.source_state.as_ptr() == buffers.destination_state.as_ptr()
            || buffers.source_inc.as_ptr() == buffers.destination_inc.as_ptr()
        {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "dense source and destination allocations alias",
            ));
        }

        Ok(RegistersRwDenseRoundInvocation {
            context: self.clone(),
            message_pipeline,
            reduction_pipeline,
            message_limits,
            reduction_limits,
            buffers,
            params,
            reduction_steps,
            challenge,
            threads,
            dynamic_threadgroup_bytes,
            final_in_a,
            storage,
            completed: Cell::new(false),
        })
    }
}

impl RegistersRwDenseRoundInvocation {
    pub const fn source_rows(&self) -> usize {
        self.params.source_rows as usize
    }

    pub const fn destination_rows(&self) -> usize {
        self.params.destination_rows as usize
    }

    pub const fn storage(&self) -> RegistersRwDenseRoundStorage {
        self.storage
    }

    pub const fn message_pipeline_limits(&self) -> PipelineLimits {
        self.message_limits
    }

    pub const fn reduction_pipeline_limits(&self) -> PipelineLimits {
        self.reduction_limits
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads
    }

    pub const fn dynamic_threadgroup_memory_bytes(&self) -> usize {
        self.dynamic_threadgroup_bytes
    }

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn source_state_allocation_identity(&self) -> usize {
        self.buffers.source_state.as_ptr() as usize
    }

    pub fn destination_state_allocation_identity(&self) -> usize {
        self.buffers.destination_state.as_ptr() as usize
    }

    pub fn execute(&self) -> Result<[AkitaField; MESSAGE_FIELDS], MetalError> {
        self.execute_timed().map(|(message, _)| message)
    }

    pub fn execute_timed(&self) -> Result<([AkitaField; MESSAGE_FIELDS], Duration), MetalError> {
        self.completed.set(false);
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.message_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.source_state), 0);
            encoder.set_buffer(1, Some(&self.buffers.source_inc), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
            set_inline_bytes(encoder, 4, &self.challenge);
            encoder.set_buffer(5, Some(&self.buffers.destination_state), 0);
            encoder.set_buffer(6, Some(&self.buffers.destination_inc), 0);
            encoder.set_buffer(7, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 8, &self.params);
            encoder.set_threadgroup_memory_length(0, self.dynamic_threadgroup_bytes as u64);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.params.e_out_length as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );

            let mut input_a = true;
            for step in &self.reduction_steps {
                encoder.set_compute_pipeline_state(&self.reduction_pipeline);
                let (input, output) = if input_a {
                    (&self.buffers.partial_a, &self.buffers.partial_b)
                } else {
                    (&self.buffers.partial_b, &self.buffers.partial_a)
                };
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                set_inline_bytes(encoder, 2, step);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: step.output_count as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: REGISTERS_RW_DENSE_SIMD_WIDTH as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                input_a = !input_a;
            }
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(command_buffer.status()));
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end });
            }
            self.completed.set(true);
            Ok((self.read_message()?, Duration::from_secs_f64(end - start)))
        })
    }

    pub fn read_destination_state(&self) -> Result<Vec<RegistersRwDenseStateWords>, MetalError> {
        self.require_completed()?;
        let entries = self.destination_rows() * REGISTERS_RW_DENSE_COLUMNS;
        // SAFETY: the shared destination buffer owns exactly `entries` values
        // and the command completed before this host read.
        let values = unsafe {
            slice::from_raw_parts(
                self.buffers
                    .destination_state
                    .contents()
                    .cast::<RegistersRwDenseStateWords>(),
                entries,
            )
        };
        self.context.validate_inputs(
            "registers read/write dense destination",
            dense_state_fields(values)?,
        )?;
        Ok(values.to_vec())
    }

    pub fn read_destination_increment(&self) -> Result<Vec<AkitaField>, MetalError> {
        self.require_completed()?;
        // SAFETY: the shared destination buffer owns exactly `destination_rows`
        // fields and the command completed before this host read.
        let values = unsafe {
            slice::from_raw_parts(
                self.buffers.destination_inc.contents().cast::<Fp128>(),
                self.destination_rows(),
            )
        };
        self.context
            .validate_inputs("registers read/write dense destination increment", values)?;
        Ok(values
            .iter()
            .map(|&value| value.into_jolt_field())
            .collect())
    }

    fn read_message(&self) -> Result<[AkitaField; MESSAGE_FIELDS], MetalError> {
        self.require_completed()?;
        let buffer = if self.final_in_a {
            &self.buffers.partial_a
        } else {
            &self.buffers.partial_b
        };
        // SAFETY: the final reduction leaves two fields at the front of the
        // selected shared buffer and the command completed before this read.
        let values =
            unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), MESSAGE_FIELDS) };
        self.context
            .validate_inputs("registers read/write dense message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }

    fn require_completed(&self) -> Result<(), MetalError> {
        if self.completed.get() {
            Ok(())
        } else {
            Err(MetalError::NotExecuted)
        }
    }
}

fn dense_state_fields(state: &[RegistersRwDenseStateWords]) -> Result<&[Fp128], MetalError> {
    let fields = state
        .len()
        .checked_mul(3)
        .ok_or(MetalError::InputTooLong(state.len()))?;
    // SAFETY: `RegistersRwDenseStateWords` is repr(C), consists of exactly
    // three aligned Fp128 values, and its ABI assertions fix the size at 48.
    Ok(unsafe { slice::from_raw_parts(state.as_ptr().cast::<Fp128>(), fields) })
}

fn encode_fields(
    context: &SolinasMetal,
    name: &'static str,
    fields: &[AkitaField],
) -> Result<Vec<Fp128>, MetalError> {
    let encoded = fields
        .iter()
        .map(Fp128::from_jolt_field)
        .collect::<Vec<_>>();
    context.validate_inputs(name, &encoded)?;
    Ok(encoded)
}

fn reduction_steps(
    mut input_count: usize,
) -> Result<(Vec<RegistersRwDenseReductionParams>, bool), MetalError> {
    let mut steps = Vec::new();
    let mut input_a = true;
    while input_count > 1 {
        let params = RegistersRwDenseReductionParams::new(input_count)?;
        input_count = params.output_count as usize;
        steps.push(params);
        input_a = !input_a;
    }
    Ok((steps, input_a))
}

fn round_storage(
    params: &RegistersRwDensePhaseParams,
) -> Result<RegistersRwDenseRoundStorage, MetalError> {
    let source_rows = params.source_rows as usize;
    let destination_rows = params.destination_rows as usize;
    let e_in = params.e_in_length as usize;
    let e_out = params.e_out_length as usize;
    let source_state_bytes = bytes::<RegistersRwDenseStateWords>(
        source_rows
            .checked_mul(REGISTERS_RW_DENSE_COLUMNS)
            .ok_or(MetalError::InputTooLong(source_rows))?,
    )?;
    let source_inc_bytes = bytes::<Fp128>(source_rows)?;
    let e_in_bytes = bytes::<Fp128>(e_in)?;
    let e_out_bytes = bytes::<Fp128>(e_out)?;
    let destination_state_bytes = bytes::<RegistersRwDenseStateWords>(
        destination_rows
            .checked_mul(REGISTERS_RW_DENSE_COLUMNS)
            .ok_or(MetalError::InputTooLong(destination_rows))?,
    )?;
    let destination_inc_bytes = bytes::<Fp128>(destination_rows)?;
    let partial_a_bytes = bytes::<Fp128>(
        MESSAGE_FIELDS
            .checked_mul(e_out)
            .ok_or(MetalError::InputTooLong(e_out))?,
    )?;
    let partial_b_bytes = bytes::<Fp128>(
        MESSAGE_FIELDS
            .checked_mul(e_out.div_ceil(REGISTERS_RW_DENSE_SIMD_WIDTH))
            .ok_or(MetalError::InputTooLong(e_out))?,
    )?;
    let parts = [
        source_state_bytes,
        source_inc_bytes,
        e_in_bytes,
        e_out_bytes,
        destination_state_bytes,
        destination_inc_bytes,
        partial_a_bytes,
        partial_b_bytes,
    ];
    let total_bytes = parts
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(MetalError::InputTooLong(source_rows))?;
    Ok(RegistersRwDenseRoundStorage {
        source_state_bytes,
        source_inc_bytes,
        e_in_bytes,
        e_out_bytes,
        destination_state_bytes,
        destination_inc_bytes,
        partial_a_bytes,
        partial_b_bytes,
        total_bytes,
    })
}

fn validate_storage(
    context: &SolinasMetal,
    storage: RegistersRwDenseRoundStorage,
) -> Result<(), MetalError> {
    for bytes in [
        storage.source_state_bytes,
        storage.source_inc_bytes,
        storage.e_in_bytes,
        storage.e_out_bytes,
        storage.destination_state_bytes,
        storage.destination_inc_bytes,
        storage.partial_a_bytes,
        storage.partial_b_bytes,
    ] {
        context.validate_buffer_length(
            u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(bytes))?,
        )?;
    }
    context.validate_additional_working_set(
        u64::try_from(storage.total_bytes)
            .map_err(|_| MetalError::InputTooLong(storage.total_bytes))?,
    )
}

fn bytes<T>(elements: usize) -> Result<usize, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .ok_or(MetalError::InputTooLong(elements))
}

fn new_buffer(context: &SolinasMetal, bytes: usize) -> Result<Buffer, MetalError> {
    let bytes = u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(bytes))?;
    context.validate_buffer_length(bytes)?;
    Ok(context
        .device
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal dense-round parity setup")]
mod tests {
    use jolt_field::AkitaField;

    use super::*;

    type DenseState = [AkitaField; 3];

    #[test]
    fn log26_round9_storage_matches_the_frozen_accounting() {
        let params = RegistersRwDensePhaseParams::new(1 << 18, 8, 1 << 13).unwrap();
        let storage = round_storage(&params).unwrap();
        assert_eq!(storage.source_state_bytes, 1_610_612_736);
        assert_eq!(storage.source_inc_bytes, 4_194_304);
        assert_eq!(storage.e_in_bytes, 128);
        assert_eq!(storage.e_out_bytes, 131_072);
        assert_eq!(storage.destination_state_bytes, 805_306_368);
        assert_eq!(storage.destination_inc_bytes, 2_097_152);
        assert_eq!(storage.partial_a_bytes, 262_144);
        assert_eq!(storage.partial_b_bytes, 8_192);
        assert_eq!(storage.total_bytes, 2_422_612_096);
    }

    #[test]
    fn dense_round_matches_an_unfactored_scalar_oracle() {
        let context = SolinasMetal::for_akita().unwrap();
        let source_rows = 8;
        let source = (0..source_rows * REGISTERS_RW_DENSE_COLUMNS)
            .map(|index| {
                let row = index / REGISTERS_RW_DENSE_COLUMNS;
                let column = index % REGISTERS_RW_DENSE_COLUMNS;
                [
                    AkitaField::from_u64((17 * row + 3 * column + 1) as u64),
                    AkitaField::from_u64((11 * row + 5 * column + 2) as u64),
                    AkitaField::from_u64((7 * row + 13 * column + 4) as u64),
                ]
            })
            .collect::<Vec<_>>();
        let source_words = source.iter().copied().map(encode_state).collect::<Vec<_>>();
        let source_inc = (0..source_rows)
            .map(|row| AkitaField::from_u64((19 * row + 6) as u64))
            .collect::<Vec<_>>();
        let e_in = [AkitaField::from_u64(23)];
        let e_out = [AkitaField::from_u64(29), AkitaField::from_u64(31)];

        for challenge in [
            AkitaField::zero(),
            AkitaField::one(),
            -AkitaField::one(),
            AkitaField::from_u64(0xfeed_beef),
        ] {
            let (expected_state, expected_inc, expected_message) =
                scalar_round(&source, &source_inc, &e_in, &e_out, challenge);
            let invocation = context
                .prepare_registers_rw_dense_round(
                    &source_words,
                    &source_inc,
                    &e_in,
                    &e_out,
                    challenge,
                )
                .unwrap();
            assert!(matches!(
                invocation.read_destination_state(),
                Err(MetalError::NotExecuted)
            ));
            assert_ne!(
                invocation.source_state_allocation_identity(),
                invocation.destination_state_allocation_identity()
            );
            assert_eq!(invocation.execute_device_buffer_allocations(), 0);
            assert_eq!(invocation.threads_per_threadgroup(), 128);
            assert_eq!(invocation.dynamic_threadgroup_memory_bytes(), 288);
            assert_eq!(invocation.execute().unwrap(), expected_message);
            assert_eq!(
                invocation
                    .read_destination_state()
                    .unwrap()
                    .into_iter()
                    .map(decode_state)
                    .collect::<Vec<_>>(),
                expected_state
            );
            assert_eq!(
                invocation.read_destination_increment().unwrap(),
                expected_inc
            );
        }
    }

    fn encode_state(state: DenseState) -> RegistersRwDenseStateWords {
        RegistersRwDenseStateWords {
            val: Fp128::from_jolt_field(&state[0]),
            ra: Fp128::from_jolt_field(&state[1]),
            wa: Fp128::from_jolt_field(&state[2]),
        }
    }

    fn decode_state(state: RegistersRwDenseStateWords) -> DenseState {
        [
            state.val.into_jolt_field(),
            state.ra.into_jolt_field(),
            state.wa.into_jolt_field(),
        ]
    }

    fn scalar_round(
        source: &[DenseState],
        source_inc: &[AkitaField],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        challenge: AkitaField,
    ) -> (Vec<DenseState>, Vec<AkitaField>, [AkitaField; 2]) {
        let destination_rows = source_inc.len() / 2;
        let bind = |low: AkitaField, high: AkitaField| low + challenge * (high - low);
        let destination = (0..destination_rows * REGISTERS_RW_DENSE_COLUMNS)
            .map(|index| {
                let row = index / REGISTERS_RW_DENSE_COLUMNS;
                let column = index % REGISTERS_RW_DENSE_COLUMNS;
                let low = source[(2 * row) * REGISTERS_RW_DENSE_COLUMNS + column];
                let high = source[(2 * row + 1) * REGISTERS_RW_DENSE_COLUMNS + column];
                std::array::from_fn(|field| bind(low[field], high[field]))
            })
            .collect::<Vec<_>>();
        let destination_inc = source_inc
            .chunks_exact(2)
            .map(|pair| bind(pair[0], pair[1]))
            .collect::<Vec<_>>();
        let mut message = [AkitaField::zero(); 2];
        for pair in 0..destination_rows / 2 {
            let mut pair_message = [AkitaField::zero(); 2];
            let even_row = 2 * pair;
            let odd_row = even_row + 1;
            for column in 0..REGISTERS_RW_DENSE_COLUMNS {
                let [val_0, ra_0, wa_0] =
                    destination[even_row * REGISTERS_RW_DENSE_COLUMNS + column];
                let [val_1, ra_1, wa_1] =
                    destination[odd_row * REGISTERS_RW_DENSE_COLUMNS + column];
                let val_m = val_1 - val_0;
                let ra_m = ra_1 - ra_0;
                let wa_m = wa_1 - wa_0;
                pair_message[0] += ra_0 * val_0 + wa_0 * (val_0 + destination_inc[even_row]);
                pair_message[1] += ra_m * val_m
                    + wa_m * (val_m + destination_inc[odd_row] - destination_inc[even_row]);
            }
            let weight = e_out[pair / e_in.len()] * e_in[pair % e_in.len()];
            message[0] += weight * pair_message[0];
            message[1] += weight * pair_message[1];
        }
        (destination, destination_inc, message)
    }
}
