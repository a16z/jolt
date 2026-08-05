use std::{
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSRange,
};

use super::{
    command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
    SpartanOuterUniskipRows,
};

pub const INSTRUCTION_INPUT_TABLES: usize = 8;
pub const INSTRUCTION_INPUT_COEFFICIENTS: usize = 3;
const INSTRUCTION_INPUT_DEVICE_BUFFERS: usize = 6;

const SIMD_WIDTH: usize = 32;
const NATIVE_MESSAGE_PIPELINE: &str = "solinas_instruction_input_native_message";
const NATIVE_TRANSITION_PIPELINE: &str = "solinas_instruction_input_native_transition";
const DENSE_TRANSITION_PIPELINE: &str = "solinas_instruction_input_dense_transition";
const REDUCTION_PIPELINE: &str = "solinas_instruction_input_reduce";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InstructionInputStorageInitialization {
    #[default]
    Lazy,
    Minimal,
    Full,
}

impl InstructionInputStorageInitialization {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lazy => "lazy",
            Self::Minimal => "minimal",
            Self::Full => "full",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionInputStorageInitializationStats {
    pub mode: InstructionInputStorageInitialization,
    pub device_buffers: usize,
    pub bytes: u64,
    pub wall: Duration,
    pub gpu_active: Duration,
    pub buffer_identities: [usize; INSTRUCTION_INPUT_DEVICE_BUFFERS],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionInputSequenceConfig {
    pub native_message_threads_per_threadgroup: Option<usize>,
    pub native_transition_threads_per_threadgroup: Option<usize>,
    pub dense_transition_threads_per_threadgroup: Option<usize>,
    pub storage_initialization: InstructionInputStorageInitialization,
}

impl Default for InstructionInputSequenceConfig {
    fn default() -> Self {
        Self {
            native_message_threads_per_threadgroup: Some(256),
            native_transition_threads_per_threadgroup: Some(128),
            dense_transition_threads_per_threadgroup: Some(128),
            storage_initialization: InstructionInputStorageInitialization::Lazy,
        }
    }
}

pub(crate) fn instruction_input_weight_capacities(
    rows: usize,
) -> Result<(usize, usize), MetalError> {
    if rows < 4 || !rows.is_power_of_two() {
        return Err(MetalError::InvalidInstructionInputRows(rows));
    }
    let e_out_capacity = 1usize << (rows.ilog2() / 2);
    let e_in_capacity = (rows / 2) / e_out_capacity;
    Ok((e_in_capacity, e_out_capacity))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct InstructionInputParams {
    source_elements: u32,
    e_in_length: u32,
    e_out_length: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ReductionParams {
    input_count: u32,
    output_count: u32,
    reserved: [u32; 2],
}

struct Pipelines {
    native_message: ComputePipelineState,
    native_transition: ComputePipelineState,
    dense_transition: ComputePipelineState,
    reduction: ComputePipelineState,
}

struct Buffers {
    dense_a: Buffer,
    dense_b: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

impl Buffers {
    fn all(&self) -> [&Buffer; INSTRUCTION_INPUT_DEVICE_BUFFERS] {
        [
            &self.dense_a,
            &self.dense_b,
            &self.e_in,
            &self.e_out,
            &self.partial_a,
            &self.partial_b,
        ]
    }

    fn identities(&self) -> [usize; INSTRUCTION_INPUT_DEVICE_BUFFERS] {
        self.all().map(|buffer| buffer.as_ptr() as usize)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InstructionInputStorageLayout {
    dense_a_elements: usize,
    dense_b_elements: usize,
    partial_elements: usize,
    buffer_bytes: [u64; INSTRUCTION_INPUT_DEVICE_BUFFERS],
    owned_bytes: u64,
}

fn instruction_input_storage_layout(
    rows: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
) -> Result<InstructionInputStorageLayout, MetalError> {
    if rows < 4 || !rows.is_power_of_two() {
        return Err(MetalError::InvalidInstructionInputRows(rows));
    }
    let covered = e_in_capacity
        .checked_mul(e_out_capacity)
        .ok_or(MetalError::InputTooLong(rows))?;
    if e_in_capacity == 0 || e_out_capacity == 0 || covered != rows / 2 {
        return Err(MetalError::InstructionInputWeightShape {
            expected: rows / 2,
            covered,
        });
    }
    let dense_a_elements = INSTRUCTION_INPUT_TABLES
        .checked_mul(rows / 2)
        .ok_or(MetalError::InputTooLong(rows))?;
    let dense_b_elements = INSTRUCTION_INPUT_TABLES
        .checked_mul(rows / 4)
        .ok_or(MetalError::InputTooLong(rows))?;
    let partial_elements = INSTRUCTION_INPUT_COEFFICIENTS
        .checked_mul(e_out_capacity)
        .ok_or(MetalError::InputTooLong(e_out_capacity))?;
    for elements in [
        rows,
        e_in_capacity,
        e_out_capacity,
        dense_a_elements,
        dense_b_elements,
        partial_elements,
    ] {
        validate_u32_element_count(elements)?;
    }
    let buffer_bytes = [
        byte_length::<Fp128>(dense_a_elements)?,
        byte_length::<Fp128>(dense_b_elements)?,
        byte_length::<Fp128>(e_in_capacity)?,
        byte_length::<Fp128>(e_out_capacity)?,
        byte_length::<Fp128>(partial_elements)?,
        byte_length::<Fp128>(partial_elements)?,
    ];
    let owned_bytes = buffer_bytes.iter().try_fold(0u64, |total, bytes| {
        total
            .checked_add(*bytes)
            .ok_or(MetalError::InputTooLong(rows))
    })?;
    Ok(InstructionInputStorageLayout {
        dense_a_elements,
        dense_b_elements,
        partial_elements,
        buffer_bytes,
        owned_bytes,
    })
}

pub(crate) fn instruction_input_sequence_storage_bytes(rows: usize) -> Result<u64, MetalError> {
    let (e_in_capacity, e_out_capacity) = instruction_input_weight_capacities(rows)?;
    Ok(instruction_input_storage_layout(rows, e_in_capacity, e_out_capacity)?.owned_bytes)
}

pub(crate) struct InstructionInputSequenceStorage {
    context: SolinasMetal,
    pipelines: Pipelines,
    native_message_limits: PipelineLimits,
    native_transition_limits: PipelineLimits,
    dense_transition_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    rows: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
    native_message_threads: usize,
    native_transition_threads: usize,
    dense_transition_threads: usize,
    config: InstructionInputSequenceConfig,
    initialization: InstructionInputStorageInitializationStats,
    owned_bytes: u64,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for InstructionInputSequenceStorage {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_storage"),
            self.owned_bytes as usize,
        );
        visitor.exit();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SequencePhase {
    BeforeMessage,
    Native,
    Dense,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DispatchKind {
    NativeMessage,
    NativeTransition,
    DenseTransition,
}

pub struct InstructionInputSequence {
    storage: InstructionInputSequenceStorage,
    resident_rows: SpartanOuterUniskipRows,
    phase: SequencePhase,
    dense_elements: usize,
    dense_in_a: bool,
    gpu_active_time: Duration,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for InstructionInputSequence {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("storage"), &self.storage);
        visitor.visit_field(allocative::Key::new("resident_rows"), &self.resident_rows);
        visitor.exit();
    }
}

impl SolinasMetal {
    /// Allocates a reusable instruction-input sequence and uploads `rows` once.
    pub fn prepare_instruction_input_sequence(
        &self,
        rows: &[super::SpartanOuterUniskipRow],
        config: InstructionInputSequenceConfig,
    ) -> Result<InstructionInputSequence, MetalError> {
        let resident_rows = self.prepare_spartan_outer_uniskip_rows(rows)?;
        let (e_in_capacity, e_out_capacity) = instruction_input_weight_capacities(rows.len())?;
        self.prepare_instruction_input_sequence_storage(
            rows.len(),
            e_in_capacity,
            e_out_capacity,
            config,
        )?
        .attach(resident_rows)
    }

    pub(crate) fn prepare_instruction_input_sequence_storage(
        &self,
        rows: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionInputSequenceConfig,
    ) -> Result<InstructionInputSequenceStorage, MetalError> {
        let layout = instruction_input_storage_layout(rows, e_in_capacity, e_out_capacity)?;

        let pipelines = Pipelines {
            native_message: self.compile_named_pipeline(NATIVE_MESSAGE_PIPELINE)?,
            native_transition: self.compile_named_pipeline(NATIVE_TRANSITION_PIPELINE)?,
            dense_transition: self.compile_named_pipeline(DENSE_TRANSITION_PIPELINE)?,
            reduction: self.compile_named_pipeline(REDUCTION_PIPELINE)?,
        };
        let native_message_limits = Self::limits(&pipelines.native_message);
        let native_transition_limits = Self::limits(&pipelines.native_transition);
        let dense_transition_limits = Self::limits(&pipelines.dense_transition);
        let reduction_limits = Self::limits(&pipelines.reduction);
        for (pipeline, limits) in [
            (NATIVE_MESSAGE_PIPELINE, native_message_limits),
            (NATIVE_TRANSITION_PIPELINE, native_transition_limits),
            (DENSE_TRANSITION_PIPELINE, dense_transition_limits),
            (REDUCTION_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedInstructionInputExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let native_message_threads = Self::resolve_threadgroup_width(
            config.native_message_threads_per_threadgroup,
            native_message_limits,
        )?;
        let native_transition_threads = Self::resolve_threadgroup_width(
            config.native_transition_threads_per_threadgroup,
            native_transition_limits,
        )?;
        let dense_transition_threads = Self::resolve_threadgroup_width(
            config.dense_transition_threads_per_threadgroup,
            dense_transition_limits,
        )?;

        let device = self.device_info();
        let _allocation_span = tracing::info_span!(
            "MetalInstructionInput::allocation_plan",
            device_buffers = INSTRUCTION_INPUT_DEVICE_BUFFERS,
            planned_device_bytes = layout.owned_bytes,
            current_device_bytes = device.current_allocated_size,
            recommended_device_bytes = device.recommended_max_working_set_size,
        )
        .entered();
        self.validate_additional_working_set(layout.owned_bytes)?;

        let buffers = Buffers {
            dense_a: new_buffer(self, layout.dense_a_elements)?,
            dense_b: new_buffer(self, layout.dense_b_elements)?,
            e_in: new_buffer(self, e_in_capacity)?,
            e_out: new_buffer(self, e_out_capacity)?,
            partial_a: new_buffer(self, layout.partial_elements)?,
            partial_b: new_buffer(self, layout.partial_elements)?,
        };
        let actual_buffer_bytes = buffers.all().map(|buffer| buffer.length());
        if actual_buffer_bytes != layout.buffer_bytes {
            return Err(MetalError::InvalidInstructionInputState(
                "allocated storage lengths disagree with the plan",
            ));
        }
        let initialization = initialize_storage(self, &buffers, config.storage_initialization)?;

        Ok(InstructionInputSequenceStorage {
            context: self.clone(),
            pipelines,
            native_message_limits,
            native_transition_limits,
            dense_transition_limits,
            reduction_limits,
            buffers,
            rows,
            e_in_capacity,
            e_out_capacity,
            native_message_threads,
            native_transition_threads,
            dense_transition_threads,
            config,
            initialization,
            owned_bytes: layout.owned_bytes,
        })
    }
}

impl InstructionInputSequenceStorage {
    pub(crate) fn matches(
        &self,
        context: &SolinasMetal,
        rows: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionInputSequenceConfig,
    ) -> bool {
        self.context.device_registry_id() == context.device_registry_id()
            && self.rows == rows
            && self.e_in_capacity == e_in_capacity
            && self.e_out_capacity == e_out_capacity
            && self.config == config
    }

    pub(crate) fn attach(
        self,
        resident_rows: SpartanOuterUniskipRows,
    ) -> Result<InstructionInputSequence, MetalError> {
        if resident_rows.len() != self.rows {
            return Err(MetalError::InvalidInstructionInputRows(resident_rows.len()));
        }
        if resident_rows.device_registry_id() != self.context.device_registry_id() {
            return Err(MetalError::InstructionInputRowsDevice {
                expected: self.context.device_registry_id(),
                got: resident_rows.device_registry_id(),
            });
        }
        Ok(InstructionInputSequence {
            storage: self,
            resident_rows,
            phase: SequencePhase::BeforeMessage,
            dense_elements: 0,
            dense_in_a: true,
            gpu_active_time: Duration::ZERO,
        })
    }

    pub(crate) const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }
}

impl InstructionInputSequence {
    /// Restores the sequence to its resident native rows without allocating buffers.
    pub fn reset(&mut self) {
        self.phase = SequencePhase::BeforeMessage;
        self.dense_elements = 0;
        self.dense_in_a = true;
        self.gpu_active_time = Duration::ZERO;
    }

    pub fn message(
        &mut self,
        gamma: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; INSTRUCTION_INPUT_COEFFICIENTS], MetalError> {
        if self.phase != SequencePhase::BeforeMessage {
            return Err(MetalError::InvalidInstructionInputState(
                "native message may execute only once",
            ));
        }
        self.execute(DispatchKind::NativeMessage, None, gamma, e_in, e_out)
    }

    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        gamma: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; INSTRUCTION_INPUT_COEFFICIENTS], MetalError> {
        let kind = match self.phase {
            SequencePhase::BeforeMessage => {
                return Err(MetalError::InvalidInstructionInputState(
                    "the native message must precede the first bind",
                ));
            }
            SequencePhase::Native => DispatchKind::NativeTransition,
            SequencePhase::Dense => DispatchKind::DenseTransition,
        };
        self.execute(kind, Some(challenge), gamma, e_in, e_out)
    }

    pub fn read_current_tables(&self, output: &mut [AkitaField]) -> Result<(), MetalError> {
        if self.phase != SequencePhase::Dense {
            return Err(MetalError::InvalidInstructionInputState(
                "native rows cannot be read as dense tables",
            ));
        }
        let expected = INSTRUCTION_INPUT_TABLES
            .checked_mul(self.dense_elements)
            .ok_or(MetalError::InputTooLong(self.dense_elements))?;
        if output.len() != expected {
            return Err(MetalError::InstructionInputStorageLength {
                expected,
                got: output.len(),
            });
        }
        let source = self.dense_source_buffer();
        // SAFETY: the active buffer contains exactly `expected` initialized fields,
        // and every dispatch waits for command completion before updating the phase.
        let values = unsafe { slice::from_raw_parts(source.contents().cast::<Fp128>(), expected) };
        self.storage
            .context
            .validate_inputs("instruction input dense tables", values)?;
        for (output, value) in output.iter_mut().zip(values) {
            *output = value.into_jolt_field();
        }
        Ok(())
    }

    pub const fn current_elements(&self) -> usize {
        match self.phase {
            SequencePhase::BeforeMessage | SequencePhase::Native => self.storage.rows,
            SequencePhase::Dense => self.dense_elements,
        }
    }

    pub const fn is_dense(&self) -> bool {
        matches!(self.phase, SequencePhase::Dense)
    }

    pub const fn gpu_active_time(&self) -> Duration {
        self.gpu_active_time
    }

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn resident_row_identity(&self) -> usize {
        self.resident_rows.allocation_identity()
    }

    pub fn static_buffer_identity(&self) -> [usize; INSTRUCTION_INPUT_DEVICE_BUFFERS] {
        self.storage.buffers.identities()
    }

    pub const fn owned_storage_bytes(&self) -> u64 {
        self.storage.owned_bytes
    }

    pub const fn storage_initialization(&self) -> InstructionInputStorageInitializationStats {
        self.storage.initialization
    }

    pub const fn native_message_pipeline_limits(&self) -> PipelineLimits {
        self.storage.native_message_limits
    }

    pub const fn native_transition_pipeline_limits(&self) -> PipelineLimits {
        self.storage.native_transition_limits
    }

    pub const fn dense_transition_pipeline_limits(&self) -> PipelineLimits {
        self.storage.dense_transition_limits
    }

    fn execute(
        &mut self,
        kind: DispatchKind,
        challenge: Option<AkitaField>,
        gamma: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; INSTRUCTION_INPUT_COEFFICIENTS], MetalError> {
        let source_elements = match kind {
            DispatchKind::NativeMessage | DispatchKind::NativeTransition => self.storage.rows,
            DispatchKind::DenseTransition => self.dense_elements,
        };
        let pair_divisor = match kind {
            DispatchKind::NativeMessage => 2,
            DispatchKind::NativeTransition | DispatchKind::DenseTransition => 4,
        };
        if source_elements < pair_divisor || !source_elements.is_power_of_two() {
            return Err(MetalError::InvalidInstructionInputRows(source_elements));
        }
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(source_elements))?;
        let expected = source_elements / pair_divisor;
        if e_in.is_empty()
            || e_out.is_empty()
            || e_in.len() > self.storage.e_in_capacity
            || e_out.len() > self.storage.e_out_capacity
            || covered != expected
        {
            return Err(MetalError::InstructionInputWeightShape { expected, covered });
        }
        write_fields(&self.storage.buffers.e_in, self.storage.e_in_capacity, e_in)?;
        write_fields(
            &self.storage.buffers.e_out,
            self.storage.e_out_capacity,
            e_out,
        )?;

        let params = InstructionInputParams {
            source_elements: u32::try_from(source_elements)
                .map_err(|_| MetalError::InputTooLong(source_elements))?,
            e_in_length: u32::try_from(e_in.len())
                .map_err(|_| MetalError::InputTooLong(e_in.len()))?,
            e_out_length: u32::try_from(e_out.len())
                .map_err(|_| MetalError::InputTooLong(e_out.len()))?,
            reserved: 0,
        };
        let gamma = Fp128::from_jolt_field(&gamma);
        let challenge = challenge.map(|value| Fp128::from_jolt_field(&value));
        let (pipeline, threads) = match kind {
            DispatchKind::NativeMessage => (
                self.storage.pipelines.native_message.clone(),
                self.storage.native_message_threads,
            ),
            DispatchKind::NativeTransition => (
                self.storage.pipelines.native_transition.clone(),
                self.storage.native_transition_threads,
            ),
            DispatchKind::DenseTransition => (
                self.storage.pipelines.dense_transition.clone(),
                self.storage.dense_transition_threads,
            ),
        };

        let queue = self.storage.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            match kind {
                DispatchKind::NativeMessage => {
                    encoder.set_buffer(0, Some(self.resident_rows.buffer()), 0);
                    encoder.set_buffer(1, Some(&self.storage.buffers.e_in), 0);
                    encoder.set_buffer(2, Some(&self.storage.buffers.e_out), 0);
                    encoder.set_buffer(3, Some(&self.storage.buffers.partial_a), 0);
                    set_inline_bytes(encoder, 4, &gamma);
                    set_inline_bytes(encoder, 5, &params);
                }
                DispatchKind::NativeTransition => {
                    encoder.set_buffer(0, Some(self.resident_rows.buffer()), 0);
                    encoder.set_buffer(1, Some(&self.storage.buffers.dense_a), 0);
                    encoder.set_buffer(2, Some(&self.storage.buffers.e_in), 0);
                    encoder.set_buffer(3, Some(&self.storage.buffers.e_out), 0);
                    encoder.set_buffer(4, Some(&self.storage.buffers.partial_a), 0);
                    set_inline_bytes(
                        encoder,
                        5,
                        challenge
                            .as_ref()
                            .ok_or(MetalError::InvalidInstructionInputState(
                                "native transition is missing its challenge",
                            ))?,
                    );
                    set_inline_bytes(encoder, 6, &gamma);
                    set_inline_bytes(encoder, 7, &params);
                }
                DispatchKind::DenseTransition => {
                    encoder.set_buffer(0, Some(self.dense_source_buffer()), 0);
                    encoder.set_buffer(1, Some(self.dense_destination_buffer()), 0);
                    encoder.set_buffer(2, Some(&self.storage.buffers.e_in), 0);
                    encoder.set_buffer(3, Some(&self.storage.buffers.e_out), 0);
                    encoder.set_buffer(4, Some(&self.storage.buffers.partial_a), 0);
                    set_inline_bytes(
                        encoder,
                        5,
                        challenge
                            .as_ref()
                            .ok_or(MetalError::InvalidInstructionInputState(
                                "dense transition is missing its challenge",
                            ))?,
                    );
                    set_inline_bytes(encoder, 6, &gamma);
                    set_inline_bytes(encoder, 7, &params);
                }
            }
            let simdgroups = threads / SIMD_WIDTH;
            encoder.set_threadgroup_memory_length(
                0,
                (INSTRUCTION_INPUT_COEFFICIENTS * simdgroups * size_of::<Fp128>()) as u64,
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: e_out.len() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            self.encode_reductions(encoder, e_out.len());
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok::<(), MetalError>(())
        })?;

        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(command_buffer.status()));
        }
        let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        self.gpu_active_time += Duration::from_secs_f64(end - start);
        let final_buffer = self.final_partial_buffer(e_out.len());
        // SAFETY: the completed reduction leaves three canonical fields at the
        // start of the selected partial buffer.
        let values = unsafe {
            slice::from_raw_parts(
                final_buffer.contents().cast::<Fp128>(),
                INSTRUCTION_INPUT_COEFFICIENTS,
            )
        };
        self.storage
            .context
            .validate_inputs("instruction input message", values)?;
        let message = std::array::from_fn(|index| values[index].into_jolt_field());

        match kind {
            DispatchKind::NativeMessage => self.phase = SequencePhase::Native,
            DispatchKind::NativeTransition => {
                self.phase = SequencePhase::Dense;
                self.dense_elements = self.storage.rows / 2;
                self.dense_in_a = true;
            }
            DispatchKind::DenseTransition => {
                self.dense_elements /= 2;
                self.dense_in_a = !self.dense_in_a;
            }
        }
        Ok(message)
    }

    fn encode_reductions(&self, encoder: &metal::ComputeCommandEncoderRef, mut count: usize) {
        let mut input_a = true;
        while count > 1 {
            let output_count = count.div_ceil(SIMD_WIDTH);
            let params = ReductionParams {
                input_count: count as u32,
                output_count: output_count as u32,
                reserved: [0; 2],
            };
            encoder.set_compute_pipeline_state(&self.storage.pipelines.reduction);
            let (input, output) = if input_a {
                (
                    &self.storage.buffers.partial_a,
                    &self.storage.buffers.partial_b,
                )
            } else {
                (
                    &self.storage.buffers.partial_b,
                    &self.storage.buffers.partial_a,
                )
            };
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(output), 0);
            set_inline_bytes(encoder, 2, &params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: output_count as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.storage.reduction_limits.thread_execution_width as u64,
                    height: 1,
                    depth: 1,
                },
            );
            count = output_count;
            input_a = !input_a;
        }
    }

    fn final_partial_buffer(&self, mut count: usize) -> &Buffer {
        let mut input_a = true;
        while count > 1 {
            count = count.div_ceil(SIMD_WIDTH);
            input_a = !input_a;
        }
        if input_a {
            &self.storage.buffers.partial_a
        } else {
            &self.storage.buffers.partial_b
        }
    }

    fn dense_source_buffer(&self) -> &Buffer {
        if self.dense_in_a {
            &self.storage.buffers.dense_a
        } else {
            &self.storage.buffers.dense_b
        }
    }

    fn dense_destination_buffer(&self) -> &Buffer {
        if self.dense_in_a {
            &self.storage.buffers.dense_b
        } else {
            &self.storage.buffers.dense_a
        }
    }
}

fn initialize_storage(
    context: &SolinasMetal,
    buffers: &Buffers,
    mode: InstructionInputStorageInitialization,
) -> Result<InstructionInputStorageInitializationStats, MetalError> {
    let buffer_identities = buffers.identities();
    let fill_lengths = buffers.all().map(|buffer| match mode {
        InstructionInputStorageInitialization::Lazy => 0,
        InstructionInputStorageInitialization::Minimal => size_of::<Fp128>() as u64,
        InstructionInputStorageInitialization::Full => buffer.length(),
    });
    let bytes = fill_lengths.iter().try_fold(0u64, |total, length| {
        total
            .checked_add(*length)
            .ok_or(MetalError::InvalidInstructionInputState(
                "storage initialization byte count overflowed",
            ))
    })?;
    let device_buffers = usize::from(mode != InstructionInputStorageInitialization::Lazy)
        * INSTRUCTION_INPUT_DEVICE_BUFFERS;
    let span = tracing::info_span!(
        "MetalInstructionInput::storage_initialize",
        mode = mode.as_str(),
        device_buffers,
        bytes,
        protocol_dispatches = 0,
        buffer_0 = buffer_identities[0],
        buffer_1 = buffer_identities[1],
        buffer_2 = buffer_identities[2],
        buffer_3 = buffer_identities[3],
        buffer_4 = buffer_identities[4],
        buffer_5 = buffer_identities[5],
    );
    let _entered = span.enter();
    let started = Instant::now();
    let gpu_active = if mode == InstructionInputStorageInitialization::Lazy {
        Duration::ZERO
    } else {
        let command_buffer = context.queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_blit_command_encoder();
            for (buffer, length) in buffers.all().into_iter().zip(fill_lengths) {
                encoder.fill_buffer(buffer, NSRange::new(0, length), 0);
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
        let gpu_active = Duration::from_secs_f64(end - start);
        let gpu_active_ns = u64::try_from(gpu_active.as_nanos()).unwrap_or(u64::MAX);
        let completion = tracing::info_span!(
            "MetalInstructionInput::storage_initialize_complete",
            mode = mode.as_str(),
            command_completed = true,
            gpu_active_ns,
        );
        let _completed = completion.enter();
        gpu_active
    };
    let wall = started.elapsed();
    Ok(InstructionInputStorageInitializationStats {
        mode,
        device_buffers,
        bytes,
        wall,
        gpu_active,
        buffer_identities,
    })
}

fn write_fields(buffer: &Buffer, capacity: usize, values: &[AkitaField]) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(MetalError::InstructionInputStorageLength {
            expected: capacity,
            got: values.len(),
        });
    }
    // SAFETY: shared storage has capacity fields and no command is active while
    // the host updates the current prefix.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity) };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}

fn new_buffer(context: &SolinasMetal, elements: usize) -> Result<Buffer, MetalError> {
    let bytes = byte_length::<Fp128>(elements)?;
    context.validate_buffer_length(bytes)?;
    Ok(context
        .device
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
}

fn validate_u32_element_count(elements: usize) -> Result<(), MetalError> {
    let _ = u32::try_from(elements).map_err(|_| MetalError::InputTooLong(elements))?;
    Ok(())
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    let bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or(MetalError::InputTooLong(elements))?;
    u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(elements))
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::expect_used, reason = "test module")]
mod tests {
    use std::{mem::size_of, slice};

    use jolt_field::AkitaField;
    use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

    use super::{
        initialize_storage, instruction_input_sequence_storage_bytes,
        instruction_input_storage_layout, instruction_input_weight_capacities,
        validate_u32_element_count, InstructionInputParams, InstructionInputSequenceConfig,
        InstructionInputStorageInitialization, ReductionParams, INSTRUCTION_INPUT_DEVICE_BUFFERS,
        INSTRUCTION_INPUT_TABLES,
    };
    use crate::metal::solinas::{MetalError, SolinasMetal, SpartanOuterUniskipRow};

    #[test]
    fn weight_capacities_cover_initial_pairs() {
        for log_rows in 2..=30 {
            let rows = 1usize << log_rows;
            let (e_in, e_out) = instruction_input_weight_capacities(rows).unwrap();
            assert_eq!(e_in * e_out, rows / 2);
        }
    }

    #[test]
    fn sequence_bytes_match_the_production_geometry() {
        let rows = 1 << 26;
        let (e_in, e_out) = instruction_input_weight_capacities(rows).unwrap();
        let layout = instruction_input_storage_layout(rows, e_in, e_out).unwrap();
        assert_eq!(
            layout.buffer_bytes,
            [
                4_294_967_296,
                2_147_483_648,
                65_536,
                131_072,
                393_216,
                393_216,
            ]
        );
        assert_eq!(layout.owned_bytes, 6_443_433_984);
        assert_eq!(
            instruction_input_sequence_storage_bytes(rows).unwrap(),
            layout.owned_bytes
        );
    }

    #[test]
    fn flattened_shader_element_counts_fit_u32() {
        let maximum = usize::try_from(u32::MAX).unwrap();
        assert!(validate_u32_element_count(maximum).is_ok());

        let too_many = maximum + 1;
        assert!(matches!(
            validate_u32_element_count(too_many),
            Err(MetalError::InputTooLong(elements)) if elements == too_many
        ));
    }

    #[test]
    fn shader_parameter_abi_is_four_words() {
        assert_eq!(size_of::<InstructionInputParams>(), 16);
        assert_eq!(size_of::<ReductionParams>(), 16);
    }

    fn relation(values: &[AkitaField; INSTRUCTION_INPUT_TABLES], gamma: AkitaField) -> AkitaField {
        let right = values[4] * values[5] + values[6] * values[7];
        let left = values[0] * values[1] + values[2] * values[3];
        right + gamma * left
    }

    fn descriptors(
        tables: &[AkitaField],
        elements: usize,
        gamma: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> [AkitaField; 3] {
        assert_eq!(tables.len(), INSTRUCTION_INPUT_TABLES * elements);
        assert_eq!(e_in.len() * e_out.len(), elements / 2);
        let mut output = [AkitaField::zero(); 3];
        for (x_out, outer_weight) in e_out.iter().enumerate() {
            for (x_in, inner_weight) in e_in.iter().enumerate() {
                let pair = x_out * e_in.len() + x_in;
                let at_0 = std::array::from_fn(|table| tables[table * elements + 2 * pair]);
                let at_1 = std::array::from_fn(|table| tables[table * elements + 2 * pair + 1]);
                let step = std::array::from_fn(|table| at_1[table] - at_0[table]);
                let weight = *outer_weight * *inner_weight;
                output[0] += weight * relation(&at_0, gamma);
                output[1] += weight * relation(&at_1, gamma);
                output[2] += weight * relation(&step, gamma);
            }
        }
        output
    }

    fn bind_tables(
        tables: &[AkitaField],
        elements: usize,
        challenge: AkitaField,
    ) -> Vec<AkitaField> {
        let mut output = Vec::with_capacity(INSTRUCTION_INPUT_TABLES * elements / 2);
        for table in 0..INSTRUCTION_INPUT_TABLES {
            for pair in 0..elements / 2 {
                let low = tables[table * elements + 2 * pair];
                let high = tables[table * elements + 2 * pair + 1];
                output.push(low + challenge * (high - low));
            }
        }
        output
    }

    fn packed_rows(rows: usize) -> Vec<SpartanOuterUniskipRow> {
        (0..rows)
            .map(|index| {
                let mut words = [0u64; 20];
                words[6] = 0x1000 + 4 * index as u64;
                let immediate = match index {
                    0 => i128::MIN,
                    1 => i128::MAX,
                    _ if index % 3 == 0 => -(index as i128) - 1,
                    _ => (index as i128) * 17 + 3,
                };
                let magnitude = immediate.unsigned_abs();
                words[7] = magnitude as u64;
                words[8] = (magnitude >> 64) as u64;
                words[9] = u64::MAX.wrapping_sub(13 * index as u64);
                words[10] = 0xABCD_0000 + index as u64;
                let mut flags = 0u64;
                flags |= u64::from(immediate >= 0) << 18;
                flags |= ((index & 1) as u64) << 20;
                flags |= (((index >> 1) & 1) as u64) << 21;
                flags |= (((index >> 2) & 1) as u64) << 22;
                flags |= (((index >> 3) & 1) as u64) << 23;
                if index == 7 {
                    flags |= 1;
                    flags &= !(1 << 22);
                }
                words[19] = flags;
                SpartanOuterUniskipRow::from_words(words)
            })
            .collect()
    }

    #[test]
    fn resident_sequence_matches_cpu_descriptors_and_tables() {
        let rows = packed_rows(16);
        let initial_tables: Vec<AkitaField> = (0..INSTRUCTION_INPUT_TABLES)
            .flat_map(|table| {
                rows.iter()
                    .map(move |row| row.instruction_input_fields::<AkitaField>()[table])
            })
            .collect();
        let gamma = AkitaField::from_u64(0xC001_CAFE);
        let r_product = [
            AkitaField::from_u64(5),
            AkitaField::from_u64(7),
            AkitaField::from_u64(11),
            AkitaField::from_u64(13),
        ];
        let mut gruen = GruenSplitEqPolynomial::new(&r_product, BindingOrder::LowToHigh);
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let resident = context
            .prepare_spartan_outer_uniskip_rows(&rows)
            .expect("rows should upload");
        let (e_in_capacity, e_out_capacity) =
            instruction_input_weight_capacities(rows.len()).unwrap();
        let storage = context
            .prepare_instruction_input_sequence_storage(
                rows.len(),
                e_in_capacity,
                e_out_capacity,
                InstructionInputSequenceConfig {
                    storage_initialization: InstructionInputStorageInitialization::Full,
                    ..InstructionInputSequenceConfig::default()
                },
            )
            .expect("sequence storage should prepare");
        let mut sequence = storage.attach(resident).expect("rows should attach");

        let expected = descriptors(
            &initial_tables,
            rows.len(),
            gamma,
            gruen.e_in_current(),
            gruen.e_out_current(),
        );
        let actual = sequence
            .message(gamma, gruen.e_in_current(), gruen.e_out_current())
            .expect("native message should execute");
        assert_eq!(actual, expected);

        let challenge_0 = AkitaField::from_u64(19);
        gruen.bind(challenge_0);
        let tables_1 = bind_tables(&initial_tables, rows.len(), challenge_0);
        let expected = descriptors(
            &tables_1,
            rows.len() / 2,
            gamma,
            gruen.e_in_current(),
            gruen.e_out_current(),
        );
        let actual = sequence
            .bind_and_message(
                challenge_0,
                gamma,
                gruen.e_in_current(),
                gruen.e_out_current(),
            )
            .expect("native transition should execute");
        assert_eq!(actual, expected);
        let mut readback = vec![AkitaField::zero(); tables_1.len()];
        sequence.read_current_tables(&mut readback).unwrap();
        assert_eq!(readback, tables_1);

        let challenge_1 = AkitaField::from_u64(23);
        gruen.bind(challenge_1);
        let tables_2 = bind_tables(&tables_1, rows.len() / 2, challenge_1);
        let expected = descriptors(
            &tables_2,
            rows.len() / 4,
            gamma,
            gruen.e_in_current(),
            gruen.e_out_current(),
        );
        let actual = sequence
            .bind_and_message(
                challenge_1,
                gamma,
                gruen.e_in_current(),
                gruen.e_out_current(),
            )
            .expect("dense transition should execute");
        assert_eq!(actual, expected);
        let mut readback = vec![AkitaField::zero(); tables_2.len()];
        sequence.read_current_tables(&mut readback).unwrap();
        assert_eq!(readback, tables_2);
    }

    #[test]
    fn full_initialization_zeroes_every_static_buffer_in_place() {
        let rows = 16;
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let (e_in_capacity, e_out_capacity) = instruction_input_weight_capacities(rows).unwrap();
        let storage = context
            .prepare_instruction_input_sequence_storage(
                rows,
                e_in_capacity,
                e_out_capacity,
                InstructionInputSequenceConfig::default(),
            )
            .expect("sequence storage should prepare");
        let identities = storage.buffers.identities();
        for buffer in storage.buffers.all() {
            let length = usize::try_from(buffer.length()).unwrap();
            // SAFETY: every test buffer uses shared storage and no command is active.
            let bytes = unsafe { slice::from_raw_parts_mut(buffer.contents().cast(), length) };
            bytes.fill(0xA5);
        }

        let stats = initialize_storage(
            &context,
            &storage.buffers,
            InstructionInputStorageInitialization::Full,
        )
        .expect("full initialization should complete");

        assert_eq!(stats.device_buffers, INSTRUCTION_INPUT_DEVICE_BUFFERS);
        assert_eq!(stats.bytes, storage.owned_bytes());
        assert_eq!(stats.buffer_identities, identities);
        assert!(stats.gpu_active <= stats.wall);
        assert_eq!(storage.buffers.identities(), identities);
        for buffer in storage.buffers.all() {
            let length = usize::try_from(buffer.length()).unwrap();
            // SAFETY: the synchronous blit completed and the shared buffer remains live.
            let bytes: &[u8] = unsafe { slice::from_raw_parts(buffer.contents().cast(), length) };
            assert!(bytes.iter().all(|byte| *byte == 0));
        }
    }
}
