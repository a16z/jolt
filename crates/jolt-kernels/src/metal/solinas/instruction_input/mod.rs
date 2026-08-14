use std::{
    ffi::c_void,
    mem::size_of,
    ops::Deref,
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use jolt_witness::witnesses::SpartanOuterRow;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBuffer,
    ComputePipelineState, MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSRange,
};

use super::instruction_input_successor::{
    model::{checked_dense_message_shape, checked_materialize_shape},
    DENSE_MESSAGE_PIPELINE as SUCCESSOR_DENSE_MESSAGE_PIPELINE,
    INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH, MATERIALIZE_PIPELINE as SUCCESSOR_MATERIALIZE_PIPELINE,
};
use super::{
    command_buffer_timestamp, spartan_outer_uniskip_residual_row_bytes, Fp128, MetalError,
    OuterResidualArenaKey, OuterResidualReleaseReceipt, PipelineLimits, SolinasMetal,
    SpartanOuterUniskipRows,
};

pub const INSTRUCTION_INPUT_TABLES: usize = 8;
pub const INSTRUCTION_INPUT_COEFFICIENTS: usize = 3;
const INSTRUCTION_INPUT_DEVICE_BUFFERS: usize = 6;
const INSTRUCTION_INPUT_ROW_WORDS: usize = 6;
const INSTRUCTION_INPUT_SUCCESSOR_MATERIALIZE_THREADS: usize = 256;
const INSTRUCTION_INPUT_SUCCESSOR_DENSE_MESSAGE_THREADS: usize = 128;
const INSTRUCTION_INPUT_SUCCESSOR_PRIMER_SOURCE_ELEMENTS: usize = 64;
const INSTRUCTION_INPUT_SUCCESSOR_PRIMER_BOUND_ELEMENTS: usize =
    INSTRUCTION_INPUT_SUCCESSOR_PRIMER_SOURCE_ELEMENTS / 2;
const INSTRUCTION_INPUT_SUCCESSOR_PRIMER_E_IN_ELEMENTS: usize = 1;
const INSTRUCTION_INPUT_SUCCESSOR_PRIMER_E_OUT_ELEMENTS: usize =
    INSTRUCTION_INPUT_SUCCESSOR_PRIMER_BOUND_ELEMENTS / 2;

const ROW_RS1: usize = 0;
const ROW_UNEXPANDED_PC: usize = 1;
const ROW_RS2: usize = 2;
const ROW_IMM_LOW: usize = 3;
const ROW_IMM_HIGH: usize = 4;
const ROW_FLAGS: usize = 5;

const FLAG_LOAD: u32 = 0;
const FLAG_IMM_POSITIVE: u32 = 18;
const FLAG_LEFT_OPERAND_IS_RS1: u32 = 20;
const FLAG_LEFT_OPERAND_IS_PC: u32 = 21;
const FLAG_RIGHT_OPERAND_IS_RS2: u32 = 22;
const FLAG_RIGHT_OPERAND_IS_IMM: u32 = 23;
const FLAG_BRANCH: u32 = 25;
const FLAG_NEXT_IS_NOOP: u32 = 26;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InstructionInputRow {
    words: [u64; INSTRUCTION_INPUT_ROW_WORDS],
}

#[derive(Clone)]
pub struct InstructionInputRows {
    buffer: Buffer,
    len: usize,
    device_registry_id: u64,
}

impl InstructionInputRow {
    pub fn from_spartan_outer(row: &SpartanOuterRow) -> Self {
        let imm = row.imm.0.unsigned_abs();
        let mut flags = 0u64;
        let mut set = |bit: u32, value: bool| flags |= u64::from(value) << bit;
        set(FLAG_LOAD, row.load.0);
        set(FLAG_IMM_POSITIVE, row.imm.0 >= 0);
        set(FLAG_LEFT_OPERAND_IS_RS1, row.left_operand_is_rs1.0);
        set(FLAG_LEFT_OPERAND_IS_PC, row.left_operand_is_pc.0);
        set(FLAG_RIGHT_OPERAND_IS_RS2, row.right_operand_is_rs2.0);
        set(FLAG_RIGHT_OPERAND_IS_IMM, row.right_operand_is_imm.0);
        set(FLAG_BRANCH, row.branch_flag.0);
        set(FLAG_NEXT_IS_NOOP, row.next_is_noop.0);
        Self {
            words: [
                row.rs1_value.0,
                row.unexpanded_pc.0,
                if row.load.0 { 0 } else { row.rs2_value.0 },
                imm as u64,
                (imm >> 64) as u64,
                flags,
            ],
        }
    }

    pub(crate) const fn from_full_words(words: [u64; 20]) -> Self {
        let flags = words[19];
        Self {
            words: [
                words[9],
                words[6],
                if flags & (1 << FLAG_LOAD) == 0 {
                    words[10]
                } else {
                    0
                },
                words[7],
                words[8],
                flags,
            ],
        }
    }

    pub const fn words(self) -> [u64; INSTRUCTION_INPUT_ROW_WORDS] {
        self.words
    }

    pub fn fields<F: jolt_field::Field>(&self) -> [F; INSTRUCTION_INPUT_TABLES] {
        let flags = self.words[ROW_FLAGS];
        let flag = |bit| F::from_u64((flags >> bit) & 1);
        let imm_magnitude =
            u128::from(self.words[ROW_IMM_LOW]) | (u128::from(self.words[ROW_IMM_HIGH]) << 64);
        let imm = F::from_u128(imm_magnitude);
        let imm = if ((flags >> FLAG_IMM_POSITIVE) & 1) != 0 {
            imm
        } else {
            -imm
        };
        [
            flag(FLAG_LEFT_OPERAND_IS_RS1),
            F::from_u64(self.words[ROW_RS1]),
            flag(FLAG_LEFT_OPERAND_IS_PC),
            F::from_u64(self.words[ROW_UNEXPANDED_PC]),
            flag(FLAG_RIGHT_OPERAND_IS_RS2),
            F::from_u64(self.words[ROW_RS2]),
            flag(FLAG_RIGHT_OPERAND_IS_IMM),
            imm,
        ]
    }
}

impl InstructionInputRows {
    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub const fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    pub fn allocation_identity(&self) -> usize {
        self.buffer.as_ptr() as usize
    }

    pub(crate) fn from_buffer(buffer: Buffer, len: usize, device_registry_id: u64) -> Self {
        Self {
            buffer,
            len,
            device_registry_id,
        }
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for InstructionInputRows {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_rows"),
            self.len * size_of::<InstructionInputRow>(),
        );
        visitor.exit();
    }
}

const SIMD_WIDTH: usize = 32;
const NATIVE_MESSAGE_PIPELINE: &str = "solinas_instruction_input_native_message";
const NATIVE_TRANSITION_PIPELINE: &str = "solinas_instruction_input_native_transition";
const DENSE_TRANSITION_PIPELINE: &str = "solinas_instruction_input_dense_transition";
const REDUCTION_PIPELINE: &str = "solinas_instruction_input_reduce";
pub(crate) const INSTRUCTION_INPUT_PRIMER_SOURCE_ELEMENTS: usize = 64;
pub(crate) const INSTRUCTION_INPUT_PRIMER_E_IN_ELEMENTS: usize = 1;
pub(crate) const INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS: usize =
    INSTRUCTION_INPUT_PRIMER_SOURCE_ELEMENTS / 2;

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
    pub buffer_offsets: [u64; INSTRUCTION_INPUT_DEVICE_BUFFERS],
    pub buffer_lengths: [u64; INSTRUCTION_INPUT_DEVICE_BUFFERS],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionInputPrimerStats {
    pub source_elements: usize,
    pub e_in_elements: usize,
    pub e_out_elements: usize,
    pub wall: Duration,
    pub submit_wall: Duration,
    pub overlap_wall: Duration,
    pub join_wall: Duration,
    pub gpu_active: Duration,
    pub completed_before_join: bool,
    pub resident_row_identity: usize,
    pub storage_buffer_identities: [usize; INSTRUCTION_INPUT_DEVICE_BUFFERS],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionInputSuccessorMaterializeStats {
    pub source_elements: usize,
    pub bound_elements: usize,
    pub wall: Duration,
    pub gpu_active: Duration,
    pub resident_row_identity: usize,
    pub dense_buffer_identity: usize,
    pub pipeline_limits: PipelineLimits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionInputSuccessorDenseMessageStats {
    pub table_elements: usize,
    pub e_in_elements: usize,
    pub e_out_elements: usize,
    pub wall: Duration,
    pub gpu_active: Duration,
    pub dense_buffer_identity: usize,
    pub dynamic_threadgroup_memory_length: usize,
    pub pipeline_limits: PipelineLimits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionInputSuccessorTransitionStats {
    pub source_elements: usize,
    pub table_elements: usize,
    pub e_in_elements: usize,
    pub e_out_elements: usize,
    pub wall: Duration,
    pub gpu_active: Duration,
    pub resident_row_identity: usize,
    pub dense_buffer_identity: usize,
    pub dynamic_threadgroup_memory_length: usize,
    pub materialize_pipeline_limits: PipelineLimits,
    pub dense_message_pipeline_limits: PipelineLimits,
}

struct InstructionInputNativePrimerCommand {
    command_buffer: CommandBuffer,
    submitted_at: Instant,
    submit_wall: Duration,
    resident_row_identity: usize,
    storage_buffer_identities: [usize; INSTRUCTION_INPUT_DEVICE_BUFFERS],
}

#[must_use = "a submitted Metal primer must be joined before the sequence is used"]
pub(crate) struct PendingInstructionInputPrimer {
    sequence: Option<InstructionInputSequence>,
    command: Option<InstructionInputNativePrimerCommand>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingInstructionInputPrimer {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(sequence) = &self.sequence {
            visitor.visit_field(allocative::Key::new("sequence"), sequence);
        }
        visitor.exit();
    }
}

impl Drop for PendingInstructionInputPrimer {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.command_buffer.wait_until_completed();
        }
    }
}

impl PendingInstructionInputPrimer {
    pub(crate) fn matches(
        &self,
        context: &SolinasMetal,
        rows: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionInputSequenceConfig,
    ) -> bool {
        self.sequence.as_ref().is_some_and(|sequence| {
            sequence
                .storage
                .matches(context, rows, e_in_capacity, e_out_capacity, config)
        })
    }

    pub(crate) fn resident_row_identity(&self) -> Option<usize> {
        self.sequence
            .as_ref()
            .map(InstructionInputSequence::resident_row_identity)
    }

    pub(crate) fn storage_initialization(
        &self,
    ) -> Option<InstructionInputStorageInitializationStats> {
        self.sequence
            .as_ref()
            .map(InstructionInputSequence::storage_initialization)
    }

    pub(crate) fn join(
        mut self,
    ) -> Result<(InstructionInputSequence, InstructionInputPrimerStats), MetalError> {
        let sequence = self
            .sequence
            .take()
            .ok_or(MetalError::InvalidInstructionInputState(
                "native pipeline primer lost its sequence",
            ))?;
        let command = self
            .command
            .take()
            .ok_or(MetalError::InvalidInstructionInputState(
                "native pipeline primer lost its command",
            ))?;
        let stats = sequence
            .storage
            .complete_native_pipeline_primer(&sequence.resident_rows, command)?;
        Ok((sequence, stats))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionInputSequenceConfig {
    pub native_message_threads_per_threadgroup: Option<usize>,
    pub native_transition_threads_per_threadgroup: Option<usize>,
    pub dense_transition_threads_per_threadgroup: Option<usize>,
    pub first_transition: InstructionInputFirstTransition,
    pub storage_initialization: InstructionInputStorageInitialization,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InstructionInputFirstTransition {
    #[default]
    Compact,
    Successor,
}

impl InstructionInputFirstTransition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Successor => "successor",
        }
    }
}

impl Default for InstructionInputSequenceConfig {
    fn default() -> Self {
        Self {
            native_message_threads_per_threadgroup: Some(256),
            native_transition_threads_per_threadgroup: Some(128),
            dense_transition_threads_per_threadgroup: Some(128),
            first_transition: InstructionInputFirstTransition::Compact,
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

struct SuccessorPipelines {
    materialize: ComputePipelineState,
    dense_message: ComputePipelineState,
    materialize_limits: PipelineLimits,
    dense_message_limits: PipelineLimits,
    materialize_threads: usize,
    dense_message_threads: usize,
}

struct BufferRegion {
    buffer: Buffer,
    offset_bytes: u64,
    length_bytes: u64,
}

impl BufferRegion {
    fn whole(buffer: Buffer) -> Self {
        let length_bytes = buffer.length();
        Self {
            buffer,
            offset_bytes: 0,
            length_bytes,
        }
    }

    fn range(buffer: &Buffer, offset_bytes: u64, length_bytes: u64) -> Result<Self, MetalError> {
        let end = offset_bytes.checked_add(length_bytes).ok_or(
            MetalError::InvalidInstructionInputState("dense buffer range overflowed"),
        )?;
        if !offset_bytes.is_multiple_of(size_of::<Fp128>() as u64) || end > buffer.length() {
            return Err(MetalError::InvalidInstructionInputState(
                "dense buffer range is misaligned or out of bounds",
            ));
        }
        let _ = usize::try_from(offset_bytes).map_err(|_| MetalError::InputTooLong(usize::MAX))?;
        Ok(Self {
            buffer: buffer.clone(),
            offset_bytes,
            length_bytes,
        })
    }

    const fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }

    const fn length(&self) -> u64 {
        self.length_bytes
    }

    fn contents(&self) -> *mut c_void {
        self.buffer
            .contents()
            .cast::<u8>()
            .wrapping_add(self.offset_bytes as usize)
            .cast()
    }

    fn allocation_identity(&self) -> usize {
        self.buffer.as_ptr() as usize
    }

    fn bind(&self, encoder: &metal::ComputeCommandEncoderRef, index: u64) {
        encoder.set_buffer(index, Some(&self.buffer), self.offset_bytes);
    }
}

impl Deref for BufferRegion {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

struct Buffers {
    dense_a: BufferRegion,
    dense_b: BufferRegion,
    e_in: BufferRegion,
    e_out: BufferRegion,
    partial_a: BufferRegion,
    partial_b: BufferRegion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DenseArenaState {
    Owned,
    OuterResidual {
        expected: OuterResidualArenaKey,
        released: bool,
    },
}

impl Buffers {
    fn all(&self) -> [&BufferRegion; INSTRUCTION_INPUT_DEVICE_BUFFERS] {
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
        self.all().map(BufferRegion::allocation_identity)
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

pub(crate) fn instruction_input_sequence_auxiliary_storage_bytes(
    rows: usize,
) -> Result<u64, MetalError> {
    let (e_in_capacity, e_out_capacity) = instruction_input_weight_capacities(rows)?;
    let layout = instruction_input_storage_layout(rows, e_in_capacity, e_out_capacity)?;
    layout
        .owned_bytes
        .checked_sub(layout.buffer_bytes[0])
        .and_then(|bytes| bytes.checked_sub(layout.buffer_bytes[1]))
        .ok_or(MetalError::InvalidInstructionInputState(
            "InstructionInput auxiliary byte count underflowed",
        ))
}

pub(crate) struct InstructionInputSequenceStorage {
    context: SolinasMetal,
    pipelines: Pipelines,
    native_message_limits: PipelineLimits,
    native_transition_limits: PipelineLimits,
    dense_transition_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    successor: Option<SuccessorPipelines>,
    buffers: Buffers,
    dense_arena: DenseArenaState,
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
    resident_rows: InstructionInputRows,
    phase: SequencePhase,
    dense_elements: usize,
    dense_in_a: bool,
    successor_materialized: bool,
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
    pub fn prepare_instruction_input_rows(
        &self,
        rows: &[InstructionInputRow],
    ) -> Result<InstructionInputRows, MetalError> {
        self.prepare_instruction_input_rows_with_fill(rows.len(), |destination| {
            destination.copy_from_slice(rows);
            Ok(())
        })
    }

    pub fn prepare_instruction_input_rows_from_spartan(
        &self,
        rows: &[super::SpartanOuterUniskipRow],
    ) -> Result<InstructionInputRows, MetalError> {
        self.prepare_instruction_input_rows_with_fill(rows.len(), |destination| {
            for (source, destination) in rows.iter().zip(destination) {
                *destination = InstructionInputRow::from_full_words(source.words());
            }
            Ok(())
        })
    }

    pub(crate) fn prepare_instruction_input_rows_with_fill(
        &self,
        rows: usize,
        fill: impl FnOnce(&mut [InstructionInputRow]) -> Result<(), MetalError>,
    ) -> Result<InstructionInputRows, MetalError> {
        if rows == 0 {
            return Err(MetalError::EmptyInput);
        }
        let row_bytes = instruction_input_row_bytes(rows)?;
        self.validate_buffer_length(row_bytes)?;
        self.validate_additional_working_set(row_bytes)?;
        let buffer = self
            .device
            .new_buffer(row_bytes, MTLResourceOptions::StorageModeShared);
        // SAFETY: the shared buffer has exactly `rows` compact elements and no
        // command buffer can observe it until `fill` returns.
        let destination = unsafe {
            slice::from_raw_parts_mut(buffer.contents().cast::<InstructionInputRow>(), rows)
        };
        fill(destination)?;
        Ok(InstructionInputRows::from_buffer(
            buffer,
            rows,
            self.device_registry_id(),
        ))
    }

    /// Allocates a reusable instruction-input sequence and uploads `rows` once.
    pub fn prepare_instruction_input_sequence(
        &self,
        rows: &[super::SpartanOuterUniskipRow],
        config: InstructionInputSequenceConfig,
    ) -> Result<InstructionInputSequence, MetalError> {
        let resident_rows = self.prepare_instruction_input_rows_from_spartan(rows)?;
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
        self.prepare_instruction_input_sequence_storage_impl(
            rows,
            e_in_capacity,
            e_out_capacity,
            config,
            None,
        )
    }

    pub(crate) fn prepare_instruction_input_sequence_storage_from_outer(
        &self,
        outer_rows: &SpartanOuterUniskipRows,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionInputSequenceConfig,
    ) -> Result<InstructionInputSequenceStorage, MetalError> {
        self.prepare_instruction_input_sequence_storage_impl(
            outer_rows.len(),
            e_in_capacity,
            e_out_capacity,
            config,
            Some(outer_rows),
        )
    }

    fn prepare_instruction_input_sequence_storage_impl(
        &self,
        rows: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionInputSequenceConfig,
        outer_rows: Option<&SpartanOuterUniskipRows>,
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
        let successor = if config.first_transition == InstructionInputFirstTransition::Successor {
            let materialize = self.compile_named_pipeline(SUCCESSOR_MATERIALIZE_PIPELINE)?;
            let dense_message = self.compile_named_pipeline(SUCCESSOR_DENSE_MESSAGE_PIPELINE)?;
            let materialize_limits = Self::limits(&materialize);
            let dense_message_limits = Self::limits(&dense_message);
            for (pipeline, limits) in [
                (SUCCESSOR_MATERIALIZE_PIPELINE, materialize_limits),
                (SUCCESSOR_DENSE_MESSAGE_PIPELINE, dense_message_limits),
            ] {
                if limits.thread_execution_width != INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH {
                    return Err(MetalError::UnsupportedInstructionInputExecutionWidth {
                        pipeline,
                        expected: INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH,
                        got: limits.thread_execution_width,
                    });
                }
            }
            Some(SuccessorPipelines {
                materialize,
                dense_message,
                materialize_limits,
                dense_message_limits,
                materialize_threads: Self::resolve_threadgroup_width(
                    Some(INSTRUCTION_INPUT_SUCCESSOR_MATERIALIZE_THREADS),
                    materialize_limits,
                )?,
                dense_message_threads: Self::resolve_threadgroup_width(
                    Some(INSTRUCTION_INPUT_SUCCESSOR_DENSE_MESSAGE_THREADS),
                    dense_message_limits,
                )?,
            })
        } else {
            None
        };

        let borrowed_dense_bytes = layout.buffer_bytes[0]
            .checked_add(layout.buffer_bytes[1])
            .ok_or(MetalError::InvalidInstructionInputState(
                "dense buffer byte count overflowed",
            ))?;
        let borrowed = outer_rows.is_some();
        let owned_bytes = if borrowed {
            layout.owned_bytes.checked_sub(borrowed_dense_bytes).ok_or(
                MetalError::InvalidInstructionInputState("owned buffer byte count underflowed"),
            )?
        } else {
            layout.owned_bytes
        };
        let device = self.device_info();
        let _allocation_span = tracing::info_span!(
            "MetalInstructionInput::allocation_plan",
            device_buffers = if borrowed {
                4
            } else {
                INSTRUCTION_INPUT_DEVICE_BUFFERS
            },
            planned_device_bytes = layout.owned_bytes,
            owned_device_bytes = owned_bytes,
            reused_device_bytes = layout.owned_bytes - owned_bytes,
            borrowed_outer_residual = borrowed,
            current_device_bytes = device.current_allocated_size,
            recommended_device_bytes = device.recommended_max_working_set_size,
        )
        .entered();
        self.validate_additional_working_set(owned_bytes)?;

        let (dense_a, dense_b, dense_arena) = if let Some(outer_rows) = outer_rows {
            if config.storage_initialization == InstructionInputStorageInitialization::Full {
                return Err(MetalError::InvalidInstructionInputState(
                    "borrowed dense storage does not support full initialization",
                ));
            }
            let expected = outer_rows.residual_arena_key();
            let expected_parent_bytes = spartan_outer_uniskip_residual_row_bytes(rows)?;
            if expected.rows != rows
                || expected.device_registry_id != self.device_registry_id()
                || expected.storage_bytes != expected_parent_bytes
                || borrowed_dense_bytes > expected.storage_bytes
            {
                return Err(MetalError::InvalidInstructionInputState(
                    "Outer residual arena has the wrong shape or device",
                ));
            }
            (
                BufferRegion::range(outer_rows.residual_buffer(), 0, layout.buffer_bytes[0])?,
                BufferRegion::range(
                    outer_rows.residual_buffer(),
                    layout.buffer_bytes[0],
                    layout.buffer_bytes[1],
                )?,
                DenseArenaState::OuterResidual {
                    expected,
                    released: false,
                },
            )
        } else {
            (
                new_buffer(self, layout.dense_a_elements)?,
                new_buffer(self, layout.dense_b_elements)?,
                DenseArenaState::Owned,
            )
        };

        let buffers = Buffers {
            dense_a,
            dense_b,
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
        let initialization =
            initialize_storage(self, &buffers, config.storage_initialization, !borrowed)?;

        Ok(InstructionInputSequenceStorage {
            context: self.clone(),
            pipelines,
            native_message_limits,
            native_transition_limits,
            dense_transition_limits,
            reduction_limits,
            successor,
            buffers,
            dense_arena,
            rows,
            e_in_capacity,
            e_out_capacity,
            native_message_threads,
            native_transition_threads,
            dense_transition_threads,
            config,
            initialization,
            owned_bytes,
        })
    }
}

impl InstructionInputSequenceStorage {
    fn successor_pipelines(&self) -> Result<&SuccessorPipelines, MetalError> {
        self.successor
            .as_ref()
            .ok_or(MetalError::InvalidInstructionInputState(
                "the InstructionInput successor was not selected during preparation",
            ))
    }

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

    pub(crate) const fn requires_outer_residual_release(&self) -> bool {
        matches!(
            self.dense_arena,
            DenseArenaState::OuterResidual {
                released: false,
                ..
            }
        )
    }

    pub(crate) fn unlock_outer_residual(
        &mut self,
        receipt: OuterResidualReleaseReceipt,
        resident_rows: &InstructionInputRows,
    ) -> Result<(), MetalError> {
        let DenseArenaState::OuterResidual { expected, released } = &mut self.dense_arena else {
            return Err(MetalError::InvalidInstructionInputState(
                "owned dense storage received an Outer release receipt",
            ));
        };
        if *released {
            return Err(MetalError::InvalidInstructionInputState(
                "Outer residual arena was released more than once",
            ));
        }
        if receipt.key != *expected
            || resident_rows.len() != expected.rows
            || resident_rows.device_registry_id() != expected.device_registry_id
            || resident_rows.allocation_identity() != expected.compact_storage_id
            || self.buffers.dense_a.allocation_identity() != expected.storage_id
            || self.buffers.dense_b.allocation_identity() != expected.storage_id
        {
            return Err(MetalError::InvalidInstructionInputState(
                "Outer residual release receipt changed before InstructionInput",
            ));
        }
        *released = true;
        Ok(())
    }

    fn submit_native_pipeline_primer(
        &self,
        resident_rows: &InstructionInputRows,
    ) -> Result<InstructionInputNativePrimerCommand, MetalError> {
        if resident_rows.len() != self.rows
            || self.rows < INSTRUCTION_INPUT_PRIMER_SOURCE_ELEMENTS
            || self.e_in_capacity < INSTRUCTION_INPUT_PRIMER_E_IN_ELEMENTS
            || self.e_out_capacity < INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS
        {
            return Err(MetalError::InvalidInstructionInputRows(resident_rows.len()));
        }
        if resident_rows.device_registry_id() != self.context.device_registry_id() {
            return Err(MetalError::InstructionInputRowsDevice {
                expected: self.context.device_registry_id(),
                got: resident_rows.device_registry_id(),
            });
        }

        let submitted_at = Instant::now();
        let zeros_in = [AkitaField::zero(); INSTRUCTION_INPUT_PRIMER_E_IN_ELEMENTS];
        let zeros_out = [AkitaField::zero(); INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS];
        write_fields(&self.buffers.e_in, self.e_in_capacity, &zeros_in)?;
        write_fields(&self.buffers.e_out, self.e_out_capacity, &zeros_out)?;
        let params = InstructionInputParams {
            source_elements: INSTRUCTION_INPUT_PRIMER_SOURCE_ELEMENTS as u32,
            e_in_length: INSTRUCTION_INPUT_PRIMER_E_IN_ELEMENTS as u32,
            e_out_length: INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS as u32,
            reserved: 0,
        };
        let reduction_params = ReductionParams {
            input_count: INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS as u32,
            output_count: 1,
            reserved: [0; 2],
        };
        let gamma = Fp128::from_jolt_field(&AkitaField::zero());
        let successor_primer = self
            .successor
            .as_ref()
            .map(|successor| -> Result<_, MetalError> {
                Ok((
                    successor,
                    checked_materialize_shape(
                        INSTRUCTION_INPUT_SUCCESSOR_PRIMER_SOURCE_ELEMENTS,
                        self.context.device.max_buffer_length(),
                    )?,
                    checked_dense_message_shape(
                        INSTRUCTION_INPUT_SUCCESSOR_PRIMER_BOUND_ELEMENTS,
                        INSTRUCTION_INPUT_SUCCESSOR_PRIMER_E_IN_ELEMENTS,
                        INSTRUCTION_INPUT_SUCCESSOR_PRIMER_E_OUT_ELEMENTS,
                        successor.dense_message_threads,
                        self.context.device.max_buffer_length(),
                    )?,
                    ReductionParams {
                        input_count: INSTRUCTION_INPUT_SUCCESSOR_PRIMER_E_OUT_ELEMENTS as u32,
                        output_count: 1,
                        reserved: [0; 2],
                    },
                ))
            })
            .transpose()?;
        let command_buffer = self.context.queue.new_command_buffer().to_owned();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.native_message);
            encoder.set_buffer(0, Some(resident_rows.buffer()), 0);
            self.buffers.e_in.bind(encoder, 1);
            self.buffers.e_out.bind(encoder, 2);
            self.buffers.partial_a.bind(encoder, 3);
            set_inline_bytes(encoder, 4, &gamma);
            set_inline_bytes(encoder, 5, &params);
            encoder.set_threadgroup_memory_length(
                0,
                (INSTRUCTION_INPUT_COEFFICIENTS
                    * (self.native_message_threads / SIMD_WIDTH)
                    * size_of::<Fp128>()) as u64,
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.native_message_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_compute_pipeline_state(&self.pipelines.reduction);
            self.buffers.partial_a.bind(encoder, 0);
            self.buffers.partial_b.bind(encoder, 1);
            set_inline_bytes(encoder, 2, &reduction_params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.reduction_limits.thread_execution_width as u64,
                    height: 1,
                    depth: 1,
                },
            );
            if let Some((successor, successor_shape, _, _)) = successor_primer {
                encoder.set_compute_pipeline_state(&successor.materialize);
                encoder.set_buffer(0, Some(resident_rows.buffer()), 0);
                self.buffers.dense_a.bind(encoder, 1);
                set_inline_bytes(encoder, 2, &gamma);
                set_inline_bytes(encoder, 3, &successor_shape.params());
                encoder.set_threadgroup_memory_length(0, 0);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: successor_shape
                            .grid_threads()
                            .div_ceil(successor.materialize_threads)
                            as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: successor.materialize_threads as u64,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.end_encoding();
            if let Some((successor, _, successor_dense_shape, successor_reduction_params)) =
                successor_primer
            {
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&successor.dense_message);
                self.buffers.dense_a.bind(encoder, 0);
                self.buffers.e_in.bind(encoder, 1);
                self.buffers.e_out.bind(encoder, 2);
                self.buffers.partial_a.bind(encoder, 3);
                set_inline_bytes(encoder, 4, &gamma);
                set_inline_bytes(encoder, 5, &successor_dense_shape.params());
                encoder.set_threadgroup_memory_length(
                    0,
                    successor_dense_shape.threadgroup_bytes() as u64,
                );
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: successor_dense_shape.grid_threadgroups() as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: successor.dense_message_threads as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.set_compute_pipeline_state(&self.pipelines.reduction);
                self.buffers.partial_a.bind(encoder, 0);
                self.buffers.partial_b.bind(encoder, 1);
                set_inline_bytes(encoder, 2, &successor_reduction_params);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.reduction_limits.thread_execution_width as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.end_encoding();
            }
            command_buffer.commit();
        });
        Ok(InstructionInputNativePrimerCommand {
            command_buffer,
            submitted_at,
            submit_wall: submitted_at.elapsed(),
            resident_row_identity: resident_rows.allocation_identity(),
            storage_buffer_identities: self.buffers.identities(),
        })
    }

    fn complete_native_pipeline_primer(
        &self,
        resident_rows: &InstructionInputRows,
        primer: InstructionInputNativePrimerCommand,
    ) -> Result<InstructionInputPrimerStats, MetalError> {
        if resident_rows.allocation_identity() != primer.resident_row_identity
            || self.buffers.identities() != primer.storage_buffer_identities
        {
            return Err(MetalError::InvalidInstructionInputState(
                "native pipeline primer resources changed before completion",
            ));
        }
        let completed_before_join =
            primer.command_buffer.status() == MTLCommandBufferStatus::Completed;
        let join_started = Instant::now();
        let overlap_wall = join_started
            .saturating_duration_since(primer.submitted_at)
            .saturating_sub(primer.submit_wall);
        primer.command_buffer.wait_until_completed();
        if primer.command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(primer.command_buffer.status()));
        }
        let gpu_start = command_buffer_timestamp(&primer.command_buffer, "GPUStartTime")?;
        let gpu_end = command_buffer_timestamp(&primer.command_buffer, "GPUEndTime")?;
        if !gpu_start.is_finite() || !gpu_end.is_finite() || gpu_start <= 0.0 || gpu_end < gpu_start
        {
            return Err(MetalError::InvalidGpuTimestamps {
                start: gpu_start,
                end: gpu_end,
            });
        }
        // SAFETY: the completed final primer reduction wrote three fields at
        // the start of partial_b, and no protocol command has started.
        let output = unsafe {
            slice::from_raw_parts(
                self.buffers.partial_b.contents().cast::<Fp128>(),
                INSTRUCTION_INPUT_COEFFICIENTS,
            )
        };
        self.context
            .validate_inputs("instruction input pipeline primer", output)?;
        if output
            .iter()
            .any(|value| value.into_jolt_field::<AkitaField>() != AkitaField::zero())
        {
            return Err(MetalError::InvalidInstructionInputState(
                "instruction input pipeline primer produced a nonzero message",
            ));
        }
        if self.successor.is_some() {
            let successor_values =
                INSTRUCTION_INPUT_TABLES * INSTRUCTION_INPUT_SUCCESSOR_PRIMER_BOUND_ELEMENTS;
            // SAFETY: the completed successor primer initialized the compact
            // 64-row fixture at the start of dense A.
            let output = unsafe {
                slice::from_raw_parts(
                    self.buffers.dense_a.contents().cast::<Fp128>(),
                    successor_values,
                )
            };
            self.context
                .validate_inputs("instruction input successor primer", output)?;
        }
        let join_wall = join_started.elapsed();
        Ok(InstructionInputPrimerStats {
            source_elements: INSTRUCTION_INPUT_PRIMER_SOURCE_ELEMENTS,
            e_in_elements: INSTRUCTION_INPUT_PRIMER_E_IN_ELEMENTS,
            e_out_elements: INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS,
            wall: primer.submitted_at.elapsed(),
            submit_wall: primer.submit_wall,
            overlap_wall,
            join_wall,
            gpu_active: Duration::from_secs_f64(gpu_end - gpu_start),
            completed_before_join,
            resident_row_identity: primer.resident_row_identity,
            storage_buffer_identities: primer.storage_buffer_identities,
        })
    }

    pub(crate) fn attach(
        self,
        resident_rows: InstructionInputRows,
    ) -> Result<InstructionInputSequence, MetalError> {
        if self.requires_outer_residual_release() {
            return Err(MetalError::InvalidInstructionInputState(
                "Outer residual arena was attached before release",
            ));
        }
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
            successor_materialized: false,
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
        self.successor_materialized = false;
        self.gpu_active_time = Duration::ZERO;
    }

    pub(crate) fn submit_native_pipeline_primer(
        self,
    ) -> Result<PendingInstructionInputPrimer, MetalError> {
        if self.phase != SequencePhase::BeforeMessage {
            return Err(MetalError::InvalidInstructionInputState(
                "native pipeline primer requires the initial sequence state",
            ));
        }
        let command = self
            .storage
            .submit_native_pipeline_primer(&self.resident_rows)?;
        Ok(PendingInstructionInputPrimer {
            sequence: Some(self),
            command: Some(command),
        })
    }

    /// Primes the InstructionInput pipelines on a zero-weighted 64-row prefix
    /// without advancing the protocol state.
    pub fn prime_native_pipeline(&mut self) -> Result<InstructionInputPrimerStats, MetalError> {
        if self.phase != SequencePhase::BeforeMessage {
            return Err(MetalError::InvalidInstructionInputState(
                "native pipeline primer requires the initial sequence state",
            ));
        }
        let command = self
            .storage
            .submit_native_pipeline_primer(&self.resident_rows)?;
        self.storage
            .complete_native_pipeline_primer(&self.resident_rows, command)
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
        if self.phase == SequencePhase::Native
            && self.storage.config.first_transition == InstructionInputFirstTransition::Successor
        {
            return self
                .run_successor_transition(challenge, gamma, e_in, e_out)
                .map(|(descriptors, _)| descriptors);
        }
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

    #[cfg(any(test, feature = "test-utils"))]
    pub fn probe_successor_materialize(
        &mut self,
        challenge: AkitaField,
        output: &mut [AkitaField],
    ) -> Result<InstructionInputSuccessorMaterializeStats, MetalError> {
        let stats = self.run_successor_materialize(challenge)?;
        self.read_successor_materialized_tables(output)?;
        Ok(stats)
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn run_successor_materialize(
        &mut self,
        challenge: AkitaField,
    ) -> Result<InstructionInputSuccessorMaterializeStats, MetalError> {
        if self.phase != SequencePhase::BeforeMessage {
            return Err(MetalError::InvalidInstructionInputState(
                "the successor materializer probe requires unbound resident rows",
            ));
        }
        self.successor_materialized = false;
        let shape = checked_materialize_shape(
            self.storage.rows,
            self.storage.context.device.max_buffer_length(),
        )?;
        if self.storage.buffers.dense_a.length() != shape.dense_table_bytes() {
            return Err(MetalError::InvalidInstructionInputState(
                "the successor materializer target differs from dense A",
            ));
        }
        let successor = self.storage.successor_pipelines()?;
        let materialize_threads = successor.materialize_threads;
        let pipeline_limits = successor.materialize_limits;

        let started = Instant::now();
        let challenge = Fp128::from_jolt_field(&challenge);
        self.storage
            .context
            .validate_inputs("instruction input successor challenge", &[challenge])?;
        let command_buffer = self.storage.context.queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&successor.materialize);
            encoder.set_buffer(0, Some(self.resident_rows.buffer()), 0);
            self.storage.buffers.dense_a.bind(encoder, 1);
            set_inline_bytes(encoder, 2, &challenge);
            set_inline_bytes(encoder, 3, &shape.params());
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: shape.grid_threads().div_ceil(materialize_threads) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: materialize_threads as u64,
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
        let gpu_start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let gpu_end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !gpu_start.is_finite() || !gpu_end.is_finite() || gpu_start <= 0.0 || gpu_end < gpu_start
        {
            return Err(MetalError::InvalidGpuTimestamps {
                start: gpu_start,
                end: gpu_end,
            });
        }
        self.successor_materialized = true;
        Ok(InstructionInputSuccessorMaterializeStats {
            source_elements: self.storage.rows,
            bound_elements: shape.grid_threads(),
            wall: started.elapsed(),
            gpu_active: Duration::from_secs_f64(gpu_end - gpu_start),
            resident_row_identity: self.resident_rows.allocation_identity(),
            dense_buffer_identity: self.storage.buffers.dense_a.allocation_identity(),
            pipeline_limits,
        })
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn read_successor_materialized_tables(
        &self,
        output: &mut [AkitaField],
    ) -> Result<(), MetalError> {
        if self.phase != SequencePhase::BeforeMessage || !self.successor_materialized {
            return Err(MetalError::InvalidInstructionInputState(
                "the successor materializer read requires a completed diagnostic dispatch",
            ));
        }
        let expected = INSTRUCTION_INPUT_TABLES
            .checked_mul(self.storage.rows / 2)
            .ok_or(MetalError::InputTooLong(self.storage.rows))?;
        if output.len() != expected {
            return Err(MetalError::InstructionInputStorageLength {
                expected,
                got: output.len(),
            });
        }
        // SAFETY: a completed successor materializer initializes exactly
        // `expected` fields at the front of dense A.
        let values = unsafe {
            slice::from_raw_parts(
                self.storage.buffers.dense_a.contents().cast::<Fp128>(),
                expected,
            )
        };
        self.storage
            .context
            .validate_inputs("instruction input successor tables", values)?;
        for (output, value) in output.iter_mut().zip(values) {
            *output = value.into_jolt_field();
        }
        Ok(())
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn run_successor_dense_message(
        &mut self,
        gamma: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<
        (
            [AkitaField; INSTRUCTION_INPUT_COEFFICIENTS],
            InstructionInputSuccessorDenseMessageStats,
        ),
        MetalError,
    > {
        if self.phase != SequencePhase::BeforeMessage || !self.successor_materialized {
            return Err(MetalError::InvalidInstructionInputState(
                "the successor dense-message probe requires materialized tables",
            ));
        }
        let successor = self.storage.successor_pipelines()?;
        let dense_message_threads = successor.dense_message_threads;
        let pipeline_limits = successor.dense_message_limits;
        let table_elements = self.storage.rows / 2;
        let shape = checked_dense_message_shape(
            table_elements,
            e_in.len(),
            e_out.len(),
            dense_message_threads,
            self.storage.context.device.max_buffer_length(),
        )?;
        if self.storage.buffers.dense_a.length() != shape.table_bytes() {
            return Err(MetalError::InvalidInstructionInputState(
                "the successor dense-message source differs from dense A",
            ));
        }
        let requested_threadgroup_bytes = u64::try_from(shape.threadgroup_bytes())
            .map_err(|_| MetalError::InputTooLong(shape.threadgroup_bytes()))?
            .checked_add(pipeline_limits.static_threadgroup_memory_length)
            .ok_or(MetalError::InputTooLong(shape.threadgroup_bytes()))?;
        let maximum_threadgroup_bytes = self.storage.context.device.max_threadgroup_memory_length();
        if requested_threadgroup_bytes > maximum_threadgroup_bytes {
            return Err(MetalError::InstructionInputSuccessorThreadgroupMemory {
                requested: requested_threadgroup_bytes,
                maximum: maximum_threadgroup_bytes,
            });
        }

        let started = Instant::now();
        write_fields(&self.storage.buffers.e_in, self.storage.e_in_capacity, e_in)?;
        write_fields(
            &self.storage.buffers.e_out,
            self.storage.e_out_capacity,
            e_out,
        )?;
        let gamma = Fp128::from_jolt_field(&gamma);
        self.storage
            .context
            .validate_inputs("instruction input successor gamma", &[gamma])?;
        let command_buffer = self.storage.context.queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&successor.dense_message);
            self.storage.buffers.dense_a.bind(encoder, 0);
            self.storage.buffers.e_in.bind(encoder, 1);
            self.storage.buffers.e_out.bind(encoder, 2);
            self.storage.buffers.partial_a.bind(encoder, 3);
            set_inline_bytes(encoder, 4, &gamma);
            set_inline_bytes(encoder, 5, &shape.params());
            encoder.set_threadgroup_memory_length(0, shape.threadgroup_bytes() as u64);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: shape.grid_threadgroups() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: dense_message_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            self.encode_reductions(encoder, e_out.len());
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(command_buffer.status()));
        }
        let gpu_start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let gpu_end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !gpu_start.is_finite() || !gpu_end.is_finite() || gpu_start <= 0.0 || gpu_end < gpu_start
        {
            return Err(MetalError::InvalidGpuTimestamps {
                start: gpu_start,
                end: gpu_end,
            });
        }
        let final_buffer = self.final_partial_buffer(e_out.len());
        // SAFETY: the completed recursive reduction leaves three canonical
        // fields at the start of the selected partial buffer.
        let values = unsafe {
            slice::from_raw_parts(
                final_buffer.contents().cast::<Fp128>(),
                INSTRUCTION_INPUT_COEFFICIENTS,
            )
        };
        self.storage
            .context
            .validate_inputs("instruction input successor dense message", values)?;
        let descriptors = std::array::from_fn(|index| values[index].into_jolt_field());
        Ok((
            descriptors,
            InstructionInputSuccessorDenseMessageStats {
                table_elements,
                e_in_elements: e_in.len(),
                e_out_elements: e_out.len(),
                wall: started.elapsed(),
                gpu_active: Duration::from_secs_f64(gpu_end - gpu_start),
                dense_buffer_identity: self.storage.buffers.dense_a.allocation_identity(),
                dynamic_threadgroup_memory_length: shape.threadgroup_bytes(),
                pipeline_limits,
            },
        ))
    }

    pub fn run_successor_transition(
        &mut self,
        challenge: AkitaField,
        gamma: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<
        (
            [AkitaField; INSTRUCTION_INPUT_COEFFICIENTS],
            InstructionInputSuccessorTransitionStats,
        ),
        MetalError,
    > {
        if self.phase != SequencePhase::Native {
            return Err(MetalError::InvalidInstructionInputState(
                "the successor transition requires the completed native message",
            ));
        }
        let successor = self.storage.successor_pipelines()?;
        let materialize_threads = successor.materialize_threads;
        let dense_message_threads = successor.dense_message_threads;
        let materialize_pipeline_limits = successor.materialize_limits;
        let dense_message_pipeline_limits = successor.dense_message_limits;
        let materialize_shape = checked_materialize_shape(
            self.storage.rows,
            self.storage.context.device.max_buffer_length(),
        )?;
        let table_elements = materialize_shape.grid_threads();
        let dense_shape = checked_dense_message_shape(
            table_elements,
            e_in.len(),
            e_out.len(),
            dense_message_threads,
            self.storage.context.device.max_buffer_length(),
        )?;
        if self.storage.buffers.dense_a.length() != materialize_shape.dense_table_bytes()
            || self.storage.buffers.dense_a.length() != dense_shape.table_bytes()
        {
            return Err(MetalError::InvalidInstructionInputState(
                "the successor transition storage differs from dense A",
            ));
        }
        let requested_threadgroup_bytes = u64::try_from(dense_shape.threadgroup_bytes())
            .map_err(|_| MetalError::InputTooLong(dense_shape.threadgroup_bytes()))?
            .checked_add(dense_message_pipeline_limits.static_threadgroup_memory_length)
            .ok_or(MetalError::InputTooLong(dense_shape.threadgroup_bytes()))?;
        let maximum_threadgroup_bytes = self.storage.context.device.max_threadgroup_memory_length();
        if requested_threadgroup_bytes > maximum_threadgroup_bytes {
            return Err(MetalError::InstructionInputSuccessorThreadgroupMemory {
                requested: requested_threadgroup_bytes,
                maximum: maximum_threadgroup_bytes,
            });
        }

        let started = Instant::now();
        write_fields(&self.storage.buffers.e_in, self.storage.e_in_capacity, e_in)?;
        write_fields(
            &self.storage.buffers.e_out,
            self.storage.e_out_capacity,
            e_out,
        )?;
        let challenge = Fp128::from_jolt_field(&challenge);
        let gamma = Fp128::from_jolt_field(&gamma);
        self.storage.context.validate_inputs(
            "instruction input successor transition scalars",
            &[challenge, gamma],
        )?;
        let command_buffer = self.storage.context.queue.new_command_buffer();
        autoreleasepool(|| {
            let materialize = command_buffer.new_compute_command_encoder();
            materialize.set_compute_pipeline_state(&successor.materialize);
            materialize.set_buffer(0, Some(self.resident_rows.buffer()), 0);
            self.storage.buffers.dense_a.bind(materialize, 1);
            set_inline_bytes(materialize, 2, &challenge);
            set_inline_bytes(materialize, 3, &materialize_shape.params());
            materialize.dispatch_thread_groups(
                MTLSize {
                    width: materialize_shape
                        .grid_threads()
                        .div_ceil(materialize_threads) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: materialize_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            materialize.end_encoding();

            let message = command_buffer.new_compute_command_encoder();
            message.set_compute_pipeline_state(&successor.dense_message);
            self.storage.buffers.dense_a.bind(message, 0);
            self.storage.buffers.e_in.bind(message, 1);
            self.storage.buffers.e_out.bind(message, 2);
            self.storage.buffers.partial_a.bind(message, 3);
            set_inline_bytes(message, 4, &gamma);
            set_inline_bytes(message, 5, &dense_shape.params());
            message.set_threadgroup_memory_length(0, dense_shape.threadgroup_bytes() as u64);
            message.dispatch_thread_groups(
                MTLSize {
                    width: dense_shape.grid_threadgroups() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: dense_message_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            self.encode_reductions(message, e_out.len());
            message.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(command_buffer.status()));
        }
        let gpu_start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let gpu_end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !gpu_start.is_finite() || !gpu_end.is_finite() || gpu_start <= 0.0 || gpu_end < gpu_start
        {
            return Err(MetalError::InvalidGpuTimestamps {
                start: gpu_start,
                end: gpu_end,
            });
        }
        let final_buffer = self.final_partial_buffer(e_out.len());
        // SAFETY: the completed recursive reduction leaves three canonical
        // fields at the start of the selected partial buffer.
        let values = unsafe {
            slice::from_raw_parts(
                final_buffer.contents().cast::<Fp128>(),
                INSTRUCTION_INPUT_COEFFICIENTS,
            )
        };
        self.storage
            .context
            .validate_inputs("instruction input successor transition", values)?;
        let descriptors = std::array::from_fn(|index| values[index].into_jolt_field());
        let gpu_active = Duration::from_secs_f64(gpu_end - gpu_start);
        self.phase = SequencePhase::Dense;
        self.dense_elements = table_elements;
        self.dense_in_a = true;
        self.successor_materialized = false;
        self.gpu_active_time += gpu_active;
        Ok((
            descriptors,
            InstructionInputSuccessorTransitionStats {
                source_elements: self.storage.rows,
                table_elements,
                e_in_elements: e_in.len(),
                e_out_elements: e_out.len(),
                wall: started.elapsed(),
                gpu_active,
                resident_row_identity: self.resident_rows.allocation_identity(),
                dense_buffer_identity: self.storage.buffers.dense_a.allocation_identity(),
                dynamic_threadgroup_memory_length: dense_shape.threadgroup_bytes(),
                materialize_pipeline_limits,
                dense_message_pipeline_limits,
            },
        ))
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

    pub const fn first_transition(&self) -> InstructionInputFirstTransition {
        self.storage.config.first_transition
    }

    #[cfg(feature = "test-utils")]
    pub fn reset_with_first_transition(
        &mut self,
        first_transition: InstructionInputFirstTransition,
    ) -> Result<(), MetalError> {
        if first_transition == InstructionInputFirstTransition::Successor
            && self.storage.successor.is_none()
        {
            return Err(MetalError::InvalidInstructionInputState(
                "the InstructionInput successor was not selected during preparation",
            ));
        }
        self.storage.config.first_transition = first_transition;
        self.reset();
        Ok(())
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

    pub fn successor_materialize_pipeline_limits(&self) -> Option<PipelineLimits> {
        self.storage
            .successor
            .as_ref()
            .map(|successor| successor.materialize_limits)
    }

    pub fn successor_dense_message_pipeline_limits(&self) -> Option<PipelineLimits> {
        self.storage
            .successor
            .as_ref()
            .map(|successor| successor.dense_message_limits)
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
                    self.storage.buffers.e_in.bind(encoder, 1);
                    self.storage.buffers.e_out.bind(encoder, 2);
                    self.storage.buffers.partial_a.bind(encoder, 3);
                    set_inline_bytes(encoder, 4, &gamma);
                    set_inline_bytes(encoder, 5, &params);
                }
                DispatchKind::NativeTransition => {
                    encoder.set_buffer(0, Some(self.resident_rows.buffer()), 0);
                    self.storage.buffers.dense_a.bind(encoder, 1);
                    self.storage.buffers.e_in.bind(encoder, 2);
                    self.storage.buffers.e_out.bind(encoder, 3);
                    self.storage.buffers.partial_a.bind(encoder, 4);
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
                    self.dense_source_buffer().bind(encoder, 0);
                    self.dense_destination_buffer().bind(encoder, 1);
                    self.storage.buffers.e_in.bind(encoder, 2);
                    self.storage.buffers.e_out.bind(encoder, 3);
                    self.storage.buffers.partial_a.bind(encoder, 4);
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
            input.bind(encoder, 0);
            output.bind(encoder, 1);
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

    fn final_partial_buffer(&self, mut count: usize) -> &BufferRegion {
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

    fn dense_source_buffer(&self) -> &BufferRegion {
        if self.dense_in_a {
            &self.storage.buffers.dense_a
        } else {
            &self.storage.buffers.dense_b
        }
    }

    fn dense_destination_buffer(&self) -> &BufferRegion {
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
    initialize_dense: bool,
) -> Result<InstructionInputStorageInitializationStats, MetalError> {
    let buffer_identities = buffers.identities();
    let fill_lengths: [u64; INSTRUCTION_INPUT_DEVICE_BUFFERS] = std::array::from_fn(|index| {
        let buffer = buffers.all()[index];
        if index < 2 && !initialize_dense {
            return 0;
        }
        match mode {
            InstructionInputStorageInitialization::Lazy => 0,
            InstructionInputStorageInitialization::Minimal => size_of::<Fp128>() as u64,
            InstructionInputStorageInitialization::Full => buffer.length(),
        }
    });
    let bytes = fill_lengths.iter().try_fold(0u64, |total, length| {
        total
            .checked_add(*length)
            .ok_or(MetalError::InvalidInstructionInputState(
                "storage initialization byte count overflowed",
            ))
    })?;
    let device_buffers = fill_lengths.iter().filter(|length| **length != 0).count();
    let span = tracing::info_span!(
        "MetalInstructionInput::storage_initialize",
        mode = %mode.as_str(),
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
                if length == 0 {
                    continue;
                }
                encoder.fill_buffer(
                    buffer.buffer(),
                    NSRange::new(buffer.offset_bytes(), length),
                    0,
                );
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
        Duration::from_secs_f64(end - start)
    };
    let wall = started.elapsed();
    let gpu_active_ns = u64::try_from(gpu_active.as_nanos()).unwrap_or(u64::MAX);
    let completion = tracing::info_span!(
        "MetalInstructionInput::storage_initialize_complete",
        mode = %mode.as_str(),
        command_completed = mode != InstructionInputStorageInitialization::Lazy,
        gpu_active_ns,
    );
    let _completed = completion.enter();
    Ok(InstructionInputStorageInitializationStats {
        mode,
        device_buffers,
        bytes,
        wall,
        gpu_active,
        buffer_identities,
        buffer_offsets: buffers.all().map(BufferRegion::offset_bytes),
        buffer_lengths: buffers.all().map(BufferRegion::length),
    })
}

fn write_fields(
    buffer: &BufferRegion,
    capacity: usize,
    values: &[AkitaField],
) -> Result<(), MetalError> {
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

fn new_buffer(context: &SolinasMetal, elements: usize) -> Result<BufferRegion, MetalError> {
    let bytes = byte_length::<Fp128>(elements)?;
    context.validate_buffer_length(bytes)?;
    Ok(BufferRegion::whole(
        context
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
    ))
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

pub(crate) fn instruction_input_row_bytes(rows: usize) -> Result<u64, MetalError> {
    byte_length::<InstructionInputRow>(rows)
}

const _: () = assert!(size_of::<InstructionInputRow>() == 48);
const _: () = assert!(std::mem::align_of::<InstructionInputRow>() == 16);

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
    use std::{
        mem::{align_of, size_of},
        slice,
        time::Duration,
    };

    use jolt_field::AkitaField;
    use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

    use super::{
        initialize_storage, instruction_input_row_bytes,
        instruction_input_sequence_auxiliary_storage_bytes,
        instruction_input_sequence_storage_bytes, instruction_input_storage_layout,
        instruction_input_weight_capacities, validate_u32_element_count,
        InstructionInputFirstTransition, InstructionInputParams, InstructionInputRow,
        InstructionInputSequenceConfig, InstructionInputStorageInitialization, ReductionParams,
        INSTRUCTION_INPUT_DEVICE_BUFFERS, INSTRUCTION_INPUT_TABLES,
    };
    use crate::metal::solinas::{
        instruction_input_successor::{
            oracle::{dense_message, materialize_first_bind, row_fields},
            InstructionInputSuccessorRow,
        },
        MetalError, OuterResidualReleaseReceipt, SolinasMetal, SpartanOuterUniskipRow,
    };

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
        assert_eq!(
            instruction_input_sequence_auxiliary_storage_bytes(rows).unwrap(),
            983_040
        );
        assert_eq!(
            instruction_input_sequence_auxiliary_storage_bytes(1 << 27).unwrap(),
            1_048_576
        );
        assert_eq!(
            instruction_input_sequence_auxiliary_storage_bytes(1 << 28).unwrap(),
            1_966_080
        );
    }

    #[test]
    fn borrowed_outer_residual_requires_release_and_preserves_guard() {
        let rows = packed_rows(16);
        let initial_tables: Vec<AkitaField> = (0..INSTRUCTION_INPUT_TABLES)
            .flat_map(|table| {
                rows.iter()
                    .map(move |row| row.instruction_input_fields::<AkitaField>()[table])
            })
            .collect();
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let mut outer = context
            .prepare_spartan_outer_uniskip_rows(&rows)
            .expect("Outer rows should prepare");
        let key = outer.residual_arena_key();
        // SAFETY: the shared parent allocation is live for `key.storage_bytes`.
        let parent_before_prepare = unsafe {
            slice::from_raw_parts(
                outer.residual_buffer().contents().cast::<u8>(),
                usize::try_from(key.storage_bytes).unwrap(),
            )
        }
        .to_vec();
        let resident = outer.share_instruction_input_rows();
        let (e_in_capacity, e_out_capacity) =
            instruction_input_weight_capacities(rows.len()).unwrap();
        let config = InstructionInputSequenceConfig {
            storage_initialization: InstructionInputStorageInitialization::Lazy,
            ..InstructionInputSequenceConfig::default()
        };
        let mut storage = context
            .prepare_instruction_input_sequence_storage_from_outer(
                &outer,
                e_in_capacity,
                e_out_capacity,
                config,
            )
            .expect("borrowed storage should prepare");
        let layout =
            instruction_input_storage_layout(rows.len(), e_in_capacity, e_out_capacity).unwrap();
        let guard_offset = layout.buffer_bytes[0] + layout.buffer_bytes[1];
        let guard_len = usize::try_from(key.storage_bytes - guard_offset).unwrap();
        // SAFETY: storage preparation must leave the still-owned parent untouched.
        let parent_after_prepare = unsafe {
            slice::from_raw_parts(
                outer.residual_buffer().contents().cast::<u8>(),
                usize::try_from(key.storage_bytes).unwrap(),
            )
        };
        assert_eq!(parent_after_prepare, parent_before_prepare);
        let guard_before = parent_before_prepare[usize::try_from(guard_offset).unwrap()..].to_vec();

        assert!(storage.requires_outer_residual_release());
        assert_eq!(storage.owned_bytes(), 480);
        assert_eq!(storage.initialization.device_buffers, 0);
        assert_eq!(storage.initialization.bytes, 0);
        assert_eq!(
            storage.buffers.dense_a.allocation_identity(),
            key.storage_id
        );
        assert_eq!(
            storage.buffers.dense_b.allocation_identity(),
            key.storage_id
        );
        assert_eq!(storage.buffers.dense_a.offset_bytes(), 0);
        assert_eq!(
            storage.buffers.dense_b.offset_bytes(),
            layout.buffer_bytes[0]
        );
        let mut wrong = key;
        wrong.rows *= 2;
        assert!(storage
            .unlock_outer_residual(OuterResidualReleaseReceipt { key: wrong }, &resident)
            .is_err());
        let mut stale = key;
        stale.generation += 1;
        assert!(storage
            .unlock_outer_residual(OuterResidualReleaseReceipt { key: stale }, &resident)
            .is_err());
        storage
            .unlock_outer_residual(OuterResidualReleaseReceipt { key }, &resident)
            .expect("the exact release receipt should unlock the arena");
        assert!(!storage.requires_outer_residual_release());
        assert!(storage
            .unlock_outer_residual(OuterResidualReleaseReceipt { key }, &resident)
            .is_err());

        let mut sequence = storage
            .attach(resident)
            .expect("released storage should attach");
        let gamma = AkitaField::from_u64(0xC001_CAFE);
        let r_product = [
            AkitaField::from_u64(5),
            AkitaField::from_u64(7),
            AkitaField::from_u64(11),
            AkitaField::from_u64(13),
        ];
        let mut gruen = GruenSplitEqPolynomial::new(&r_product, BindingOrder::LowToHigh);
        let expected = descriptors(
            &initial_tables,
            rows.len(),
            gamma,
            gruen.e_in_current(),
            gruen.e_out_current(),
        );
        assert_eq!(
            sequence
                .message(gamma, gruen.e_in_current(), gruen.e_out_current())
                .expect("native message should execute"),
            expected
        );

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
        assert_eq!(
            sequence
                .bind_and_message(
                    challenge_0,
                    gamma,
                    gruen.e_in_current(),
                    gruen.e_out_current(),
                )
                .expect("native transition should execute"),
            expected
        );
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
        assert_eq!(
            sequence
                .bind_and_message(
                    challenge_1,
                    gamma,
                    gruen.e_in_current(),
                    gruen.e_out_current(),
                )
                .expect("dense transition should execute"),
            expected
        );
        let mut readback = vec![AkitaField::zero(); tables_2.len()];
        sequence.read_current_tables(&mut readback).unwrap();
        assert_eq!(readback, tables_2);

        // SAFETY: the parent allocation and checked guard range remain live
        // after the synchronous commands complete.
        let guard_after = unsafe {
            slice::from_raw_parts(
                outer
                    .residual_buffer()
                    .contents()
                    .cast::<u8>()
                    .add(usize::try_from(guard_offset).unwrap()),
                guard_len,
            )
        };
        assert_eq!(guard_after, guard_before);
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
    fn compact_rows_preserve_all_instruction_input_fields() {
        for row in packed_rows(16) {
            let compact = InstructionInputRow::from_full_words(row.words());
            let successor = InstructionInputSuccessorRow::from_words(compact.words());
            assert_eq!(
                compact.fields::<AkitaField>(),
                row.instruction_input_fields::<AkitaField>()
            );
            assert_eq!(successor.words(), compact.words());
            assert_eq!(
                row_fields::<AkitaField>(successor).unwrap(),
                compact.fields()
            );
        }
        assert_eq!(
            size_of::<InstructionInputSuccessorRow>(),
            size_of::<InstructionInputRow>()
        );
        assert_eq!(
            align_of::<InstructionInputSuccessorRow>(),
            align_of::<InstructionInputRow>()
        );
        assert_eq!(instruction_input_row_bytes(1 << 26).unwrap(), 3_221_225_472);
        assert_eq!(
            instruction_input_row_bytes(1 << 28).unwrap(),
            12_884_901_888
        );
    }

    #[test]
    fn successor_phases_and_fused_transition_match_production_rows() {
        let rows = packed_rows(4096);
        let successor_rows = rows
            .iter()
            .map(|row| {
                InstructionInputSuccessorRow::from_words(
                    InstructionInputRow::from_full_words(row.words()).words(),
                )
            })
            .collect::<Vec<_>>();
        let initial_tables = (0..INSTRUCTION_INPUT_TABLES)
            .flat_map(|table| {
                rows.iter()
                    .map(move |row| row.instruction_input_fields::<AkitaField>()[table])
            })
            .collect::<Vec<_>>();
        let challenge = AkitaField::from_u64(0x1234_5678_9abc_def0);
        let expected = materialize_first_bind(&successor_rows, challenge).unwrap();
        let gamma = AkitaField::from_u64(0xC001_CAFE);
        let r_product = (0..rows.len().ilog2())
            .map(|index| AkitaField::from_u64(5 + 2 * u64::from(index)))
            .collect::<Vec<_>>();
        let gruen = GruenSplitEqPolynomial::new(&r_product, BindingOrder::LowToHigh);
        let mut bound_gruen = GruenSplitEqPolynomial::new(&r_product, BindingOrder::LowToHigh);
        bound_gruen.bind(challenge);
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let mut sequence = context
            .prepare_instruction_input_sequence(
                &rows,
                InstructionInputSequenceConfig {
                    first_transition: InstructionInputFirstTransition::Successor,
                    storage_initialization: InstructionInputStorageInitialization::Minimal,
                    ..InstructionInputSequenceConfig::default()
                },
            )
            .expect("sequence should prepare");
        let resident_identity = sequence.resident_row_identity();
        let buffer_identities = sequence.static_buffer_identity();
        let mut actual = vec![AkitaField::zero(); expected.len()];

        assert!(matches!(
            sequence.run_successor_dense_message(
                gamma,
                bound_gruen.e_in_current(),
                bound_gruen.e_out_current(),
            ),
            Err(MetalError::InvalidInstructionInputState(_))
        ));

        let stats = sequence
            .probe_successor_materialize(challenge, &mut actual)
            .expect("the successor materializer should execute");

        assert_eq!(actual, expected);
        assert_eq!(stats.source_elements, rows.len());
        assert_eq!(stats.bound_elements, rows.len() / 2);
        assert_eq!(stats.resident_row_identity, resident_identity);
        assert_eq!(stats.dense_buffer_identity, buffer_identities[0]);
        assert_eq!(stats.pipeline_limits.thread_execution_width, 32);
        assert!(stats.pipeline_limits.max_total_threads_per_threadgroup >= 256);
        assert!(stats.gpu_active > Duration::ZERO);
        assert!(stats.gpu_active <= stats.wall);
        assert_eq!(sequence.resident_row_identity(), resident_identity);
        assert_eq!(sequence.static_buffer_identity(), buffer_identities);

        let expected_dense_message = dense_message(
            &expected,
            rows.len() / 2,
            bound_gruen.e_in_current(),
            bound_gruen.e_out_current(),
            gamma,
        )
        .unwrap()
        .values();
        let (actual_dense_message, dense_stats) = sequence
            .run_successor_dense_message(
                gamma,
                bound_gruen.e_in_current(),
                bound_gruen.e_out_current(),
            )
            .expect("the successor dense message should execute");
        assert_eq!(actual_dense_message, expected_dense_message);
        assert_eq!(dense_stats.table_elements, rows.len() / 2);
        assert_eq!(dense_stats.dense_buffer_identity, buffer_identities[0]);
        assert_eq!(dense_stats.dynamic_threadgroup_memory_length, 192);
        assert_eq!(dense_stats.pipeline_limits.thread_execution_width, 32);
        assert!(
            dense_stats
                .pipeline_limits
                .max_total_threads_per_threadgroup
                >= 128
        );
        assert!(dense_stats.gpu_active > Duration::ZERO);
        assert!(dense_stats.gpu_active <= dense_stats.wall);
        assert_eq!(sequence.resident_row_identity(), resident_identity);
        assert_eq!(sequence.static_buffer_identity(), buffer_identities);

        let expected_message = descriptors(
            &initial_tables,
            rows.len(),
            gamma,
            gruen.e_in_current(),
            gruen.e_out_current(),
        );
        assert_eq!(
            sequence
                .message(gamma, gruen.e_in_current(), gruen.e_out_current())
                .expect("the production message should still be first"),
            expected_message
        );
        let (transition_message, transition_stats) = sequence
            .run_successor_transition(
                challenge,
                gamma,
                bound_gruen.e_in_current(),
                bound_gruen.e_out_current(),
            )
            .expect("the fused successor transition should execute");
        assert_eq!(transition_message, expected_dense_message);
        assert_eq!(transition_stats.source_elements, rows.len());
        assert_eq!(transition_stats.table_elements, rows.len() / 2);
        assert_eq!(transition_stats.resident_row_identity, resident_identity);
        assert_eq!(transition_stats.dense_buffer_identity, buffer_identities[0]);
        assert_eq!(transition_stats.dynamic_threadgroup_memory_length, 192);
        assert!(transition_stats.gpu_active > Duration::ZERO);
        assert!(transition_stats.gpu_active <= transition_stats.wall);
        assert!(sequence.is_dense());
        assert_eq!(sequence.current_elements(), rows.len() / 2);
        let mut transitioned_tables = vec![AkitaField::zero(); expected.len()];
        sequence
            .read_current_tables(&mut transitioned_tables)
            .expect("the fused successor tables should remain resident");
        assert_eq!(transitioned_tables, expected);
        assert_eq!(sequence.resident_row_identity(), resident_identity);
        assert_eq!(sequence.static_buffer_identity(), buffer_identities);
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
            .prepare_instruction_input_rows_from_spartan(&rows)
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
    fn native_pipeline_primer_preserves_the_first_protocol_message() {
        let rows = packed_rows(4096);
        let initial_tables: Vec<AkitaField> = (0..INSTRUCTION_INPUT_TABLES)
            .flat_map(|table| {
                rows.iter()
                    .map(move |row| row.instruction_input_fields::<AkitaField>()[table])
            })
            .collect();
        let gamma = AkitaField::from_u64(0xC001_CAFE);
        let r_product = (0..rows.len().ilog2())
            .map(|index| AkitaField::from_u64(5 + 2 * u64::from(index)))
            .collect::<Vec<_>>();
        let gruen = GruenSplitEqPolynomial::new(&r_product, BindingOrder::LowToHigh);
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let sequence = context
            .prepare_instruction_input_sequence(
                &rows,
                InstructionInputSequenceConfig {
                    storage_initialization: InstructionInputStorageInitialization::Lazy,
                    ..InstructionInputSequenceConfig::default()
                },
            )
            .expect("sequence should prepare");
        let resident_identity = sequence.resident_row_identity();
        let storage_identities = sequence.static_buffer_identity();

        let pending = sequence
            .submit_native_pipeline_primer()
            .expect("native pipeline primer should submit");
        let (mut sequence, primer) = pending
            .join()
            .expect("native pipeline primer should complete");

        assert_eq!(primer.source_elements, 64);
        assert_eq!(primer.e_in_elements, 1);
        assert_eq!(primer.e_out_elements, 32);
        assert_eq!(primer.resident_row_identity, resident_identity);
        assert_eq!(primer.storage_buffer_identities, storage_identities);
        assert!(primer.submit_wall <= primer.wall);
        assert!(primer.overlap_wall <= primer.wall);
        assert!(primer.join_wall <= primer.wall);
        assert!(primer.submit_wall + primer.overlap_wall + primer.join_wall <= primer.wall);
        assert!(primer.gpu_active > Duration::ZERO);
        assert!(primer.gpu_active <= primer.wall);
        assert_eq!(sequence.resident_row_identity(), resident_identity);
        assert_eq!(sequence.static_buffer_identity(), storage_identities);

        let expected = descriptors(
            &initial_tables,
            rows.len(),
            gamma,
            gruen.e_in_current(),
            gruen.e_out_current(),
        );
        let actual = sequence
            .message(gamma, gruen.e_in_current(), gruen.e_out_current())
            .expect("first protocol message should execute after the primer");
        assert_eq!(actual, expected);
        assert!(matches!(
            sequence.submit_native_pipeline_primer(),
            Err(MetalError::InvalidInstructionInputState(_))
        ));
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
            true,
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
