//! Resident lazy-prefix sequence for production-G4 Instruction RA.
//!
//! The lazy prefix gathers from the stage-5 lookup plane while each bind doubles
//! the branch tables. At the configured width, the final gather writes
//! factor-major dense tables before releasing the lookup plane.

use std::{ffi::c_void, mem::size_of, slice};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, FunctionConstantValues, MTLDataType,
    MTLResourceOptions, MTLSize,
};

use super::{
    encode_column_reductions, set_inline_bytes, validate_completed_command, Fp128, MetalError,
    PipelineLimits, ResidentLookupIndexPlane, SolinasMetal,
};

const FACTORS: usize = 16;
const BINS: usize = 256;
const SAMPLES: usize = 4;
const SIMD_WIDTH: usize = 32;
const DEFAULT_MESSAGE_THREADS: usize = 128;
const DEFAULT_MATERIALIZE_THREADS: usize = 64;
const BRANCH_THREADS: usize = 256;
const MESSAGE_WIDTH_1_PIPELINE: &str = "solinas_instruction_ra_first_message";
const MESSAGE_WIDTH_2_PIPELINE: &str = "solinas_instruction_ra_message_width_2";
const MESSAGE_WIDTH_4_PIPELINE: &str = "solinas_instruction_ra_message_width_4";
const MESSAGE_WIDTH_8_PIPELINE: &str = "solinas_instruction_ra_message_width_8";
const MESSAGE_WIDE_PIPELINE: &str = "solinas_instruction_ra_message_wide";
const DOUBLE_PIPELINE: &str = "solinas_instruction_ra_double_branches";
const MATERIALIZE_WIDTH_16_PIPELINE: &str = "solinas_instruction_ra_materialize_width_16";
const MATERIALIZE_WIDE_PIPELINE: &str = "solinas_instruction_ra_materialize_wide";
const DENSE_TRANSITION_PIPELINE: &str = "solinas_instruction_ra_dense_transition";
const REDUCE_PIPELINE: &str = "solinas_instruction_ra_reduce";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum InstructionRaMaterializeWidth {
    W16 = 16,
    W32 = 32,
    W64 = 64,
    W128 = 128,
    W256 = 256,
    W512 = 512,
}

impl InstructionRaMaterializeWidth {
    pub const fn elements(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionRaSequenceConfig {
    pub message_threads_per_threadgroup: Option<usize>,
    pub materialize_threads_per_threadgroup: Option<usize>,
    pub materialize_width: InstructionRaMaterializeWidth,
    /// Overwrites the resident inverse map after its final gather; the plane is one-shot.
    pub reuse_inverse_for_dense: bool,
}

impl Default for InstructionRaSequenceConfig {
    fn default() -> Self {
        Self {
            message_threads_per_threadgroup: Some(DEFAULT_MESSAGE_THREADS),
            materialize_threads_per_threadgroup: Some(DEFAULT_MATERIALIZE_THREADS),
            materialize_width: InstructionRaMaterializeWidth::W16,
            reuse_inverse_for_dense: false,
        }
    }
}

#[cfg(test)]
fn instruction_ra_weight_capacities(rows: usize) -> Result<(usize, usize), MetalError> {
    if rows < 32 || !rows.is_power_of_two() {
        return Err(MetalError::InvalidInstructionRaRows(rows));
    }
    let e_out_capacity = 1usize << (rows.ilog2() / 2);
    let e_in_capacity = (rows / 2) / e_out_capacity;
    Ok((e_in_capacity, e_out_capacity))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionRaSequenceScratchLayout {
    pub branch_a_bytes: u64,
    pub branch_b_bytes: u64,
    pub dense_a_bytes: u64,
    pub dense_b_active_bytes: u64,
    pub dense_b_owned_bytes: u64,
    pub dense_b_physical_bytes: u64,
}

impl InstructionRaSequenceScratchLayout {
    pub const fn owned_bytes(self) -> u64 {
        self.branch_a_bytes + self.branch_b_bytes + self.dense_a_bytes + self.dense_b_owned_bytes
    }

    pub const fn resident_bytes_after_handoff(self) -> u64 {
        self.branch_a_bytes + self.branch_b_bytes + self.dense_a_bytes + self.dense_b_physical_bytes
    }
}

impl InstructionRaSequenceConfig {
    pub fn scratch_layout(
        self,
        rows: usize,
    ) -> Result<InstructionRaSequenceScratchLayout, MetalError> {
        let materialize_width = self.materialize_width.elements();
        if rows < 2 * materialize_width || !rows.is_power_of_two() {
            return Err(MetalError::InvalidInstructionRaRows(rows));
        }
        if self.reuse_inverse_for_dense && materialize_width == 16 {
            return Err(MetalError::InvalidInstructionRaState(
                "the resident inverse buffer is too small for width-16 dense ping-pong",
            ));
        }
        let (branch_a_width, branch_b_width) = branch_capacity_widths(materialize_width);
        let branch_a_values = FACTORS
            .checked_mul(branch_a_width)
            .and_then(|values| values.checked_mul(BINS))
            .ok_or(MetalError::InputTooLong(rows))?;
        let branch_b_values = FACTORS
            .checked_mul(branch_b_width)
            .and_then(|values| values.checked_mul(BINS))
            .ok_or(MetalError::InputTooLong(rows))?;
        let dense_a_values = FACTORS
            .checked_mul(rows / materialize_width)
            .ok_or(MetalError::InputTooLong(rows))?;
        let dense_b_active_values = dense_a_values / 2;
        Ok(InstructionRaSequenceScratchLayout {
            branch_a_bytes: byte_length::<Fp128>(branch_a_values)?,
            branch_b_bytes: byte_length::<Fp128>(branch_b_values)?,
            dense_a_bytes: byte_length::<Fp128>(dense_a_values)?,
            dense_b_active_bytes: byte_length::<Fp128>(dense_b_active_values)?,
            dense_b_owned_bytes: if self.reuse_inverse_for_dense {
                0
            } else {
                byte_length::<Fp128>(dense_b_active_values)?
            },
            dense_b_physical_bytes: if self.reuse_inverse_for_dense {
                byte_length::<u32>(rows)?
            } else {
                byte_length::<Fp128>(dense_b_active_values)?
            },
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MessageParams {
    e_in_length: u32,
    e_out_length: u32,
    _reserved: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BranchParams {
    branch_width: u32,
    _reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MaterializeParams {
    source_elements: u32,
    e_in_length: u32,
    e_out_length: u32,
    _reserved: u32,
}

struct Pipelines {
    width_1: ComputePipelineState,
    width_2: ComputePipelineState,
    width_4: ComputePipelineState,
    width_8: ComputePipelineState,
    wide_messages: Vec<(usize, ComputePipelineState)>,
    double: ComputePipelineState,
    materialize: ComputePipelineState,
    dense_transition: ComputePipelineState,
    reduce: ComputePipelineState,
}

struct Buffers {
    branches_a: Buffer,
    branches_b: Buffer,
    dense_a: Buffer,
    dense_b: Option<Buffer>,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub(crate) struct InstructionRaSequenceStorage {
    context: SolinasMetal,
    config: InstructionRaSequenceConfig,
    pipelines: Pipelines,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    rows: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
    message_threads_per_threadgroup: usize,
    materialize_threads_per_threadgroup: usize,
    branch_threads_per_threadgroup: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for InstructionRaSequenceStorage {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_storage"),
            device_storage_bytes(&self.buffers),
        );
        visitor.exit();
    }
}

pub struct InstructionRaSequence {
    context: SolinasMetal,
    lookup_plane: Option<ResidentLookupIndexPlane>,
    pipelines: Pipelines,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    rows: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
    message_threads_per_threadgroup: usize,
    materialize_threads_per_threadgroup: usize,
    branch_threads_per_threadgroup: usize,
    materialize_width: usize,
    reuse_inverse_for_dense: bool,
    branch_width: usize,
    branches_in_a: bool,
    dense: bool,
    dense_in_a: bool,
    dense_elements: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for InstructionRaSequence {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_storage"),
            device_storage_bytes(&self.buffers),
        );
        if let Some(plane) = &self.lookup_plane {
            visitor.visit_field(allocative::Key::new("lookup_plane"), plane);
        }
        visitor.exit();
    }
}

impl SolinasMetal {
    fn compile_instruction_ra_width_pipeline(
        &self,
        name: &'static str,
        width: usize,
    ) -> Result<ComputePipelineState, MetalError> {
        let _span = tracing::info_span!(
            "MetalSolinas::pipeline_compile",
            pipeline = name,
            specialized = true,
            width
        )
        .entered();
        let width = u32::try_from(width).map_err(|_| MetalError::InputTooLong(width))?;
        let constants = FunctionConstantValues::new();
        constants.set_constant_value_at_index(
            std::ptr::from_ref(&width).cast::<c_void>(),
            MTLDataType::UInt,
            0,
        );
        let function = self
            .library
            .get_function(name, Some(constants))
            .map_err(|message| MetalError::FunctionLookup { name, message })?;
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|message| MetalError::PipelineCompilation { name, message })
    }

    pub(crate) fn prepare_instruction_ra_sequence_with_plane(
        &self,
        plane: ResidentLookupIndexPlane,
        chunk_tables: &[AkitaField],
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionRaSequenceConfig,
    ) -> Result<InstructionRaSequence, MetalError> {
        let rows = plane.len();
        let storage = self.prepare_instruction_ra_sequence_storage(
            rows,
            e_in_capacity,
            e_out_capacity,
            config,
        )?;
        storage.attach(plane, chunk_tables)
    }

    pub(crate) fn prepare_instruction_ra_sequence_storage(
        &self,
        rows: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionRaSequenceConfig,
    ) -> Result<InstructionRaSequenceStorage, MetalError> {
        let materialize_width = config.materialize_width.elements();
        if rows < 2 * materialize_width || !rows.is_power_of_two() {
            return Err(MetalError::InvalidInstructionRaRows(rows));
        }
        if config.reuse_inverse_for_dense && materialize_width == 16 {
            return Err(MetalError::InvalidInstructionRaState(
                "the resident inverse buffer is too small for width-16 dense ping-pong",
            ));
        }
        let covered = e_in_capacity
            .checked_mul(e_out_capacity)
            .ok_or(MetalError::InputTooLong(rows))?;
        if e_in_capacity == 0 || e_out_capacity == 0 || covered != rows / 2 {
            return Err(MetalError::InstructionRaWeightShape {
                expected: rows / 2,
                covered,
            });
        }

        let mut wide_messages = Vec::new();
        let mut width = 16;
        while width < materialize_width {
            wide_messages.push((
                width,
                self.compile_instruction_ra_width_pipeline(MESSAGE_WIDE_PIPELINE, width)?,
            ));
            width *= 2;
        }
        let (materialize_pipeline_name, materialize) = if materialize_width == 16 {
            (
                MATERIALIZE_WIDTH_16_PIPELINE,
                self.compile_named_pipeline(MATERIALIZE_WIDTH_16_PIPELINE)?,
            )
        } else {
            (
                MATERIALIZE_WIDE_PIPELINE,
                self.compile_instruction_ra_width_pipeline(
                    MATERIALIZE_WIDE_PIPELINE,
                    materialize_width,
                )?,
            )
        };
        let pipelines = Pipelines {
            width_1: self.compile_named_pipeline(MESSAGE_WIDTH_1_PIPELINE)?,
            width_2: self.compile_named_pipeline(MESSAGE_WIDTH_2_PIPELINE)?,
            width_4: self.compile_named_pipeline(MESSAGE_WIDTH_4_PIPELINE)?,
            width_8: self.compile_named_pipeline(MESSAGE_WIDTH_8_PIPELINE)?,
            wide_messages,
            double: self.compile_named_pipeline(DOUBLE_PIPELINE)?,
            materialize,
            dense_transition: self.compile_named_pipeline(DENSE_TRANSITION_PIPELINE)?,
            reduce: self.compile_named_pipeline(REDUCE_PIPELINE)?,
        };
        let message_limits = Self::limits(&pipelines.width_1);
        let materialize_limits = Self::limits(&pipelines.materialize);
        let reduction_limits = Self::limits(&pipelines.reduce);
        for (pipeline, limits) in [
            (MESSAGE_WIDTH_1_PIPELINE, message_limits),
            (MESSAGE_WIDTH_2_PIPELINE, Self::limits(&pipelines.width_2)),
            (MESSAGE_WIDTH_4_PIPELINE, Self::limits(&pipelines.width_4)),
            (MESSAGE_WIDTH_8_PIPELINE, Self::limits(&pipelines.width_8)),
            (DOUBLE_PIPELINE, Self::limits(&pipelines.double)),
            (materialize_pipeline_name, materialize_limits),
            (
                DENSE_TRANSITION_PIPELINE,
                Self::limits(&pipelines.dense_transition),
            ),
            (REDUCE_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedInstructionRaExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        for (_, pipeline) in &pipelines.wide_messages {
            let limits = Self::limits(pipeline);
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedInstructionRaExecutionWidth {
                    pipeline: MESSAGE_WIDE_PIPELINE,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let message_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.message_threads_per_threadgroup,
            message_limits,
        )?;
        let materialize_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.materialize_threads_per_threadgroup,
            materialize_limits,
        )?;
        let branch_threads_per_threadgroup =
            Self::resolve_threadgroup_width(Some(BRANCH_THREADS), Self::limits(&pipelines.double))?;

        let (branch_a_width, branch_b_width) = branch_capacity_widths(materialize_width);
        let branch_a_capacity = FACTORS * branch_a_width * BINS;
        let branch_b_capacity = FACTORS * branch_b_width * BINS;
        let dense_capacity = FACTORS
            .checked_mul(rows / materialize_width)
            .ok_or(MetalError::InputTooLong(rows))?;
        let partial_capacity = SAMPLES
            .checked_mul(e_out_capacity)
            .ok_or(MetalError::InputTooLong(e_out_capacity))?;
        Ok(InstructionRaSequenceStorage {
            context: self.clone(),
            pipelines,
            reduction_limits,
            buffers: Buffers {
                branches_a: new_buffer(self, branch_a_capacity)?,
                branches_b: new_buffer(self, branch_b_capacity)?,
                dense_a: new_buffer(self, dense_capacity)?,
                dense_b: if config.reuse_inverse_for_dense {
                    None
                } else {
                    Some(new_buffer(self, dense_capacity / 2)?)
                },
                e_in: new_buffer(self, e_in_capacity)?,
                e_out: new_buffer(self, e_out_capacity)?,
                partial_a: new_buffer(self, partial_capacity)?,
                partial_b: new_buffer(self, partial_capacity)?,
            },
            rows,
            e_in_capacity,
            e_out_capacity,
            message_threads_per_threadgroup,
            materialize_threads_per_threadgroup,
            branch_threads_per_threadgroup,
            config,
        })
    }
}

impl InstructionRaSequenceStorage {
    pub(crate) fn matches(
        &self,
        context: &SolinasMetal,
        rows: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionRaSequenceConfig,
    ) -> bool {
        self.context.device_registry_id() == context.device_registry_id()
            && self.rows == rows
            && self.e_in_capacity == e_in_capacity
            && self.e_out_capacity == e_out_capacity
            && self.config == config
    }

    pub(crate) fn attach(
        self,
        plane: ResidentLookupIndexPlane,
        chunk_tables: &[AkitaField],
    ) -> Result<InstructionRaSequence, MetalError> {
        if plane.len() != self.rows {
            return Err(MetalError::InvalidInstructionRaState(
                "resident lookup plane does not match the preallocated sequence",
            ));
        }
        if chunk_tables.len() != FACTORS * BINS {
            return Err(MetalError::InstructionRaStorageLength {
                expected: FACTORS * BINS,
                got: chunk_tables.len(),
            });
        }
        validate_plane(&self.context, &plane)?;
        write_fields(&self.buffers.branches_a, FACTORS * BINS, chunk_tables)?;
        if self.buffers.dense_b.is_none() {
            let required = byte_length::<Fp128>(
                FACTORS * (self.rows / self.config.materialize_width.elements()) / 2,
            )?;
            let inverse = plane.cycle_to_table_major();
            if inverse.length() < required {
                return Err(MetalError::InstructionRaPlaneLength {
                    name: "cycle-to-table-major dense reuse",
                    expected: required,
                    got: inverse.length(),
                });
            }
        }

        Ok(InstructionRaSequence {
            context: self.context,
            lookup_plane: Some(plane),
            pipelines: self.pipelines,
            reduction_limits: self.reduction_limits,
            buffers: self.buffers,
            rows: self.rows,
            e_in_capacity: self.e_in_capacity,
            e_out_capacity: self.e_out_capacity,
            message_threads_per_threadgroup: self.message_threads_per_threadgroup,
            materialize_threads_per_threadgroup: self.materialize_threads_per_threadgroup,
            branch_threads_per_threadgroup: self.branch_threads_per_threadgroup,
            materialize_width: self.config.materialize_width.elements(),
            reuse_inverse_for_dense: self.config.reuse_inverse_for_dense,
            branch_width: 1,
            branches_in_a: true,
            dense: false,
            dense_in_a: true,
            dense_elements: 0,
        })
    }
}

#[cfg(feature = "allocative")]
fn device_storage_bytes(buffers: &Buffers) -> usize {
    let fixed = [
        &buffers.branches_a,
        &buffers.branches_b,
        &buffers.dense_a,
        &buffers.e_in,
        &buffers.e_out,
        &buffers.partial_a,
        &buffers.partial_b,
    ]
    .into_iter()
    .fold(0usize, |bytes, buffer| {
        bytes.saturating_add(buffer.length() as usize)
    });
    buffers.dense_b.as_ref().map_or(fixed, |buffer| {
        fixed.saturating_add(buffer.length() as usize)
    })
}

impl InstructionRaSequence {
    pub fn message(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        self.execute_lazy(None, e_in, e_out)
    }

    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        if self.dense {
            self.execute_dense(challenge, e_in, e_out)
        } else {
            self.execute_lazy(Some(challenge), e_in, e_out)
        }
    }

    pub fn read_current_tables(&self, output: &mut [AkitaField]) -> Result<(), MetalError> {
        if !self.dense {
            return Err(MetalError::InvalidInstructionRaState(
                "lazy tables cannot be read as dense tables",
            ));
        }
        let elements = FACTORS * self.dense_elements;
        if output.len() != elements {
            return Err(MetalError::InstructionRaStorageLength {
                expected: elements,
                got: output.len(),
            });
        }
        // SAFETY: materialization completed synchronously before `dense` was
        // set, and the buffer contains `elements` initialized fields.
        let values = unsafe {
            slice::from_raw_parts(
                self.dense_source_buffer()?.contents().cast::<Fp128>(),
                elements,
            )
        };
        self.context
            .validate_inputs("instruction RA dense tables", values)?;
        for (output, value) in output.iter_mut().zip(values) {
            *output = value.into_jolt_field();
        }
        Ok(())
    }

    pub const fn current_elements(&self) -> usize {
        if self.dense {
            self.dense_elements
        } else {
            self.rows / self.branch_width
        }
    }

    copy_field_getters! { pub, {
        branch_width: usize,
        is_dense => dense: bool,
    }}

    fn execute_lazy(
        &mut self,
        challenge: Option<AkitaField>,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        if self.dense {
            return Err(MetalError::InvalidInstructionRaState(
                "the lazy prefix has already materialized",
            ));
        }
        let next_width = if challenge.is_some() {
            self.branch_width * 2
        } else {
            self.branch_width
        };
        if next_width > self.materialize_width {
            return Err(MetalError::InvalidInstructionRaState(
                "branch width exceeds the materialization point",
            ));
        }
        let source_elements = self.rows / next_width;
        self.validate_weights(source_elements / 2, e_in, e_out)?;
        self.write_weights(e_in, e_out)?;
        let materialize = next_width == self.materialize_width;
        let message_params = MessageParams {
            e_in_length: u32::try_from(e_in.len())
                .map_err(|_| MetalError::InputTooLong(e_in.len()))?,
            e_out_length: u32::try_from(e_out.len())
                .map_err(|_| MetalError::InputTooLong(e_out.len()))?,
            _reserved: [0; 2],
        };
        let materialize_params = MaterializeParams {
            source_elements: u32::try_from(source_elements)
                .map_err(|_| MetalError::InputTooLong(source_elements))?,
            e_in_length: message_params.e_in_length,
            e_out_length: message_params.e_out_length,
            _reserved: 0,
        };
        let message_pipeline = if materialize {
            None
        } else {
            Some(self.message_pipeline(next_width)?.clone())
        };
        let plane = self
            .lookup_plane
            .as_ref()
            .ok_or(MetalError::InvalidInstructionRaState(
                "the resident lookup plane is missing",
            ))?;

        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            let mut message_branches_in_a = self.branches_in_a;
            if let Some(challenge) = challenge {
                let params = BranchParams {
                    branch_width: self.branch_width as u32,
                    _reserved: [0; 3],
                };
                encoder.set_compute_pipeline_state(&self.pipelines.double);
                encoder.set_buffer(0, Some(self.branch_source_buffer()), 0);
                encoder.set_buffer(1, Some(self.branch_destination_buffer()), 0);
                set_inline_bytes(encoder, 2, &Fp128::from_jolt_field(&challenge));
                set_inline_bytes(encoder, 3, &params);
                let elements = FACTORS * self.branch_width * BINS;
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: elements.div_ceil(self.branch_threads_per_threadgroup) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.branch_threads_per_threadgroup as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                message_branches_in_a = !message_branches_in_a;
            }

            if materialize {
                encoder.set_compute_pipeline_state(&self.pipelines.materialize);
                encoder.set_buffer(0, Some(plane.lookups()), 0);
                encoder.set_buffer(1, Some(plane.cycle_to_table_major()), 0);
                encoder.set_buffer(2, Some(self.branch_buffer(message_branches_in_a)), 0);
                encoder.set_buffer(3, Some(&self.buffers.dense_a), 0);
                encoder.set_buffer(4, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(5, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(6, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 7, &materialize_params);
                Self::encode_message_dispatch(
                    encoder,
                    e_out.len(),
                    self.materialize_threads_per_threadgroup,
                );
            } else {
                encoder.set_compute_pipeline_state(message_pipeline.as_ref().ok_or(
                    MetalError::InvalidInstructionRaState("the lazy message pipeline is missing"),
                )?);
                encoder.set_buffer(0, Some(plane.lookups()), 0);
                encoder.set_buffer(1, Some(plane.cycle_to_table_major()), 0);
                encoder.set_buffer(2, Some(self.branch_buffer(message_branches_in_a)), 0);
                encoder.set_buffer(3, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(4, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 6, &message_params);
                Self::encode_message_dispatch(
                    encoder,
                    e_out.len(),
                    self.message_threads_per_threadgroup,
                );
            }
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduce,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                SAMPLES,
                self.reduction_limits.thread_execution_width,
            )?;
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok::<bool, MetalError>(final_in_a)
        })?;

        let message = self.finish_command(command_buffer, final_in_a)?;
        if challenge.is_some() {
            self.branch_width = next_width;
            self.branches_in_a = !self.branches_in_a;
        }
        if materialize {
            if self.reuse_inverse_for_dense {
                let inverse = self
                    .lookup_plane
                    .as_ref()
                    .ok_or(MetalError::InvalidInstructionRaState(
                        "the resident inverse buffer is missing at materialization",
                    ))?
                    .cycle_to_table_major()
                    .clone();
                self.buffers.dense_b = Some(inverse);
            }
            self.dense = true;
            self.dense_in_a = true;
            self.dense_elements = source_elements;
            // `finish_command` observed completion before the inverse buffer was
            // repurposed and the lookup-plane ownership was released.
            let _ = self.lookup_plane.take();
        }
        Ok(message)
    }

    fn execute_dense(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        if !self.dense || self.dense_elements < 4 {
            return Err(MetalError::InvalidInstructionRaState(
                "dense transition needs at least four elements per factor",
            ));
        }
        self.validate_weights(self.dense_elements / 4, e_in, e_out)?;
        self.write_weights(e_in, e_out)?;
        let params = MaterializeParams {
            source_elements: u32::try_from(self.dense_elements)
                .map_err(|_| MetalError::InputTooLong(self.dense_elements))?,
            e_in_length: u32::try_from(e_in.len())
                .map_err(|_| MetalError::InputTooLong(e_in.len()))?,
            e_out_length: u32::try_from(e_out.len())
                .map_err(|_| MetalError::InputTooLong(e_out.len()))?,
            _reserved: 0,
        };

        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.dense_transition);
            encoder.set_buffer(0, Some(self.dense_source_buffer()?), 0);
            encoder.set_buffer(1, Some(self.dense_destination_buffer()?), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 5, &Fp128::from_jolt_field(&challenge));
            set_inline_bytes(encoder, 6, &params);
            Self::encode_message_dispatch(
                encoder,
                e_out.len(),
                self.message_threads_per_threadgroup,
            );
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduce,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                SAMPLES,
                self.reduction_limits.thread_execution_width,
            )?;
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok::<bool, MetalError>(final_in_a)
        })?;

        let message = self.finish_command(command_buffer, final_in_a)?;
        self.dense_elements /= 2;
        self.dense_in_a = !self.dense_in_a;
        Ok(message)
    }

    fn validate_weights(
        &self,
        expected: usize,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<(), MetalError> {
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(expected))?;
        if e_in.is_empty()
            || e_out.is_empty()
            || e_in.len() > self.e_in_capacity
            || e_out.len() > self.e_out_capacity
            || covered != expected
        {
            return Err(MetalError::InstructionRaWeightShape { expected, covered });
        }
        Ok(())
    }

    fn write_weights(&self, e_in: &[AkitaField], e_out: &[AkitaField]) -> Result<(), MetalError> {
        write_fields(&self.buffers.e_in, self.e_in_capacity, e_in)?;
        write_fields(&self.buffers.e_out, self.e_out_capacity, e_out)
    }

    fn message_pipeline(&self, width: usize) -> Result<&ComputePipelineState, MetalError> {
        match width {
            1 => Ok(&self.pipelines.width_1),
            2 => Ok(&self.pipelines.width_2),
            4 => Ok(&self.pipelines.width_4),
            8 => Ok(&self.pipelines.width_8),
            _ => self
                .pipelines
                .wide_messages
                .iter()
                .find_map(|(pipeline_width, pipeline)| {
                    (*pipeline_width == width).then_some(pipeline)
                })
                .ok_or(MetalError::InvalidInstructionRaState(
                    "no lazy message pipeline for this branch width",
                )),
        }
    }

    fn encode_message_dispatch(
        encoder: &metal::ComputeCommandEncoderRef,
        groups: usize,
        threads_per_threadgroup: usize,
    ) {
        let simdgroups = threads_per_threadgroup / SIMD_WIDTH;
        encoder
            .set_threadgroup_memory_length(0, (SAMPLES * simdgroups * size_of::<Fp128>()) as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: groups as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_threadgroup as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn finish_command(
        &mut self,
        command_buffer: &metal::CommandBufferRef,
        final_in_a: bool,
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        validate_completed_command(command_buffer)?;
        let buffer = if final_in_a {
            &self.buffers.partial_a
        } else {
            &self.buffers.partial_b
        };
        // SAFETY: the completed reduction leaves four fields at the front of
        // the selected shared buffer.
        let values = unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), SAMPLES) };
        self.context
            .validate_inputs("instruction RA lazy message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }

    fn branch_source_buffer(&self) -> &Buffer {
        self.branch_buffer(self.branches_in_a)
    }

    fn branch_destination_buffer(&self) -> &Buffer {
        self.branch_buffer(!self.branches_in_a)
    }

    fn branch_buffer(&self, in_a: bool) -> &Buffer {
        if in_a {
            &self.buffers.branches_a
        } else {
            &self.buffers.branches_b
        }
    }

    fn dense_source_buffer(&self) -> Result<&Buffer, MetalError> {
        if self.dense_in_a {
            Ok(&self.buffers.dense_a)
        } else {
            self.dense_b_buffer()
        }
    }

    fn dense_destination_buffer(&self) -> Result<&Buffer, MetalError> {
        if self.dense_in_a {
            self.dense_b_buffer()
        } else {
            Ok(&self.buffers.dense_a)
        }
    }

    fn dense_b_buffer(&self) -> Result<&Buffer, MetalError> {
        self.buffers
            .dense_b
            .as_ref()
            .ok_or(MetalError::InvalidInstructionRaState(
                "the dense destination buffer is missing",
            ))
    }
}

fn branch_capacity_widths(materialize_width: usize) -> (usize, usize) {
    if materialize_width == 16 {
        return (16, 16);
    }
    let materializes_in_a = materialize_width.trailing_zeros().is_multiple_of(2);
    if materializes_in_a {
        (materialize_width, materialize_width / 2)
    } else {
        (materialize_width / 2, materialize_width)
    }
}

fn validate_plane(
    context: &SolinasMetal,
    plane: &ResidentLookupIndexPlane,
) -> Result<(), MetalError> {
    let expected_lookups = byte_length::<[u64; 2]>(plane.len())?;
    let expected_inverse = byte_length::<u32>(plane.len())?;
    if plane.lookups().length() != expected_lookups {
        return Err(MetalError::InstructionRaPlaneLength {
            name: "lookup-index",
            expected: expected_lookups,
            got: plane.lookups().length(),
        });
    }
    if plane.cycle_to_table_major().length() != expected_inverse {
        return Err(MetalError::InstructionRaPlaneLength {
            name: "cycle-to-table-major",
            expected: expected_inverse,
            got: plane.cycle_to_table_major().length(),
        });
    }
    let expected = context.device.registry_id();
    let got = plane.device_registry_id();
    if got != expected {
        return Err(MetalError::InstructionRaPlaneDevice { expected, got });
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

fn write_fields(buffer: &Buffer, capacity: usize, values: &[AkitaField]) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(MetalError::InstructionRaStorageLength {
            expected: capacity,
            got: values.len(),
        });
    }
    // SAFETY: the shared buffer has `capacity` fields and no command is using
    // it while the host writes the active prefix.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity) };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

const _: () = assert!(size_of::<MessageParams>() == 16);
const _: () = assert!(size_of::<BranchParams>() == 16);
const _: () = assert!(size_of::<MaterializeParams>() == 16);

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use jolt_field::Prime128OffsetA7F7 as AkitaField;
    use jolt_field::Zero as _;
    use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

    use super::*;

    #[test]
    fn weight_capacities_match_the_low_to_high_gruen_split() {
        for log_t in 5..=30 {
            let gruen = GruenSplitEqPolynomial::<AkitaField>::new(
                &vec![AkitaField::zero(); log_t],
                BindingOrder::LowToHigh,
            );
            assert_eq!(
                instruction_ra_weight_capacities(1 << log_t).ok(),
                Some((gruen.e_in_current_len(), gruen.e_out_current_len()))
            );
        }
        assert!(matches!(
            instruction_ra_weight_capacities(31),
            Err(MetalError::InvalidInstructionRaRows(31))
        ));
    }

    #[test]
    fn t26_scratch_layout_tracks_materialization_width() {
        let cases = [
            (InstructionRaMaterializeWidth::W16, 1538),
            (InstructionRaMaterializeWidth::W32, 771),
            (InstructionRaMaterializeWidth::W64, 390),
            (InstructionRaMaterializeWidth::W128, 204),
            (InstructionRaMaterializeWidth::W256, 120),
            (InstructionRaMaterializeWidth::W512, 96),
        ];
        for (materialize_width, expected_mib) in cases {
            let config = InstructionRaSequenceConfig {
                materialize_width,
                ..InstructionRaSequenceConfig::default()
            };
            let layout = config
                .scratch_layout(1 << 26)
                .expect("the T26 layout should fit");
            assert_eq!(layout.owned_bytes() >> 20, expected_mib);
        }
    }

    #[test]
    fn inverse_reuse_removes_only_the_owned_dense_destination() {
        let config = InstructionRaSequenceConfig {
            materialize_width: InstructionRaMaterializeWidth::W256,
            reuse_inverse_for_dense: true,
            ..InstructionRaSequenceConfig::default()
        };
        let layout = config
            .scratch_layout(1 << 26)
            .expect("the T26 layout should fit");
        assert_eq!(layout.dense_b_active_bytes >> 20, 32);
        assert_eq!(layout.dense_b_owned_bytes, 0);
        assert_eq!(layout.dense_b_physical_bytes >> 20, 256);
        assert_eq!(layout.owned_bytes() >> 20, 88);
        assert_eq!(layout.resident_bytes_after_handoff() >> 20, 344);
    }

    #[test]
    fn width_16_rejects_inverse_reuse() {
        let config = InstructionRaSequenceConfig {
            reuse_inverse_for_dense: true,
            ..InstructionRaSequenceConfig::default()
        };
        assert!(matches!(
            config.scratch_layout(1 << 26),
            Err(MetalError::InvalidInstructionRaState(_))
        ));
    }

    #[test]
    fn wide_materialization_pipelines_compile() {
        let rows = 1 << 10;
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let (e_in, e_out) =
            instruction_ra_weight_capacities(rows).expect("weight capacities should derive");
        let _storage = context
            .prepare_instruction_ra_sequence_storage(
                rows,
                e_in,
                e_out,
                InstructionRaSequenceConfig {
                    materialize_width: InstructionRaMaterializeWidth::W32,
                    reuse_inverse_for_dense: true,
                    ..InstructionRaSequenceConfig::default()
                },
            )
            .expect("wide Instruction RA pipelines should compile");
    }
}
