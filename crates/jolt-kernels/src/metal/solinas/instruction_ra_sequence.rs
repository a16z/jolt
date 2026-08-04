//! Resident lazy-prefix sequence for production-G4 Instruction RA.
//!
//! Widths 1, 2, 4, and 8 gather from the stage-5 lookup plane. Each bind
//! doubles the small branch tables. The fourth bind produces width-16 branches,
//! writes factor-major dense tables, and computes the following message from the
//! gathered values before releasing the lookup plane.

use std::{mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::{
    command_buffer_timestamp, Fp128, MetalError, PipelineLimits, ResidentLookupIndexPlane,
    SolinasMetal,
};

const FACTORS: usize = 16;
const BINS: usize = 256;
const SAMPLES: usize = 4;
const MATERIALIZE_WIDTH: usize = 16;
const SIMD_WIDTH: usize = 32;
const DEFAULT_MESSAGE_THREADS: usize = 128;
const DEFAULT_MATERIALIZE_THREADS: usize = 64;
const BRANCH_THREADS: usize = 256;
const MESSAGE_WIDTH_1_PIPELINE: &str = "solinas_instruction_ra_first_message";
const MESSAGE_WIDTH_2_PIPELINE: &str = "solinas_instruction_ra_message_width_2";
const MESSAGE_WIDTH_4_PIPELINE: &str = "solinas_instruction_ra_message_width_4";
const MESSAGE_WIDTH_8_PIPELINE: &str = "solinas_instruction_ra_message_width_8";
const DOUBLE_PIPELINE: &str = "solinas_instruction_ra_double_branches";
const MATERIALIZE_PIPELINE: &str = "solinas_instruction_ra_materialize_width_16";
const DENSE_TRANSITION_PIPELINE: &str = "solinas_instruction_ra_dense_transition";
const REDUCE_PIPELINE: &str = "solinas_instruction_ra_reduce";

#[repr(C)]
struct LookupAbi {
    limbs: [u64; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionRaSequenceConfig {
    pub message_threads_per_threadgroup: Option<usize>,
    pub materialize_threads_per_threadgroup: Option<usize>,
}

impl Default for InstructionRaSequenceConfig {
    fn default() -> Self {
        Self {
            message_threads_per_threadgroup: Some(DEFAULT_MESSAGE_THREADS),
            materialize_threads_per_threadgroup: Some(DEFAULT_MATERIALIZE_THREADS),
        }
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

#[repr(C)]
#[derive(Clone, Copy)]
struct ReductionParams {
    input_count: u32,
    output_count: u32,
    _reserved: [u32; 2],
}

struct Pipelines {
    width_1: ComputePipelineState,
    width_2: ComputePipelineState,
    width_4: ComputePipelineState,
    width_8: ComputePipelineState,
    double: ComputePipelineState,
    materialize: ComputePipelineState,
    dense_transition: ComputePipelineState,
    reduce: ComputePipelineState,
}

struct Buffers {
    branches_a: Buffer,
    branches_b: Buffer,
    dense_a: Buffer,
    dense_b: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct InstructionRaSequence {
    context: SolinasMetal,
    lookup_plane: Option<ResidentLookupIndexPlane>,
    pipelines: Pipelines,
    message_limits: PipelineLimits,
    materialize_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    rows: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
    message_threads_per_threadgroup: usize,
    materialize_threads_per_threadgroup: usize,
    branch_threads_per_threadgroup: usize,
    branch_width: usize,
    branches_in_a: bool,
    dense: bool,
    dense_in_a: bool,
    dense_elements: usize,
    gpu_active_time: Duration,
}

impl SolinasMetal {
    pub(crate) fn prepare_instruction_ra_sequence_with_plane(
        &self,
        plane: ResidentLookupIndexPlane,
        chunk_tables: &[AkitaField],
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionRaSequenceConfig,
    ) -> Result<InstructionRaSequence, MetalError> {
        let rows = plane.len();
        if rows < 2 * MATERIALIZE_WIDTH || !rows.is_power_of_two() {
            return Err(MetalError::InvalidInstructionRaRows(rows));
        }
        if chunk_tables.len() != FACTORS * BINS {
            return Err(MetalError::InstructionRaStorageLength {
                expected: FACTORS * BINS,
                got: chunk_tables.len(),
            });
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
        validate_plane(self, &plane)?;

        let pipelines = Pipelines {
            width_1: self.compile_named_pipeline(MESSAGE_WIDTH_1_PIPELINE)?,
            width_2: self.compile_named_pipeline(MESSAGE_WIDTH_2_PIPELINE)?,
            width_4: self.compile_named_pipeline(MESSAGE_WIDTH_4_PIPELINE)?,
            width_8: self.compile_named_pipeline(MESSAGE_WIDTH_8_PIPELINE)?,
            double: self.compile_named_pipeline(DOUBLE_PIPELINE)?,
            materialize: self.compile_named_pipeline(MATERIALIZE_PIPELINE)?,
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
            (MATERIALIZE_PIPELINE, materialize_limits),
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

        let branch_capacity = FACTORS * MATERIALIZE_WIDTH * BINS;
        let dense_capacity = FACTORS
            .checked_mul(rows / MATERIALIZE_WIDTH)
            .ok_or(MetalError::InputTooLong(rows))?;
        let partial_capacity = SAMPLES
            .checked_mul(e_out_capacity)
            .ok_or(MetalError::InputTooLong(e_out_capacity))?;
        let branches_a = new_buffer(self, branch_capacity)?;
        write_fields(&branches_a, branch_capacity, chunk_tables)?;

        Ok(InstructionRaSequence {
            context: self.clone(),
            lookup_plane: Some(plane),
            pipelines,
            message_limits,
            materialize_limits,
            reduction_limits,
            buffers: Buffers {
                branches_a,
                branches_b: new_buffer(self, branch_capacity)?,
                dense_a: new_buffer(self, dense_capacity)?,
                dense_b: new_buffer(self, dense_capacity / 2)?,
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
            branch_width: 1,
            branches_in_a: true,
            dense: false,
            dense_in_a: true,
            dense_elements: 0,
            gpu_active_time: Duration::ZERO,
        })
    }
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
                self.dense_source_buffer().contents().cast::<Fp128>(),
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

    pub const fn branch_width(&self) -> usize {
        self.branch_width
    }

    pub const fn is_dense(&self) -> bool {
        self.dense
    }

    pub const fn lookup_plane_is_resident(&self) -> bool {
        self.lookup_plane.is_some()
    }

    pub const fn gpu_active_time(&self) -> Duration {
        self.gpu_active_time
    }

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    pub const fn message_pipeline_limits(&self) -> PipelineLimits {
        self.message_limits
    }

    pub const fn materialize_pipeline_limits(&self) -> PipelineLimits {
        self.materialize_limits
    }

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
        if next_width > MATERIALIZE_WIDTH {
            return Err(MetalError::InvalidInstructionRaState(
                "branch width exceeds the materialization point",
            ));
        }
        let source_elements = self.rows / next_width;
        self.validate_weights(source_elements / 2, e_in, e_out)?;
        self.write_weights(e_in, e_out)?;
        let materialize = next_width == MATERIALIZE_WIDTH;
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
        autoreleasepool(|| {
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
            self.encode_reductions(encoder, e_out.len());
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok::<(), MetalError>(())
        })?;

        let message = self.finish_command(command_buffer, e_out.len())?;
        if challenge.is_some() {
            self.branch_width = next_width;
            self.branches_in_a = !self.branches_in_a;
        }
        if materialize {
            self.dense = true;
            self.dense_in_a = true;
            self.dense_elements = source_elements;
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
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.dense_transition);
            encoder.set_buffer(0, Some(self.dense_source_buffer()), 0);
            encoder.set_buffer(1, Some(self.dense_destination_buffer()), 0);
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
            self.encode_reductions(encoder, e_out.len());
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok::<(), MetalError>(())
        })?;

        let message = self.finish_command(command_buffer, e_out.len())?;
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
            _ => Err(MetalError::InvalidInstructionRaState(
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

    fn encode_reductions(&self, encoder: &metal::ComputeCommandEncoderRef, mut count: usize) {
        let mut input_a = true;
        while count > 1 {
            let output_count = count.div_ceil(self.reduction_limits.thread_execution_width);
            let params = ReductionParams {
                input_count: count as u32,
                output_count: output_count as u32,
                _reserved: [0; 2],
            };
            encoder.set_compute_pipeline_state(&self.pipelines.reduce);
            let (input, output) = if input_a {
                (&self.buffers.partial_a, &self.buffers.partial_b)
            } else {
                (&self.buffers.partial_b, &self.buffers.partial_a)
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
                    width: self.reduction_limits.thread_execution_width as u64,
                    height: 1,
                    depth: 1,
                },
            );
            count = output_count;
            input_a = !input_a;
        }
    }

    fn finish_command(
        &mut self,
        command_buffer: &metal::CommandBufferRef,
        reduction_input_count: usize,
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(command_buffer.status()));
        }
        let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        self.gpu_active_time += Duration::from_secs_f64(end - start);

        let mut count = reduction_input_count;
        let mut final_in_a = true;
        while count > 1 {
            count = count.div_ceil(self.reduction_limits.thread_execution_width);
            final_in_a = !final_in_a;
        }
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

    fn dense_source_buffer(&self) -> &Buffer {
        if self.dense_in_a {
            &self.buffers.dense_a
        } else {
            &self.buffers.dense_b
        }
    }

    fn dense_destination_buffer(&self) -> &Buffer {
        if self.dense_in_a {
            &self.buffers.dense_b
        } else {
            &self.buffers.dense_a
        }
    }
}

fn validate_plane(
    context: &SolinasMetal,
    plane: &ResidentLookupIndexPlane,
) -> Result<(), MetalError> {
    let expected_lookups = byte_length::<LookupAbi>(plane.len())?;
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

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

const _: () = assert!(size_of::<LookupAbi>() == 16);
const _: () = assert!(size_of::<MessageParams>() == 16);
const _: () = assert!(size_of::<BranchParams>() == 16);
const _: () = assert!(size_of::<MaterializeParams>() == 16);
const _: () = assert!(size_of::<ReductionParams>() == 16);
