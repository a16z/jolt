use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::super::booleanity::write_fields;
use super::super::{
    command_buffer_timestamp, BooleanityRows, Fp128, HammingHotRows, MetalError, PipelineLimits,
    SolinasMetal, AKITA_OFFSET_FFFFA7F7,
};
use super::{
    BooleanityAddressSuccessorBufferLengths, BooleanityAddressSuccessorConfig,
    BooleanityAddressSuccessorDispatchPlan, BooleanityAddressSuccessorError,
    BooleanityAddressSuccessorGeometry, BooleanityAddressSuccessorParams,
    BOOLEANITY_ADDRESS_SUCCESSOR_FINALIZE_THREADGROUP_BYTES,
    BOOLEANITY_ADDRESS_SUCCESSOR_SIMD_WIDTH, BOOLEANITY_ADDRESS_SUCCESSOR_THREADGROUP_BYTES,
    FINALIZE_BUFFER_OUTPUT, FINALIZE_BUFFER_PARAMS, FINALIZE_BUFFER_PARTIALS, FINALIZE_PIPELINE,
    PACKED_TILES_BUFFER_E_IN, PACKED_TILES_BUFFER_E_OUT, PACKED_TILES_BUFFER_HOT,
    PACKED_TILES_BUFFER_PARAMS, PACKED_TILES_BUFFER_PARTIALS, PACKED_TILES_BUFFER_VALIDITY,
    PACKED_TILES_PIPELINE, PACK_AND_FIRST_BUFFER_E_IN, PACK_AND_FIRST_BUFFER_E_OUT,
    PACK_AND_FIRST_BUFFER_HOT, PACK_AND_FIRST_BUFFER_PARAMS, PACK_AND_FIRST_BUFFER_PARTIALS,
    PACK_AND_FIRST_BUFFER_ROWS, PACK_AND_FIRST_BUFFER_VALIDITY, PACK_AND_FIRST_PIPELINE,
};

#[derive(Debug, thiserror::Error)]
pub enum BooleanityAddressSuccessorRuntimeError {
    #[error("invalid Booleanity-address successor plan: {0:?}")]
    Plan(BooleanityAddressSuccessorError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("Booleanity-address successor reference point has length {got}, expected {expected}")]
    ReferencePointLength { expected: usize, got: usize },
    #[error(
        "Booleanity-address successor pipeline `{pipeline}` requires SIMD width {expected}, got {got}"
    )]
    ExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "Booleanity-address successor pipeline `{pipeline}` needs {requested} threads, maximum is {maximum}"
    )]
    ThreadLimit {
        pipeline: &'static str,
        requested: usize,
        maximum: usize,
    },
    #[error(
        "Booleanity-address successor pipeline `{pipeline}` needs {requested} threadgroup bytes, device maximum is {maximum}"
    )]
    ThreadgroupMemory {
        pipeline: &'static str,
        requested: u64,
        maximum: u64,
    },
    #[error("invalid Booleanity-address successor state: {0}")]
    InvalidState(&'static str),
}

struct Buffers {
    rows: BooleanityRows,
    hot_rows: HammingHotRows,
    validity: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partials: Buffer,
    output: Buffer,
}

pub struct BooleanityAddressSuccessorInvocation {
    context: SolinasMetal,
    pack_pipeline: ComputePipelineState,
    packed_pipeline: ComputePipelineState,
    finalize_pipeline: ComputePipelineState,
    pack_limits: PipelineLimits,
    packed_limits: PipelineLimits,
    finalize_limits: PipelineLimits,
    buffers: Buffers,
    geometry: BooleanityAddressSuccessorGeometry,
    params: BooleanityAddressSuccessorParams,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub fn prepare_booleanity_address_successor(
        &self,
        rows: BooleanityRows,
        reference_cycle: &[AkitaField],
        config: BooleanityAddressSuccessorConfig,
    ) -> Result<BooleanityAddressSuccessorInvocation, BooleanityAddressSuccessorRuntimeError> {
        if self.offset != AKITA_OFFSET_FFFFA7F7 {
            return Err(MetalError::UnexpectedSolinasOffset {
                expected: AKITA_OFFSET_FFFFA7F7,
                got: self.offset,
            }
            .into());
        }
        self.validate_booleanity_rows(&rows)?;
        let geometry = BooleanityAddressSuccessorGeometry::new(rows.len(), config)?;
        let expected_point = rows.len().ilog2() as usize;
        if reference_cycle.len() != expected_point {
            return Err(
                BooleanityAddressSuccessorRuntimeError::ReferencePointLength {
                    expected: expected_point,
                    got: reference_cycle.len(),
                },
            );
        }
        let split = reference_cycle.len() - config.inner_log2;
        let (out_point, in_point) = reference_cycle.split_at(split);
        let e_in = EqPolynomial::evals(in_point, None);
        let e_out = EqPolynomial::evals(out_point, None);

        let pack_pipeline = self.compile_named_pipeline(PACK_AND_FIRST_PIPELINE)?;
        let packed_pipeline = self.compile_named_pipeline(PACKED_TILES_PIPELINE)?;
        let finalize_pipeline = self.compile_named_pipeline(FINALIZE_PIPELINE)?;
        let pack_limits = Self::limits(&pack_pipeline);
        let packed_limits = Self::limits(&packed_pipeline);
        let finalize_limits = Self::limits(&finalize_pipeline);
        validate_pipeline(
            PACK_AND_FIRST_PIPELINE,
            pack_limits,
            config.accumulator_threads_per_threadgroup,
            BOOLEANITY_ADDRESS_SUCCESSOR_THREADGROUP_BYTES as u64,
            self.device.max_threadgroup_memory_length(),
        )?;
        validate_pipeline(
            PACKED_TILES_PIPELINE,
            packed_limits,
            config.accumulator_threads_per_threadgroup,
            BOOLEANITY_ADDRESS_SUCCESSOR_THREADGROUP_BYTES as u64,
            self.device.max_threadgroup_memory_length(),
        )?;
        validate_pipeline(
            FINALIZE_PIPELINE,
            finalize_limits,
            config.finalize_threads_per_threadgroup,
            BOOLEANITY_ADDRESS_SUCCESSOR_FINALIZE_THREADGROUP_BYTES as u64,
            self.device.max_threadgroup_memory_length(),
        )?;

        let lengths = geometry.buffer_lengths()?;
        for bytes in [
            lengths.hot_bytes,
            lengths.validity_bytes,
            field_bytes(lengths.e_in_fields)?,
            field_bytes(lengths.e_out_fields)?,
            field_bytes(lengths.partial_fields)?,
            field_bytes(lengths.output_fields)?,
        ] {
            self.validate_buffer_length(bytes)?;
        }
        self.validate_additional_working_set(lengths.owned_bytes()?)?;

        let e_in_buffer = self.device.new_buffer(
            field_bytes(lengths.e_in_fields)?,
            MTLResourceOptions::StorageModeShared,
        );
        let e_out_buffer = self.device.new_buffer(
            field_bytes(lengths.e_out_fields)?,
            MTLResourceOptions::StorageModeShared,
        );
        write_fields(&e_in_buffer, e_in.len(), &e_in)?;
        write_fields(&e_out_buffer, e_out.len(), &e_out)?;
        let hot_buffer = self
            .device
            .new_buffer(lengths.hot_bytes, MTLResourceOptions::StorageModePrivate);
        let source_rows_storage_id = rows.allocation_identity();
        let hot_rows = HammingHotRows::new(
            hot_buffer,
            rows.len(),
            self.device_registry_id(),
            source_rows_storage_id,
        );
        let buffers = Buffers {
            rows,
            hot_rows,
            validity: self.device.new_buffer(
                lengths.validity_bytes,
                MTLResourceOptions::StorageModePrivate,
            ),
            e_in: e_in_buffer,
            e_out: e_out_buffer,
            partials: self.device.new_buffer(
                field_bytes(lengths.partial_fields)?,
                MTLResourceOptions::StorageModePrivate,
            ),
            output: self.device.new_buffer(
                field_bytes(lengths.output_fields)?,
                MTLResourceOptions::StorageModeShared,
            ),
        };

        Ok(BooleanityAddressSuccessorInvocation {
            context: self.clone(),
            pack_pipeline,
            packed_pipeline,
            finalize_pipeline,
            pack_limits,
            packed_limits,
            finalize_limits,
            buffers,
            geometry,
            params: geometry.params()?,
            completed: Cell::new(false),
        })
    }
}

impl BooleanityAddressSuccessorInvocation {
    pub const fn pack_pipeline_limits(&self) -> PipelineLimits {
        self.pack_limits
    }

    pub const fn packed_pipeline_limits(&self) -> PipelineLimits {
        self.packed_limits
    }

    pub const fn finalize_pipeline_limits(&self) -> PipelineLimits {
        self.finalize_limits
    }

    pub const fn output_elements(&self) -> usize {
        super::BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS * super::BOOLEANITY_ADDRESS_SUCCESSOR_BINS
    }

    pub fn buffer_lengths(
        &self,
    ) -> Result<BooleanityAddressSuccessorBufferLengths, BooleanityAddressSuccessorRuntimeError>
    {
        self.geometry
            .buffer_lengths()
            .map_err(BooleanityAddressSuccessorRuntimeError::Plan)
    }

    pub fn dispatch_plan(
        &self,
    ) -> Result<BooleanityAddressSuccessorDispatchPlan, BooleanityAddressSuccessorRuntimeError>
    {
        self.geometry
            .dispatch_plan()
            .map_err(BooleanityAddressSuccessorRuntimeError::Plan)
    }

    pub fn source_rows_storage_id(&self) -> usize {
        self.buffers.rows.allocation_identity()
    }

    pub fn hot_rows_storage_id(&self) -> usize {
        self.buffers.hot_rows.allocation_identity()
    }

    pub fn execute_timed(&self) -> Result<Duration, BooleanityAddressSuccessorRuntimeError> {
        self.completed.set(false);
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();

            let pack = command_buffer.new_compute_command_encoder();
            pack.set_compute_pipeline_state(&self.pack_pipeline);
            pack.set_buffer(
                PACK_AND_FIRST_BUFFER_ROWS,
                Some(self.buffers.rows.buffer()),
                0,
            );
            pack.set_buffer(PACK_AND_FIRST_BUFFER_E_IN, Some(&self.buffers.e_in), 0);
            pack.set_buffer(PACK_AND_FIRST_BUFFER_E_OUT, Some(&self.buffers.e_out), 0);
            pack.set_buffer(
                PACK_AND_FIRST_BUFFER_HOT,
                Some(self.buffers.hot_rows.buffer()),
                0,
            );
            pack.set_buffer(
                PACK_AND_FIRST_BUFFER_VALIDITY,
                Some(&self.buffers.validity),
                0,
            );
            pack.set_buffer(
                PACK_AND_FIRST_BUFFER_PARTIALS,
                Some(&self.buffers.partials),
                0,
            );
            set_inline_bytes(pack, PACK_AND_FIRST_BUFFER_PARAMS, &self.params);
            pack.set_threadgroup_memory_length(
                0,
                BOOLEANITY_ADDRESS_SUCCESSOR_THREADGROUP_BYTES as u64,
            );
            pack.dispatch_thread_groups(
                MTLSize {
                    width: self.geometry.e_out_length() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.geometry.accumulator_threads_per_threadgroup() as u64,
                    height: 1,
                    depth: 1,
                },
            );
            pack.end_encoding();

            let packed = command_buffer.new_compute_command_encoder();
            packed.set_compute_pipeline_state(&self.packed_pipeline);
            packed.set_buffer(
                PACKED_TILES_BUFFER_HOT,
                Some(self.buffers.hot_rows.buffer()),
                0,
            );
            packed.set_buffer(
                PACKED_TILES_BUFFER_VALIDITY,
                Some(&self.buffers.validity),
                0,
            );
            packed.set_buffer(PACKED_TILES_BUFFER_E_IN, Some(&self.buffers.e_in), 0);
            packed.set_buffer(PACKED_TILES_BUFFER_E_OUT, Some(&self.buffers.e_out), 0);
            packed.set_buffer(
                PACKED_TILES_BUFFER_PARTIALS,
                Some(&self.buffers.partials),
                0,
            );
            set_inline_bytes(packed, PACKED_TILES_BUFFER_PARAMS, &self.params);
            packed.set_threadgroup_memory_length(
                0,
                BOOLEANITY_ADDRESS_SUCCESSOR_THREADGROUP_BYTES as u64,
            );
            packed.dispatch_thread_groups(
                MTLSize {
                    width: (self.geometry.e_out_length()
                        * super::BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES)
                        as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.geometry.accumulator_threads_per_threadgroup() as u64,
                    height: 1,
                    depth: 1,
                },
            );
            packed.end_encoding();

            let finalize = command_buffer.new_compute_command_encoder();
            finalize.set_compute_pipeline_state(&self.finalize_pipeline);
            finalize.set_buffer(FINALIZE_BUFFER_PARTIALS, Some(&self.buffers.partials), 0);
            finalize.set_buffer(FINALIZE_BUFFER_OUTPUT, Some(&self.buffers.output), 0);
            set_inline_bytes(finalize, FINALIZE_BUFFER_PARAMS, &self.params);
            finalize.set_threadgroup_memory_length(
                0,
                BOOLEANITY_ADDRESS_SUCCESSOR_FINALIZE_THREADGROUP_BYTES as u64,
            );
            finalize.dispatch_thread_groups(
                MTLSize {
                    width: super::BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.geometry.finalize_threads_per_threadgroup() as u64,
                    height: 1,
                    depth: 1,
                },
            );
            finalize.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(status).into());
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end }.into());
            }
            self.completed.set(true);
            Ok(Duration::from_secs_f64(end - start))
        })
    }

    pub fn read_masses(&self) -> Result<Vec<AkitaField>, BooleanityAddressSuccessorRuntimeError> {
        if !self.completed.get() {
            return Err(BooleanityAddressSuccessorRuntimeError::InvalidState(
                "readback before command completion",
            ));
        }
        // SAFETY: the shared output allocation contains exactly output_elements
        // fields, and execute_timed completed the only writer command.
        let values = unsafe {
            slice::from_raw_parts(
                self.buffers.output.contents().cast::<Fp128>(),
                self.output_elements(),
            )
        };
        self.context
            .validate_inputs("Booleanity address successor output", values)?;
        Ok(values
            .iter()
            .map(|value| (*value).into_jolt_field())
            .collect())
    }

    pub fn completed_hot_rows(
        &self,
    ) -> Result<HammingHotRows, BooleanityAddressSuccessorRuntimeError> {
        if !self.completed.get() {
            return Err(BooleanityAddressSuccessorRuntimeError::InvalidState(
                "projection requested before command completion",
            ));
        }
        Ok(self.buffers.hot_rows.clone())
    }
}

fn validate_pipeline(
    pipeline: &'static str,
    limits: PipelineLimits,
    threads: usize,
    dynamic_threadgroup_bytes: u64,
    maximum_threadgroup_bytes: u64,
) -> Result<(), BooleanityAddressSuccessorRuntimeError> {
    if limits.thread_execution_width != BOOLEANITY_ADDRESS_SUCCESSOR_SIMD_WIDTH {
        return Err(BooleanityAddressSuccessorRuntimeError::ExecutionWidth {
            pipeline,
            expected: BOOLEANITY_ADDRESS_SUCCESSOR_SIMD_WIDTH,
            got: limits.thread_execution_width,
        });
    }
    if threads > limits.max_total_threads_per_threadgroup {
        return Err(BooleanityAddressSuccessorRuntimeError::ThreadLimit {
            pipeline,
            requested: threads,
            maximum: limits.max_total_threads_per_threadgroup,
        });
    }
    let requested = limits
        .static_threadgroup_memory_length
        .checked_add(dynamic_threadgroup_bytes)
        .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)?;
    if requested > maximum_threadgroup_bytes {
        return Err(BooleanityAddressSuccessorRuntimeError::ThreadgroupMemory {
            pipeline,
            requested,
            maximum: maximum_threadgroup_bytes,
        });
    }
    Ok(())
}

fn field_bytes(fields: u64) -> Result<u64, BooleanityAddressSuccessorError> {
    fields
        .checked_mul(size_of::<Fp128>() as u64)
        .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

impl From<BooleanityAddressSuccessorError> for BooleanityAddressSuccessorRuntimeError {
    fn from(error: BooleanityAddressSuccessorError) -> Self {
        Self::Plan(error)
    }
}
