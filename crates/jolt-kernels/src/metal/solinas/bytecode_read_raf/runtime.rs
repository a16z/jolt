use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSRange,
};

use super::super::{
    buffer_from_slice, command_buffer_timestamp, BooleanityRows, Fp128, MetalError, PipelineLimits,
    SolinasMetal,
};
use super::{
    BytecodeReadRafConfig, BytecodeReadRafError, BytecodeReadRafFusedProductPath,
    BytecodeReadRafLongWorkerSlicePlan, BytecodeReadRafPushforwardParams, BytecodeReadRafRun,
    BytecodeReadRafShape, BytecodeReadRafSplitEqTables, BytecodeReadRafTopology,
    BYTECODE_ADDRESS_DOMAIN, BYTECODE_ADDRESS_SIMD_WIDTH, BYTECODE_ADDRESS_STAGES,
    FINALIZE_PIPELINE, LONG_FULL_PIPELINE, LONG_U64_PIPELINE,
};

const FINALIZE_THREADS: usize = 256;

#[derive(Debug, thiserror::Error)]
pub enum BytecodeReadRafSliceRuntimeError {
    #[error(transparent)]
    Plan(#[from] BytecodeReadRafError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("{name} has {got} stages, expected {expected}")]
    StageCount {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("{name} stage {stage} has {got} elements, expected {expected}")]
    TableLength {
        name: &'static str,
        stage: usize,
        expected: usize,
        got: usize,
    },
    #[error("pipeline `{pipeline}` requires SIMD width {expected}, got {got}")]
    ExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("pipeline `{pipeline}` needs {requested} threads, maximum is {maximum}")]
    ThreadLimit {
        pipeline: &'static str,
        requested: usize,
        maximum: usize,
    },
    #[error("bytecode read/RAF slice byte length overflow")]
    ByteLengthOverflow,
    #[error("invalid bytecode read/RAF slice state: {0}")]
    InvalidState(&'static str),
}

struct SliceBuffers {
    rows: BooleanityRows,
    occurrences: Buffer,
    runs: Buffer,
    counters: Buffer,
    e_lo: Buffer,
    e_hi: Buffer,
    deferred: Buffer,
    output: Buffer,
}

pub struct BytecodeReadRafLongWorkerSliceInvocation {
    context: SolinasMetal,
    long_pipeline: ComputePipelineState,
    finalize_pipeline: ComputePipelineState,
    long_limits: PipelineLimits,
    finalize_limits: PipelineLimits,
    buffers: SliceBuffers,
    plan: BytecodeReadRafLongWorkerSlicePlan,
    params: BytecodeReadRafPushforwardParams,
    finalize_threads: usize,
    owned_bytes: usize,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub fn prepare_bytecode_read_raf_long_worker_slice(
        &self,
        rows: BooleanityRows,
        topology: &BytecodeReadRafTopology,
        tables: &BytecodeReadRafSplitEqTables<AkitaField>,
        config: BytecodeReadRafConfig,
        product_path: BytecodeReadRafFusedProductPath,
    ) -> Result<BytecodeReadRafLongWorkerSliceInvocation, BytecodeReadRafSliceRuntimeError> {
        self.validate_booleanity_rows(&rows)?;
        let shape = BytecodeReadRafShape::new(rows.len(), BYTECODE_ADDRESS_DOMAIN)?;
        let plan = BytecodeReadRafLongWorkerSlicePlan::new(shape, config)?;
        plan.validate_topology(topology)?;

        let e_lo = flatten_tables("E_lo", &tables.e_lo, shape.inner_length())?;
        let e_hi = flatten_tables("E_hi", &tables.e_hi, shape.outer_length())?;
        let e_lo = encode_fields(self, "bytecode address E_lo", &e_lo)?;
        let e_hi = encode_fields(self, "bytecode address E_hi", &e_hi)?;

        let long_name = match product_path {
            BytecodeReadRafFusedProductPath::FullWidth => LONG_FULL_PIPELINE,
            BytecodeReadRafFusedProductPath::ExactU64 => LONG_U64_PIPELINE,
        };
        let long_pipeline = self.compile_named_pipeline(long_name)?;
        let finalize_pipeline = self.compile_named_pipeline(FINALIZE_PIPELINE)?;
        let long_limits = Self::limits(&long_pipeline);
        let finalize_limits = Self::limits(&finalize_pipeline);
        validate_pipeline(long_name, long_limits, plan.long_threads() as usize, true)?;
        let finalize_threads =
            FINALIZE_THREADS.min(finalize_limits.max_total_threads_per_threadgroup);
        validate_pipeline(FINALIZE_PIPELINE, finalize_limits, finalize_threads, false)?;

        let storage = shape.storage_plan()?;
        for bytes in [
            storage.occurrence_bytes,
            storage.run_bytes,
            storage.e_lo_bytes,
            storage.e_hi_bytes,
            storage.deferred_output_bytes,
            storage.output_bytes,
            storage.status_bytes,
        ] {
            self.validate_buffer_length(bytes_u64(bytes)?)?;
        }
        self.validate_additional_working_set(bytes_u64(storage.owned_bytes)?)?;

        let occurrences = buffer_from_slice(&self.device, &topology.occurrences);
        let runs = self.device.new_buffer(
            bytes_u64(storage.run_bytes)?,
            MTLResourceOptions::StorageModeShared,
        );
        write_long_runs(&runs, plan, topology)?;
        let counters = buffer_from_slice(&self.device, &[plan.worker_counters()]);
        let buffers = SliceBuffers {
            rows,
            occurrences,
            runs,
            counters,
            e_lo: buffer_from_slice(&self.device, &e_lo),
            e_hi: buffer_from_slice(&self.device, &e_hi),
            deferred: self.device.new_buffer(
                bytes_u64(storage.deferred_output_bytes)?,
                MTLResourceOptions::StorageModePrivate,
            ),
            output: self.device.new_buffer(
                bytes_u64(storage.output_bytes)?,
                MTLResourceOptions::StorageModeShared,
            ),
        };

        Ok(BytecodeReadRafLongWorkerSliceInvocation {
            context: self.clone(),
            long_pipeline,
            finalize_pipeline,
            long_limits,
            finalize_limits,
            buffers,
            plan,
            params: plan.pushforward_params(),
            finalize_threads,
            owned_bytes: storage.owned_bytes,
            completed: Cell::new(false),
        })
    }
}

impl BytecodeReadRafLongWorkerSliceInvocation {
    pub const fn shape(&self) -> BytecodeReadRafShape {
        self.plan.shape()
    }

    pub const fn long_pipeline_limits(&self) -> PipelineLimits {
        self.long_limits
    }

    pub const fn finalize_pipeline_limits(&self) -> PipelineLimits {
        self.finalize_limits
    }

    pub const fn owned_bytes(&self) -> usize {
        self.owned_bytes
    }

    pub fn source_rows_storage_id(&self) -> usize {
        self.buffers.rows.allocation_identity()
    }

    pub fn static_buffer_identities(&self) -> [usize; 7] {
        [
            self.buffers.occurrences.as_ptr() as usize,
            self.buffers.runs.as_ptr() as usize,
            self.buffers.counters.as_ptr() as usize,
            self.buffers.e_lo.as_ptr() as usize,
            self.buffers.e_hi.as_ptr() as usize,
            self.buffers.deferred.as_ptr() as usize,
            self.buffers.output.as_ptr() as usize,
        ]
    }

    pub fn execute_timed(&self) -> Result<Duration, BytecodeReadRafSliceRuntimeError> {
        self.completed.set(false);
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let blit = command_buffer.new_blit_command_encoder();
            blit.fill_buffer(
                &self.buffers.deferred,
                NSRange::new(0, self.buffers.deferred.length()),
                0,
            );
            blit.end_encoding();

            let long = command_buffer.new_compute_command_encoder();
            long.set_compute_pipeline_state(&self.long_pipeline);
            long.set_buffer(0, Some(self.buffers.rows.buffer()), 0);
            long.set_buffer(1, Some(&self.buffers.occurrences), 0);
            long.set_buffer(2, Some(&self.buffers.runs), 0);
            long.set_buffer(3, Some(&self.buffers.counters), 0);
            long.set_buffer(4, Some(&self.buffers.e_lo), 0);
            long.set_buffer(5, Some(&self.buffers.e_hi), 0);
            long.set_buffer(6, Some(&self.buffers.deferred), 0);
            set_inline_bytes(long, 7, &self.params);
            let grid = self.plan.long_grid();
            long.dispatch_thread_groups(
                MTLSize {
                    width: grid.threadgroups_x as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.plan.long_threads() as u64,
                    height: 1,
                    depth: 1,
                },
            );
            long.end_encoding();

            let finalize = command_buffer.new_compute_command_encoder();
            finalize.set_compute_pipeline_state(&self.finalize_pipeline);
            finalize.set_buffer(0, Some(&self.buffers.deferred), 0);
            finalize.set_buffer(1, Some(&self.buffers.output), 0);
            set_inline_bytes(finalize, 2, &self.params);
            finalize.dispatch_thread_groups(
                MTLSize {
                    width: self.output_elements().div_ceil(self.finalize_threads) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.finalize_threads as u64,
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

    pub fn read_output(&self) -> Result<Vec<AkitaField>, BytecodeReadRafSliceRuntimeError> {
        if !self.completed.get() {
            return Err(BytecodeReadRafSliceRuntimeError::InvalidState(
                "readback before command completion",
            ));
        }
        // SAFETY: the shared buffer contains `output_elements` fields and the
        // only writer command has completed.
        let values = unsafe {
            slice::from_raw_parts(
                self.buffers.output.contents().cast::<Fp128>(),
                self.output_elements(),
            )
        };
        self.context
            .validate_inputs("bytecode address long-worker output", values)?;
        Ok(values
            .iter()
            .map(|value| (*value).into_jolt_field())
            .collect())
    }

    const fn output_elements(&self) -> usize {
        BYTECODE_ADDRESS_STAGES * self.plan.shape().addresses()
    }
}

fn flatten_tables(
    name: &'static str,
    tables: &[Vec<AkitaField>],
    expected_length: usize,
) -> Result<Vec<AkitaField>, BytecodeReadRafSliceRuntimeError> {
    if tables.len() != BYTECODE_ADDRESS_STAGES {
        return Err(BytecodeReadRafSliceRuntimeError::StageCount {
            name,
            expected: BYTECODE_ADDRESS_STAGES,
            got: tables.len(),
        });
    }
    let mut flattened = Vec::with_capacity(BYTECODE_ADDRESS_STAGES * expected_length);
    for (stage, table) in tables.iter().enumerate() {
        if table.len() != expected_length {
            return Err(BytecodeReadRafSliceRuntimeError::TableLength {
                name,
                stage,
                expected: expected_length,
                got: table.len(),
            });
        }
        flattened.extend_from_slice(table);
    }
    Ok(flattened)
}

fn encode_fields(
    context: &SolinasMetal,
    name: &'static str,
    values: &[AkitaField],
) -> Result<Vec<Fp128>, MetalError> {
    let encoded = values
        .iter()
        .map(Fp128::from_jolt_field)
        .collect::<Vec<_>>();
    context.validate_inputs(name, &encoded)?;
    Ok(encoded)
}

fn write_long_runs(
    buffer: &Buffer,
    plan: BytecodeReadRafLongWorkerSlicePlan,
    topology: &BytecodeReadRafTopology,
) -> Result<(), BytecodeReadRafSliceRuntimeError> {
    plan.validate_topology(topology)?;
    for (run_index, run) in topology.long_runs.iter().copied().enumerate() {
        let arena_index = plan
            .run_arena_index(run_index)
            .ok_or(BytecodeReadRafError::TopologyInvariant)?;
        // SAFETY: `arena_index` is within the allocation-sized run arena and
        // the buffer has no concurrent GPU readers during preparation.
        unsafe {
            buffer
                .contents()
                .cast::<BytecodeReadRafRun>()
                .add(arena_index)
                .write(run);
        }
    }
    Ok(())
}

fn validate_pipeline(
    pipeline: &'static str,
    limits: PipelineLimits,
    requested: usize,
    require_simd_width: bool,
) -> Result<(), BytecodeReadRafSliceRuntimeError> {
    if require_simd_width && limits.thread_execution_width != BYTECODE_ADDRESS_SIMD_WIDTH {
        return Err(BytecodeReadRafSliceRuntimeError::ExecutionWidth {
            pipeline,
            expected: BYTECODE_ADDRESS_SIMD_WIDTH,
            got: limits.thread_execution_width,
        });
    }
    if requested == 0 || requested > limits.max_total_threads_per_threadgroup {
        return Err(BytecodeReadRafSliceRuntimeError::ThreadLimit {
            pipeline,
            requested,
            maximum: limits.max_total_threads_per_threadgroup,
        });
    }
    Ok(())
}

fn bytes_u64(bytes: usize) -> Result<u64, BytecodeReadRafSliceRuntimeError> {
    u64::try_from(bytes).map_err(|_| BytecodeReadRafSliceRuntimeError::ByteLengthOverflow)
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}
