use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use thiserror::Error;

use super::{carrier::AddressMajorShape, oracle::HostAddressMajorCarrier};
use crate::metal::solinas::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

pub const BYTECODE_ADDRESS_MAJOR_STAGES: usize = 9;
pub const BYTECODE_ADDRESS_MAJOR_BASE_STAGES: usize = 5;
pub const BYTECODE_ADDRESS_MAJOR_THREADS: usize = 256;
pub const BYTECODE_ADDRESS_MAJOR_SIMD_WIDTH: usize = 32;
pub const BYTECODE_ADDRESS_MAJOR_SIMDGROUPS: usize =
    BYTECODE_ADDRESS_MAJOR_THREADS / BYTECODE_ADDRESS_MAJOR_SIMD_WIDTH;

const WORKER_PIPELINE: &str = "solinas_bytecode_address_major_worker_5_4";
const REDUCE_PIPELINE: &str = "solinas_bytecode_address_major_reduce_tiles";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeAddressMajorConfig {
    pub outer_tiles: usize,
}

impl Default for BytecodeAddressMajorConfig {
    fn default() -> Self {
        Self { outer_tiles: 8 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BytecodeAddressMajorParams {
    rows: u32,
    addresses: u32,
    inner_length: u32,
    outer_length: u32,
    outer_tiles: u32,
    stages: u32,
    base_stages: u32,
    reserved: u32,
}

const _: [(); 32] = [(); size_of::<BytecodeAddressMajorParams>()];

impl BytecodeAddressMajorParams {
    fn new(
        shape: AddressMajorShape,
        config: BytecodeAddressMajorConfig,
    ) -> Result<Self, BytecodeAddressMajorRuntimeError> {
        let rows = shape.rows()?;
        let addresses = shape.addresses()?;
        let inner_length = shape.inner_length()?;
        let outer_length = shape.outer_length()?;
        if config.outer_tiles == 0 || config.outer_tiles > outer_length {
            return Err(BytecodeAddressMajorRuntimeError::InvalidOuterTiles {
                tiles: config.outer_tiles,
                outer_length,
            });
        }
        Ok(Self {
            rows: shader_count("rows", rows)?,
            addresses: shader_count("addresses", addresses)?,
            inner_length: shader_count("inner length", inner_length)?,
            outer_length: shader_count("outer length", outer_length)?,
            outer_tiles: shader_count("outer tiles", config.outer_tiles)?,
            stages: BYTECODE_ADDRESS_MAJOR_STAGES as u32,
            base_stages: BYTECODE_ADDRESS_MAJOR_BASE_STAGES as u32,
            reserved: 0,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeAddressMajorStorage {
    pub carrier_bytes: usize,
    pub equality_bytes: usize,
    pub partial_bytes: usize,
    pub output_bytes: usize,
    pub owned_bytes: usize,
}

struct BytecodeAddressMajorBuffers {
    cells: Buffer,
    inner_sign: Buffer,
    magnitude: Buffer,
    e_lo: Buffer,
    e_hi: Buffer,
    partials: Buffer,
    output: Buffer,
}

pub struct BytecodeAddressMajorInvocation {
    context: SolinasMetal,
    worker_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    worker_limits: PipelineLimits,
    reduce_limits: PipelineLimits,
    buffers: BytecodeAddressMajorBuffers,
    params: BytecodeAddressMajorParams,
    storage: BytecodeAddressMajorStorage,
    completed: Cell<bool>,
}

impl SolinasMetal {
    /// Upload-backed bring-up path for the production address-major worker.
    /// The stage-6a integration must replace these uploads with checked owner buffers.
    #[doc(hidden)]
    pub fn prepare_bytecode_address_major_probe(
        &self,
        carrier: &HostAddressMajorCarrier,
        e_lo: &[Vec<AkitaField>],
        e_hi: &[Vec<AkitaField>],
        config: BytecodeAddressMajorConfig,
    ) -> Result<BytecodeAddressMajorInvocation, BytecodeAddressMajorRuntimeError> {
        let shape = carrier.shape();
        let params = BytecodeAddressMajorParams::new(shape, config)?;
        let rows = shape.rows()?;
        let addresses = shape.addresses()?;
        let inner_length = shape.inner_length()?;
        let outer_length = shape.outer_length()?;
        validate_table_shape("E_lo", e_lo, inner_length)?;
        validate_table_shape("E_hi", e_hi, outer_length)?;

        let e_lo = flatten_tables(e_lo);
        let e_hi = flatten_tables(e_hi);
        self.validate_inputs("bytecode address-major E_lo", &e_lo)?;
        self.validate_inputs("bytecode address-major E_hi", &e_hi)?;

        let output_fields = checked_mul("output fields", BYTECODE_ADDRESS_MAJOR_STAGES, addresses)?;
        let partial_fields = checked_mul("partial fields", output_fields, config.outer_tiles)?;
        let partial_bytes = field_bytes(partial_fields)?;
        let output_bytes = field_bytes(output_fields)?;
        let carrier_bytes = checked_add(
            "carrier bytes",
            checked_add(
                "carrier bytes",
                byte_len("cells", carrier.cells().len(), size_of::<u32>())?,
                byte_len("inner/sign", carrier.inner_sign().len(), size_of::<u32>())?,
            )?,
            byte_len("magnitude", carrier.magnitude().len(), size_of::<u64>())?,
        )?;
        if carrier.inner_sign().len() != rows || carrier.magnitude().len() != rows {
            return Err(BytecodeAddressMajorRuntimeError::InvalidCarrierLength);
        }
        let equality_bytes = checked_add(
            "equality bytes",
            field_bytes(e_lo.len())?,
            field_bytes(e_hi.len())?,
        )?;
        let owned_bytes = [carrier_bytes, equality_bytes, partial_bytes, output_bytes]
            .into_iter()
            .try_fold(0usize, |sum, bytes| checked_add("owned bytes", sum, bytes))?;
        self.validate_additional_working_set(
            u64::try_from(owned_bytes)
                .map_err(|_| BytecodeAddressMajorRuntimeError::SizeOverflow("working set"))?,
        )?;
        for bytes in [partial_bytes, output_bytes] {
            self.validate_buffer_length(
                u64::try_from(bytes)
                    .map_err(|_| BytecodeAddressMajorRuntimeError::SizeOverflow("buffer"))?,
            )?;
        }

        let worker_pipeline = self.compile_named_pipeline(WORKER_PIPELINE)?;
        let reduce_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let worker_limits = Self::limits(&worker_pipeline);
        let reduce_limits = Self::limits(&reduce_pipeline);
        validate_pipeline(
            WORKER_PIPELINE,
            worker_limits,
            BYTECODE_ADDRESS_MAJOR_THREADS,
        )?;
        validate_pipeline(
            REDUCE_PIPELINE,
            reduce_limits,
            BYTECODE_ADDRESS_MAJOR_THREADS,
        )?;

        Ok(BytecodeAddressMajorInvocation {
            context: self.clone(),
            worker_pipeline,
            reduce_pipeline,
            worker_limits,
            reduce_limits,
            buffers: BytecodeAddressMajorBuffers {
                cells: buffer_from_slice(&self.device, carrier.cells()),
                inner_sign: buffer_from_slice(&self.device, carrier.inner_sign()),
                magnitude: buffer_from_slice(&self.device, carrier.magnitude()),
                e_lo: buffer_from_slice(&self.device, &e_lo),
                e_hi: buffer_from_slice(&self.device, &e_hi),
                partials: self
                    .device
                    .new_buffer(partial_bytes as u64, MTLResourceOptions::StorageModeShared),
                output: self
                    .device
                    .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared),
            },
            params,
            storage: BytecodeAddressMajorStorage {
                carrier_bytes,
                equality_bytes,
                partial_bytes,
                output_bytes,
                owned_bytes,
            },
            completed: Cell::new(false),
        })
    }
}

impl BytecodeAddressMajorInvocation {
    pub fn execute(&self) -> Result<Vec<AkitaField>, BytecodeAddressMajorRuntimeError> {
        self.execute_timed().map(|(output, _)| output)
    }

    pub fn execute_timed(
        &self,
    ) -> Result<(Vec<AkitaField>, Duration), BytecodeAddressMajorRuntimeError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let worker = command_buffer.new_compute_command_encoder();
            worker.set_compute_pipeline_state(&self.worker_pipeline);
            worker.set_buffer(0, Some(&self.buffers.cells), 0);
            worker.set_buffer(1, Some(&self.buffers.inner_sign), 0);
            worker.set_buffer(2, Some(&self.buffers.magnitude), 0);
            worker.set_buffer(3, Some(&self.buffers.e_lo), 0);
            worker.set_buffer(4, Some(&self.buffers.e_hi), 0);
            worker.set_buffer(5, Some(&self.buffers.partials), 0);
            set_inline_bytes(worker, 6, &self.params);
            worker.set_threadgroup_memory_length(0, threadgroup_bytes() as u64);
            worker.dispatch_thread_groups(
                MTLSize {
                    width: u64::from(self.params.addresses) * u64::from(self.params.outer_tiles),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: BYTECODE_ADDRESS_MAJOR_THREADS as u64,
                    height: 1,
                    depth: 1,
                },
            );
            worker.end_encoding();

            let reduce = command_buffer.new_compute_command_encoder();
            reduce.set_compute_pipeline_state(&self.reduce_pipeline);
            reduce.set_buffer(0, Some(&self.buffers.partials), 0);
            reduce.set_buffer(1, Some(&self.buffers.output), 0);
            set_inline_bytes(reduce, 2, &self.params);
            let output_fields = self.output_fields();
            reduce.dispatch_thread_groups(
                MTLSize {
                    width: output_fields.div_ceil(BYTECODE_ADDRESS_MAJOR_THREADS) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: BYTECODE_ADDRESS_MAJOR_THREADS as u64,
                    height: 1,
                    depth: 1,
                },
            );
            reduce.end_encoding();

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
            Ok((self.read_output()?, Duration::from_secs_f64(end - start)))
        })
    }

    pub fn read_output(&self) -> Result<Vec<AkitaField>, BytecodeAddressMajorRuntimeError> {
        if !self.completed.get() {
            return Err(BytecodeAddressMajorRuntimeError::NotExecuted);
        }
        let output_fields = self.output_fields();
        // SAFETY: the shared buffer holds `output_fields` values and the writer
        // command has completed before this method is called.
        let output = unsafe {
            slice::from_raw_parts(
                self.buffers.output.contents().cast::<Fp128>(),
                output_fields,
            )
        };
        self.context
            .validate_inputs("bytecode address-major output", output)?;
        Ok(output
            .iter()
            .map(|value| (*value).into_jolt_field())
            .collect())
    }

    pub const fn storage(&self) -> BytecodeAddressMajorStorage {
        self.storage
    }

    pub const fn outer_tiles(&self) -> usize {
        self.params.outer_tiles as usize
    }

    pub const fn worker_pipeline_limits(&self) -> PipelineLimits {
        self.worker_limits
    }

    pub const fn reduce_pipeline_limits(&self) -> PipelineLimits {
        self.reduce_limits
    }

    pub const fn threadgroup_memory_bytes(&self) -> usize {
        threadgroup_bytes()
    }

    const fn output_fields(&self) -> usize {
        self.params.stages as usize * self.params.addresses as usize
    }
}

#[derive(Debug, Error)]
pub enum BytecodeAddressMajorRuntimeError {
    #[error(transparent)]
    Carrier(#[from] super::carrier::CarrierError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("bytecode address-major {0} size overflow")]
    SizeOverflow(&'static str),
    #[error("bytecode address-major {table} has {got} stages, expected {expected}")]
    StageCount {
        table: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("bytecode address-major {table} stage {stage} has {got} fields, expected {expected}")]
    TableLength {
        table: &'static str,
        stage: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "bytecode address-major outer tile count {tiles} is invalid for {outer_length} outer blocks"
    )]
    InvalidOuterTiles { tiles: usize, outer_length: usize },
    #[error("bytecode address-major carrier planes have inconsistent lengths")]
    InvalidCarrierLength,
    #[error(
        "bytecode address-major pipeline `{pipeline}` has SIMD width {got_width} and thread limit {got_threads}; expected width 32 and at least {required_threads} threads"
    )]
    UnsupportedPipeline {
        pipeline: &'static str,
        got_width: usize,
        got_threads: usize,
        required_threads: usize,
    },
    #[error("bytecode address-major output read before execution")]
    NotExecuted,
}

fn validate_table_shape(
    table: &'static str,
    tables: &[Vec<AkitaField>],
    expected_length: usize,
) -> Result<(), BytecodeAddressMajorRuntimeError> {
    if tables.len() != BYTECODE_ADDRESS_MAJOR_STAGES {
        return Err(BytecodeAddressMajorRuntimeError::StageCount {
            table,
            expected: BYTECODE_ADDRESS_MAJOR_STAGES,
            got: tables.len(),
        });
    }
    for (stage, values) in tables.iter().enumerate() {
        if values.len() != expected_length {
            return Err(BytecodeAddressMajorRuntimeError::TableLength {
                table,
                stage,
                expected: expected_length,
                got: values.len(),
            });
        }
    }
    Ok(())
}

fn flatten_tables(tables: &[Vec<AkitaField>]) -> Vec<Fp128> {
    tables
        .iter()
        .flat_map(|table| table.iter().map(Fp128::from_jolt_field))
        .collect()
}

fn validate_pipeline(
    pipeline: &'static str,
    limits: PipelineLimits,
    required_threads: usize,
) -> Result<(), BytecodeAddressMajorRuntimeError> {
    if limits.thread_execution_width != BYTECODE_ADDRESS_MAJOR_SIMD_WIDTH
        || limits.max_total_threads_per_threadgroup < required_threads
    {
        return Err(BytecodeAddressMajorRuntimeError::UnsupportedPipeline {
            pipeline,
            got_width: limits.thread_execution_width,
            got_threads: limits.max_total_threads_per_threadgroup,
            required_threads,
        });
    }
    Ok(())
}

const fn threadgroup_bytes() -> usize {
    BYTECODE_ADDRESS_MAJOR_SIMDGROUPS * BYTECODE_ADDRESS_MAJOR_BASE_STAGES * size_of::<Fp128>()
}

fn shader_count(name: &'static str, value: usize) -> Result<u32, BytecodeAddressMajorRuntimeError> {
    u32::try_from(value).map_err(|_| BytecodeAddressMajorRuntimeError::SizeOverflow(name))
}

fn checked_mul(
    name: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, BytecodeAddressMajorRuntimeError> {
    left.checked_mul(right)
        .ok_or(BytecodeAddressMajorRuntimeError::SizeOverflow(name))
}

fn checked_add(
    name: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, BytecodeAddressMajorRuntimeError> {
    left.checked_add(right)
        .ok_or(BytecodeAddressMajorRuntimeError::SizeOverflow(name))
}

fn byte_len(
    name: &'static str,
    elements: usize,
    element_bytes: usize,
) -> Result<usize, BytecodeAddressMajorRuntimeError> {
    checked_mul(name, elements, element_bytes)
}

fn field_bytes(elements: usize) -> Result<usize, BytecodeAddressMajorRuntimeError> {
    byte_len("field buffer", elements, size_of::<Fp128>())
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast(),
    );
}
