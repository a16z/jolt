use std::{cell::Cell, mem::size_of, slice};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLResourceOptions, MTLSize,
};
use thiserror::Error;

use super::{
    carrier::AddressMajorShape,
    worklist::{BYTECODE_ADDRESS_BASE_STAGES, BYTECODE_ADDRESS_PUSHFORWARD_STAGES},
    worklist_owner::{BytecodeAddressSparseStage1Carrier, BytecodeAddressSparseStage1Receipt},
};
use crate::metal::solinas::{
    buffer_from_slice, set_inline_bytes, validate_completed_command, Fp128, MetalError,
    PipelineLimits, SolinasMetal,
};

const WORKER_PIPELINE: &str = "solinas_bytecode_address_sparse_worker_packed_4_5_4";
const REDUCE_PIPELINE: &str = "solinas_bytecode_address_sparse_reduce";
const WORKER_THREADS: usize = 128;
const WORKER_ITEMS_PER_THREADGROUP: usize = 4;
const REDUCE_THREADS: usize = 256;
const SIMD_WIDTH: usize = 32;
const THREADGROUP_BYTES: usize = 0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BytecodeAddressSparseParams {
    physical_rows: u32,
    addresses: u32,
    inner_length: u32,
    outer_length: u32,
    work_items: u32,
    stages: u32,
    base_stages: u32,
    reserved: u32,
}

const _: [(); 32] = [(); size_of::<BytecodeAddressSparseParams>()];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BytecodeAddressSparseObservation {
    pub(crate) output: Vec<AkitaField>,
    pub(crate) receipt: BytecodeAddressSparseStage1Receipt,
}

struct BytecodeAddressSparseBuffers {
    occurrences: Buffer,
    magnitudes: Buffer,
    work_items: Buffer,
    address_offsets: Buffer,
    e_lo: Buffer,
    e_hi: Buffer,
    padding: Buffer,
    partials: Buffer,
    output: Buffer,
}

struct BytecodeAddressSparseResidentInput {
    stage1_receipt: BytecodeAddressSparseStage1Receipt,
    shape: AddressMajorShape,
    physical_rows: usize,
    work_items: usize,
    occurrences: Buffer,
    magnitudes: Buffer,
    items: Buffer,
    address_offsets: Buffer,
}

pub(crate) struct BytecodeAddressSparseInvocation {
    context: SolinasMetal,
    worker_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    buffers: BytecodeAddressSparseBuffers,
    params: BytecodeAddressSparseParams,
    source_rows_storage_id: usize,
    source_device_registry_id: u64,
    source_generation: u64,
    source_completion_serial: u64,
    stage1_receipt: BytecodeAddressSparseStage1Receipt,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub(crate) fn prepare_bytecode_address_sparse_resident(
        &self,
        carrier: BytecodeAddressSparseStage1Carrier,
        e_lo: &[Vec<AkitaField>],
        e_hi: &[Vec<AkitaField>],
    ) -> Result<BytecodeAddressSparseInvocation, BytecodeAddressSparseRuntimeError> {
        let parts = carrier.into_parts();
        let receipt = parts.receipt;
        self.prepare_bytecode_address_sparse_buffers(
            BytecodeAddressSparseResidentInput {
                stage1_receipt: receipt,
                shape: receipt.shape(),
                physical_rows: receipt.physical_rows(),
                work_items: receipt.work_items(),
                occurrences: parts.occurrences,
                magnitudes: parts.magnitudes,
                items: parts.work_items,
                address_offsets: parts.address_offsets,
            },
            e_lo,
            e_hi,
        )
    }

    fn prepare_bytecode_address_sparse_buffers(
        &self,
        input: BytecodeAddressSparseResidentInput,
        e_lo: &[Vec<AkitaField>],
        e_hi: &[Vec<AkitaField>],
    ) -> Result<BytecodeAddressSparseInvocation, BytecodeAddressSparseRuntimeError> {
        let padded_rows = input.shape.rows()?;
        let addresses = input.shape.addresses()?;
        let inner_length = input.shape.inner_length()?;
        let outer_length = input.shape.outer_length()?;
        if input.physical_rows == 0 || input.physical_rows > padded_rows || input.work_items == 0 {
            return Err(BytecodeAddressSparseRuntimeError::InvalidRows {
                physical: input.physical_rows,
                padded: padded_rows,
            });
        }
        validate_table_shape("E_lo", e_lo, inner_length)?;
        validate_table_shape("E_hi", e_hi, outer_length)?;
        let flat_e_lo = flatten_tables(e_lo);
        let flat_e_hi = flatten_tables(e_hi);
        let padding = padding_base_terms(input.physical_rows, e_lo, e_hi)?;
        self.validate_inputs("bytecode sparse E_lo", &flat_e_lo)?;
        self.validate_inputs("bytecode sparse E_hi", &flat_e_hi)?;
        self.validate_inputs("bytecode sparse padding", &padding)?;

        let occurrence_bytes = input.physical_rows.checked_mul(size_of::<u16>()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("occurrence bytes"),
        )?;
        let magnitude_bytes = input.physical_rows.checked_mul(size_of::<u64>()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("magnitude bytes"),
        )?;
        let work_item_bytes = input.work_items.checked_mul(8).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("work item bytes"),
        )?;
        let address_offset_bytes = (addresses + 1).checked_mul(size_of::<u32>()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("address offset bytes"),
        )?;
        let equality_bytes = field_bytes(flat_e_lo.len().checked_add(flat_e_hi.len()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("equality fields"),
        )?)?;
        let padding_bytes = field_bytes(padding.len())?;
        let partial_fields = BYTECODE_ADDRESS_PUSHFORWARD_STAGES
            .checked_mul(input.work_items)
            .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                "partial fields",
            ))?;
        let output_fields = BYTECODE_ADDRESS_PUSHFORWARD_STAGES
            .checked_mul(addresses)
            .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                "output fields",
            ))?;
        let partial_bytes = field_bytes(partial_fields)?;
        let output_bytes = field_bytes(output_fields)?;
        let member_owned_bytes = [equality_bytes, padding_bytes, partial_bytes, output_bytes]
            .into_iter()
            .try_fold(0usize, |total, bytes| {
                total
                    .checked_add(bytes)
                    .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                        "member owned bytes",
                    ))
            })?;
        self.validate_additional_working_set(to_u64("working set", member_owned_bytes)?)?;
        for bytes in [
            occurrence_bytes,
            magnitude_bytes,
            work_item_bytes,
            address_offset_bytes,
            equality_bytes,
            padding_bytes,
            partial_bytes,
            output_bytes,
        ] {
            self.validate_buffer_length(to_u64("buffer", bytes)?)?;
        }

        let worker_pipeline = self.compile_named_pipeline(WORKER_PIPELINE)?;
        let reduce_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let worker_limits = Self::limits(&worker_pipeline);
        let reduce_limits = Self::limits(&reduce_pipeline);
        validate_pipeline(WORKER_PIPELINE, worker_limits, WORKER_THREADS)?;
        validate_pipeline(REDUCE_PIPELINE, reduce_limits, REDUCE_THREADS)?;
        let maximum_threadgroup_bytes = self.device_info().max_threadgroup_memory_length;
        let requested_threadgroup_bytes = worker_limits
            .static_threadgroup_memory_length
            .checked_add(THREADGROUP_BYTES as u64)
            .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                "threadgroup memory",
            ))?;
        if requested_threadgroup_bytes > maximum_threadgroup_bytes {
            return Err(BytecodeAddressSparseRuntimeError::ThreadgroupMemory {
                requested: requested_threadgroup_bytes,
                maximum: maximum_threadgroup_bytes,
            });
        }

        let params = BytecodeAddressSparseParams {
            physical_rows: shader_count("physical rows", input.physical_rows)?,
            addresses: shader_count("addresses", addresses)?,
            inner_length: shader_count("inner length", inner_length)?,
            outer_length: shader_count("outer length", outer_length)?,
            work_items: shader_count("work items", input.work_items)?,
            stages: BYTECODE_ADDRESS_PUSHFORWARD_STAGES as u32,
            base_stages: BYTECODE_ADDRESS_BASE_STAGES as u32,
            reserved: 0,
        };
        let source_rows_storage_id = input.stage1_receipt.source_rows_storage_id();
        let source_device_registry_id = input.stage1_receipt.device_registry_id();
        let source_generation = input.stage1_receipt.source_generation();
        let source_completion_serial = input.stage1_receipt.source_completion_serial();
        Ok(BytecodeAddressSparseInvocation {
            context: self.clone(),
            worker_pipeline,
            reduce_pipeline,
            buffers: BytecodeAddressSparseBuffers {
                occurrences: input.occurrences,
                magnitudes: input.magnitudes,
                work_items: input.items,
                address_offsets: input.address_offsets,
                e_lo: buffer_from_slice(&self.device, &flat_e_lo),
                e_hi: buffer_from_slice(&self.device, &flat_e_hi),
                padding: buffer_from_slice(&self.device, &padding),
                partials: self
                    .device
                    .new_buffer(partial_bytes as u64, MTLResourceOptions::StorageModePrivate),
                output: self
                    .device
                    .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared),
            },
            params,
            source_rows_storage_id,
            source_device_registry_id,
            source_generation,
            source_completion_serial,
            stage1_receipt: input.stage1_receipt,
            completed: Cell::new(false),
        })
    }
}

impl BytecodeAddressSparseInvocation {
    pub(crate) fn execute_timed(
        &self,
    ) -> Result<BytecodeAddressSparseObservation, BytecodeAddressSparseRuntimeError> {
        autoreleasepool(|| {
            self.validate_source()?;
            self.completed.set(false);
            let command_buffer = self.context.queue.new_command_buffer();
            self.encode(command_buffer);
            command_buffer.commit();
            command_buffer.wait_until_completed();
            validate_completed_command(command_buffer)?;
            self.validate_source()?;
            self.completed.set(true);
            Ok(BytecodeAddressSparseObservation {
                output: self.read_output()?,
                receipt: self.stage1_receipt,
            })
        })
    }

    fn encode(&self, command_buffer: &metal::CommandBufferRef) {
        let worker = command_buffer.new_compute_command_encoder();
        worker.set_compute_pipeline_state(&self.worker_pipeline);
        worker.set_buffer(0, Some(&self.buffers.occurrences), 0);
        worker.set_buffer(1, Some(&self.buffers.magnitudes), 0);
        worker.set_buffer(2, Some(&self.buffers.work_items), 0);
        worker.set_buffer(3, Some(&self.buffers.e_lo), 0);
        worker.set_buffer(4, Some(&self.buffers.e_hi), 0);
        worker.set_buffer(5, Some(&self.buffers.partials), 0);
        set_inline_bytes(worker, 6, &self.params);
        worker.dispatch_thread_groups(
            MTLSize {
                width: u64::from(self.params.work_items)
                    .div_ceil(WORKER_ITEMS_PER_THREADGROUP as u64),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: WORKER_THREADS as u64,
                height: 1,
                depth: 1,
            },
        );
        worker.end_encoding();

        let reduce = command_buffer.new_compute_command_encoder();
        reduce.set_compute_pipeline_state(&self.reduce_pipeline);
        reduce.set_buffer(0, Some(&self.buffers.partials), 0);
        reduce.set_buffer(1, Some(&self.buffers.address_offsets), 0);
        reduce.set_buffer(2, Some(&self.buffers.padding), 0);
        reduce.set_buffer(3, Some(&self.buffers.output), 0);
        set_inline_bytes(reduce, 4, &self.params);
        let output_fields = u64::from(self.params.stages) * u64::from(self.params.addresses);
        reduce.dispatch_thread_groups(
            MTLSize {
                width: output_fields.div_ceil(REDUCE_THREADS as u64),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: REDUCE_THREADS as u64,
                height: 1,
                depth: 1,
            },
        );
        reduce.end_encoding();
    }

    fn validate_source(&self) -> Result<(), BytecodeAddressSparseRuntimeError> {
        let receipt = self.stage1_receipt;
        if receipt.device_registry_id() != self.context.device_registry_id()
            || receipt.device_registry_id() != self.source_device_registry_id
            || receipt.source_rows_storage_id() != self.source_rows_storage_id
            || receipt.source_generation() != self.source_generation
            || receipt.source_completion_serial() != self.source_completion_serial
            || receipt.completion_serial() == 0
            || !receipt.complete_overwrite()
            || receipt.covered_rows() != self.params.physical_rows as usize
            || receipt.work_items() != self.params.work_items as usize
            || receipt.occurrence_storage_id() != self.buffers.occurrences.as_ptr() as usize
            || receipt.magnitude_storage_id() != self.buffers.magnitudes.as_ptr() as usize
            || receipt.work_item_storage_id() != self.buffers.work_items.as_ptr() as usize
            || receipt.address_offset_storage_id() != self.buffers.address_offsets.as_ptr() as usize
            || receipt.occurrence_bytes() as u64 != self.buffers.occurrences.length()
            || receipt.magnitude_bytes() as u64 != self.buffers.magnitudes.length()
            || receipt.work_item_bytes() as u64 != self.buffers.work_items.length()
            || receipt.address_offset_bytes() as u64 != self.buffers.address_offsets.length()
        {
            return Err(BytecodeAddressSparseRuntimeError::InvalidState);
        }
        Ok(())
    }

    fn read_output(&self) -> Result<Vec<AkitaField>, BytecodeAddressSparseRuntimeError> {
        if !self.completed.get() {
            return Err(BytecodeAddressSparseRuntimeError::NotExecuted);
        }
        let fields = self.params.stages as usize * self.params.addresses as usize;
        // SAFETY: the shared output buffer contains exactly `fields` Fp128
        // values and its command has completed before this read.
        let output = unsafe {
            slice::from_raw_parts(self.buffers.output.contents().cast::<Fp128>(), fields)
        };
        self.context
            .validate_inputs("bytecode sparse output", output)?;
        Ok(output
            .iter()
            .map(|value| (*value).into_jolt_field())
            .collect())
    }
}

#[derive(Debug, Error)]
pub(crate) enum BytecodeAddressSparseRuntimeError {
    #[error(transparent)]
    Carrier(#[from] super::carrier::CarrierError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("bytecode sparse address has {physical} physical rows in a {padded}-row domain")]
    InvalidRows { physical: usize, padded: usize },
    #[error("bytecode sparse address {table} has {got} stages, expected {expected}")]
    StageCount {
        table: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("bytecode sparse address {table} stage {stage} has {got} fields, expected {expected}")]
    TableLength {
        table: &'static str,
        stage: usize,
        expected: usize,
        got: usize,
    },
    #[error("bytecode sparse address {0} size overflow")]
    SizeOverflow(&'static str),
    #[error(
        "bytecode sparse address pipeline `{pipeline}` has SIMD width {got_width} and thread limit {got_threads}; expected width 32 and at least {expected_threads} threads"
    )]
    UnsupportedPipeline {
        pipeline: &'static str,
        expected_threads: usize,
        got_width: usize,
        got_threads: usize,
    },
    #[error(
        "bytecode sparse address worker needs {requested} threadgroup bytes, device maximum is {maximum}"
    )]
    ThreadgroupMemory { requested: u64, maximum: u64 },
    #[error("bytecode sparse address invocation has inconsistent resources or state")]
    InvalidState,
    #[error("bytecode sparse address output read before execution")]
    NotExecuted,
}

fn validate_table_shape(
    table: &'static str,
    tables: &[Vec<AkitaField>],
    expected_length: usize,
) -> Result<(), BytecodeAddressSparseRuntimeError> {
    if tables.len() != BYTECODE_ADDRESS_PUSHFORWARD_STAGES {
        return Err(BytecodeAddressSparseRuntimeError::StageCount {
            table,
            expected: BYTECODE_ADDRESS_PUSHFORWARD_STAGES,
            got: tables.len(),
        });
    }
    for (stage, values) in tables.iter().enumerate() {
        if values.len() != expected_length {
            return Err(BytecodeAddressSparseRuntimeError::TableLength {
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

fn padding_base_terms(
    physical_rows: usize,
    e_lo: &[Vec<AkitaField>],
    e_hi: &[Vec<AkitaField>],
) -> Result<Vec<Fp128>, BytecodeAddressSparseRuntimeError> {
    let inner_length = e_lo
        .first()
        .map(Vec::len)
        .ok_or(BytecodeAddressSparseRuntimeError::InvalidState)?;
    let outer_length = e_hi
        .first()
        .map(Vec::len)
        .ok_or(BytecodeAddressSparseRuntimeError::InvalidState)?;
    let padded_rows = inner_length.checked_mul(outer_length).ok_or(
        BytecodeAddressSparseRuntimeError::SizeOverflow("padded rows"),
    )?;
    if physical_rows > padded_rows {
        return Err(BytecodeAddressSparseRuntimeError::InvalidRows {
            physical: physical_rows,
            padded: padded_rows,
        });
    }
    let full_outers = physical_rows / inner_length;
    let partial_inner = physical_rows % inner_length;
    let mut padding = Vec::with_capacity(BYTECODE_ADDRESS_BASE_STAGES);
    for stage in 0..BYTECODE_ADDRESS_BASE_STAGES {
        let lo_total = e_lo[stage]
            .iter()
            .copied()
            .fold(AkitaField::zero(), |sum, value| sum + value);
        let first_padding_outer = full_outers + usize::from(partial_inner != 0);
        let later_hi = e_hi[stage][first_padding_outer..]
            .iter()
            .copied()
            .fold(AkitaField::zero(), |sum, value| sum + value);
        let partial = if partial_inner == 0 {
            AkitaField::zero()
        } else {
            let lo_suffix = e_lo[stage][partial_inner..]
                .iter()
                .copied()
                .fold(AkitaField::zero(), |sum, value| sum + value);
            e_hi[stage][full_outers] * lo_suffix
        };
        padding.push(Fp128::from_jolt_field(&(partial + later_hi * lo_total)));
    }
    Ok(padding)
}

fn validate_pipeline(
    pipeline: &'static str,
    limits: PipelineLimits,
    expected_threads: usize,
) -> Result<(), BytecodeAddressSparseRuntimeError> {
    if limits.thread_execution_width != SIMD_WIDTH
        || limits.max_total_threads_per_threadgroup < expected_threads
    {
        return Err(BytecodeAddressSparseRuntimeError::UnsupportedPipeline {
            pipeline,
            expected_threads,
            got_width: limits.thread_execution_width,
            got_threads: limits.max_total_threads_per_threadgroup,
        });
    }
    Ok(())
}

fn field_bytes(fields: usize) -> Result<usize, BytecodeAddressSparseRuntimeError> {
    fields
        .checked_mul(size_of::<Fp128>())
        .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
            "field buffer",
        ))
}

fn shader_count(
    name: &'static str,
    value: usize,
) -> Result<u32, BytecodeAddressSparseRuntimeError> {
    u32::try_from(value).map_err(|_| BytecodeAddressSparseRuntimeError::SizeOverflow(name))
}

fn to_u64(name: &'static str, value: usize) -> Result<u64, BytecodeAddressSparseRuntimeError> {
    u64::try_from(value).map_err(|_| BytecodeAddressSparseRuntimeError::SizeOverflow(name))
}
