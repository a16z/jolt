use std::{
    cell::Cell,
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use thiserror::Error;

use super::{
    carrier::AddressMajorShape,
    worklist::{
        SparseAddressRow, SparseAddressWorklist, SparseAddressWorklistError,
        BYTECODE_ADDRESS_BASE_STAGES, BYTECODE_ADDRESS_PUSHFORWARD_STAGES,
    },
};
use crate::metal::solinas::{
    buffer_from_slice, command_buffer_timestamp, BooleanityRow, BooleanityRows, Fp128, MetalError,
    PipelineLimits, SolinasMetal,
};

const WORKER_PIPELINE: &str = "solinas_bytecode_address_sparse_worker_5_4";
const REDUCE_PIPELINE: &str = "solinas_bytecode_address_sparse_reduce";
const THREADS: usize = 256;
const SIMD_WIDTH: usize = 32;
const SIMDGROUPS: usize = THREADS / SIMD_WIDTH;
const THREADGROUP_BYTES: usize = SIMDGROUPS * BYTECODE_ADDRESS_BASE_STAGES * size_of::<Fp128>();

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[expect(
    clippy::struct_field_names,
    reason = "the storage ledger keeps byte units explicit at every accounting seam"
)]
pub(crate) struct BytecodeAddressSparseStorage {
    pub(crate) occurrence_bytes: usize,
    pub(crate) magnitude_bytes: usize,
    pub(crate) work_item_bytes: usize,
    pub(crate) address_offset_bytes: usize,
    pub(crate) equality_bytes: usize,
    pub(crate) padding_bytes: usize,
    pub(crate) partial_bytes: usize,
    pub(crate) output_bytes: usize,
    pub(crate) owned_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BytecodeAddressSparseObservation {
    pub(crate) output: Vec<AkitaField>,
    pub(crate) gpu_active: Duration,
    pub(crate) resident_wall: Duration,
    pub(crate) source_rows_storage_id: usize,
    pub(crate) source_device_registry_id: u64,
    pub(crate) physical_rows: usize,
    pub(crate) work_items: usize,
    pub(crate) static_buffer_identities: [usize; 8],
}

struct BytecodeAddressSparseBuffers {
    source_rows: BooleanityRows,
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

pub(crate) struct BytecodeAddressSparseInvocation {
    context: SolinasMetal,
    worker_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    worker_limits: PipelineLimits,
    reduce_limits: PipelineLimits,
    buffers: BytecodeAddressSparseBuffers,
    params: BytecodeAddressSparseParams,
    storage: BytecodeAddressSparseStorage,
    threadgroup_memory_bytes: usize,
    source_rows_storage_id: usize,
    source_device_registry_id: u64,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub(crate) fn build_bytecode_address_sparse_worklist(
        &self,
        rows: &BooleanityRows,
        physical_rows: usize,
    ) -> Result<SparseAddressWorklist, BytecodeAddressSparseRuntimeError> {
        self.validate_booleanity_rows(rows)?;
        if rows.len() < 1usize << 15 || !rows.len().is_power_of_two() {
            return Err(BytecodeAddressSparseRuntimeError::InvalidRows {
                physical: physical_rows,
                padded: rows.len(),
            });
        }
        if physical_rows > rows.len() {
            return Err(BytecodeAddressSparseRuntimeError::InvalidRows {
                physical: physical_rows,
                padded: rows.len(),
            });
        }
        let shape = AddressMajorShape::production(rows.len().ilog2())?;
        // SAFETY: every BooleanityRows constructor publishes a fully initialized
        // shared allocation whose checked length is `rows.len()`.
        let source = unsafe {
            slice::from_raw_parts(rows.buffer().contents().cast::<BooleanityRow>(), rows.len())
        };
        Ok(SparseAddressWorklist::try_build_with(
            physical_rows,
            shape,
            |index| {
                let row = source[index];
                let words = row.words();
                SparseAddressRow::with_magnitude(row.mapped_pc(), words[3], words[4] >> 63 != 0)
            },
        )?)
    }

    pub(crate) fn prepare_bytecode_address_sparse_probe(
        &self,
        source_rows: BooleanityRows,
        worklist: &SparseAddressWorklist,
        e_lo: &[Vec<AkitaField>],
        e_hi: &[Vec<AkitaField>],
    ) -> Result<BytecodeAddressSparseInvocation, BytecodeAddressSparseRuntimeError> {
        self.validate_booleanity_rows(&source_rows)?;
        let shape = worklist.shape();
        let padded_rows = shape.rows()?;
        let addresses = shape.addresses()?;
        let inner_length = shape.inner_length()?;
        let outer_length = shape.outer_length()?;
        if source_rows.len() != padded_rows
            || worklist.padded_rows() != padded_rows
            || worklist.physical_rows() > padded_rows
            || worklist.work_items() == 0
            || worklist.address_offsets().len() != addresses + 1
        {
            return Err(BytecodeAddressSparseRuntimeError::InvalidRows {
                physical: worklist.physical_rows(),
                padded: padded_rows,
            });
        }
        validate_table_shape("E_lo", e_lo, inner_length)?;
        validate_table_shape("E_hi", e_hi, outer_length)?;
        let flat_e_lo = flatten_tables(e_lo);
        let flat_e_hi = flatten_tables(e_hi);
        let padding = padding_base_terms(worklist.physical_rows(), e_lo, e_hi)?;
        self.validate_inputs("bytecode sparse E_lo", &flat_e_lo)?;
        self.validate_inputs("bytecode sparse E_hi", &flat_e_hi)?;
        self.validate_inputs("bytecode sparse padding", &padding)?;

        let occurrence_bytes = worklist.ledger().occurrence_bytes();
        let magnitude_bytes = worklist.ledger().magnitude_bytes();
        let work_item_bytes = worklist.ledger().work_item_bytes();
        let address_offset_bytes = worklist.ledger().descriptor_offset_bytes();
        let equality_bytes = field_bytes(flat_e_lo.len().checked_add(flat_e_hi.len()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("equality fields"),
        )?)?;
        let padding_bytes = field_bytes(padding.len())?;
        let partial_fields = BYTECODE_ADDRESS_PUSHFORWARD_STAGES
            .checked_mul(worklist.work_items())
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
        let owned_bytes = [
            occurrence_bytes,
            magnitude_bytes,
            work_item_bytes,
            address_offset_bytes,
            equality_bytes,
            padding_bytes,
            partial_bytes,
            output_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                    "owned bytes",
                ))
        })?;
        self.validate_additional_working_set(to_u64("working set", owned_bytes)?)?;
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
        validate_pipeline(WORKER_PIPELINE, worker_limits)?;
        validate_pipeline(REDUCE_PIPELINE, reduce_limits)?;
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
            physical_rows: shader_count("physical rows", worklist.physical_rows())?,
            addresses: shader_count("addresses", addresses)?,
            inner_length: shader_count("inner length", inner_length)?,
            outer_length: shader_count("outer length", outer_length)?,
            work_items: shader_count("work items", worklist.work_items())?,
            stages: BYTECODE_ADDRESS_PUSHFORWARD_STAGES as u32,
            base_stages: BYTECODE_ADDRESS_BASE_STAGES as u32,
            reserved: 0,
        };
        let source_rows_storage_id = source_rows.allocation_identity();
        let source_device_registry_id = source_rows.device_registry_id();
        Ok(BytecodeAddressSparseInvocation {
            context: self.clone(),
            worker_pipeline,
            reduce_pipeline,
            worker_limits,
            reduce_limits,
            buffers: BytecodeAddressSparseBuffers {
                source_rows,
                occurrences: buffer_from_slice(&self.device, worklist.occurrences()),
                magnitudes: buffer_from_slice(&self.device, worklist.magnitudes()),
                work_items: buffer_from_slice(&self.device, worklist.items()),
                address_offsets: buffer_from_slice(&self.device, worklist.address_offsets()),
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
            storage: BytecodeAddressSparseStorage {
                occurrence_bytes,
                magnitude_bytes,
                work_item_bytes,
                address_offset_bytes,
                equality_bytes,
                padding_bytes,
                partial_bytes,
                output_bytes,
                owned_bytes,
            },
            threadgroup_memory_bytes: THREADGROUP_BYTES,
            source_rows_storage_id,
            source_device_registry_id,
            completed: Cell::new(false),
        })
    }
}

impl BytecodeAddressSparseInvocation {
    pub(crate) fn execute_timed(
        &self,
    ) -> Result<BytecodeAddressSparseObservation, BytecodeAddressSparseRuntimeError> {
        let resident_started = Instant::now();
        autoreleasepool(|| {
            self.validate_source()?;
            self.completed.set(false);
            let command_buffer = self.context.queue.new_command_buffer();
            self.encode(command_buffer);
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let gpu_active = completed_gpu_active(command_buffer)?;
            self.validate_source()?;
            self.completed.set(true);
            Ok(BytecodeAddressSparseObservation {
                output: self.read_output()?,
                gpu_active,
                resident_wall: resident_started.elapsed(),
                source_rows_storage_id: self.source_rows_storage_id,
                source_device_registry_id: self.source_device_registry_id,
                physical_rows: self.params.physical_rows as usize,
                work_items: self.params.work_items as usize,
                static_buffer_identities: self.static_buffer_identities(),
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
        worker.set_threadgroup_memory_length(0, THREADGROUP_BYTES as u64);
        worker.dispatch_thread_groups(
            MTLSize {
                width: u64::from(self.params.work_items),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADS as u64,
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
                width: output_fields.div_ceil(THREADS as u64),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADS as u64,
                height: 1,
                depth: 1,
            },
        );
        reduce.end_encoding();
    }

    fn validate_source(&self) -> Result<(), BytecodeAddressSparseRuntimeError> {
        self.context
            .validate_booleanity_rows(&self.buffers.source_rows)?;
        if self.buffers.source_rows.len()
            != self.params.inner_length as usize * self.params.outer_length as usize
            || self.buffers.source_rows.allocation_identity() != self.source_rows_storage_id
            || self.buffers.source_rows.device_registry_id() != self.source_device_registry_id
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

    pub(crate) const fn storage(&self) -> BytecodeAddressSparseStorage {
        self.storage
    }

    pub(crate) const fn worker_pipeline_limits(&self) -> PipelineLimits {
        self.worker_limits
    }

    pub(crate) const fn reduce_pipeline_limits(&self) -> PipelineLimits {
        self.reduce_limits
    }

    pub(crate) const fn threadgroup_memory_bytes(&self) -> usize {
        self.threadgroup_memory_bytes
    }

    fn static_buffer_identities(&self) -> [usize; 8] {
        [
            self.buffers.occurrences.as_ptr() as usize,
            self.buffers.magnitudes.as_ptr() as usize,
            self.buffers.work_items.as_ptr() as usize,
            self.buffers.address_offsets.as_ptr() as usize,
            self.buffers.e_lo.as_ptr() as usize,
            self.buffers.e_hi.as_ptr() as usize,
            self.buffers.partials.as_ptr() as usize,
            self.buffers.output.as_ptr() as usize,
        ]
    }
}

#[derive(Debug, Error)]
pub(crate) enum BytecodeAddressSparseRuntimeError {
    #[error(transparent)]
    Worklist(#[from] SparseAddressWorklistError),
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
        "bytecode sparse address pipeline `{pipeline}` has SIMD width {got_width} and thread limit {got_threads}; expected width 32 and at least 256 threads"
    )]
    UnsupportedPipeline {
        pipeline: &'static str,
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
) -> Result<(), BytecodeAddressSparseRuntimeError> {
    if limits.thread_execution_width != SIMD_WIDTH
        || limits.max_total_threads_per_threadgroup < THREADS
    {
        return Err(BytecodeAddressSparseRuntimeError::UnsupportedPipeline {
            pipeline,
            got_width: limits.thread_execution_width,
            got_threads: limits.max_total_threads_per_threadgroup,
        });
    }
    Ok(())
}

fn completed_gpu_active(
    command_buffer: &metal::CommandBufferRef,
) -> Result<Duration, BytecodeAddressSparseRuntimeError> {
    let status = command_buffer.status();
    if status != MTLCommandBufferStatus::Completed {
        return Err(MetalError::CommandFailed(status).into());
    }
    let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
    let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
        return Err(MetalError::InvalidGpuTimestamps { start, end }.into());
    }
    Ok(Duration::from_secs_f64(end - start))
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

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast(),
    );
}
