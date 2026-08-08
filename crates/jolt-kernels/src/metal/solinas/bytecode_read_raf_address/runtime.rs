use std::{
    cell::Cell,
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBuffer,
    ComputePipelineState, MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSRange,
};
use thiserror::Error;

use super::{carrier::AddressMajorShape, oracle::HostAddressMajorCarrier};
use crate::metal::solinas::{
    buffer_from_slice, command_buffer_timestamp, BooleanityRows, Fp128, MetalError, PipelineLimits,
    SolinasMetal,
};

pub const BYTECODE_ADDRESS_MAJOR_STAGES: usize = 9;
pub const BYTECODE_ADDRESS_MAJOR_BASE_STAGES: usize = 5;
pub const BYTECODE_ADDRESS_MAJOR_THREADS: usize = 256;
pub const BYTECODE_ADDRESS_MAJOR_SIMD_WIDTH: usize = 32;
pub const BYTECODE_ADDRESS_MAJOR_SIMDGROUPS: usize =
    BYTECODE_ADDRESS_MAJOR_THREADS / BYTECODE_ADDRESS_MAJOR_SIMD_WIDTH;

const WORKER_PIPELINE: &str = "solinas_bytecode_address_major_worker_5_4";
const REDUCE_PIPELINE: &str = "solinas_bytecode_address_major_reduce_tiles";
const PRODUCER_PIPELINE: &str = "solinas_bytecode_address_major_build_compact_support";
const PRODUCER_THREADS: usize = 256;
const MAX_PRODUCER_ACTIVE_ADDRESSES: usize = 32;
const PRODUCER_SCAN_WORDS: usize = PRODUCER_THREADS / BYTECODE_ADDRESS_MAJOR_SIMD_WIDTH + 1;
const PRODUCER_STATUS_WORDS: usize = 3;

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
    max_active_addresses: u32,
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
            max_active_addresses: 0,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeAddressMajorStorage {
    pub carrier_bytes: usize,
    pub equality_bytes: usize,
    pub partial_bytes: usize,
    pub output_bytes: usize,
    pub producer_status_bytes: usize,
    pub producer_support_bytes: usize,
    pub owned_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BytecodeAddressMajorProducerStatus {
    pub invalid_rows: u32,
    pub completed_outer_blocks: u32,
    pub emitted_rows: u32,
}

const _: [(); PRODUCER_STATUS_WORDS * size_of::<u32>()] =
    [(); size_of::<BytecodeAddressMajorProducerStatus>()];

struct BytecodeAddressMajorBuffers {
    rows: Option<BooleanityRows>,
    cells: Buffer,
    inner_sign: Buffer,
    magnitude: Buffer,
    e_lo: Buffer,
    e_hi: Buffer,
    partials: Buffer,
    output: Buffer,
    producer_status: Option<Buffer>,
    active_addresses: Option<Buffer>,
    support_offsets: Option<Buffer>,
}

pub struct BytecodeAddressMajorInvocation {
    context: SolinasMetal,
    producer_pipeline: Option<ComputePipelineState>,
    worker_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    producer_limits: Option<PipelineLimits>,
    worker_limits: PipelineLimits,
    reduce_limits: PipelineLimits,
    buffers: BytecodeAddressMajorBuffers,
    params: BytecodeAddressMajorParams,
    storage: BytecodeAddressMajorStorage,
    completed: Cell<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BytecodeAddressMajorObservation {
    pub output: Vec<AkitaField>,
    pub producer_status: Option<BytecodeAddressMajorProducerStatus>,
    pub submit_wall: Duration,
    pub overlap_wall: Duration,
    pub join_wall: Duration,
    pub total_wall: Duration,
    pub gpu_active: Duration,
    pub completed_before_join: bool,
    pub source_rows_device_registry_id: Option<u64>,
    pub source_rows_storage_id: Option<usize>,
    pub max_active_addresses: Option<usize>,
    pub producer_threadgroup_bytes: Option<usize>,
    pub static_buffer_identities: [usize; 10],
}

struct BytecodeAddressMajorSubmittedCommand {
    command_buffer: CommandBuffer,
    submitted_at: Instant,
    submit_wall: Duration,
    source_rows_device_registry_id: Option<u64>,
    source_rows_storage_id: Option<usize>,
    static_buffer_identities: [usize; 10],
}

#[must_use = "a submitted bytecode address-major execution must be joined"]
pub struct PendingBytecodeAddressMajorInvocation {
    invocation: Option<BytecodeAddressMajorInvocation>,
    command: Option<BytecodeAddressMajorSubmittedCommand>,
}

impl Drop for PendingBytecodeAddressMajorInvocation {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.command_buffer.wait_until_completed();
        }
    }
}

enum BytecodeAddressMajorCarrierInput<'a> {
    Upload(&'a HostAddressMajorCarrier),
    Resident(BooleanityRows),
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
        self.prepare_bytecode_address_major(
            BytecodeAddressMajorCarrierInput::Upload(carrier),
            e_lo,
            e_hi,
            config,
        )
    }

    /// Builds the compact carrier from the existing stage-5 row allocation and
    /// consumes it in the same command buffer as the address-major worker.
    #[doc(hidden)]
    pub fn prepare_bytecode_address_major_resident_shadow(
        &self,
        rows: BooleanityRows,
        e_lo: &[Vec<AkitaField>],
        e_hi: &[Vec<AkitaField>],
        config: BytecodeAddressMajorConfig,
    ) -> Result<BytecodeAddressMajorInvocation, BytecodeAddressMajorRuntimeError> {
        self.prepare_bytecode_address_major(
            BytecodeAddressMajorCarrierInput::Resident(rows),
            e_lo,
            e_hi,
            config,
        )
    }

    fn prepare_bytecode_address_major(
        &self,
        carrier: BytecodeAddressMajorCarrierInput<'_>,
        e_lo: &[Vec<AkitaField>],
        e_hi: &[Vec<AkitaField>],
        config: BytecodeAddressMajorConfig,
    ) -> Result<BytecodeAddressMajorInvocation, BytecodeAddressMajorRuntimeError> {
        let shape = match &carrier {
            BytecodeAddressMajorCarrierInput::Upload(carrier) => carrier.shape(),
            BytecodeAddressMajorCarrierInput::Resident(rows) => {
                self.validate_booleanity_rows(rows)?;
                if rows.len() < 1usize << super::carrier::INNER_LOG2
                    || !rows.len().is_power_of_two()
                {
                    return Err(BytecodeAddressMajorRuntimeError::InvalidResidentRows(
                        rows.len(),
                    ));
                }
                AddressMajorShape::production(rows.len().ilog2())?
            }
        };
        let mut params = BytecodeAddressMajorParams::new(shape, config)?;
        let rows = shape.rows()?;
        let addresses = shape.addresses()?;
        let inner_length = shape.inner_length()?;
        let outer_length = shape.outer_length()?;
        let producer_support = match &carrier {
            BytecodeAddressMajorCarrierInput::Upload(_) => None,
            BytecodeAddressMajorCarrierInput::Resident(rows) => {
                let (support_offsets, active_addresses, max_active_addresses) =
                    compact_support(rows, addresses, outer_length)?;
                params.max_active_addresses =
                    shader_count("maximum active addresses", max_active_addresses)?;
                Some((support_offsets, active_addresses))
            }
        };
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
        let cell_bytes = byte_len("cells", shape.cells()?, size_of::<u32>())?;
        let inner_sign_bytes = byte_len("inner/sign", rows, size_of::<u32>())?;
        let magnitude_bytes = byte_len("magnitude", rows, size_of::<u64>())?;
        let carrier_bytes = checked_add(
            "carrier bytes",
            checked_add("carrier bytes", cell_bytes, inner_sign_bytes)?,
            magnitude_bytes,
        )?;
        if let BytecodeAddressMajorCarrierInput::Upload(carrier) = &carrier {
            if carrier.cells().len() != shape.cells()?
                || carrier.inner_sign().len() != rows
                || carrier.magnitude().len() != rows
            {
                return Err(BytecodeAddressMajorRuntimeError::InvalidCarrierLength);
            }
        }
        let equality_bytes = checked_add(
            "equality bytes",
            field_bytes(e_lo.len())?,
            field_bytes(e_hi.len())?,
        )?;
        let producer_status_bytes = match &carrier {
            BytecodeAddressMajorCarrierInput::Upload(_) => 0,
            BytecodeAddressMajorCarrierInput::Resident(_) => {
                size_of::<BytecodeAddressMajorProducerStatus>()
            }
        };
        let producer_support_bytes = producer_support.as_ref().map_or(0, |(offsets, active)| {
            (offsets.len() + active.len()) * size_of::<u32>()
        });
        let owned_bytes = [
            carrier_bytes,
            equality_bytes,
            partial_bytes,
            output_bytes,
            producer_status_bytes,
            producer_support_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |sum, bytes| checked_add("owned bytes", sum, bytes))?;
        self.validate_additional_working_set(
            u64::try_from(owned_bytes)
                .map_err(|_| BytecodeAddressMajorRuntimeError::SizeOverflow("working set"))?,
        )?;
        for bytes in [
            cell_bytes,
            inner_sign_bytes,
            magnitude_bytes,
            equality_bytes,
            partial_bytes,
            output_bytes,
            producer_status_bytes,
            producer_support_bytes,
        ] {
            if bytes == 0 {
                continue;
            }
            self.validate_buffer_length(
                u64::try_from(bytes)
                    .map_err(|_| BytecodeAddressMajorRuntimeError::SizeOverflow("buffer"))?,
            )?;
        }

        let producer_pipeline = match &carrier {
            BytecodeAddressMajorCarrierInput::Upload(_) => None,
            BytecodeAddressMajorCarrierInput::Resident(_) => {
                Some(self.compile_named_pipeline(PRODUCER_PIPELINE)?)
            }
        };
        let worker_pipeline = self.compile_named_pipeline(WORKER_PIPELINE)?;
        let reduce_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let producer_limits = producer_pipeline.as_ref().map(Self::limits);
        let worker_limits = Self::limits(&worker_pipeline);
        let reduce_limits = Self::limits(&reduce_pipeline);
        if let Some(limits) = producer_limits {
            validate_pipeline(PRODUCER_PIPELINE, limits, PRODUCER_THREADS)?;
            let requested = producer_threadgroup_bytes_from_params(params);
            let maximum = usize::try_from(self.device.max_threadgroup_memory_length())
                .map_err(|_| BytecodeAddressMajorRuntimeError::SizeOverflow("threadgroup"))?;
            if requested > maximum {
                return Err(BytecodeAddressMajorRuntimeError::ThreadgroupMemory {
                    requested,
                    maximum,
                });
            }
        }
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

        let (
            resident_rows,
            cells,
            inner_sign,
            magnitude,
            producer_status,
            active_addresses,
            support_offsets,
        ) = match (carrier, producer_support) {
            (BytecodeAddressMajorCarrierInput::Upload(carrier), None) => (
                None,
                buffer_from_slice(&self.device, carrier.cells()),
                buffer_from_slice(&self.device, carrier.inner_sign()),
                buffer_from_slice(&self.device, carrier.magnitude()),
                None,
                None,
                None,
            ),
            (
                BytecodeAddressMajorCarrierInput::Resident(rows),
                Some((support_offsets, active_addresses)),
            ) => (
                Some(rows),
                self.device
                    .new_buffer(cell_bytes as u64, MTLResourceOptions::StorageModePrivate),
                self.device.new_buffer(
                    inner_sign_bytes as u64,
                    MTLResourceOptions::StorageModePrivate,
                ),
                self.device.new_buffer(
                    magnitude_bytes as u64,
                    MTLResourceOptions::StorageModePrivate,
                ),
                Some(self.device.new_buffer(
                    producer_status_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                )),
                Some(buffer_from_slice(&self.device, &active_addresses)),
                Some(buffer_from_slice(&self.device, &support_offsets)),
            ),
            _ => return Err(BytecodeAddressMajorRuntimeError::InvalidState),
        };

        Ok(BytecodeAddressMajorInvocation {
            context: self.clone(),
            producer_pipeline,
            worker_pipeline,
            reduce_pipeline,
            producer_limits,
            worker_limits,
            reduce_limits,
            buffers: BytecodeAddressMajorBuffers {
                rows: resident_rows,
                cells,
                inner_sign,
                magnitude,
                e_lo: buffer_from_slice(&self.device, &e_lo),
                e_hi: buffer_from_slice(&self.device, &e_hi),
                partials: self
                    .device
                    .new_buffer(partial_bytes as u64, MTLResourceOptions::StorageModeShared),
                output: self
                    .device
                    .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared),
                producer_status,
                active_addresses,
                support_offsets,
            },
            params,
            storage: BytecodeAddressMajorStorage {
                carrier_bytes,
                equality_bytes,
                partial_bytes,
                output_bytes,
                producer_status_bytes,
                producer_support_bytes,
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
            self.validate_source()?;
            self.completed.set(false);
            self.encode(command_buffer)?;
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let gpu_active = completed_gpu_active(command_buffer)?;
            let _ = self.read_producer_status()?;
            self.completed.set(true);
            Ok((self.read_output()?, gpu_active))
        })
    }

    pub fn submit(
        self,
    ) -> Result<PendingBytecodeAddressMajorInvocation, BytecodeAddressMajorRuntimeError> {
        self.validate_source()?;
        self.completed.set(false);
        let submitted_at = Instant::now();
        let command_buffer = self.context.queue.new_command_buffer().to_owned();
        autoreleasepool(|| {
            self.encode(&command_buffer)?;
            command_buffer.commit();
            Ok::<(), BytecodeAddressMajorRuntimeError>(())
        })?;
        let command = BytecodeAddressMajorSubmittedCommand {
            command_buffer,
            submitted_at,
            submit_wall: submitted_at.elapsed(),
            source_rows_device_registry_id: self.source_rows_device_registry_id(),
            source_rows_storage_id: self.source_rows_storage_id(),
            static_buffer_identities: self.static_buffer_identities(),
        };
        Ok(PendingBytecodeAddressMajorInvocation {
            invocation: Some(self),
            command: Some(command),
        })
    }

    fn encode(
        &self,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<(), BytecodeAddressMajorRuntimeError> {
        match (
            &self.producer_pipeline,
            &self.buffers.rows,
            &self.buffers.producer_status,
            &self.buffers.active_addresses,
            &self.buffers.support_offsets,
        ) {
            (None, None, None, None, None) => {}
            (
                Some(pipeline),
                Some(rows),
                Some(status),
                Some(active_addresses),
                Some(support_offsets),
            ) => {
                let blit = command_buffer.new_blit_command_encoder();
                blit.fill_buffer(status, NSRange::new(0, status.length()), 0);
                blit.fill_buffer(
                    &self.buffers.cells,
                    NSRange::new(0, self.buffers.cells.length()),
                    0,
                );
                blit.end_encoding();

                let producer = command_buffer.new_compute_command_encoder();
                producer.set_compute_pipeline_state(pipeline);
                producer.set_buffer(0, Some(rows.buffer()), 0);
                producer.set_buffer(1, Some(&self.buffers.cells), 0);
                producer.set_buffer(2, Some(&self.buffers.inner_sign), 0);
                producer.set_buffer(3, Some(&self.buffers.magnitude), 0);
                set_inline_bytes(producer, 4, &self.params);
                producer.set_buffer(5, Some(status), 0);
                producer.set_buffer(6, Some(active_addresses), 0);
                producer.set_buffer(7, Some(support_offsets), 0);
                producer.set_threadgroup_memory_length(
                    0,
                    producer_threadgroup_bytes_from_params(self.params) as u64,
                );
                producer.dispatch_thread_groups(
                    MTLSize {
                        width: u64::from(self.params.outer_length),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: PRODUCER_THREADS as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                producer.end_encoding();
            }
            _ => return Err(BytecodeAddressMajorRuntimeError::InvalidState),
        }

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
        reduce.dispatch_thread_groups(
            MTLSize {
                width: self
                    .output_fields()
                    .div_ceil(BYTECODE_ADDRESS_MAJOR_THREADS) as u64,
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
        Ok(())
    }

    fn validate_source(&self) -> Result<(), BytecodeAddressMajorRuntimeError> {
        if let Some(rows) = &self.buffers.rows {
            self.context.validate_booleanity_rows(rows)?;
            if rows.len() != self.params.rows as usize {
                return Err(BytecodeAddressMajorRuntimeError::InvalidResidentRows(
                    rows.len(),
                ));
            }
            if rows.bytecode_outer_support().map(|(_, _, maximum)| maximum)
                != Some(self.params.max_active_addresses as usize)
            {
                return Err(BytecodeAddressMajorRuntimeError::InvalidState);
            }
        }
        Ok(())
    }

    fn read_producer_status(
        &self,
    ) -> Result<Option<BytecodeAddressMajorProducerStatus>, BytecodeAddressMajorRuntimeError> {
        let Some(buffer) = &self.buffers.producer_status else {
            return Ok(None);
        };
        // SAFETY: the shared buffer has exactly one status record and its writer
        // command has completed before this method is called.
        let status = unsafe {
            *buffer
                .contents()
                .cast::<BytecodeAddressMajorProducerStatus>()
        };
        if status.invalid_rows != 0
            || status.completed_outer_blocks != self.params.outer_length
            || status.emitted_rows != self.params.rows
        {
            return Err(BytecodeAddressMajorRuntimeError::ProducerFailed(status));
        }
        Ok(Some(status))
    }

    fn source_rows_device_registry_id(&self) -> Option<u64> {
        self.buffers
            .rows
            .as_ref()
            .map(BooleanityRows::device_registry_id)
    }

    fn source_rows_storage_id(&self) -> Option<usize> {
        self.buffers
            .rows
            .as_ref()
            .map(BooleanityRows::allocation_identity)
    }

    fn static_buffer_identities(&self) -> [usize; 10] {
        [
            self.buffers.cells.as_ptr() as usize,
            self.buffers.inner_sign.as_ptr() as usize,
            self.buffers.magnitude.as_ptr() as usize,
            self.buffers.e_lo.as_ptr() as usize,
            self.buffers.e_hi.as_ptr() as usize,
            self.buffers.partials.as_ptr() as usize,
            self.buffers.output.as_ptr() as usize,
            self.buffers
                .producer_status
                .as_ref()
                .map_or(0, |buffer| buffer.as_ptr() as usize),
            self.buffers
                .active_addresses
                .as_ref()
                .map_or(0, |buffer| buffer.as_ptr() as usize),
            self.buffers
                .support_offsets
                .as_ref()
                .map_or(0, |buffer| buffer.as_ptr() as usize),
        ]
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

    pub const fn producer_pipeline_limits(&self) -> Option<PipelineLimits> {
        self.producer_limits
    }

    pub const fn max_active_addresses(&self) -> Option<usize> {
        if self.params.max_active_addresses == 0 {
            None
        } else {
            Some(self.params.max_active_addresses as usize)
        }
    }

    pub const fn producer_threadgroup_memory_bytes(&self) -> Option<usize> {
        if self.params.max_active_addresses == 0 {
            None
        } else {
            Some(producer_threadgroup_bytes_from_params(self.params))
        }
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

impl PendingBytecodeAddressMajorInvocation {
    pub fn join(
        mut self,
    ) -> Result<
        (
            BytecodeAddressMajorInvocation,
            BytecodeAddressMajorObservation,
        ),
        BytecodeAddressMajorRuntimeError,
    > {
        let invocation = self
            .invocation
            .take()
            .ok_or(BytecodeAddressMajorRuntimeError::InvalidState)?;
        let command = self
            .command
            .take()
            .ok_or(BytecodeAddressMajorRuntimeError::InvalidState)?;
        let observation = invocation.complete_submission(command)?;
        Ok((invocation, observation))
    }
}

impl BytecodeAddressMajorInvocation {
    fn complete_submission(
        &self,
        command: BytecodeAddressMajorSubmittedCommand,
    ) -> Result<BytecodeAddressMajorObservation, BytecodeAddressMajorRuntimeError> {
        let completed_before_join =
            command.command_buffer.status() == MTLCommandBufferStatus::Completed;
        let join_started = Instant::now();
        let overlap_wall = join_started
            .saturating_duration_since(command.submitted_at)
            .saturating_sub(command.submit_wall);
        command.command_buffer.wait_until_completed();
        let gpu_active = completed_gpu_active(&command.command_buffer)?;
        self.validate_source()?;
        if command.source_rows_device_registry_id != self.source_rows_device_registry_id()
            || command.source_rows_storage_id != self.source_rows_storage_id()
            || command.static_buffer_identities != self.static_buffer_identities()
        {
            return Err(BytecodeAddressMajorRuntimeError::InvalidState);
        }
        let producer_status = self.read_producer_status()?;
        self.completed.set(true);
        let output = self.read_output()?;
        let join_wall = join_started.elapsed();
        let total_wall = command.submitted_at.elapsed();
        Ok(BytecodeAddressMajorObservation {
            output,
            producer_status,
            submit_wall: command.submit_wall,
            overlap_wall,
            join_wall,
            total_wall,
            gpu_active,
            completed_before_join,
            source_rows_device_registry_id: command.source_rows_device_registry_id,
            source_rows_storage_id: command.source_rows_storage_id,
            max_active_addresses: self.max_active_addresses(),
            producer_threadgroup_bytes: self.producer_threadgroup_memory_bytes(),
            static_buffer_identities: command.static_buffer_identities,
        })
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
    #[error("bytecode address-major resident source has invalid row count {0}")]
    InvalidResidentRows(usize),
    #[error("bytecode address-major mapped PC {0} exceeds the address domain")]
    InvalidSourcePc(usize),
    #[error("bytecode address-major packed selector {0:#x} is malformed")]
    InvalidSourceSelector(u16),
    #[error("bytecode address-major resident source has no checked PC-address support")]
    MissingBytecodeAddressSupport,
    #[error(
        "bytecode address-major compact producer supports at most {maximum} active addresses, got {got}"
    )]
    TooManyActiveAddresses { got: usize, maximum: usize },
    #[error("bytecode address-major PC-address support is malformed")]
    InvalidBytecodeAddressSupport,
    #[error(
        "bytecode address-major producer needs {requested} threadgroup bytes, device maximum is {maximum}"
    )]
    ThreadgroupMemory { requested: usize, maximum: usize },
    #[error(
        "bytecode address-major pipeline `{pipeline}` has SIMD width {got_width} and thread limit {got_threads}; expected width 32 and at least {required_threads} threads"
    )]
    UnsupportedPipeline {
        pipeline: &'static str,
        got_width: usize,
        got_threads: usize,
        required_threads: usize,
    },
    #[error("bytecode address-major resident producer failed: {0:?}")]
    ProducerFailed(BytecodeAddressMajorProducerStatus),
    #[error("bytecode address-major invocation has inconsistent resources or state")]
    InvalidState,
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

fn compact_support(
    rows: &BooleanityRows,
    addresses: usize,
    outer_length: usize,
) -> Result<(Vec<u32>, Vec<u32>, usize), BytecodeAddressMajorRuntimeError> {
    let (offsets, active, max_active_addresses) = rows
        .bytecode_outer_support()
        .ok_or(BytecodeAddressMajorRuntimeError::MissingBytecodeAddressSupport)?;
    if max_active_addresses > MAX_PRODUCER_ACTIVE_ADDRESSES {
        return Err(BytecodeAddressMajorRuntimeError::TooManyActiveAddresses {
            got: max_active_addresses,
            maximum: MAX_PRODUCER_ACTIVE_ADDRESSES,
        });
    }
    if offsets.len() != outer_length + 1
        || offsets.first() != Some(&0)
        || offsets.last().copied() != u32::try_from(active.len()).ok()
        || active.is_empty()
        || active.iter().any(|address| *address as usize >= addresses)
    {
        return Err(BytecodeAddressMajorRuntimeError::InvalidBytecodeAddressSupport);
    }
    for pair in offsets.windows(2) {
        let Some(segment) = active.get(pair[0] as usize..pair[1] as usize) else {
            return Err(BytecodeAddressMajorRuntimeError::InvalidBytecodeAddressSupport);
        };
        if segment.is_empty() || segment.windows(2).any(|entry| entry[0] >= entry[1]) {
            return Err(BytecodeAddressMajorRuntimeError::InvalidBytecodeAddressSupport);
        }
    }
    Ok((offsets.to_vec(), active.to_vec(), max_active_addresses))
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

const fn producer_threadgroup_bytes_from_params(params: BytecodeAddressMajorParams) -> usize {
    let active = params.max_active_addresses as usize;
    active * PRODUCER_THREADS * size_of::<u16>() + (PRODUCER_SCAN_WORDS + active) * size_of::<u32>()
}

fn completed_gpu_active(
    command_buffer: &metal::CommandBufferRef,
) -> Result<Duration, BytecodeAddressMajorRuntimeError> {
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
