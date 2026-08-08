use std::{
    cell::Cell,
    mem::{size_of, MaybeUninit},
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBuffer,
    ComputePipelineState, MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSRange,
};
use thiserror::Error;

use super::{
    carrier::{
        AddressMajorShape, ConsumerBinding, CountPublication, PackedCell, PackedInnerSign,
        PlaneReceipt, ProducerIdentity, ScatterPublication, TopologyScheduleReceipt,
        ValidatedAddressMajorCarrier, ValidatedProducerCounts, CELL_BYTES, INNER_SIGN_BYTES,
        MAGNITUDE_BYTES, RESIDENT_ROW_BYTES, SHORT_THRESHOLD, SIMD_WIDTH,
    },
    oracle::HostAddressMajorCarrier,
};
use crate::metal::solinas::{
    buffer_from_slice, command_buffer_timestamp, BooleanityRows, Fp128,
    InstructionReadRafStage1Owner, InstructionReadRafStage1Receipt, MetalError, PipelineLimits,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BytecodeAddressMajorSourceRow {
    pub mapped_pc: Option<usize>,
    pub fused_inc_negative: bool,
}

pub(crate) struct BytecodeAddressMajorResidentStorage {
    shape: AddressMajorShape,
    device_registry_id: u64,
    cells: Buffer,
    inner_sign: Buffer,
    magnitude: Buffer,
    topology: Option<TopologyScheduleReceipt>,
}

pub(crate) struct BytecodeAddressMajorOuterWriter<'a> {
    cells: BytecodeAddressMajorCellPlane,
    outer: usize,
    outer_length: usize,
    addresses: usize,
    inner_sign: &'a mut [MaybeUninit<u32>],
    magnitude: &'a mut [MaybeUninit<u64>],
    topology: Option<TopologyScheduleReceipt>,
    complete: bool,
}

pub(crate) struct BytecodeAddressMajorResidentCarrier {
    source_rows: BooleanityRows,
    cells: Buffer,
    inner_sign: Buffer,
    magnitude: Buffer,
    receipt: ValidatedAddressMajorCarrier,
    source_receipt: InstructionReadRafStage1Receipt,
}

#[derive(Clone, Copy)]
struct BytecodeAddressMajorCellPlane {
    pointer: *mut u32,
    len: usize,
}

// SAFETY: each outer writer receives a unique outer index and writes only
// `address * outer_length + outer`, so concurrent writers never alias.
unsafe impl Send for BytecodeAddressMajorCellPlane {}

impl BytecodeAddressMajorSourceRow {
    pub(crate) fn selector(self) -> Result<u16, BytecodeAddressMajorRuntimeError> {
        let address = self.mapped_pc.unwrap_or(0);
        if address >= 1usize << super::carrier::ADDRESS_LOG2 {
            return Err(BytecodeAddressMajorRuntimeError::InvalidSourcePc(address));
        }
        Ok(address as u16 | (u16::from(self.fused_inc_negative) << 13))
    }
}

impl BytecodeAddressMajorOuterWriter<'_> {
    pub(crate) fn publish(
        &mut self,
        selectors: &[u16],
        magnitudes: &[u64],
    ) -> Result<(), BytecodeAddressMajorRuntimeError> {
        if self.complete
            || selectors.len() != self.inner_sign.len()
            || magnitudes.len() != self.magnitude.len()
            || selectors.len() != magnitudes.len()
        {
            return Err(BytecodeAddressMajorRuntimeError::InvalidState);
        }
        let mut counts = vec![0u16; self.addresses];
        for &selector in selectors {
            if selector >> 14 != 0 {
                return Err(BytecodeAddressMajorRuntimeError::InvalidSourceSelector(
                    selector,
                ));
            }
            let address = usize::from(selector & 0x1fff);
            counts[address] = counts[address]
                .checked_add(1)
                .ok_or(BytecodeAddressMajorRuntimeError::InvalidState)?;
        }
        let mut start = 0usize;
        let mut topology = TopologyScheduleReceipt {
            short_occurrences: 0,
            long_occurrences: 0,
            short_runs: 0,
            long_runs: 0,
            padded_short_lanes: 0,
            padded_long_lanes: 0,
            maximum_run: 0,
        };
        for (address, count_or_cursor) in counts.iter_mut().enumerate() {
            let count = usize::from(*count_or_cursor);
            if count != 0 {
                let count = count as u64;
                topology.maximum_run = topology.maximum_run.max(count);
                if count <= SHORT_THRESHOLD as u64 {
                    topology.short_occurrences += count;
                    topology.short_runs += 1;
                    topology.padded_short_lanes += SIMD_WIDTH as u64;
                } else {
                    topology.long_occurrences += count;
                    topology.long_runs += 1;
                    topology.padded_long_lanes +=
                        count.div_ceil(SIMD_WIDTH as u64) * SIMD_WIDTH as u64;
                }
            }
            let cell = PackedCell::new(start, count)?;
            let cell_index = address * self.outer_length + self.outer;
            if cell_index >= self.cells.len {
                return Err(BytecodeAddressMajorRuntimeError::InvalidState);
            }
            // SAFETY: the storage creates one writer per outer block. This writer
            // is the only one that writes `address * outer_length + self.outer`.
            unsafe { self.cells.pointer.add(cell_index).write(cell.word()) };
            *count_or_cursor =
                u16::try_from(start).map_err(|_| BytecodeAddressMajorRuntimeError::InvalidState)?;
            start += count;
        }
        if start != selectors.len() {
            return Err(BytecodeAddressMajorRuntimeError::InvalidState);
        }
        for (inner, (&selector, &magnitude)) in selectors.iter().zip(magnitudes).enumerate() {
            let address = usize::from(selector & 0x1fff);
            let destination = usize::from(counts[address]);
            counts[address] = counts[address]
                .checked_add(1)
                .ok_or(BytecodeAddressMajorRuntimeError::InvalidState)?;
            if destination >= self.inner_sign.len() {
                return Err(BytecodeAddressMajorRuntimeError::InvalidState);
            }
            let packed = PackedInnerSign::new(inner, selector & (1 << 13) != 0)?;
            let _ = self.inner_sign[destination].write(packed.word());
            let _ = self.magnitude[destination].write(magnitude);
        }
        self.topology = Some(topology);
        self.complete = true;
        Ok(())
    }
}

impl BytecodeAddressMajorResidentStorage {
    pub(crate) fn with_outer_writers<R>(
        &mut self,
        fill: impl FnOnce(&mut [BytecodeAddressMajorOuterWriter<'_>]) -> Result<R, MetalError>,
    ) -> Result<R, MetalError> {
        let rows = self
            .shape
            .rows()
            .map_err(|error| MetalError::InvalidInstructionReadRafGrouped(error.to_string()))?;
        let inner_length = self
            .shape
            .inner_length()
            .map_err(|error| MetalError::InvalidInstructionReadRafGrouped(error.to_string()))?;
        let outer_length = self
            .shape
            .outer_length()
            .map_err(|error| MetalError::InvalidInstructionReadRafGrouped(error.to_string()))?;
        let addresses = self
            .shape
            .addresses()
            .map_err(|error| MetalError::InvalidInstructionReadRafGrouped(error.to_string()))?;
        // SAFETY: storage is unpublished and exclusively borrowed. The buffers
        // have the exact lengths checked when they are allocated.
        let inner_sign = unsafe {
            slice::from_raw_parts_mut(self.inner_sign.contents().cast::<MaybeUninit<u32>>(), rows)
        };
        // SAFETY: as above; the magnitude allocation is disjoint.
        let magnitude = unsafe {
            slice::from_raw_parts_mut(self.magnitude.contents().cast::<MaybeUninit<u64>>(), rows)
        };
        let cells = BytecodeAddressMajorCellPlane {
            pointer: self.cells.contents().cast::<u32>(),
            len: self
                .shape
                .cells()
                .map_err(|error| MetalError::InvalidInstructionReadRafGrouped(error.to_string()))?,
        };
        let mut writers = inner_sign
            .chunks_mut(inner_length)
            .zip(magnitude.chunks_mut(inner_length))
            .enumerate()
            .map(
                |(outer, (inner_sign, magnitude))| BytecodeAddressMajorOuterWriter {
                    cells,
                    outer,
                    outer_length,
                    addresses,
                    inner_sign,
                    magnitude,
                    topology: None,
                    complete: false,
                },
            )
            .collect::<Vec<_>>();
        if writers.len() != outer_length {
            return Err(MetalError::InvalidInstructionReadRafGrouped(
                "bytecode carrier outer geometry changed".to_owned(),
            ));
        }
        let output = fill(&mut writers)?;
        if writers.iter().any(|writer| !writer.complete) {
            return Err(MetalError::InvalidInstructionReadRafGrouped(
                "bytecode carrier did not initialize every outer block".to_owned(),
            ));
        }
        let topology = writers.iter().try_fold(
            TopologyScheduleReceipt {
                short_occurrences: 0,
                long_occurrences: 0,
                short_runs: 0,
                long_runs: 0,
                padded_short_lanes: 0,
                padded_long_lanes: 0,
                maximum_run: 0,
            },
            |mut total, writer| {
                let current = writer.topology.ok_or_else(|| {
                    MetalError::InvalidInstructionReadRafGrouped(
                        "bytecode carrier topology was not published".to_owned(),
                    )
                })?;
                total.short_occurrences += current.short_occurrences;
                total.long_occurrences += current.long_occurrences;
                total.short_runs += current.short_runs;
                total.long_runs += current.long_runs;
                total.padded_short_lanes += current.padded_short_lanes;
                total.padded_long_lanes += current.padded_long_lanes;
                total.maximum_run = total.maximum_run.max(current.maximum_run);
                Ok::<_, MetalError>(total)
            },
        )?;
        topology
            .validate(self.shape)
            .map_err(|error| MetalError::InvalidInstructionReadRafGrouped(error.to_string()))?;
        self.topology = Some(topology);
        Ok(output)
    }

    pub(crate) fn seal(
        self,
        source_owner: &InstructionReadRafStage1Owner,
    ) -> Result<BytecodeAddressMajorResidentCarrier, BytecodeAddressMajorRuntimeError> {
        let source_receipt = source_owner.receipt();
        let source_rows = source_owner.booleanity_rows();
        let rows = self.shape.rows()?;
        if source_receipt.completion_serial() == 0
            || source_rows.len() != rows
            || source_rows.device_registry_id() != self.device_registry_id
            || source_rows.allocation_identity() != source_receipt.row_allocation_identity()
            || source_rows.buffer().length() != source_receipt.row_bytes()
        {
            return Err(BytecodeAddressMajorRuntimeError::InvalidState);
        }
        let source = ProducerIdentity::new(
            self.device_registry_id,
            source_rows.allocation_identity(),
            rows * RESIDENT_ROW_BYTES,
            source_receipt.source_generation(),
            rows,
        )?;
        let cell_count = self.shape.cells()?;
        let cells = PlaneReceipt::new(
            self.cells.as_ptr() as usize,
            cell_count,
            cell_count * CELL_BYTES,
        )?;
        let counts = ValidatedProducerCounts::publish(
            self.shape,
            source,
            cells,
            CountPublication {
                initialized_cells: cell_count,
                count_updates: rows,
                counted_rows: rows,
                completed_outer_blocks: self.shape.outer_length()?,
                invalid_rows: 0,
                reserved: [0; 3],
                additional_source_scans: 0,
                member_source_read_bytes: 0,
                host_staging_bytes: 0,
                upload_copy_bytes: 0,
            },
        )?;
        let inner_sign = PlaneReceipt::new(
            self.inner_sign.as_ptr() as usize,
            rows,
            rows * INNER_SIGN_BYTES,
        )?;
        let magnitude = PlaneReceipt::new(
            self.magnitude.as_ptr() as usize,
            rows,
            rows * MAGNITUDE_BYTES,
        )?;
        let topology = self
            .topology
            .ok_or(BytecodeAddressMajorRuntimeError::InvalidState)?;
        let first_push_pc = first_push_pc(&source_rows)?;
        let receipt = ValidatedAddressMajorCarrier::publish(
            counts,
            inner_sign,
            magnitude,
            topology,
            ScatterPublication {
                scattered_rows: rows,
                cursor_updates: rows,
                completed_outer_blocks: self.shape.outer_length()?,
                invalid_rows: 0,
                reserved: [0; 3],
                producer_resident_scans: 1,
                member_resident_scans: 0,
                source_requested_bytes: 16 * rows as u64,
                compact_write_bytes: 12 * rows as u64,
                cell_write_bytes: 4 * cell_count as u64,
                member_source_read_bytes: 0,
                host_staging_bytes: 0,
                upload_copy_bytes: 0,
                first_push_pc,
                producer_incremental_wall_ns: None,
                producer_gpu_active_ns: None,
            },
        )?;
        Ok(BytecodeAddressMajorResidentCarrier {
            source_rows,
            cells: self.cells,
            inner_sign: self.inner_sign,
            magnitude: self.magnitude,
            receipt,
            source_receipt,
        })
    }
}

impl BytecodeAddressMajorResidentCarrier {
    pub(crate) const fn receipt(&self) -> ValidatedAddressMajorCarrier {
        self.receipt
    }

    fn into_parts(
        self,
    ) -> (
        BooleanityRows,
        Buffer,
        Buffer,
        Buffer,
        ValidatedAddressMajorCarrier,
        InstructionReadRafStage1Receipt,
    ) {
        (
            self.source_rows,
            self.cells,
            self.inner_sign,
            self.magnitude,
            self.receipt,
            self.source_receipt,
        )
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for BytecodeAddressMajorResidentCarrier {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("cells"), self.cells.length() as usize);
        visitor.visit_simple(
            allocative::Key::new("inner_sign"),
            self.inner_sign.length() as usize,
        );
        visitor.visit_simple(
            allocative::Key::new("magnitude"),
            self.magnitude.length() as usize,
        );
        visitor.visit_field(allocative::Key::new("source_rows"), &self.source_rows);
        visitor.exit();
    }
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
    resident_source_rows: Option<BooleanityRows>,
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
    resident_receipt: Option<(
        ValidatedAddressMajorCarrier,
        InstructionReadRafStage1Receipt,
    )>,
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
    Prebuilt(Box<BytecodeAddressMajorResidentCarrier>),
}

impl SolinasMetal {
    pub(crate) fn prepare_bytecode_address_major_resident_storage(
        &self,
        rows: usize,
    ) -> Result<BytecodeAddressMajorResidentStorage, BytecodeAddressMajorRuntimeError> {
        if rows < 1usize << super::carrier::INNER_LOG2 || !rows.is_power_of_two() {
            return Err(BytecodeAddressMajorRuntimeError::InvalidResidentRows(rows));
        }
        let shape = AddressMajorShape::production(rows.ilog2())?;
        let cell_bytes = byte_len("cells", shape.cells()?, CELL_BYTES)?;
        let inner_sign_bytes = byte_len("inner/sign", rows, INNER_SIGN_BYTES)?;
        let magnitude_bytes = byte_len("magnitude", rows, MAGNITUDE_BYTES)?;
        let owned_bytes = [cell_bytes, inner_sign_bytes, magnitude_bytes]
            .into_iter()
            .try_fold(0usize, |sum, bytes| {
                checked_add("carrier bytes", sum, bytes)
            })?;
        self.validate_additional_working_set(
            u64::try_from(owned_bytes)
                .map_err(|_| BytecodeAddressMajorRuntimeError::SizeOverflow("carrier"))?,
        )?;
        for bytes in [cell_bytes, inner_sign_bytes, magnitude_bytes] {
            self.validate_buffer_length(
                u64::try_from(bytes)
                    .map_err(|_| BytecodeAddressMajorRuntimeError::SizeOverflow("carrier"))?,
            )?;
        }
        Ok(BytecodeAddressMajorResidentStorage {
            shape,
            device_registry_id: self.device.registry_id(),
            cells: self
                .device
                .new_buffer(cell_bytes as u64, MTLResourceOptions::StorageModeShared),
            inner_sign: self.device.new_buffer(
                inner_sign_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            magnitude: self.device.new_buffer(
                magnitude_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            topology: None,
        })
    }

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

    pub(crate) fn prepare_bytecode_address_major_resident_carrier(
        &self,
        carrier: BytecodeAddressMajorResidentCarrier,
        e_lo: &[Vec<AkitaField>],
        e_hi: &[Vec<AkitaField>],
        config: BytecodeAddressMajorConfig,
    ) -> Result<BytecodeAddressMajorInvocation, BytecodeAddressMajorRuntimeError> {
        self.prepare_bytecode_address_major(
            BytecodeAddressMajorCarrierInput::Prebuilt(Box::new(carrier)),
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
            BytecodeAddressMajorCarrierInput::Prebuilt(carrier) => carrier.receipt.shape(),
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
            BytecodeAddressMajorCarrierInput::Prebuilt(_) => None,
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
        let borrows_prebuilt_carrier =
            matches!(&carrier, BytecodeAddressMajorCarrierInput::Prebuilt(_));
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
            BytecodeAddressMajorCarrierInput::Prebuilt(_) => 0,
        };
        let producer_support_bytes = producer_support.as_ref().map_or(0, |(offsets, active)| {
            (offsets.len() + active.len()) * size_of::<u32>()
        });
        let owned_bytes = [
            if borrows_prebuilt_carrier {
                0
            } else {
                carrier_bytes
            },
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
            BytecodeAddressMajorCarrierInput::Prebuilt(_) => None,
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
            resident_source_rows,
            cells,
            inner_sign,
            magnitude,
            producer_status,
            active_addresses,
            support_offsets,
            resident_receipt,
        ) = match (carrier, producer_support) {
            (BytecodeAddressMajorCarrierInput::Upload(carrier), None) => (
                None,
                None,
                buffer_from_slice(&self.device, carrier.cells()),
                buffer_from_slice(&self.device, carrier.inner_sign()),
                buffer_from_slice(&self.device, carrier.magnitude()),
                None,
                None,
                None,
                None,
            ),
            (
                BytecodeAddressMajorCarrierInput::Resident(rows),
                Some((support_offsets, active_addresses)),
            ) => (
                Some(rows),
                None,
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
                None,
            ),
            (BytecodeAddressMajorCarrierInput::Prebuilt(carrier), None) => {
                let (source_rows, cells, inner_sign, magnitude, receipt, completion) =
                    (*carrier).into_parts();
                (
                    None,
                    Some(source_rows),
                    cells,
                    inner_sign,
                    magnitude,
                    None,
                    None,
                    None,
                    Some((receipt, completion)),
                )
            }
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
                resident_source_rows,
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
                resident_receipt,
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
        if let Some((receipt, source_receipt)) = self.buffers.resident_receipt {
            let source_rows = self
                .buffers
                .resident_source_rows
                .as_ref()
                .ok_or(BytecodeAddressMajorRuntimeError::InvalidState)?;
            self.context.validate_booleanity_rows(source_rows)?;
            let producer = receipt.producer();
            if source_receipt.completion_serial() == 0
                || source_receipt.rows() != self.params.rows as usize
                || source_rows.len() != self.params.rows as usize
                || source_receipt.device_registry_id() != source_rows.device_registry_id()
                || source_receipt.row_allocation_identity() != source_rows.allocation_identity()
                || source_receipt.row_bytes() != source_rows.buffer().length()
                || source_receipt.source_generation() != producer.generation()
                || source_receipt.device_registry_id() != producer.device_registry_id()
                || source_receipt.row_allocation_identity() != producer.source_allocation_identity()
                || source_receipt.row_bytes() as usize != producer.source_allocation_bytes()
                || self.buffers.cells.device().registry_id() != self.context.device.registry_id()
                || self.buffers.inner_sign.device().registry_id()
                    != self.context.device.registry_id()
                || self.buffers.magnitude.device().registry_id()
                    != self.context.device.registry_id()
            {
                return Err(BytecodeAddressMajorRuntimeError::InvalidState);
            }
            receipt.validate_consumer(ConsumerBinding {
                device_registry_id: source_receipt.device_registry_id(),
                source_allocation_identity: source_receipt.row_allocation_identity(),
                source_allocation_bytes: source_receipt.row_bytes() as usize,
                generation: source_receipt.source_generation(),
                cells: PlaneReceipt::new(
                    self.buffers.cells.as_ptr() as usize,
                    receipt.cells().elements(),
                    self.buffers.cells.length() as usize,
                )?,
                inner_sign: PlaneReceipt::new(
                    self.buffers.inner_sign.as_ptr() as usize,
                    receipt.inner_sign().elements(),
                    self.buffers.inner_sign.length() as usize,
                )?,
                magnitude: PlaneReceipt::new(
                    self.buffers.magnitude.as_ptr() as usize,
                    receipt.magnitude().elements(),
                    self.buffers.magnitude.length() as usize,
                )?,
            })?;
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
            .or_else(|| {
                self.buffers
                    .resident_source_rows
                    .as_ref()
                    .map(BooleanityRows::device_registry_id)
            })
            .or_else(|| {
                self.buffers
                    .resident_receipt
                    .map(|(receipt, _)| receipt.producer().device_registry_id())
            })
    }

    fn source_rows_storage_id(&self) -> Option<usize> {
        self.buffers
            .rows
            .as_ref()
            .map(BooleanityRows::allocation_identity)
            .or_else(|| {
                self.buffers
                    .resident_source_rows
                    .as_ref()
                    .map(BooleanityRows::allocation_identity)
            })
            .or_else(|| {
                self.buffers
                    .resident_receipt
                    .map(|(receipt, _)| receipt.producer().source_allocation_identity())
            })
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

fn first_push_pc(rows: &BooleanityRows) -> Result<usize, BytecodeAddressMajorRuntimeError> {
    let address = rows.first_mapped_pc().unwrap_or(0);
    if address >= 1usize << super::carrier::ADDRESS_LOG2 {
        return Err(BytecodeAddressMajorRuntimeError::InvalidSourcePc(address));
    }
    Ok(address)
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
