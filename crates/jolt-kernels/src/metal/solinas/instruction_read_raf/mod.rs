use std::mem::{size_of, MaybeUninit};
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use metal::{foreign_types::ForeignType, Buffer, MTLResourceOptions};

use super::{BooleanityRow, BooleanityRows, MetalError, SolinasMetal};

#[cfg(feature = "test-utils")]
mod probe;
mod scatter;

pub(super) const SOURCE: &str = include_str!("shader.metal");
#[cfg(feature = "test-utils")]
pub use probe::{run_instruction_read_raf_stage1_probe, InstructionReadRafStage1ProbeResult};
pub(crate) use scatter::{
    InstructionReadRafCompatibilityScatterConfig, InstructionReadRafDenseGroupedPlanes,
    InstructionReadRafDenseGroupedReceipt, InstructionReadRafFusedBytecodeReceipt,
};

pub(crate) const INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS: usize = 1 << 12;
pub(crate) const INSTRUCTION_READ_RAF_TABLES: usize = 40;
pub(crate) const INSTRUCTION_READ_RAF_SEGMENTS: usize = 82;
const MAX_ROWS: usize = 1 << 28;

static NEXT_SOURCE_GENERATION: AtomicU64 = AtomicU64::new(1);
static NEXT_COMPLETION_SERIAL: AtomicU64 = AtomicU64::new(1);

pub(crate) type InstructionReadRafChunkCounts = [u32; INSTRUCTION_READ_RAF_SEGMENTS];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InstructionReadRafCountOrder {
    TableMajorThenNoneV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InstructionReadRafPublicationKind {
    HostFillV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InstructionReadRafStage1Receipt {
    rows: usize,
    row_bytes: u64,
    claim_bytes: u64,
    row_allocation_identity: usize,
    claim_allocation_identity: usize,
    device_registry_id: u64,
    source_generation: u64,
    completion_serial: u64,
    count_order: InstructionReadRafCountOrder,
    publication_kind: InstructionReadRafPublicationKind,
    count_chunks: usize,
    count_bytes: usize,
    count_allocation_identity: usize,
    resident_device_bytes: u64,
    host_row_write_bytes: u64,
    host_claim_write_bytes: u64,
    host_count_update_bytes: u64,
    complete_overwrite: bool,
    source_windows: usize,
    member_upload_bytes: u64,
    projection_dispatches: usize,
}

impl InstructionReadRafStage1Receipt {
    pub(crate) const fn rows(self) -> usize {
        self.rows
    }

    pub(crate) const fn row_bytes(self) -> u64 {
        self.row_bytes
    }

    pub(crate) const fn claim_bytes(self) -> u64 {
        self.claim_bytes
    }

    pub(crate) const fn row_allocation_identity(self) -> usize {
        self.row_allocation_identity
    }

    pub(crate) const fn claim_allocation_identity(self) -> usize {
        self.claim_allocation_identity
    }

    pub(crate) const fn device_registry_id(self) -> u64 {
        self.device_registry_id
    }

    pub(crate) const fn source_generation(self) -> u64 {
        self.source_generation
    }

    pub(crate) const fn completion_serial(self) -> u64 {
        self.completion_serial
    }

    pub(crate) const fn count_order(self) -> InstructionReadRafCountOrder {
        self.count_order
    }

    pub(crate) const fn count_chunks(self) -> usize {
        self.count_chunks
    }

    pub(crate) const fn count_bytes(self) -> usize {
        self.count_bytes
    }

    pub(crate) const fn count_allocation_identity(self) -> usize {
        self.count_allocation_identity
    }

    pub(crate) const fn resident_device_bytes(self) -> u64 {
        self.resident_device_bytes
    }

    pub(crate) const fn host_row_write_bytes(self) -> u64 {
        self.host_row_write_bytes
    }

    pub(crate) const fn host_claim_write_bytes(self) -> u64 {
        self.host_claim_write_bytes
    }

    pub(crate) const fn host_count_update_bytes(self) -> u64 {
        self.host_count_update_bytes
    }

    pub(crate) const fn complete_overwrite(self) -> bool {
        self.complete_overwrite
    }

    pub(crate) const fn source_windows(self) -> usize {
        self.source_windows
    }

    pub(crate) const fn member_upload_bytes(self) -> u64 {
        self.member_upload_bytes
    }

    pub(crate) const fn projection_dispatches(self) -> usize {
        self.projection_dispatches
    }
}

struct InstructionReadRafStage1Inner {
    rows: BooleanityRows,
    claims: Buffer,
    counts: Box<[InstructionReadRafChunkCounts]>,
    receipt: InstructionReadRafStage1Receipt,
}

#[derive(Clone)]
pub(crate) struct InstructionReadRafStage1Owner(Arc<InstructionReadRafStage1Inner>);

pub(crate) struct InstructionReadRafStage1Lease {
    owner: InstructionReadRafStage1Owner,
}

pub(crate) struct InstructionReadRafStage1Storage {
    rows: usize,
    device_registry_id: u64,
    row_buffer: Buffer,
    claim_buffer: Buffer,
    counts: Box<[InstructionReadRafChunkCounts]>,
}

pub(crate) struct InstructionReadRafStage1ChunkWriter<'a> {
    rows: &'a mut [MaybeUninit<BooleanityRow>],
    claims: &'a mut [MaybeUninit<u8>],
    counts: &'a mut InstructionReadRafChunkCounts,
    written: usize,
}

impl SolinasMetal {
    pub(crate) fn prepare_instruction_read_raf_stage1_storage(
        &self,
        rows: usize,
    ) -> Result<InstructionReadRafStage1Storage, MetalError> {
        validate_rows(rows)?;
        let row_bytes = instruction_read_raf_stage1_row_bytes(rows)?;
        let claim_bytes = instruction_read_raf_stage1_claim_bytes(rows)?;
        self.validate_buffer_length(row_bytes)?;
        self.validate_buffer_length(claim_bytes)?;
        self.validate_additional_working_set(
            row_bytes
                .checked_add(claim_bytes)
                .ok_or(MetalError::InputTooLong(rows))?,
        )?;
        let chunks = rows.div_ceil(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS);
        Ok(InstructionReadRafStage1Storage {
            rows,
            device_registry_id: self.device.registry_id(),
            row_buffer: self
                .device
                .new_buffer(row_bytes, MTLResourceOptions::StorageModeShared),
            claim_buffer: self
                .device
                .new_buffer(claim_bytes, MTLResourceOptions::StorageModeShared),
            counts: vec![[0; INSTRUCTION_READ_RAF_SEGMENTS]; chunks].into_boxed_slice(),
        })
    }
}

impl InstructionReadRafStage1Storage {
    pub(crate) fn with_chunk_writers<R>(
        &mut self,
        fill: impl FnOnce(&mut [InstructionReadRafStage1ChunkWriter<'_>]) -> Result<R, MetalError>,
    ) -> Result<R, MetalError> {
        // SAFETY: storage is unpublished and exclusively borrowed. The row
        // allocation has the exact length validated at allocation.
        let rows = unsafe {
            slice::from_raw_parts_mut(
                self.row_buffer
                    .contents()
                    .cast::<MaybeUninit<BooleanityRow>>(),
                self.rows,
            )
        };
        // SAFETY: the unpublished claim allocation has exactly one byte per
        // row and is disjoint from the row allocation.
        let claims = unsafe {
            slice::from_raw_parts_mut(
                self.claim_buffer.contents().cast::<MaybeUninit<u8>>(),
                self.rows,
            )
        };
        let mut chunks: Vec<_> = rows
            .chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
            .zip(claims.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
            .zip(self.counts.iter_mut())
            .map(
                |((rows, claims), counts)| InstructionReadRafStage1ChunkWriter {
                    rows,
                    claims,
                    counts,
                    written: 0,
                },
            )
            .collect();
        let output = fill(&mut chunks)?;
        if chunks.iter().any(|chunk| chunk.written != chunk.rows.len()) {
            return Err(invalid_source(
                "Stage-1 source fill did not initialize every row exactly once",
            ));
        }
        Ok(output)
    }

    pub(crate) fn seal(self) -> Result<InstructionReadRafStage1Owner, MetalError> {
        for (chunk, counts) in self.counts.iter().enumerate() {
            let expected = ((chunk + 1) * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS).min(self.rows)
                - chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
            let observed = counts
                .iter()
                .try_fold(0usize, |sum, &count| sum.checked_add(count as usize));
            if observed != Some(expected) {
                return Err(invalid_source(
                    "chunk selector counts do not cover every row",
                ));
            }
        }
        let row_bytes = instruction_read_raf_stage1_row_bytes(self.rows)?;
        let claim_bytes = instruction_read_raf_stage1_claim_bytes(self.rows)?;
        let count_bytes = instruction_read_raf_stage1_count_bytes(self.rows)?;
        let row_allocation_identity = self.row_buffer.as_ptr() as usize;
        let claim_allocation_identity = self.claim_buffer.as_ptr() as usize;
        let count_allocation_identity = self.counts.as_ptr() as usize;
        if row_allocation_identity == 0
            || claim_allocation_identity == 0
            || count_allocation_identity == 0
            || row_allocation_identity == claim_allocation_identity
        {
            return Err(invalid_source("source allocations are missing or alias"));
        }
        let source_generation = next_nonzero(&NEXT_SOURCE_GENERATION)?;
        let completion_serial = next_nonzero(&NEXT_COMPLETION_SERIAL)?;
        let receipt = InstructionReadRafStage1Receipt {
            rows: self.rows,
            row_bytes,
            claim_bytes,
            row_allocation_identity,
            claim_allocation_identity,
            device_registry_id: self.device_registry_id,
            source_generation,
            completion_serial,
            count_order: InstructionReadRafCountOrder::TableMajorThenNoneV1,
            publication_kind: InstructionReadRafPublicationKind::HostFillV1,
            count_chunks: self.counts.len(),
            count_bytes,
            count_allocation_identity,
            resident_device_bytes: row_bytes + claim_bytes,
            host_row_write_bytes: row_bytes,
            host_claim_write_bytes: claim_bytes,
            host_count_update_bytes: u64::try_from(self.rows)
                .ok()
                .and_then(|rows| rows.checked_mul(size_of::<u32>() as u64))
                .ok_or(MetalError::InputTooLong(self.rows))?,
            complete_overwrite: true,
            source_windows: self.rows,
            member_upload_bytes: 0,
            projection_dispatches: 0,
        };
        let rows = BooleanityRows::from_initialized_buffer(
            self.row_buffer,
            self.rows,
            self.device_registry_id,
        )?;
        Ok(InstructionReadRafStage1Owner(Arc::new(
            InstructionReadRafStage1Inner {
                rows,
                claims: self.claim_buffer,
                counts: self.counts,
                receipt,
            },
        )))
    }
}

impl InstructionReadRafStage1ChunkWriter<'_> {
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn push(
        &mut self,
        row: BooleanityRow,
        table_plus_one: u8,
        raf: bool,
    ) -> Result<(), MetalError> {
        self.push_with_bytecode_chunk_rank(row, table_plus_one, raf, 0)
    }

    pub(crate) fn push_with_bytecode_chunk_rank(
        &mut self,
        row: BooleanityRow,
        table_plus_one: u8,
        raf: bool,
        rank: u8,
    ) -> Result<(), MetalError> {
        if self.written == self.rows.len() {
            return Err(invalid_source(
                "Stage-1 source chunk received too many rows",
            ));
        }
        let (claim, count_rank) = instruction_read_raf_claim_and_count_rank(table_plus_one, raf)
            .ok_or_else(|| invalid_source("Stage-1 source selector exceeds the table domain"))?;
        let row = row.with_bytecode_chunk_rank_low7(rank);
        let claim = claim | ((rank & 0x80) >> 1);
        let _ = self.rows[self.written].write(row);
        let _ = self.claims[self.written].write(claim);
        self.counts[count_rank] += 1;
        self.written += 1;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) const fn instruction_read_raf_bytecode_chunk_rank(row: BooleanityRow, claim: u8) -> u8 {
    row.bytecode_chunk_rank_low7() | ((claim & 0x40) << 1)
}

impl InstructionReadRafStage1Owner {
    pub(crate) fn receipt(&self) -> InstructionReadRafStage1Receipt {
        self.0.receipt
    }

    pub(crate) fn booleanity_rows(&self) -> BooleanityRows {
        self.0.rows.clone()
    }

    pub(crate) fn lease(
        &self,
        expected_rows: usize,
        expected_device_registry_id: u64,
    ) -> Result<InstructionReadRafStage1Lease, MetalError> {
        validate_owner(self, expected_rows, expected_device_registry_id)?;
        Ok(InstructionReadRafStage1Lease {
            owner: self.clone(),
        })
    }
}

impl InstructionReadRafStage1Lease {
    pub(crate) fn receipt(&self) -> InstructionReadRafStage1Receipt {
        self.owner.0.receipt
    }

    pub(crate) fn row_buffer(&self) -> &Buffer {
        self.owner.0.rows.buffer()
    }

    pub(crate) fn claim_buffer(&self) -> &Buffer {
        &self.owner.0.claims
    }

    pub(crate) fn counts(&self) -> &[InstructionReadRafChunkCounts] {
        &self.owner.0.counts
    }

    pub(crate) fn claim_slice(&self) -> &[u8] {
        // SAFETY: sealing publishes a completely initialized shared buffer;
        // the immutable owner keeps it alive for the returned slice.
        unsafe {
            slice::from_raw_parts(
                self.owner.0.claims.contents().cast::<u8>(),
                self.owner.0.receipt.rows,
            )
        }
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for InstructionReadRafStage1Owner {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(mut shared) = visitor.enter_shared(
            allocative::Key::new("owner"),
            size_of::<*const InstructionReadRafStage1Inner>(),
            Arc::as_ptr(&self.0).cast(),
        ) {
            shared.visit_simple(
                allocative::Key::new("ArcInner"),
                2 * size_of::<usize>() + size_of::<InstructionReadRafStage1Inner>(),
            );
            shared.visit_field(allocative::Key::new("rows"), &self.0.rows);
            shared.visit_simple(
                allocative::Key::new("device_claims"),
                self.0.receipt.claim_bytes as usize,
            );
            shared.visit_simple(
                allocative::Key::new("host_counts"),
                self.0.receipt.count_bytes,
            );
            shared.exit();
        }
        visitor.exit();
    }
}

pub(crate) fn instruction_read_raf_stage1_row_bytes(rows: usize) -> Result<u64, MetalError> {
    byte_length::<BooleanityRow>(rows)
}

pub(crate) fn instruction_read_raf_stage1_claim_bytes(rows: usize) -> Result<u64, MetalError> {
    byte_length::<u8>(rows)
}

pub(crate) fn instruction_read_raf_stage1_device_bytes(rows: usize) -> Result<u64, MetalError> {
    instruction_read_raf_stage1_row_bytes(rows)?
        .checked_add(instruction_read_raf_stage1_claim_bytes(rows)?)
        .ok_or(MetalError::InputTooLong(rows))
}

pub(crate) fn instruction_read_raf_stage1_count_bytes(rows: usize) -> Result<usize, MetalError> {
    rows.div_ceil(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
        .checked_mul(INSTRUCTION_READ_RAF_SEGMENTS)
        .and_then(|elements| elements.checked_mul(size_of::<u32>()))
        .ok_or(MetalError::InputTooLong(rows))
}

pub(crate) const fn instruction_read_raf_claim_and_count_rank(
    table_plus_one: u8,
    raf: bool,
) -> Option<(u8, usize)> {
    if table_plus_one as usize > INSTRUCTION_READ_RAF_TABLES {
        return None;
    }
    let logical = 2 * table_plus_one as usize + raf as usize;
    let physical = if logical < 2 {
        logical + 2 * INSTRUCTION_READ_RAF_TABLES
    } else {
        logical - 2
    };
    Some((table_plus_one | ((raf as u8) << 7), physical))
}

fn validate_owner(
    owner: &InstructionReadRafStage1Owner,
    expected_rows: usize,
    expected_device_registry_id: u64,
) -> Result<(), MetalError> {
    let receipt = owner.receipt();
    if receipt.rows != expected_rows
        || owner.0.rows.len() != expected_rows
        || owner.0.claims.length() != receipt.claim_bytes
        || owner.0.rows.buffer().length() != receipt.row_bytes
        || owner.0.counts.len() != expected_rows.div_ceil(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
        || owner.0.counts.len() != receipt.count_chunks
        || instruction_read_raf_stage1_count_bytes(expected_rows)? != receipt.count_bytes
        || instruction_read_raf_stage1_row_bytes(expected_rows)? != receipt.row_bytes
        || instruction_read_raf_stage1_claim_bytes(expected_rows)? != receipt.claim_bytes
    {
        return Err(invalid_source("source receipt shape changed"));
    }
    if receipt.device_registry_id == 0
        || receipt.device_registry_id != expected_device_registry_id
        || owner.0.rows.device_registry_id() != expected_device_registry_id
        || owner.0.claims.device().registry_id() != expected_device_registry_id
    {
        return Err(invalid_source(
            "source receipt belongs to another Metal device",
        ));
    }
    if receipt.source_generation == 0
        || receipt.completion_serial == 0
        || !receipt.complete_overwrite
        || receipt.publication_kind != InstructionReadRafPublicationKind::HostFillV1
        || receipt.source_windows != expected_rows
        || receipt.member_upload_bytes != 0
        || receipt.projection_dispatches != 0
        || receipt.resident_device_bytes != receipt.row_bytes + receipt.claim_bytes
        || receipt.host_row_write_bytes != receipt.row_bytes
        || receipt.host_claim_write_bytes != receipt.claim_bytes
    {
        return Err(invalid_source(
            "source receipt has no publication generation",
        ));
    }
    if receipt.count_order != InstructionReadRafCountOrder::TableMajorThenNoneV1 {
        return Err(invalid_source("source selector-count order is unsupported"));
    }
    if owner.0.rows.allocation_identity() != receipt.row_allocation_identity
        || owner.0.claims.as_ptr() as usize != receipt.claim_allocation_identity
        || owner.0.counts.as_ptr() as usize != receipt.count_allocation_identity
        || receipt.row_allocation_identity == receipt.claim_allocation_identity
    {
        return Err(invalid_source("source allocation identity changed"));
    }
    Ok(())
}

fn validate_rows(rows: usize) -> Result<(), MetalError> {
    if rows == 0 || !rows.is_power_of_two() || rows > MAX_ROWS {
        return Err(invalid_source(
            "row count must be a power of two in 1..=2^28",
        ));
    }
    Ok(())
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

fn next_nonzero(counter: &AtomicU64) -> Result<u64, MetalError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| invalid_source("source publication counter is exhausted"))
}

fn invalid_source(message: &'static str) -> MetalError {
    MetalError::InvalidInstructionReadRafGrouped(message.to_owned())
}

#[cfg(test)]
mod tests;
