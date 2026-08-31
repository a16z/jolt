use std::mem::{size_of, MaybeUninit};
use std::slice;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBuffer,
    MTLResourceOptions, MTLSize,
};

use super::{
    completed_command_gpu_time, BooleanityRow, BooleanityRows, MetalError, SolinasMetal,
    BOOLEANITY_SOURCE_ROW_BYTES, BOOLEANITY_SOURCE_WORDS,
};

mod scatter;

pub(super) const SOURCE: &str = include_str!("shader.metal");
#[cfg(test)]
pub(crate) use scatter::validate_bytecode_topology_admission;
pub(crate) use scatter::{
    InstructionReadRafCompatibilityScatterConfig, InstructionReadRafDenseGroupedPlanes,
    InstructionReadRafDenseGroupedReceipt, InstructionReadRafFusedBytecodeReceipt,
};

pub(crate) const INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS: usize = 1 << 12;
pub(crate) const INSTRUCTION_READ_RAF_TABLES: usize = 40;
pub(crate) const INSTRUCTION_READ_RAF_SEGMENTS: usize = 82;
const MAX_ROWS: usize = 1 << 28;
const SOURCE_PRIMER_PIPELINE: &str = "solinas_instruction_read_raf_source_primer";
const SOURCE_PRIMER_PAGE_BYTES: usize = 16 * 1024;
const SOURCE_PRIMER_THREADS: usize = 256 * 256;

static NEXT_SOURCE_GENERATION: AtomicU64 = AtomicU64::new(1);
static NEXT_COMPLETION_SERIAL: AtomicU64 = AtomicU64::new(1);

pub(crate) type InstructionReadRafChunkCounts = [u32; INSTRUCTION_READ_RAF_SEGMENTS];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InstructionReadRafCountOrder {
    TableMajorThenNoneV1,
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
    count_chunks: usize,
    count_bytes: usize,
    count_allocation_identity: usize,
}

impl InstructionReadRafStage1Receipt {
    copy_field_getters! { pub(crate), {
        rows: usize,
        row_bytes: u64,
        row_allocation_identity: usize,
        claim_allocation_identity: usize,
        device_registry_id: u64,
        source_generation: u64,
        completion_serial: u64,
        count_order: InstructionReadRafCountOrder,
    } }
}

struct InstructionReadRafStage1Inner {
    rows: BooleanityRows,
    claims: Buffer,
    counts: Box<[InstructionReadRafChunkCounts]>,
    ram_remap_compatible: bool,
    receipt: InstructionReadRafStage1Receipt,
}

#[derive(Clone)]
#[doc(hidden)]
pub struct InstructionReadRafStage1Owner(Arc<InstructionReadRafStage1Inner>);

pub(crate) struct InstructionReadRafStage1Lease {
    owner: InstructionReadRafStage1Owner,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SourcePrimerParams {
    word_counts: [u64; 2],
    page_words: u32,
    total_threads: u32,
}

const _: [(); 24] = [(); size_of::<SourcePrimerParams>()];

#[must_use = "the source primer must be joined before Instruction Read-RAF"]
pub(crate) struct PendingInstructionReadRafSourcePrimer {
    source: InstructionReadRafStage1Lease,
    command: Option<CommandBuffer>,
    checksums: Buffer,
    source_identities: [usize; 2],
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingInstructionReadRafSourcePrimer {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.enter_self_sized::<Self>().exit();
    }
}

impl Drop for PendingInstructionReadRafSourcePrimer {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.wait_until_completed();
        }
    }
}

pub(crate) struct InstructionReadRafStage1Storage {
    rows: usize,
    device_registry_id: u64,
    row_buffer: Buffer,
    claim_buffer: Buffer,
    counts: Box<[InstructionReadRafChunkCounts]>,
    ram_remap_compatible: AtomicBool,
}

pub(crate) struct InstructionReadRafStage1ChunkWriter<'a> {
    lookup_lo: &'a mut [MaybeUninit<u64>],
    lookup_hi: &'a mut [MaybeUninit<u64>],
    fused_inc_magnitude: &'a mut [MaybeUninit<u64>],
    packed_metadata: &'a mut [MaybeUninit<u64>],
    claims: &'a mut [MaybeUninit<u8>],
    counts: &'a mut InstructionReadRafChunkCounts,
    ram_remap_compatible: &'a AtomicBool,
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
            ram_remap_compatible: AtomicBool::new(true),
        })
    }
}

impl InstructionReadRafStage1Storage {
    pub(crate) fn with_chunk_writers<R>(
        &mut self,
        fill: impl FnOnce(&mut [InstructionReadRafStage1ChunkWriter<'_>]) -> Result<R, MetalError>,
    ) -> Result<R, MetalError> {
        // SAFETY: storage is unpublished and exclusively borrowed. The row
        // allocation contains four disjoint u64 columns of exactly self.rows.
        let row_words = unsafe {
            slice::from_raw_parts_mut(
                self.row_buffer.contents().cast::<MaybeUninit<u64>>(),
                BOOLEANITY_SOURCE_WORDS * self.rows,
            )
        };
        let (lookup_lo, row_words) = row_words.split_at_mut(self.rows);
        let (lookup_hi, row_words) = row_words.split_at_mut(self.rows);
        let (fused_inc_magnitude, packed_metadata) = row_words.split_at_mut(self.rows);
        // SAFETY: the unpublished claim allocation has exactly one byte per
        // row and is disjoint from the row allocation.
        let claims = unsafe {
            slice::from_raw_parts_mut(
                self.claim_buffer.contents().cast::<MaybeUninit<u8>>(),
                self.rows,
            )
        };
        let mut chunks: Vec<_> = lookup_lo
            .chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
            .zip(lookup_hi.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
            .zip(fused_inc_magnitude.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
            .zip(packed_metadata.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
            .zip(claims.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
            .zip(self.counts.iter_mut())
            .map(
                |(
                    ((((lookup_lo, lookup_hi), fused_inc_magnitude), packed_metadata), claims),
                    counts,
                )| InstructionReadRafStage1ChunkWriter {
                    lookup_lo,
                    lookup_hi,
                    fused_inc_magnitude,
                    packed_metadata,
                    claims,
                    counts,
                    ram_remap_compatible: &self.ram_remap_compatible,
                    written: 0,
                },
            )
            .collect();
        let output = fill(&mut chunks)?;
        if chunks
            .iter()
            .any(|chunk| chunk.written != chunk.fused_inc_magnitude.len())
        {
            return Err(invalid_source(
                "Stage-1 source fill did not initialize every row exactly once",
            ));
        }
        Ok(output)
    }

    pub(crate) fn seal(self) -> Result<InstructionReadRafStage1Owner, MetalError> {
        validate_chunk_counts(self.rows, &self.counts)?;
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
            count_chunks: self.counts.len(),
            count_bytes,
            count_allocation_identity,
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
                ram_remap_compatible: self.ram_remap_compatible.load(Ordering::Relaxed),
                receipt,
            },
        )))
    }
}

impl InstructionReadRafStage1ChunkWriter<'_> {
    pub fn len(&self) -> usize {
        self.fused_inc_magnitude.len()
    }

    pub(crate) fn record_ram_remap_compatibility(&self, compatible: bool) {
        if !compatible {
            self.ram_remap_compatible.store(false, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    pub(crate) fn push_with_bytecode_chunk_rank(
        &mut self,
        row: BooleanityRow,
        table_plus_one: u8,
        raf: bool,
        rank: u8,
    ) -> Result<(), MetalError> {
        self.push_with_source_metadata(row, table_plus_one, raf, rank, None)
    }

    pub fn push_with_register_write(
        &mut self,
        row: BooleanityRow,
        table_plus_one: u8,
        raf: bool,
        rank: u8,
        rd_write: Option<(u8, u64, u64)>,
    ) -> Result<(), MetalError> {
        self.push_with_source_metadata(row, table_plus_one, raf, rank, rd_write)
    }

    pub(crate) fn fill_repeated_with_register_write(
        &mut self,
        row: BooleanityRow,
        table_plus_one: u8,
        raf: bool,
        rank: u8,
        rd_write: Option<(u8, u64, u64)>,
        count: usize,
    ) -> Result<(), MetalError> {
        if count == 0 {
            return Ok(());
        }
        let end = self
            .written
            .checked_add(count)
            .filter(|&end| end <= self.fused_inc_magnitude.len())
            .ok_or_else(|| invalid_source("Stage-1 source chunk received too many rows"))?;
        let (claim, count_rank) = instruction_read_raf_claim_and_count_rank(table_plus_one, raf)
            .ok_or_else(|| invalid_source("Stage-1 source selector exceeds the table domain"))?;
        let row = row.with_bytecode_chunk_rank_low7(rank);
        let claim = claim | ((rank & 0x80) >> 1);
        let words = row.instruction_source_words(rd_write)?;
        self.lookup_lo[self.written..end].fill(MaybeUninit::new(words[0]));
        self.lookup_hi[self.written..end].fill(MaybeUninit::new(words[1]));
        self.fused_inc_magnitude[self.written..end].fill(MaybeUninit::new(words[2]));
        self.packed_metadata[self.written..end].fill(MaybeUninit::new(words[3]));
        self.claims[self.written..end].fill(MaybeUninit::new(claim));
        let count = u32::try_from(count)
            .map_err(|_| invalid_source("Stage-1 repeated source count exceeds u32"))?;
        self.counts[count_rank] = self.counts[count_rank]
            .checked_add(count)
            .ok_or_else(|| invalid_source("Stage-1 source selector count overflowed"))?;
        self.written = end;
        Ok(())
    }

    fn push_with_source_metadata(
        &mut self,
        row: BooleanityRow,
        table_plus_one: u8,
        raf: bool,
        rank: u8,
        rd_write: Option<(u8, u64, u64)>,
    ) -> Result<(), MetalError> {
        if self.written == self.fused_inc_magnitude.len() {
            return Err(invalid_source(
                "Stage-1 source chunk received too many rows",
            ));
        }
        let (claim, count_rank) = instruction_read_raf_claim_and_count_rank(table_plus_one, raf)
            .ok_or_else(|| invalid_source("Stage-1 source selector exceeds the table domain"))?;
        let row = row.with_bytecode_chunk_rank_low7(rank);
        let claim = claim | ((rank & 0x80) >> 1);
        let words = row.instruction_source_words(rd_write)?;
        let _ = self.lookup_lo[self.written].write(words[0]);
        let _ = self.lookup_hi[self.written].write(words[1]);
        let _ = self.fused_inc_magnitude[self.written].write(words[2]);
        let _ = self.packed_metadata[self.written].write(words[3]);
        let _ = self.claims[self.written].write(claim);
        self.counts[count_rank] += 1;
        self.written += 1;
        Ok(())
    }
}

fn validate_chunk_counts(
    rows: usize,
    counts: &[InstructionReadRafChunkCounts],
) -> Result<(), MetalError> {
    for (chunk, counts) in counts.iter().enumerate() {
        let expected = ((chunk + 1) * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS).min(rows)
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
    Ok(())
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

    pub(crate) fn packed_metadata(&self) -> &[u64] {
        let rows = self.0.receipt.rows;
        // SAFETY: the immutable Stage-1 owner stores four complete SoA u64
        // columns; packed metadata is the fourth column and the owner keeps
        // the shared allocation alive for the returned slice.
        unsafe {
            slice::from_raw_parts(
                self.0.rows.buffer().contents().cast::<u64>().add(3 * rows),
                rows,
            )
        }
    }

    pub(crate) fn ram_remap_compatible(&self) -> bool {
        self.0.ram_remap_compatible
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

impl PendingInstructionReadRafSourcePrimer {
    pub(crate) fn join(mut self) -> Result<(), MetalError> {
        let source_identities = [
            self.source.row_buffer().as_ptr() as usize,
            self.source.claim_buffer().as_ptr() as usize,
        ];
        if source_identities != self.source_identities
            || self.checksums.length() != byte_length::<u32>(SOURCE_PRIMER_THREADS)?
        {
            return Err(invalid_source(
                "source primer resources changed before completion",
            ));
        }
        let command = self
            .command
            .take()
            .ok_or_else(|| invalid_source("source primer command was already joined"))?;
        command.wait_until_completed();
        let _ = completed_command_gpu_time(&command)?;
        Ok(())
    }
}

impl SolinasMetal {
    pub(crate) fn submit_instruction_read_raf_source_primer(
        &self,
        owner: &InstructionReadRafStage1Owner,
    ) -> Result<PendingInstructionReadRafSourcePrimer, MetalError> {
        let receipt = owner.receipt();
        let source = owner.lease(receipt.rows(), self.device_registry_id())?;
        let sources = [source.row_buffer(), source.claim_buffer()];
        let source_identities = sources.map(|buffer| buffer.as_ptr() as usize);
        let params = SourcePrimerParams {
            word_counts: sources.map(|buffer| buffer.length() / size_of::<u32>() as u64),
            page_words: (SOURCE_PRIMER_PAGE_BYTES / size_of::<u32>()) as u32,
            total_threads: SOURCE_PRIMER_THREADS as u32,
        };
        let pipeline = self.compile_named_pipeline(SOURCE_PRIMER_PIPELINE)?;
        let limits = Self::limits(&pipeline);
        if limits.thread_execution_width != 32 || limits.max_total_threads_per_threadgroup < 256 {
            return Err(invalid_source("source primer pipeline limits changed"));
        }
        let checksum_bytes = byte_length::<u32>(SOURCE_PRIMER_THREADS)?;
        self.validate_additional_working_set(checksum_bytes)?;
        let checksums = self
            .device
            .new_buffer(checksum_bytes, MTLResourceOptions::StorageModePrivate);

        let command = self.queue.new_command_buffer().to_owned();
        autoreleasepool(|| {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(sources[0]), 0);
            encoder.set_buffer(1, Some(sources[1]), 0);
            encoder.set_buffer(2, Some(&checksums), 0);
            encoder.set_bytes(
                3,
                size_of::<SourcePrimerParams>() as u64,
                std::ptr::from_ref(&params).cast::<std::ffi::c_void>(),
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command.commit();
        });
        Ok(PendingInstructionReadRafSourcePrimer {
            source,
            command: Some(command),
            checksums,
            source_identities,
        })
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
    rows.checked_mul(BOOLEANITY_SOURCE_ROW_BYTES)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(rows))
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
    if receipt.source_generation == 0 || receipt.completion_serial == 0 {
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
