use std::{
    ffi::c_void,
    mem::{size_of, size_of_val},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use metal::{foreign_types::ForeignType, Buffer, MTLResourceOptions};

use super::{
    carrier::{AddressMajorShape, ADDRESS_LOG2, INNER_LOG2},
    worklist::{BytecodeAddressWorkItem, BYTECODE_ADDRESS_WORK_ITEM_ROWS},
};
use crate::metal::solinas::{
    InstructionReadRafStage1Lease, InstructionReadRafStage1Owner, InstructionReadRafStage1Receipt,
    MetalError, SolinasMetal, INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS,
};

const ADDRESS_COUNT: usize = 1 << ADDRESS_LOG2;
const INNER_ROWS: usize = 1 << INNER_LOG2;
const RANK_RADIX: usize = 1 << 8;
const SENTINEL_ADDRESS: u16 = u16::MAX;
const BYTECODE_ADDRESS_STAGE1_MAX_ROWS: usize = 1 << 28;
pub(crate) const BYTECODE_ADDRESS_DESCRIPTOR_COUNT_SHIFT: u32 = 20;
pub(crate) const BYTECODE_ADDRESS_DESCRIPTOR_PIVOT_START_MASK: u32 =
    (1 << BYTECODE_ADDRESS_DESCRIPTOR_COUNT_SHIFT) - 1;
pub(crate) const BYTECODE_ADDRESS_DESCRIPTOR_MAX_COUNT: usize =
    1 << (u32::BITS - BYTECODE_ADDRESS_DESCRIPTOR_COUNT_SHIFT);
pub(crate) const BYTECODE_ADDRESS_DESCRIPTOR_MAX_PIVOT_START: usize =
    BYTECODE_ADDRESS_DESCRIPTOR_PIVOT_START_MASK as usize;

static NEXT_COMPLETION_SERIAL: AtomicU64 = AtomicU64::new(1);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BytecodeAddressChunkDescriptor {
    pub(crate) address: u16,
    pub(crate) base: u16,
    pub(crate) packed_count_and_pivot_start: u32,
}

const _: [(); 8] = [(); size_of::<BytecodeAddressChunkDescriptor>()];
const _: [(); BYTECODE_ADDRESS_DESCRIPTOR_MAX_COUNT] =
    [(); INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS];

impl BytecodeAddressChunkDescriptor {
    pub(crate) fn new(
        address: u16,
        base: u16,
        pivot_start: usize,
        count: usize,
    ) -> Result<Self, MetalError> {
        if pivot_start > BYTECODE_ADDRESS_DESCRIPTOR_MAX_PIVOT_START {
            return Err(invalid(
                "bytecode Stage-1 descriptor pivot start exceeds 20 bits",
            ));
        }
        if !(1..=BYTECODE_ADDRESS_DESCRIPTOR_MAX_COUNT).contains(&count) {
            return Err(invalid(
                "bytecode Stage-1 descriptor count is outside 1..=4096",
            ));
        }
        let pivot_start = u32::try_from(pivot_start)
            .map_err(|_| invalid("bytecode Stage-1 descriptor pivot start exceeds u32"))?;
        let count_minus_one = u32::try_from(count - 1)
            .map_err(|_| invalid("bytecode Stage-1 descriptor count exceeds u32"))?;
        Ok(Self {
            address,
            base,
            packed_count_and_pivot_start: (count_minus_one
                << BYTECODE_ADDRESS_DESCRIPTOR_COUNT_SHIFT)
                | pivot_start,
        })
    }

    pub(crate) const fn pivot_start(self) -> usize {
        (self.packed_count_and_pivot_start & BYTECODE_ADDRESS_DESCRIPTOR_PIVOT_START_MASK) as usize
    }

    pub(crate) const fn count(self) -> usize {
        ((self.packed_count_and_pivot_start >> BYTECODE_ADDRESS_DESCRIPTOR_COUNT_SHIFT) + 1)
            as usize
    }
}

#[derive(Clone, Copy)]
struct ChunkEntry {
    address: u16,
    count: u16,
    base: u16,
    pivot_start: u32,
}

struct ChunkTopology {
    chunk: usize,
    rows: usize,
    first_address: Option<usize>,
    entries: Vec<ChunkEntry>,
    pivots: Vec<u16>,
}

struct ChunkTopologyBuilder {
    chunk: usize,
    expected_rows: usize,
    rows: usize,
    first_address: Option<usize>,
}

pub(crate) struct BytecodeAddressStage1TopologyScratch {
    counts: Vec<u16>,
    touched: Vec<u16>,
    pivots: Vec<(u16, u16)>,
}

pub(crate) struct BytecodeAddressStage1TopologyChunkWriter<'a> {
    builder: Option<ChunkTopologyBuilder>,
    output: &'a mut Option<ChunkTopology>,
}

struct BytecodeAddressStage1TopologyData {
    shape: AddressMajorShape,
    physical_rows: usize,
    first_push_pc: usize,
    chunks: usize,
    real_descriptors: usize,
    real_pivots: usize,
    max_descriptors_per_chunk: usize,
    max_pivots_per_chunk: usize,
    descriptors: Vec<BytecodeAddressChunkDescriptor>,
    pivots: Vec<u16>,
    chunk_offsets: Vec<u32>,
    work_items: Vec<BytecodeAddressWorkItem>,
    address_offsets: Vec<u32>,
}

pub(crate) struct BytecodeAddressStage1TopologyStorage {
    context: SolinasMetal,
    shape: AddressMajorShape,
    physical_rows: usize,
    chunks: Vec<Option<ChunkTopology>>,
    data: Option<BytecodeAddressStage1TopologyData>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BytecodeAddressStage1TopologyReceipt {
    shape: AddressMajorShape,
    padded_rows: usize,
    physical_rows: usize,
    first_push_pc: usize,
    chunks: usize,
    descriptors: usize,
    descriptor_elements: usize,
    descriptor_bytes: usize,
    descriptor_allocation_identity: usize,
    pivots: usize,
    pivot_elements: usize,
    pivot_bytes: usize,
    pivot_allocation_identity: usize,
    chunk_offset_elements: usize,
    chunk_offset_bytes: usize,
    chunk_offset_allocation_identity: usize,
    work_items: usize,
    work_item_bytes: usize,
    work_item_allocation_identity: usize,
    address_offset_elements: usize,
    address_offset_bytes: usize,
    address_offset_allocation_identity: usize,
    max_descriptors_per_chunk: usize,
    max_pivots_per_chunk: usize,
    source_receipt: InstructionReadRafStage1Receipt,
    device_registry_id: u64,
    source_generation: u64,
    source_completion_serial: u64,
    source_rows_storage_id: usize,
    source_claim_storage_id: usize,
    source_windows: usize,
    completion_serial: u64,
    complete_overwrite: bool,
    covered_rows: usize,
    shared_source_row_scans: usize,
    additional_source_row_scans: usize,
    member_upload_bytes: usize,
}

struct BytecodeAddressStage1TopologyInner {
    descriptors: Buffer,
    pivots: Buffer,
    chunk_offsets: Buffer,
    work_items: Buffer,
    address_offsets: Buffer,
    receipt: BytecodeAddressStage1TopologyReceipt,
}

#[derive(Clone)]
pub(crate) struct BytecodeAddressStage1TopologyOwner(Arc<BytecodeAddressStage1TopologyInner>);

pub(crate) struct BytecodeAddressStage1TopologyLease {
    owner: BytecodeAddressStage1TopologyOwner,
    source: InstructionReadRafStage1Lease,
}

impl BytecodeAddressStage1TopologyReceipt {
    copy_field_getters! { pub(crate), {
        shape: AddressMajorShape,
        padded_rows: usize,
        physical_rows: usize,
        first_push_pc: usize,
        chunks: usize,
        descriptors: usize,
        descriptor_elements: usize,
        descriptor_bytes: usize,
        descriptor_allocation_identity: usize,
        pivots: usize,
        pivot_elements: usize,
        pivot_bytes: usize,
        pivot_allocation_identity: usize,
        chunk_offset_elements: usize,
        chunk_offset_bytes: usize,
        chunk_offset_allocation_identity: usize,
        work_items: usize,
        work_item_bytes: usize,
        work_item_allocation_identity: usize,
        address_offset_elements: usize,
        address_offset_bytes: usize,
        address_offset_allocation_identity: usize,
        max_descriptors_per_chunk: usize,
        max_pivots_per_chunk: usize,
        source_receipt: InstructionReadRafStage1Receipt,
        device_registry_id: u64,
        source_generation: u64,
        source_completion_serial: u64,
        source_rows_storage_id: usize,
        source_claim_storage_id: usize,
        source_windows: usize,
        completion_serial: u64,
        complete_overwrite: bool,
        covered_rows: usize,
        shared_source_row_scans: usize,
        additional_source_row_scans: usize,
        member_upload_bytes: usize,
    } }
}

pub(crate) fn bytecode_address_stage1_topology_max_plane_bytes(
    physical_rows: usize,
) -> Result<[u64; 5], MetalError> {
    if physical_rows == 0 || physical_rows > BYTECODE_ADDRESS_STAGE1_MAX_ROWS {
        return Err(MetalError::InputTooLong(physical_rows));
    }
    let chunks = physical_rows.div_ceil(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS);
    let full_chunks = physical_rows / INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
    let partial_rows = physical_rows % INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
    let pivots = full_chunks
        .checked_mul((INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS - 1) / RANK_RADIX)
        .and_then(|count| count.checked_add(partial_rows.saturating_sub(1) / RANK_RADIX))
        .ok_or(MetalError::InputTooLong(physical_rows))?;
    let full_outers = physical_rows / INNER_ROWS;
    let partial_outer_rows = physical_rows % INNER_ROWS;
    let items_for_outer = |rows: usize| -> Result<usize, MetalError> {
        let nonempty = rows.min(ADDRESS_COUNT);
        nonempty
            .checked_add((rows - nonempty) / BYTECODE_ADDRESS_WORK_ITEM_ROWS)
            .ok_or(MetalError::InputTooLong(physical_rows))
    };
    let partial_work_items = items_for_outer(partial_outer_rows)?;
    let work_items = full_outers
        .checked_mul(items_for_outer(INNER_ROWS)?)
        .and_then(|items| items.checked_add(partial_work_items))
        .ok_or(MetalError::InputTooLong(physical_rows))?;
    let bytes = |elements: usize, element_bytes: usize| {
        elements
            .checked_mul(element_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(physical_rows))
    };
    let descriptor_elements = physical_rows
        .checked_add(chunks)
        .ok_or(MetalError::InputTooLong(physical_rows))?;
    let chunk_offset_elements = chunks
        .checked_mul(2)
        .ok_or(MetalError::InputTooLong(physical_rows))?;
    Ok([
        bytes(
            descriptor_elements,
            size_of::<BytecodeAddressChunkDescriptor>(),
        )?,
        bytes(pivots + 1, size_of::<u16>())?,
        bytes(chunk_offset_elements, size_of::<u32>())?,
        bytes(work_items, size_of::<BytecodeAddressWorkItem>())?,
        bytes(ADDRESS_COUNT + 1, size_of::<u32>())?,
    ])
}

pub(crate) fn bytecode_address_stage1_topology_max_bytes(
    physical_rows: usize,
) -> Result<u64, MetalError> {
    bytecode_address_stage1_topology_max_plane_bytes(physical_rows)?
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(MetalError::InputTooLong(physical_rows))
        })
}

impl SolinasMetal {
    pub(crate) fn prepare_bytecode_address_stage1_topology_storage(
        &self,
        padded_rows: usize,
        physical_rows: usize,
    ) -> Result<BytecodeAddressStage1TopologyStorage, MetalError> {
        if padded_rows < INNER_ROWS
            || !padded_rows.is_power_of_two()
            || physical_rows == 0
            || physical_rows > padded_rows
            || padded_rows > BYTECODE_ADDRESS_STAGE1_MAX_ROWS
        {
            return Err(invalid("bytecode Stage-1 topology row geometry is invalid"));
        }
        let shape = AddressMajorShape::production(padded_rows.ilog2())
            .map_err(|error| invalid(error.to_string()))?;
        let chunks = padded_rows / INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
        Ok(BytecodeAddressStage1TopologyStorage {
            context: self.clone(),
            shape,
            physical_rows,
            chunks: (0..chunks).map(|_| None).collect(),
            data: None,
        })
    }
}

impl BytecodeAddressStage1TopologyStorage {
    pub(crate) fn with_chunk_writers<R>(
        &mut self,
        fill: impl FnOnce(&mut [BytecodeAddressStage1TopologyChunkWriter<'_>]) -> Result<R, MetalError>,
    ) -> Result<R, MetalError> {
        if self.data.is_some() || self.chunks.iter().any(Option::is_some) {
            return Err(invalid("bytecode Stage-1 topology was already filled"));
        }
        let mut writers = self
            .chunks
            .iter_mut()
            .enumerate()
            .map(|(chunk, output)| {
                let base = chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
                let chunk_rows = self
                    .physical_rows
                    .saturating_sub(base)
                    .min(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS);
                BytecodeAddressStage1TopologyChunkWriter {
                    builder: Some(ChunkTopologyBuilder::new(chunk, chunk_rows)),
                    output,
                }
            })
            .collect::<Vec<_>>();
        let result = fill(&mut writers)?;
        if writers.iter().any(|writer| writer.builder.is_some()) {
            return Err(invalid(
                "bytecode Stage-1 topology did not finish every chunk",
            ));
        }
        drop(writers);
        self.data = Some(merge_chunk_topologies(
            self.shape,
            self.physical_rows,
            &mut self.chunks,
        )?);
        Ok(result)
    }
}

impl BytecodeAddressStage1TopologyChunkWriter<'_> {
    pub(crate) fn record(
        &mut self,
        scratch: &mut BytecodeAddressStage1TopologyScratch,
        address: usize,
    ) -> Result<u8, MetalError> {
        self.builder
            .as_mut()
            .ok_or_else(|| invalid("bytecode Stage-1 topology chunk is already finished"))?
            .record(scratch, address)
    }

    pub(crate) fn finish(
        &mut self,
        scratch: &mut BytecodeAddressStage1TopologyScratch,
    ) -> Result<(), MetalError> {
        let builder = self
            .builder
            .take()
            .ok_or_else(|| invalid("bytecode Stage-1 topology chunk was finished twice"))?;
        *self.output = Some(builder.finish(scratch)?);
        Ok(())
    }
}

impl BytecodeAddressStage1TopologyScratch {
    pub(crate) const fn new() -> Self {
        Self {
            counts: Vec::new(),
            touched: Vec::new(),
            pivots: Vec::new(),
        }
    }

    fn reset(&mut self) {
        for &address in &self.touched {
            self.counts[usize::from(address)] = 0;
        }
        self.touched.clear();
        self.pivots.clear();
    }
}

impl ChunkTopologyBuilder {
    fn new(chunk: usize, expected_rows: usize) -> Self {
        Self {
            chunk,
            expected_rows,
            rows: 0,
            first_address: None,
        }
    }

    fn record(
        &mut self,
        scratch: &mut BytecodeAddressStage1TopologyScratch,
        address: usize,
    ) -> Result<u8, MetalError> {
        if self.rows == self.expected_rows || address >= ADDRESS_COUNT {
            return Err(invalid(
                "bytecode Stage-1 topology chunk row is out of range",
            ));
        }
        if scratch.counts.is_empty() {
            scratch.counts.resize(ADDRESS_COUNT, 0);
        }
        let address_u16 = address as u16;
        let count = &mut scratch.counts[address];
        if *count == 0 {
            scratch.touched.push(address_u16);
        }
        let rank = *count;
        if rank != 0 && usize::from(rank).is_multiple_of(RANK_RADIX) {
            scratch.pivots.push((address_u16, self.rows as u16));
        }
        *count = count
            .checked_add(1)
            .ok_or_else(|| invalid("bytecode Stage-1 chunk address count overflowed"))?;
        if self.first_address.is_none() {
            self.first_address = Some(address);
        }
        self.rows += 1;
        Ok(rank as u8)
    }

    fn finish(
        self,
        scratch: &mut BytecodeAddressStage1TopologyScratch,
    ) -> Result<ChunkTopology, MetalError> {
        let result = self.finish_inner(scratch);
        scratch.reset();
        result
    }

    fn finish_inner(
        self,
        scratch: &mut BytecodeAddressStage1TopologyScratch,
    ) -> Result<ChunkTopology, MetalError> {
        if self.rows != self.expected_rows {
            return Err(invalid(
                "bytecode Stage-1 topology chunk does not cover its physical rows",
            ));
        }
        if self.rows == 0 {
            return Ok(ChunkTopology {
                chunk: self.chunk,
                rows: 0,
                first_address: None,
                entries: Vec::new(),
                pivots: Vec::new(),
            });
        }
        scratch.touched.sort_unstable();
        scratch.pivots.sort_unstable();
        let mut pivot_index = 0;
        let mut entries = Vec::with_capacity(scratch.touched.len());
        let mut compact_pivots = Vec::with_capacity(scratch.pivots.len());
        for &address in &scratch.touched {
            let count = scratch.counts[usize::from(address)];
            let pivot_start = u32::try_from(compact_pivots.len())
                .map_err(|_| invalid("bytecode Stage-1 pivot count exceeds u32"))?;
            while scratch
                .pivots
                .get(pivot_index)
                .is_some_and(|pivot| pivot.0 == address)
            {
                compact_pivots.push(scratch.pivots[pivot_index].1);
                pivot_index += 1;
            }
            if compact_pivots.len() - pivot_start as usize != (usize::from(count) - 1) / RANK_RADIX
            {
                return Err(invalid("bytecode Stage-1 chunk pivots are incomplete"));
            }
            entries.push(ChunkEntry {
                address,
                count,
                base: 0,
                pivot_start,
            });
        }
        if pivot_index != scratch.pivots.len() {
            return Err(invalid("bytecode Stage-1 chunk pivot address changed"));
        }
        Ok(ChunkTopology {
            chunk: self.chunk,
            rows: self.rows,
            first_address: self.first_address,
            entries,
            pivots: compact_pivots,
        })
    }
}

fn merge_chunk_topologies(
    shape: AddressMajorShape,
    physical_rows: usize,
    chunks_in: &mut [Option<ChunkTopology>],
) -> Result<BytecodeAddressStage1TopologyData, MetalError> {
    let padded_rows = shape.rows().map_err(|error| invalid(error.to_string()))?;
    if physical_rows == 0
        || physical_rows > padded_rows
        || padded_rows > BYTECODE_ADDRESS_STAGE1_MAX_ROWS
    {
        return Err(invalid(
            "bytecode Stage-1 topology exceeds its packed descriptor capacity",
        ));
    }
    let mut descriptors = Vec::new();
    let mut pivots = Vec::new();
    let mut chunk_offsets = Vec::new();
    let mut work_items = Vec::new();
    let mut chunks = 0usize;
    let mut real_descriptors = 0usize;
    let mut max_descriptors_per_chunk = 0usize;
    let mut max_pivots_per_chunk = 0usize;
    let mut first_push_pc = None;
    let physical_chunks = physical_rows.div_ceil(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS);
    let physical_outers = physical_rows.div_ceil(INNER_ROWS);
    let chunks_per_outer = INNER_ROWS / INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
    let mut outer_counts = vec![0u16; ADDRESS_COUNT];
    let mut outer_cursors = vec![0u16; ADDRESS_COUNT];
    let mut outer_touched = Vec::new();

    for outer in 0..physical_outers {
        let chunk_begin = outer * chunks_per_outer;
        let chunk_end = (chunk_begin + chunks_per_outer).min(physical_chunks);
        let outer_rows = physical_rows
            .saturating_sub(outer * INNER_ROWS)
            .min(INNER_ROWS);
        let mut outer_chunks = Vec::with_capacity(chunk_end - chunk_begin);
        for (chunk_offset, slot) in chunks_in[chunk_begin..chunk_end].iter_mut().enumerate() {
            let expected_chunk = chunk_begin + chunk_offset;
            let chunk = slot
                .take()
                .ok_or_else(|| invalid("bytecode Stage-1 topology chunk is unpublished"))?;
            if chunk.chunk != expected_chunk || chunk.rows == 0 {
                return Err(invalid("bytecode Stage-1 topology chunk order changed"));
            }
            if expected_chunk == 0 {
                first_push_pc = chunk.first_address;
            }
            for entry in &chunk.entries {
                let address = usize::from(entry.address);
                if outer_counts[address] == 0 {
                    outer_touched.push(entry.address);
                }
                outer_counts[address] = outer_counts[address]
                    .checked_add(entry.count)
                    .ok_or_else(|| invalid("bytecode Stage-1 outer count overflowed"))?;
            }
            outer_chunks.push(chunk);
        }
        outer_touched.sort_unstable();
        let mut cursor = 0usize;
        for &address_u16 in &outer_touched {
            let address = usize::from(address_u16);
            outer_cursors[address] = u16::try_from(cursor)
                .map_err(|_| invalid("bytecode Stage-1 outer cursor exceeds u16"))?;
            let mut item_start = cursor;
            let mut remaining = usize::from(outer_counts[address]);
            while remaining != 0 {
                let count = remaining.min(BYTECODE_ADDRESS_WORK_ITEM_ROWS);
                work_items.push(
                    BytecodeAddressWorkItem::new(address, outer, item_start, count, outer_rows)
                        .map_err(|error| invalid(error.to_string()))?,
                );
                item_start += count;
                remaining -= count;
            }
            cursor += usize::from(outer_counts[address]);
        }
        if cursor != outer_rows {
            return Err(invalid(
                "bytecode Stage-1 outer counts do not cover its rows",
            ));
        }

        for mut chunk in outer_chunks {
            for entry in &mut chunk.entries {
                let address = usize::from(entry.address);
                entry.base = outer_cursors[address];
                outer_cursors[address] = outer_cursors[address]
                    .checked_add(entry.count)
                    .ok_or_else(|| invalid("bytecode Stage-1 descriptor base overflowed"))?;
            }
            let begin = u32::try_from(descriptors.len())
                .map_err(|_| invalid("bytecode Stage-1 descriptor count exceeds u32"))?;
            max_descriptors_per_chunk = max_descriptors_per_chunk.max(chunk.entries.len());
            max_pivots_per_chunk = max_pivots_per_chunk.max(chunk.pivots.len());
            real_descriptors += chunk.entries.len();
            let pivot_base = pivots.len();
            let mut described_rows = 0usize;
            for (index, entry) in chunk.entries.iter().copied().enumerate() {
                let next_base = chunk
                    .entries
                    .get(index + 1)
                    .map_or(outer_rows, |next| usize::from(next.base));
                let count = usize::from(entry.count);
                let pivot_start = pivot_base
                    .checked_add(entry.pivot_start as usize)
                    .ok_or_else(|| invalid("bytecode Stage-1 pivot count overflowed"))?;
                let descriptor = BytecodeAddressChunkDescriptor::new(
                    entry.address,
                    entry.base,
                    pivot_start,
                    count,
                )?;
                let cell_end = usize::from(descriptor.base)
                    .checked_add(descriptor.count())
                    .ok_or_else(|| invalid("bytecode Stage-1 descriptor end overflowed"))?;
                if cell_end > next_base || cell_end > outer_rows {
                    return Err(invalid("bytecode Stage-1 descriptor bounds overlap"));
                }
                described_rows = described_rows
                    .checked_add(descriptor.count())
                    .ok_or_else(|| invalid("bytecode Stage-1 descriptor row count overflowed"))?;
                descriptors.push(descriptor);
            }
            if described_rows != chunk.rows {
                return Err(invalid(
                    "bytecode Stage-1 descriptors do not cover their physical chunk",
                ));
            }
            pivots.extend(chunk.pivots);
            let end = u32::try_from(descriptors.len())
                .map_err(|_| invalid("bytecode Stage-1 descriptor count exceeds u32"))?;
            descriptors.push(BytecodeAddressChunkDescriptor::new(
                SENTINEL_ADDRESS,
                u16::try_from(outer_rows)
                    .map_err(|_| invalid("bytecode Stage-1 outer rows exceed u16"))?,
                pivots.len(),
                1,
            )?);
            chunk_offsets.extend([begin, end]);
            chunks += 1;
        }
        for &address_u16 in &outer_touched {
            let address = usize::from(address_u16);
            outer_counts[address] = 0;
            outer_cursors[address] = 0;
        }
        outer_touched.clear();
    }
    for (chunk, output) in chunks_in.iter_mut().enumerate().skip(physical_chunks) {
        let output = output
            .take()
            .ok_or_else(|| invalid("bytecode Stage-1 padding chunk is unpublished"))?;
        if output.chunk != chunk || output.rows != 0 || !output.entries.is_empty() {
            return Err(invalid("bytecode Stage-1 padding chunk is nonempty"));
        }
    }
    if first_push_pc.is_none() {
        return Err(invalid(
            "bytecode Stage-1 topology does not cover every physical row",
        ));
    }
    let real_pivots = pivots.len();
    validate_packed_descriptor_stream(physical_rows, &descriptors, real_pivots, &chunk_offsets)?;
    pivots.push(u16::MAX);

    work_items.sort_by_key(|item| (item.address, item.outer, item.start));
    let mut address_offsets = vec![0u32; ADDRESS_COUNT + 1];
    let mut item = 0usize;
    for (address, offset) in address_offsets[..ADDRESS_COUNT].iter_mut().enumerate() {
        *offset = u32::try_from(item)
            .map_err(|_| invalid("bytecode Stage-1 work-item count exceeds u32"))?;
        while work_items
            .get(item)
            .is_some_and(|candidate| usize::from(candidate.address) == address)
        {
            item += 1;
        }
    }
    address_offsets[ADDRESS_COUNT] = u32::try_from(work_items.len())
        .map_err(|_| invalid("bytecode Stage-1 work-item count exceeds u32"))?;
    if item != work_items.len()
        || chunks != physical_rows.div_ceil(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
        || chunk_offsets.len() != 2 * chunks
    {
        return Err(invalid("bytecode Stage-1 topology offsets are invalid"));
    }

    Ok(BytecodeAddressStage1TopologyData {
        shape,
        physical_rows,
        first_push_pc: first_push_pc.unwrap_or_default(),
        chunks,
        real_descriptors,
        real_pivots,
        max_descriptors_per_chunk,
        max_pivots_per_chunk,
        descriptors,
        pivots,
        chunk_offsets,
        work_items,
        address_offsets,
    })
}

fn validate_packed_descriptor_stream(
    physical_rows: usize,
    descriptors: &[BytecodeAddressChunkDescriptor],
    real_pivots: usize,
    chunk_offsets: &[u32],
) -> Result<(), MetalError> {
    let chunks = physical_rows.div_ceil(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS);
    if chunk_offsets.len() != 2 * chunks {
        return Err(invalid(
            "bytecode Stage-1 packed descriptor offsets are incomplete",
        ));
    }
    let mut next_descriptor = 0usize;
    let mut next_pivot = 0usize;
    for chunk in 0..chunks {
        let begin = chunk_offsets[2 * chunk] as usize;
        let end = chunk_offsets[2 * chunk + 1] as usize;
        if begin != next_descriptor || begin >= end || end >= descriptors.len() {
            return Err(invalid(
                "bytecode Stage-1 packed descriptor range is invalid",
            ));
        }
        let outer_begin =
            (chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS / INNER_ROWS) * INNER_ROWS;
        let outer_rows = physical_rows.saturating_sub(outer_begin).min(INNER_ROWS);
        let chunk_rows = physical_rows
            .saturating_sub(chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
            .min(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS);
        let sentinel = descriptors[end];
        if sentinel.address != SENTINEL_ADDRESS
            || usize::from(sentinel.base) != outer_rows
            || sentinel.count() != 1
        {
            return Err(invalid(
                "bytecode Stage-1 packed descriptor sentinel is invalid",
            ));
        }
        let mut described_rows = 0usize;
        let mut previous_address = None;
        for index in begin..end {
            let descriptor = descriptors[index];
            let next = descriptors[index + 1];
            let address = usize::from(descriptor.address);
            let pivot_start = descriptor.pivot_start();
            let pivot_end = next.pivot_start();
            let count = descriptor.count();
            let cell_end = usize::from(descriptor.base)
                .checked_add(count)
                .ok_or_else(|| invalid("bytecode Stage-1 packed descriptor end overflowed"))?;
            if address >= ADDRESS_COUNT
                || previous_address.is_some_and(|previous| previous >= address)
                || pivot_start != next_pivot
                || pivot_end < pivot_start
                || pivot_end > real_pivots
                || pivot_end - pivot_start != (count - 1) / RANK_RADIX
                || cell_end > usize::from(next.base)
                || cell_end > outer_rows
            {
                return Err(invalid(
                    "bytecode Stage-1 packed descriptor contents are invalid",
                ));
            }
            described_rows = described_rows
                .checked_add(count)
                .ok_or_else(|| invalid("bytecode Stage-1 packed row count overflowed"))?;
            previous_address = Some(address);
            next_pivot = pivot_end;
        }
        if described_rows != chunk_rows {
            return Err(invalid(
                "bytecode Stage-1 packed descriptors do not cover their chunk",
            ));
        }
        next_descriptor = end + 1;
    }
    if next_descriptor != descriptors.len() || next_pivot != real_pivots {
        return Err(invalid(
            "bytecode Stage-1 packed descriptor stream has trailing data",
        ));
    }
    Ok(())
}

impl BytecodeAddressStage1TopologyStorage {
    pub(crate) fn seal(
        mut self,
        source_owner: &InstructionReadRafStage1Owner,
    ) -> Result<BytecodeAddressStage1TopologyOwner, MetalError> {
        let data = self
            .data
            .take()
            .ok_or_else(|| invalid("bytecode Stage-1 topology was not filled"))?;
        let padded_rows = data
            .shape
            .rows()
            .map_err(|error| invalid(error.to_string()))?;
        if data.shape != self.shape || data.physical_rows != self.physical_rows {
            return Err(invalid(
                "bytecode Stage-1 topology shape changed before seal",
            ));
        }
        let source_receipt = source_owner.receipt();
        let source = source_owner.lease(padded_rows, self.context.device.registry_id())?;
        drop(source);

        let descriptor_bytes = byte_size(&data.descriptors)?;
        let pivot_bytes = byte_size(&data.pivots)?;
        let chunk_offset_bytes = byte_size(&data.chunk_offsets)?;
        let work_item_bytes = byte_size(&data.work_items)?;
        let address_offset_bytes = byte_size(&data.address_offsets)?;
        let persistent_bytes = [
            descriptor_bytes,
            pivot_bytes,
            chunk_offset_bytes,
            work_item_bytes,
            address_offset_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .ok_or(MetalError::InputTooLong(data.physical_rows))?;
        self.context
            .validate_additional_working_set(persistent_bytes as u64)?;

        let descriptors = shared_buffer(&self.context, &data.descriptors)?;
        let pivots = shared_buffer(&self.context, &data.pivots)?;
        let chunk_offsets = shared_buffer(&self.context, &data.chunk_offsets)?;
        let work_items = shared_buffer(&self.context, &data.work_items)?;
        let address_offsets = shared_buffer(&self.context, &data.address_offsets)?;
        let ids = [
            descriptors.as_ptr() as usize,
            pivots.as_ptr() as usize,
            chunk_offsets.as_ptr() as usize,
            work_items.as_ptr() as usize,
            address_offsets.as_ptr() as usize,
        ];
        if ids.contains(&0)
            || ids
                .iter()
                .enumerate()
                .any(|(index, id)| ids[..index].contains(id))
            || ids.contains(&source_receipt.row_allocation_identity())
            || ids.contains(&source_receipt.claim_allocation_identity())
        {
            return Err(invalid("bytecode Stage-1 topology allocations alias"));
        }
        let receipt = BytecodeAddressStage1TopologyReceipt {
            shape: data.shape,
            padded_rows,
            physical_rows: data.physical_rows,
            first_push_pc: data.first_push_pc,
            chunks: data.chunks,
            descriptors: data.real_descriptors,
            descriptor_elements: data.descriptors.len(),
            descriptor_bytes,
            descriptor_allocation_identity: ids[0],
            pivots: data.real_pivots,
            pivot_elements: data.pivots.len(),
            pivot_bytes,
            pivot_allocation_identity: ids[1],
            chunk_offset_elements: data.chunk_offsets.len(),
            chunk_offset_bytes,
            chunk_offset_allocation_identity: ids[2],
            work_items: data.work_items.len(),
            work_item_bytes,
            work_item_allocation_identity: ids[3],
            address_offset_elements: data.address_offsets.len(),
            address_offset_bytes,
            address_offset_allocation_identity: ids[4],
            max_descriptors_per_chunk: data.max_descriptors_per_chunk,
            max_pivots_per_chunk: data.max_pivots_per_chunk,
            source_receipt,
            device_registry_id: source_receipt.device_registry_id(),
            source_generation: source_receipt.source_generation(),
            source_completion_serial: source_receipt.completion_serial(),
            source_rows_storage_id: source_receipt.row_allocation_identity(),
            source_claim_storage_id: source_receipt.claim_allocation_identity(),
            source_windows: source_receipt.rows(),
            completion_serial: next_nonzero()?,
            complete_overwrite: true,
            covered_rows: data.physical_rows,
            shared_source_row_scans: 1,
            additional_source_row_scans: 0,
            member_upload_bytes: 0,
        };
        Ok(BytecodeAddressStage1TopologyOwner(Arc::new(
            BytecodeAddressStage1TopologyInner {
                descriptors,
                pivots,
                chunk_offsets,
                work_items,
                address_offsets,
                receipt,
            },
        )))
    }
}

impl BytecodeAddressStage1TopologyOwner {
    pub(crate) fn receipt(&self) -> BytecodeAddressStage1TopologyReceipt {
        self.0.receipt
    }

    pub(crate) fn lease(
        &self,
        source: InstructionReadRafStage1Lease,
    ) -> Result<BytecodeAddressStage1TopologyLease, MetalError> {
        validate_owner(self, &source)?;
        Ok(BytecodeAddressStage1TopologyLease {
            owner: self.clone(),
            source,
        })
    }
}

impl BytecodeAddressStage1TopologyLease {
    pub(crate) fn receipt(&self) -> BytecodeAddressStage1TopologyReceipt {
        self.owner.0.receipt
    }

    ref_field_getters! { pub(crate), { source: InstructionReadRafStage1Lease }}

    pub(crate) fn descriptors_buffer(&self) -> &Buffer {
        &self.owner.0.descriptors
    }

    pub(crate) fn pivots_buffer(&self) -> &Buffer {
        &self.owner.0.pivots
    }

    pub(crate) fn chunk_offsets_buffer(&self) -> &Buffer {
        &self.owner.0.chunk_offsets
    }

    pub(crate) fn work_items_buffer(&self) -> &Buffer {
        &self.owner.0.work_items
    }

    pub(crate) fn address_offsets_buffer(&self) -> &Buffer {
        &self.owner.0.address_offsets
    }
}

fn validate_owner(
    owner: &BytecodeAddressStage1TopologyOwner,
    source: &InstructionReadRafStage1Lease,
) -> Result<(), MetalError> {
    let receipt = owner.receipt();
    let source_receipt = source.receipt();
    let buffers = [
        (&owner.0.descriptors, receipt.descriptor_bytes),
        (&owner.0.pivots, receipt.pivot_bytes),
        (&owner.0.chunk_offsets, receipt.chunk_offset_bytes),
        (&owner.0.work_items, receipt.work_item_bytes),
        (&owner.0.address_offsets, receipt.address_offset_bytes),
    ];
    if source_receipt != receipt.source_receipt
        || receipt.source_generation != source_receipt.source_generation()
        || receipt.source_completion_serial != source_receipt.completion_serial()
        || receipt.source_rows_storage_id != source_receipt.row_allocation_identity()
        || receipt.source_claim_storage_id != source_receipt.claim_allocation_identity()
        || receipt.source_windows != source_receipt.rows()
        || receipt.padded_rows != source_receipt.rows()
        || receipt.device_registry_id != source_receipt.device_registry_id()
    {
        return Err(invalid(
            "bytecode Stage-1 topology source provenance changed",
        ));
    }
    if receipt.physical_rows == 0
        || receipt.physical_rows > receipt.padded_rows
        || receipt.padded_rows > BYTECODE_ADDRESS_STAGE1_MAX_ROWS
        || receipt.pivots > BYTECODE_ADDRESS_DESCRIPTOR_MAX_PIVOT_START
        || receipt.covered_rows != receipt.physical_rows
        || receipt.chunks
            != receipt
                .physical_rows
                .div_ceil(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
        || receipt.descriptor_elements != receipt.descriptors + receipt.chunks
        || receipt.pivot_elements != receipt.pivots + 1
        || receipt.chunk_offset_elements != 2 * receipt.chunks
        || receipt.address_offset_elements != ADDRESS_COUNT + 1
        || receipt.work_items == 0
        || receipt.max_descriptors_per_chunk == 0
        || !receipt.complete_overwrite
        || receipt.shared_source_row_scans != 1
        || receipt.additional_source_row_scans != 0
        || receipt.member_upload_bytes != 0
        || receipt.completion_serial == 0
    {
        return Err(invalid("bytecode Stage-1 topology receipt is incomplete"));
    }
    let expected_bytes = [
        receipt.descriptor_elements * size_of::<BytecodeAddressChunkDescriptor>(),
        receipt.pivot_elements * size_of::<u16>(),
        receipt.chunk_offset_elements * size_of::<u32>(),
        receipt.work_items * size_of::<BytecodeAddressWorkItem>(),
        receipt.address_offset_elements * size_of::<u32>(),
    ];
    let expected_ids = [
        receipt.descriptor_allocation_identity,
        receipt.pivot_allocation_identity,
        receipt.chunk_offset_allocation_identity,
        receipt.work_item_allocation_identity,
        receipt.address_offset_allocation_identity,
    ];
    for (index, ((buffer, bytes), expected)) in buffers.into_iter().zip(expected_bytes).enumerate()
    {
        if bytes != expected
            || buffer.length() != expected as u64
            || buffer.as_ptr() as usize != expected_ids[index]
            || buffer.device().registry_id() != receipt.device_registry_id
        {
            return Err(invalid("bytecode Stage-1 topology buffer receipt changed"));
        }
    }
    Ok(())
}

fn shared_buffer<T>(context: &SolinasMetal, values: &[T]) -> Result<Buffer, MetalError> {
    if values.is_empty() {
        return Err(invalid("bytecode Stage-1 topology buffer is empty"));
    }
    let bytes = byte_size(values)?;
    context.validate_buffer_length(bytes as u64)?;
    Ok(context.device.new_buffer_with_data(
        values.as_ptr().cast::<c_void>(),
        bytes as u64,
        MTLResourceOptions::StorageModeShared,
    ))
}

fn byte_size<T>(values: &[T]) -> Result<usize, MetalError> {
    let bytes = size_of_val(values);
    if bytes == 0 {
        Err(MetalError::InputTooLong(values.len()))
    } else {
        Ok(bytes)
    }
}

fn next_nonzero() -> Result<u64, MetalError> {
    NEXT_COMPLETION_SERIAL
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| invalid("bytecode Stage-1 topology serial is exhausted"))
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for BytecodeAddressStage1TopologyOwner {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(mut shared) = visitor.enter_shared(
            allocative::Key::new("topology"),
            size_of::<*const BytecodeAddressStage1TopologyInner>(),
            Arc::as_ptr(&self.0).cast(),
        ) {
            shared.visit_simple(
                allocative::Key::new("device_buffers"),
                self.0.receipt.descriptor_bytes
                    + self.0.receipt.pivot_bytes
                    + self.0.receipt.chunk_offset_bytes
                    + self.0.receipt.work_item_bytes
                    + self.0.receipt.address_offset_bytes,
            );
            shared.exit();
        }
        visitor.exit();
    }
}

fn invalid(message: impl Into<String>) -> MetalError {
    MetalError::InvalidInstructionReadRafGrouped(message.into())
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "compact topology fixtures are exact")]
mod tests {
    use super::*;
    use crate::metal::solinas::validate_bytecode_topology_admission;

    fn build_data(
        padded_rows: usize,
        physical_rows: usize,
        mut address_at: impl FnMut(usize) -> usize,
    ) -> (BytecodeAddressStage1TopologyData, Vec<u8>) {
        let shape = AddressMajorShape::production(padded_rows.ilog2()).unwrap();
        let padded_chunks = padded_rows / INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
        let mut chunks = Vec::with_capacity(padded_chunks);
        let mut ranks = Vec::with_capacity(physical_rows);
        let mut scratch = BytecodeAddressStage1TopologyScratch::new();
        for chunk in 0..padded_chunks {
            let start = chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
            let rows = physical_rows
                .saturating_sub(start)
                .min(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS);
            let mut builder = ChunkTopologyBuilder::new(chunk, rows);
            for row in 0..rows {
                ranks.push(
                    builder
                        .record(&mut scratch, address_at(start + row))
                        .unwrap(),
                );
            }
            chunks.push(Some(builder.finish(&mut scratch).unwrap()));
        }
        (
            merge_chunk_topologies(shape, physical_rows, &mut chunks).unwrap(),
            ranks,
        )
    }

    #[test]
    fn chunk_descriptors_have_local_sentinels_and_stable_bases() {
        let (data, ranks) = build_data(1 << 15, 8192, |row| if row % 2 == 0 { 3 } else { 1 });

        assert_eq!(&data.chunk_offsets, &[0, 2, 3, 5]);
        assert_eq!(data.real_descriptors, 4);
        assert_eq!(data.descriptors.len(), 6);
        assert_eq!(data.descriptors[0].address, 1);
        assert_eq!(data.descriptors[0].base, 0);
        assert_eq!(data.descriptors[0].count(), 2048);
        assert_eq!(data.descriptors[1].address, 3);
        assert_eq!(data.descriptors[1].base, 4096);
        assert_eq!(data.descriptors[1].count(), 2048);
        assert_eq!(data.descriptors[2].address, SENTINEL_ADDRESS);
        assert_eq!(data.descriptors[2].base, 8192);
        assert_eq!(data.descriptors[2].count(), 1);
        assert_eq!(data.descriptors[3].address, 1);
        assert_eq!(data.descriptors[3].base, 2048);
        assert_eq!(data.descriptors[3].count(), 2048);
        assert_eq!(data.descriptors[4].address, 3);
        assert_eq!(data.descriptors[4].base, 6144);
        assert_eq!(data.descriptors[4].count(), 2048);
        assert_eq!(data.descriptors[5].address, SENTINEL_ADDRESS);
        assert_eq!(data.descriptors[5].base, 8192);
        assert_eq!(data.descriptors[5].count(), 1);
        let first_cell_end = usize::from(data.descriptors[0].base) + data.descriptors[0].count();
        assert_eq!(first_cell_end, 2048);
        assert!(first_cell_end < usize::from(data.descriptors[1].base));
        let mut corrupted = data.descriptors.clone();
        corrupted[0] = BytecodeAddressChunkDescriptor::new(
            corrupted[0].address,
            corrupted[0].base,
            corrupted[0].pivot_start(),
            4096,
        )
        .unwrap();
        assert!(validate_packed_descriptor_stream(
            data.physical_rows,
            &corrupted,
            data.real_pivots,
            &data.chunk_offsets,
        )
        .is_err());
        assert_eq!(ranks[0], 0);
        assert_eq!(ranks[512], 0);
        assert_eq!(ranks[4096], 0);
        assert_eq!(data.work_items.len(), 2);
        assert_eq!(data.work_items[0].address, 1);
        assert_eq!(data.work_items[0].start, 0);
        assert_eq!(data.work_items[0].count, 4096);
        assert_eq!(data.work_items[1].address, 3);
        assert_eq!(data.work_items[1].start, 4096);
        assert_eq!(data.work_items[1].count, 4096);
    }

    #[test]
    fn pivots_reconstruct_rank_wraps_and_work_items_split_at_4096() {
        let (data, ranks) = build_data(1 << 15, 4097, |_| 7);

        assert_eq!(ranks[127], 127);
        assert_eq!(ranks[128], 128);
        assert_eq!(ranks[255], 255);
        assert_eq!(ranks[256], 0);
        assert_eq!(ranks[4095], 255);
        assert_eq!(ranks[4096], 0);
        assert_eq!(data.real_pivots, 15);
        assert_eq!(
            &data.pivots[..15],
            &(256u16..=3840).step_by(256).collect::<Vec<_>>()
        );
        assert_eq!(data.pivots[15], u16::MAX);
        assert_eq!(data.descriptors[0].base, 0);
        assert_eq!(data.descriptors[0].count(), 4096);
        assert_eq!(data.descriptors[2].base, 4096);
        assert_eq!(data.descriptors[2].count(), 1);
        assert_eq!(data.work_items.len(), 2);
        assert_eq!(data.work_items[0].start, 0);
        assert_eq!(data.work_items[0].count, 4096);
        assert_eq!(data.work_items[1].start, 4096);
        assert_eq!(data.work_items[1].count, 1);
    }

    #[test]
    fn production_census_descriptor_shape_passes_scatter_admission() {
        const HOT_ROWS: usize = 2_817;
        const TAIL_ADDRESSES: usize = 503;
        let (data, _) = build_data(1 << 15, 4096, |row| {
            if row < HOT_ROWS {
                0
            } else {
                1 + (row - HOT_ROWS) % TAIL_ADDRESSES
            }
        });

        assert_eq!(data.max_descriptors_per_chunk, 504);
        assert_eq!(data.max_pivots_per_chunk, 11);
        validate_bytecode_topology_admission(
            data.max_descriptors_per_chunk,
            data.max_pivots_per_chunk,
        )
        .unwrap();
    }

    #[test]
    fn log27_census_descriptor_shape_passes_scatter_admission() {
        const HOT_ROWS: usize = 3_243;
        const TAIL_ADDRESSES: usize = 853;
        let (data, _) = build_data(1 << 15, 4096, |row| {
            if row < HOT_ROWS {
                0
            } else {
                1 + (row - HOT_ROWS) % TAIL_ADDRESSES
            }
        });

        assert_eq!(data.max_descriptors_per_chunk, 854);
        assert_eq!(data.max_pivots_per_chunk, 12);
        validate_bytecode_topology_admission(
            data.max_descriptors_per_chunk,
            data.max_pivots_per_chunk,
        )
        .unwrap();
    }

    #[test]
    fn structural_descriptor_cap_plus_one_reports_the_admission_clause() {
        let error = validate_bytecode_topology_admission(4097, 0).unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid grouped InstructionReadRaf state: fused bytecode topology admission failed: clause=max_descriptors_per_chunk observed=4097 allowed=1..=4096"
        );
    }

    #[test]
    fn address_major_offsets_and_bases_reset_for_the_second_outer() {
        let physical_rows = INNER_ROWS + 2;
        let (data, _) = build_data(1 << 16, physical_rows, |row| {
            if row < INNER_ROWS {
                row % 2
            } else {
                5
            }
        });

        assert_eq!(data.work_items.last().unwrap().outer, 1);
        assert_eq!(data.work_items.last().unwrap().address, 5);
        assert_eq!(data.work_items.last().unwrap().start, 0);
        assert_eq!(data.work_items.last().unwrap().count, 2);
        let last_chunk_begin = data.chunk_offsets[data.chunk_offsets.len() - 2] as usize;
        let last_chunk_end = data.chunk_offsets[data.chunk_offsets.len() - 1] as usize;
        assert_eq!(data.descriptors[last_chunk_begin].address, 5);
        assert_eq!(data.descriptors[last_chunk_begin].base, 0);
        assert_eq!(data.descriptors[last_chunk_end].base, 2);
        assert_eq!(data.address_offsets[5] + 1, data.address_offsets[6]);
    }

    #[test]
    fn shader_rank_formula_matches_a_stable_address_major_bijection() {
        let physical_rows = INNER_ROWS + 4097;
        let addresses = (0..physical_rows)
            .map(|row| {
                if row < INNER_ROWS {
                    7
                } else if row.is_multiple_of(3) {
                    2
                } else {
                    5
                }
            })
            .collect::<Vec<_>>();
        let (data, rank_low) = build_data(1 << 16, physical_rows, |row| addresses[row]);
        let mut actual = vec![usize::MAX; physical_rows];
        let mut reconstructed_ranks = vec![0usize; physical_rows];

        for row in 0..physical_rows {
            let chunk = row / INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
            let cycle = (row % INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS) as u16;
            let begin = data.chunk_offsets[2 * chunk] as usize;
            let end = data.chunk_offsets[2 * chunk + 1] as usize;
            let descriptor_index = data.descriptors[begin..end]
                .binary_search_by_key(&(addresses[row] as u16), |descriptor| descriptor.address)
                .unwrap()
                + begin;
            let descriptor = data.descriptors[descriptor_index];
            let pivot_end = data.descriptors[descriptor_index + 1].pivot_start();
            let pivot_start = descriptor.pivot_start();
            let rank_high =
                data.pivots[pivot_start..pivot_end].partition_point(|pivot| *pivot <= cycle);
            let rank = rank_high * RANK_RADIX + usize::from(rank_low[row]);
            assert!(rank < descriptor.count());
            reconstructed_ranks[row] = rank;
            actual[row] = (row / INNER_ROWS) * INNER_ROWS + usize::from(descriptor.base) + rank;
        }

        let mut expected = vec![usize::MAX; physical_rows];
        for outer in 0..physical_rows.div_ceil(INNER_ROWS) {
            let start = outer * INNER_ROWS;
            let end = (start + INNER_ROWS).min(physical_rows);
            let mut by_address = vec![Vec::new(); ADDRESS_COUNT];
            for row in start..end {
                by_address[addresses[row]].push(row);
            }
            let mut destination = start;
            for rows in by_address {
                for row in rows {
                    expected[row] = destination;
                    destination += 1;
                }
            }
            assert_eq!(destination, end);
        }

        assert_eq!(actual, expected);
        let mut bijection = actual;
        bijection.sort_unstable();
        assert_eq!(bijection, (0..physical_rows).collect::<Vec<_>>());
        for row in [127, 128, 255, 256, 4095, 4096, INNER_ROWS] {
            assert_eq!(
                reconstructed_ranks[row],
                row % INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS
            );
        }
    }

    #[test]
    fn chunk_builder_rejects_bad_shape_and_incomplete_fill() {
        let mut builder = ChunkTopologyBuilder::new(0, 2);
        let mut scratch = BytecodeAddressStage1TopologyScratch::new();
        assert!(builder.record(&mut scratch, ADDRESS_COUNT).is_err());
        assert_eq!(builder.record(&mut scratch, 0).unwrap(), 0);
        assert!(builder.finish(&mut scratch).is_err());
    }

    #[test]
    fn descriptor_packing_is_exact_and_capacity_checked() {
        let descriptor = BytecodeAddressChunkDescriptor::new(
            8191,
            u16::MAX,
            BYTECODE_ADDRESS_DESCRIPTOR_MAX_PIVOT_START,
            BYTECODE_ADDRESS_DESCRIPTOR_MAX_COUNT,
        )
        .unwrap();

        assert_eq!(descriptor.packed_count_and_pivot_start, u32::MAX);
        assert_eq!(
            descriptor.pivot_start(),
            BYTECODE_ADDRESS_DESCRIPTOR_MAX_PIVOT_START
        );
        assert_eq!(descriptor.count(), BYTECODE_ADDRESS_DESCRIPTOR_MAX_COUNT);
        assert!(BytecodeAddressChunkDescriptor::new(0, 0, 0, 0).is_err());
        assert!(BytecodeAddressChunkDescriptor::new(
            0,
            0,
            0,
            BYTECODE_ADDRESS_DESCRIPTOR_MAX_COUNT + 1,
        )
        .is_err());
        assert!(BytecodeAddressChunkDescriptor::new(
            0,
            0,
            BYTECODE_ADDRESS_DESCRIPTOR_MAX_PIVOT_START + 1,
            1,
        )
        .is_err());
    }

    #[test]
    fn topology_capacity_is_capped_at_log28() {
        assert!(bytecode_address_stage1_topology_max_plane_bytes(1 << 28).is_ok());
        assert!(bytecode_address_stage1_topology_max_plane_bytes((1 << 28) + 1).is_err());

        let log29 = AddressMajorShape::production(29).unwrap();
        assert!(merge_chunk_topologies(log29, 1, &mut []).is_err());
    }
}
