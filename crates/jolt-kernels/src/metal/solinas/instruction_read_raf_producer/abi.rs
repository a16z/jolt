use std::mem::{align_of, size_of};

use super::{
    validate_rows, ProducerLayoutError, Result, GROUPED_SEGMENTS, GROUPED_SEGMENT_OFFSETS,
    MAX_BUFFER_BYTES, MAX_SHARD_ROWS, PRODUCER_CHUNK_ROWS, PRODUCER_INPUT_BYTES_PER_ROW,
    PRODUCER_OUTPUT_BYTES_PER_ROW, PRODUCER_THREADS_PER_GROUP,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlaneRole {
    CycleLookupLo,
    CycleLookupHi,
    CycleClaims,
    GroupedLookupLo,
    GroupedLookupHi,
    CycleToGroupedLocal,
    SegmentOffsets,
    ChunkSegmentBases,
    Status,
}

pub const SCATTER_BUFFER_ROLES: [PlaneRole; 9] = [
    PlaneRole::CycleLookupLo,
    PlaneRole::CycleLookupHi,
    PlaneRole::CycleClaims,
    PlaneRole::ChunkSegmentBases,
    PlaneRole::SegmentOffsets,
    PlaneRole::GroupedLookupLo,
    PlaneRole::GroupedLookupHi,
    PlaneRole::CycleToGroupedLocal,
    PlaneRole::Status,
];

impl PlaneRole {
    const fn element_bytes(self) -> usize {
        match self {
            Self::CycleClaims => size_of::<u8>(),
            Self::CycleLookupLo
            | Self::CycleLookupHi
            | Self::GroupedLookupLo
            | Self::GroupedLookupHi => size_of::<u64>(),
            Self::CycleToGroupedLocal
            | Self::SegmentOffsets
            | Self::ChunkSegmentBases
            | Self::Status => size_of::<u32>(),
        }
    }

    pub const fn metal_buffer_slot(self) -> usize {
        match self {
            Self::CycleLookupLo => 0,
            Self::CycleLookupHi => 1,
            Self::CycleClaims => 2,
            Self::ChunkSegmentBases => 3,
            Self::SegmentOffsets => 4,
            Self::GroupedLookupLo => 5,
            Self::GroupedLookupHi => 6,
            Self::CycleToGroupedLocal => 7,
            Self::Status => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferShape {
    role: PlaneRole,
    elements: usize,
    bytes: usize,
}

impl BufferShape {
    fn new(role: PlaneRole, elements: usize) -> Result<Self> {
        let bytes = elements
            .checked_mul(role.element_bytes())
            .ok_or(ProducerLayoutError::SizeOverflow("buffer bytes"))?;
        if bytes > MAX_BUFFER_BYTES {
            return Err(ProducerLayoutError::BufferTooLarge { plane: role, bytes });
        }
        Ok(Self {
            role,
            elements,
            bytes,
        })
    }

    pub const fn role(self) -> PlaneRole {
        self.role
    }

    pub const fn elements(self) -> usize {
        self.elements
    }

    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerGeometry {
    total_rows: usize,
    shard_count: usize,
}

impl ProducerGeometry {
    pub fn new(total_rows: usize) -> Result<Self> {
        validate_rows(total_rows)?;
        Ok(Self {
            total_rows,
            shard_count: total_rows.div_ceil(MAX_SHARD_ROWS),
        })
    }

    pub const fn total_rows(self) -> usize {
        self.total_rows
    }

    pub const fn shard_count(self) -> usize {
        self.shard_count
    }

    pub fn shard(self, index: usize) -> Result<ProducerShardPlan> {
        if index >= self.shard_count {
            return Err(ProducerLayoutError::InvalidShardIndex {
                index,
                shards: self.shard_count,
            });
        }
        let absolute_row_start = index
            .checked_mul(MAX_SHARD_ROWS)
            .ok_or(ProducerLayoutError::SizeOverflow("shard row start"))?;
        let rows = (self.total_rows - absolute_row_start).min(MAX_SHARD_ROWS);
        let shard = ProducerShardPlan {
            total_rows: self.total_rows,
            shard_index: index,
            absolute_row_start,
            rows,
            chunks: rows.div_ceil(PRODUCER_CHUNK_ROWS),
        };
        let _buffer_shapes = shard.buffer_shapes()?;
        Ok(shard)
    }

    pub fn shards(self) -> Result<Vec<ProducerShardPlan>> {
        (0..self.shard_count)
            .map(|index| self.shard(index))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerShardPlan {
    total_rows: usize,
    shard_index: usize,
    absolute_row_start: usize,
    rows: usize,
    chunks: usize,
}

impl ProducerShardPlan {
    pub const fn total_rows(self) -> usize {
        self.total_rows
    }

    pub const fn shard_index(self) -> usize {
        self.shard_index
    }

    pub const fn absolute_row_start(self) -> usize {
        self.absolute_row_start
    }

    pub fn absolute_row_end(self) -> Result<usize> {
        self.absolute_row_start
            .checked_add(self.rows)
            .ok_or(ProducerLayoutError::SizeOverflow("shard row end"))
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn chunks(self) -> usize {
        self.chunks
    }

    pub fn buffer_shape(self, role: PlaneRole) -> Result<BufferShape> {
        let elements = match role {
            PlaneRole::CycleLookupLo
            | PlaneRole::CycleLookupHi
            | PlaneRole::CycleClaims
            | PlaneRole::GroupedLookupLo
            | PlaneRole::GroupedLookupHi
            | PlaneRole::CycleToGroupedLocal => self.rows,
            PlaneRole::SegmentOffsets => GROUPED_SEGMENT_OFFSETS,
            PlaneRole::ChunkSegmentBases => self
                .chunks
                .checked_mul(GROUPED_SEGMENTS)
                .ok_or(ProducerLayoutError::SizeOverflow("chunk segment bases"))?,
            PlaneRole::Status => 1,
        };
        BufferShape::new(role, elements)
    }

    pub fn buffer_shapes(self) -> Result<[BufferShape; 9]> {
        Ok([
            self.buffer_shape(PlaneRole::CycleLookupLo)?,
            self.buffer_shape(PlaneRole::CycleLookupHi)?,
            self.buffer_shape(PlaneRole::CycleClaims)?,
            self.buffer_shape(PlaneRole::ChunkSegmentBases)?,
            self.buffer_shape(PlaneRole::SegmentOffsets)?,
            self.buffer_shape(PlaneRole::GroupedLookupLo)?,
            self.buffer_shape(PlaneRole::GroupedLookupHi)?,
            self.buffer_shape(PlaneRole::CycleToGroupedLocal)?,
            self.buffer_shape(PlaneRole::Status)?,
        ])
    }
}

/// Per-chunk counts accumulated while the producer writes cycle claims.
///
/// Construction checks shape and row totals. The scatter's count status is
/// still required to prove that these counts match the bound claim buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkSegmentCounts {
    shard: ProducerShardPlan,
    counts: Vec<[u32; GROUPED_SEGMENTS]>,
}

impl ChunkSegmentCounts {
    pub fn new(shard: ProducerShardPlan, counts: Vec<[u32; GROUPED_SEGMENTS]>) -> Result<Self> {
        if counts.len() != shard.chunks {
            return Err(ProducerLayoutError::InvalidLayout(
                "chunk-count rows must equal the shard chunk count",
            ));
        }
        for (chunk, segment_counts) in counts.iter().enumerate() {
            let expected =
                ((chunk + 1) * PRODUCER_CHUNK_ROWS).min(shard.rows) - chunk * PRODUCER_CHUNK_ROWS;
            let observed = segment_counts
                .iter()
                .try_fold(0usize, |sum, &count| sum.checked_add(count as usize));
            if observed != Some(expected) {
                return Err(ProducerLayoutError::InvalidLayout(
                    "each chunk's segment counts must sum to its source rows",
                ));
            }
        }
        Ok(Self { shard, counts })
    }

    pub const fn shard(&self) -> ProducerShardPlan {
        self.shard
    }

    pub fn counts(&self) -> &[[u32; GROUPED_SEGMENTS]] {
        &self.counts
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScatterLayout {
    shard: ProducerShardPlan,
    segment_offsets: [u32; GROUPED_SEGMENT_OFFSETS],
    chunk_segment_bases: Vec<[u32; GROUPED_SEGMENTS]>,
}

impl ScatterLayout {
    pub fn from_chunk_counts(counts: ChunkSegmentCounts) -> Result<Self> {
        let shard = counts.shard;
        let mut totals = [0u32; GROUPED_SEGMENTS];
        for chunk_counts in &counts.counts {
            for (total, &count) in totals.iter_mut().zip(chunk_counts) {
                *total = total
                    .checked_add(count)
                    .ok_or(ProducerLayoutError::SizeOverflow("segment count"))?;
            }
        }

        let mut segment_offsets = [0u32; GROUPED_SEGMENT_OFFSETS];
        for segment in 0..GROUPED_SEGMENTS {
            segment_offsets[segment + 1] = segment_offsets[segment]
                .checked_add(totals[segment])
                .ok_or(ProducerLayoutError::SizeOverflow("segment offset"))?;
        }

        let mut running: [u32; GROUPED_SEGMENTS] =
            std::array::from_fn(|segment| segment_offsets[segment]);
        let mut chunk_segment_bases = counts.counts;
        for bases in &mut chunk_segment_bases {
            let chunk_counts = *bases;
            *bases = running;
            for (running, count) in running.iter_mut().zip(chunk_counts) {
                *running = running
                    .checked_add(count)
                    .ok_or(ProducerLayoutError::SizeOverflow("chunk segment base"))?;
            }
        }
        Ok(Self::from_validated_parts(
            shard,
            &segment_offsets,
            chunk_segment_bases,
        ))
    }

    pub(super) fn from_validated_parts(
        shard: ProducerShardPlan,
        segment_offsets: &[u32; GROUPED_SEGMENT_OFFSETS],
        chunk_segment_bases: Vec<[u32; GROUPED_SEGMENTS]>,
    ) -> Self {
        Self {
            shard,
            segment_offsets: *segment_offsets,
            chunk_segment_bases,
        }
    }

    pub const fn shard(&self) -> ProducerShardPlan {
        self.shard
    }

    pub const fn segment_offsets(&self) -> &[u32; GROUPED_SEGMENT_OFFSETS] {
        &self.segment_offsets
    }

    pub fn chunk_segment_bases(&self) -> &[[u32; GROUPED_SEGMENTS]] {
        &self.chunk_segment_bases
    }

    pub fn chunk_segment_count(&self, chunk: usize, segment: usize) -> Option<u32> {
        let start = *self.chunk_segment_bases.get(chunk)?.get(segment)?;
        let end = if chunk + 1 < self.chunk_segment_bases.len() {
            self.chunk_segment_bases[chunk + 1][segment]
        } else {
            *self.segment_offsets.get(segment + 1)?
        };
        end.checked_sub(start)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterParams {
    total_rows: u32,
    shard_row_start: u32,
    shard_rows: u32,
    chunks: u32,
    segments: u32,
    chunk_rows: u32,
    lookup_lo_elements: u32,
    lookup_hi_elements: u32,
    cycle_claim_elements: u32,
    grouped_lo_elements: u32,
    grouped_hi_elements: u32,
    cycle_to_grouped_elements: u32,
    chunk_base_elements: u32,
    segment_offset_elements: u32,
    status_elements: u32,
    reserved: u32,
}

impl ScatterParams {
    fn new(shard: ProducerShardPlan) -> Result<Self> {
        let element_count = |role| {
            u32::try_from(shard.buffer_shape(role)?.elements())
                .map_err(|_| ProducerLayoutError::SizeOverflow("shader element count"))
        };
        Ok(Self {
            total_rows: u32::try_from(shard.total_rows)
                .map_err(|_| ProducerLayoutError::SizeOverflow("total rows"))?,
            shard_row_start: u32::try_from(shard.absolute_row_start)
                .map_err(|_| ProducerLayoutError::SizeOverflow("shard row start"))?,
            shard_rows: u32::try_from(shard.rows)
                .map_err(|_| ProducerLayoutError::SizeOverflow("shard rows"))?,
            chunks: u32::try_from(shard.chunks)
                .map_err(|_| ProducerLayoutError::SizeOverflow("shard chunks"))?,
            segments: GROUPED_SEGMENTS as u32,
            chunk_rows: PRODUCER_CHUNK_ROWS as u32,
            lookup_lo_elements: element_count(PlaneRole::CycleLookupLo)?,
            lookup_hi_elements: element_count(PlaneRole::CycleLookupHi)?,
            cycle_claim_elements: element_count(PlaneRole::CycleClaims)?,
            grouped_lo_elements: element_count(PlaneRole::GroupedLookupLo)?,
            grouped_hi_elements: element_count(PlaneRole::GroupedLookupHi)?,
            cycle_to_grouped_elements: element_count(PlaneRole::CycleToGroupedLocal)?,
            chunk_base_elements: element_count(PlaneRole::ChunkSegmentBases)?,
            segment_offset_elements: element_count(PlaneRole::SegmentOffsets)?,
            status_elements: element_count(PlaneRole::Status)?,
            reserved: 0,
        })
    }

    pub const fn words(self) -> [u32; 16] {
        [
            self.total_rows,
            self.shard_row_start,
            self.shard_rows,
            self.chunks,
            self.segments,
            self.chunk_rows,
            self.lookup_lo_elements,
            self.lookup_hi_elements,
            self.cycle_claim_elements,
            self.grouped_lo_elements,
            self.grouped_hi_elements,
            self.cycle_to_grouped_elements,
            self.chunk_base_elements,
            self.segment_offset_elements,
            self.status_elements,
            self.reserved,
        ]
    }
}

const _: () = assert!(size_of::<ScatterParams>() == 64);
const _: () = assert!(align_of::<ScatterParams>() == 4);

/// Non-owning predispatch plan. A future owner must bind and retain real Metal
/// buffers after checking their device, generation, and byte lengths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScatterDispatchPlan {
    shard: ProducerShardPlan,
    layout: Box<ScatterLayout>,
    params: ScatterParams,
    required_buffers: [BufferShape; 9],
}

impl ScatterDispatchPlan {
    pub fn new(shard: ProducerShardPlan, layout: &ScatterLayout) -> Result<Self> {
        if layout.shard != shard {
            return Err(ProducerLayoutError::ShardMismatch);
        }
        validate_layout_shape(shard, layout)?;
        Ok(Self {
            shard,
            layout: Box::new(layout.clone()),
            params: ScatterParams::new(shard)?,
            required_buffers: shard.buffer_shapes()?,
        })
    }

    pub const fn shard(&self) -> ProducerShardPlan {
        self.shard
    }

    pub fn layout(&self) -> &ScatterLayout {
        &self.layout
    }

    pub const fn params(&self) -> ScatterParams {
        self.params
    }

    pub const fn threadgroups(&self) -> usize {
        self.shard.chunks
    }

    pub const fn threads_per_group(&self) -> usize {
        PRODUCER_THREADS_PER_GROUP
    }

    pub const fn required_buffers(&self) -> &[BufferShape; 9] {
        &self.required_buffers
    }
}

/// Optimistic external-memory traffic for scatter only.
///
/// Layout uploads are separate. The model excludes source production,
/// cache-line amplification, metadata reads served from cache, and atomics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterTraffic {
    rows: u64,
    input_bytes: u64,
    output_bytes: u64,
    payload_bytes: u64,
    layout_upload_bytes: u64,
}

impl ScatterTraffic {
    pub fn for_geometry(geometry: ProducerGeometry) -> Result<Self> {
        let rows = geometry.total_rows as u64;
        let input_bytes = rows
            .checked_mul(PRODUCER_INPUT_BYTES_PER_ROW as u64)
            .ok_or(ProducerLayoutError::SizeOverflow("scatter input traffic"))?;
        let output_bytes = rows
            .checked_mul(PRODUCER_OUTPUT_BYTES_PER_ROW as u64)
            .ok_or(ProducerLayoutError::SizeOverflow("scatter output traffic"))?;
        let payload_bytes = input_bytes
            .checked_add(output_bytes)
            .ok_or(ProducerLayoutError::SizeOverflow("scatter payload traffic"))?;
        let mut layout_upload_bytes = 0u64;
        for shard in geometry.shards()? {
            for role in [PlaneRole::SegmentOffsets, PlaneRole::ChunkSegmentBases] {
                layout_upload_bytes = layout_upload_bytes
                    .checked_add(shard.buffer_shape(role)?.bytes() as u64)
                    .ok_or(ProducerLayoutError::SizeOverflow("layout upload traffic"))?;
            }
        }
        Ok(Self {
            rows,
            input_bytes,
            output_bytes,
            payload_bytes,
            layout_upload_bytes,
        })
    }

    pub const fn rows(self) -> u64 {
        self.rows
    }

    pub const fn input_bytes(self) -> u64 {
        self.input_bytes
    }

    pub const fn output_bytes(self) -> u64 {
        self.output_bytes
    }

    pub const fn payload_bytes(self) -> u64 {
        self.payload_bytes
    }

    pub const fn layout_upload_bytes(self) -> u64 {
        self.layout_upload_bytes
    }
}

fn validate_layout_shape(shard: ProducerShardPlan, layout: &ScatterLayout) -> Result<()> {
    let offsets = &layout.segment_offsets;
    if offsets[0] != 0 || offsets[GROUPED_SEGMENTS] as usize != shard.rows {
        return Err(ProducerLayoutError::InvalidLayout(
            "segment offsets must cover exactly the shard rows",
        ));
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(ProducerLayoutError::InvalidLayout(
            "segment offsets must be monotone",
        ));
    }
    if layout.chunk_segment_bases.len() != shard.chunks {
        return Err(ProducerLayoutError::InvalidLayout(
            "chunk-base rows must equal the shard chunk count",
        ));
    }
    for segment in 0..GROUPED_SEGMENTS {
        let mut minimum = offsets[segment];
        let end = offsets[segment + 1];
        for (chunk, bases) in layout.chunk_segment_bases.iter().enumerate() {
            let base = bases[segment];
            if base < minimum || base > end || (chunk == 0 && base != offsets[segment]) {
                return Err(ProducerLayoutError::InvalidLayout(
                    "chunk bases must monotonically partition each segment",
                ));
            }
            minimum = base;
        }
    }
    Ok(())
}
