use super::{
    ChunkSegmentCounts, PlaneRole, ProducerLayoutError, ProducerShardPlan, Result, ScatterLayout,
    GROUPED_SEGMENTS, GROUPED_SEGMENT_OFFSETS, LOOKUP_TABLES, PRODUCER_CHUNK_ROWS,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerSelector {
    table_index: Option<usize>,
    raf_flag: bool,
}

impl ProducerSelector {
    pub fn new(table_index: Option<usize>, raf_flag: bool) -> Result<Self> {
        if let Some(table_index) = table_index.filter(|&table| table >= LOOKUP_TABLES) {
            return Err(ProducerLayoutError::InvalidTableIndex(table_index));
        }
        Ok(Self {
            table_index,
            raf_flag,
        })
    }

    pub const fn table_index(self) -> Option<usize> {
        self.table_index
    }

    pub const fn raf_flag(self) -> bool {
        self.raf_flag
    }

    pub const fn table_plus_one(self) -> usize {
        match self.table_index {
            Some(table) => table + 1,
            None => 0,
        }
    }

    pub const fn segment(self) -> usize {
        2 * self.table_plus_one() + self.raf_flag as usize
    }

    pub const fn claim(self) -> u8 {
        self.table_plus_one() as u8 | ((self.raf_flag as u8) << 7)
    }
}

pub fn decode_claim(claim: u8) -> Result<ProducerSelector> {
    let table_plus_one = usize::from(claim & 0x7f);
    if table_plus_one > LOOKUP_TABLES {
        return Err(ProducerLayoutError::InvalidClaim(claim));
    }
    Ok(ProducerSelector {
        table_index: table_plus_one.checked_sub(1),
        raf_flag: claim & 0x80 != 0,
    })
}

impl ScatterLayout {
    pub fn from_cycle_claims(shard: ProducerShardPlan, cycle_claims: &[u8]) -> Result<Self> {
        validate_plane_len(PlaneRole::CycleClaims, cycle_claims.len(), shard.rows())?;
        Self::from_chunk_counts(count_cycle_claims(shard, cycle_claims)?)
    }

    pub fn from_checked_parts(
        shard: ProducerShardPlan,
        cycle_claims: &[u8],
        segment_offsets: &[u32; GROUPED_SEGMENT_OFFSETS],
        chunk_segment_bases: Vec<[u32; GROUPED_SEGMENTS]>,
    ) -> Result<Self> {
        validate_plane_len(PlaneRole::CycleClaims, cycle_claims.len(), shard.rows())?;
        let expected = Self::from_cycle_claims(shard, cycle_claims)?;
        if segment_offsets != expected.segment_offsets()
            || chunk_segment_bases.as_slice() != expected.chunk_segment_bases()
        {
            return Err(ProducerLayoutError::InvalidLayout(
                "offsets and chunk bases must match the cycle claim counts",
            ));
        }
        Ok(Self::from_validated_parts(
            shard,
            segment_offsets,
            chunk_segment_bases,
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostScatter {
    layout: ScatterLayout,
    grouped_lookup_lo: Vec<u64>,
    grouped_lookup_hi: Vec<u64>,
    cycle_to_grouped_local: Vec<u32>,
}

impl HostScatter {
    pub fn from_cycle_planes(
        shard: ProducerShardPlan,
        cycle_lookup_lo: &[u64],
        cycle_lookup_hi: &[u64],
        cycle_claims: &[u8],
    ) -> Result<Self> {
        validate_source_planes(shard, cycle_lookup_lo, cycle_lookup_hi, cycle_claims)?;
        let layout = ScatterLayout::from_cycle_claims(shard, cycle_claims)?;
        let mut grouped_lookup_lo = vec![0u64; shard.rows()];
        let mut grouped_lookup_hi = vec![0u64; shard.rows()];
        let mut cycle_to_grouped_local = vec![0u32; shard.rows()];
        let mut next = layout.chunk_segment_bases().to_vec();

        for cycle in 0..shard.rows() {
            let chunk = cycle / PRODUCER_CHUNK_ROWS;
            let segment = decode_claim(cycle_claims[cycle])?.segment();
            let grouped = next[chunk][segment];
            next[chunk][segment] = grouped
                .checked_add(1)
                .ok_or(ProducerLayoutError::SizeOverflow("grouped local index"))?;
            let grouped = grouped as usize;
            grouped_lookup_lo[grouped] = cycle_lookup_lo[cycle];
            grouped_lookup_hi[grouped] = cycle_lookup_hi[cycle];
            cycle_to_grouped_local[cycle] = grouped as u32;
        }

        Self::from_checked_parts(
            &layout,
            grouped_lookup_lo,
            grouped_lookup_hi,
            cycle_to_grouped_local,
            cycle_lookup_lo,
            cycle_lookup_hi,
            cycle_claims,
        )
    }

    pub fn from_checked_parts(
        layout: &ScatterLayout,
        grouped_lookup_lo: Vec<u64>,
        grouped_lookup_hi: Vec<u64>,
        cycle_to_grouped_local: Vec<u32>,
        cycle_lookup_lo: &[u64],
        cycle_lookup_hi: &[u64],
        cycle_claims: &[u8],
    ) -> Result<Self> {
        let scatter = Self {
            layout: layout.clone(),
            grouped_lookup_lo,
            grouped_lookup_hi,
            cycle_to_grouped_local,
        };
        scatter.validate_against(cycle_lookup_lo, cycle_lookup_hi, cycle_claims)?;
        Ok(scatter)
    }

    pub const fn layout(&self) -> &ScatterLayout {
        &self.layout
    }

    pub fn grouped_lookup_lo(&self) -> &[u64] {
        &self.grouped_lookup_lo
    }

    pub fn grouped_lookup_hi(&self) -> &[u64] {
        &self.grouped_lookup_hi
    }

    pub fn cycle_to_grouped_local(&self) -> &[u32] {
        &self.cycle_to_grouped_local
    }

    pub fn validate_against(
        &self,
        cycle_lookup_lo: &[u64],
        cycle_lookup_hi: &[u64],
        cycle_claims: &[u8],
    ) -> Result<()> {
        let shard = self.layout.shard();
        validate_source_planes(shard, cycle_lookup_lo, cycle_lookup_hi, cycle_claims)?;
        validate_plane_len(
            PlaneRole::GroupedLookupLo,
            self.grouped_lookup_lo.len(),
            shard.rows(),
        )?;
        validate_plane_len(
            PlaneRole::GroupedLookupHi,
            self.grouped_lookup_hi.len(),
            shard.rows(),
        )?;
        validate_plane_len(
            PlaneRole::CycleToGroupedLocal,
            self.cycle_to_grouped_local.len(),
            shard.rows(),
        )?;

        let mut seen = vec![false; shard.rows()];
        for cycle in 0..shard.rows() {
            let segment = decode_claim(cycle_claims[cycle])?.segment();
            let chunk = cycle / PRODUCER_CHUNK_ROWS;
            let range_start = self.layout.chunk_segment_bases()[chunk][segment];
            let range_end = if chunk + 1 < shard.chunks() {
                self.layout.chunk_segment_bases()[chunk + 1][segment]
            } else {
                self.layout.segment_offsets()[segment + 1]
            };
            let grouped = self.cycle_to_grouped_local[cycle];
            if grouped < range_start || grouped >= range_end || grouped as usize >= shard.rows() {
                return Err(ProducerLayoutError::ScatterInvariant(
                    "inverse must stay inside its chunk and segment range",
                ));
            }
            let grouped = grouped as usize;
            if seen[grouped] {
                return Err(ProducerLayoutError::ScatterInvariant(
                    "cycle-to-grouped-local must be a permutation",
                ));
            }
            seen[grouped] = true;
            if self.grouped_lookup_lo[grouped] != cycle_lookup_lo[cycle]
                || self.grouped_lookup_hi[grouped] != cycle_lookup_hi[cycle]
            {
                return Err(ProducerLayoutError::ScatterInvariant(
                    "grouped lookup limbs must match their source cycle",
                ));
            }
        }
        Ok(())
    }
}

fn count_cycle_claims(shard: ProducerShardPlan, cycle_claims: &[u8]) -> Result<ChunkSegmentCounts> {
    let mut chunk_counts = vec![[0u32; GROUPED_SEGMENTS]; shard.chunks()];
    for (cycle, &claim) in cycle_claims.iter().enumerate() {
        let segment = decode_claim(claim)?.segment();
        chunk_counts[cycle / PRODUCER_CHUNK_ROWS][segment] += 1;
    }
    ChunkSegmentCounts::new(shard, chunk_counts)
}

fn validate_source_planes(
    shard: ProducerShardPlan,
    cycle_lookup_lo: &[u64],
    cycle_lookup_hi: &[u64],
    cycle_claims: &[u8],
) -> Result<()> {
    validate_plane_len(
        PlaneRole::CycleLookupLo,
        cycle_lookup_lo.len(),
        shard.rows(),
    )?;
    validate_plane_len(
        PlaneRole::CycleLookupHi,
        cycle_lookup_hi.len(),
        shard.rows(),
    )?;
    validate_plane_len(PlaneRole::CycleClaims, cycle_claims.len(), shard.rows())
}

fn validate_plane_len(role: PlaneRole, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(ProducerLayoutError::PlaneElements {
            plane: role,
            expected,
            got,
        });
    }
    Ok(())
}
