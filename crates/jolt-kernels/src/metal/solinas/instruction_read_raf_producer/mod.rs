mod address_atoms;
#[cfg(test)]
mod address_atoms_tests;

pub use address_atoms::{
    split_equality_weight, AddressAtomBufferShape, AddressAtomCycleRow, AddressAtomCycleSource,
    AddressAtomError, AddressAtomLookup, AddressAtomMassReceipt, AddressAtomPartitionPenalty,
    AddressAtomPlaneReceipt, AddressAtomPlaneRole, AddressAtomResult, AddressAtomShape,
    AddressAtomSourceProvenance, AddressAtomTopology, AddressAtomTopologyBatchReceipt,
    AddressAtomTopologyParts, AddressAtomTopologyReceipt, AddressAtomTraffic,
    ADDRESS_ATOM_MASS_BYTES, ADDRESS_ATOM_PLANE_ROLES,
};

use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use thiserror::Error;

pub const LOOKUP_TABLES: usize = LookupTableKind::<RISCV_XLEN>::COUNT;
pub const TABLE_SELECTOR_VALUES: usize = LOOKUP_TABLES + 1;
pub const RAF_SELECTOR_VALUES: usize = 2;
pub const GROUPED_SEGMENTS: usize = TABLE_SELECTOR_VALUES * RAF_SELECTOR_VALUES;
pub const GROUPED_SEGMENT_OFFSETS: usize = GROUPED_SEGMENTS + 1;
pub const PRODUCER_CHUNK_ROWS: usize = 4096;
pub const MAX_TOTAL_ROWS: usize = 1 << 28;
pub const MAX_SHARD_ROWS: usize = 1 << 26;
pub const MAX_BUFFER_BYTES: usize = 2 * 1024 * 1024 * 1024;
pub const PRODUCER_INPUT_BYTES_PER_ROW: usize = 8 + 8 + 1;

const _: () = assert!(LOOKUP_TABLES == 40);
const _: () = assert!(GROUPED_SEGMENTS == 82);
const _: () = assert!(GROUPED_SEGMENT_OFFSETS == 83);

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ProducerLayoutError {
    #[error("row count must be a nonzero power of two no larger than 2^28, got {0}")]
    InvalidRowCount(usize),
    #[error("shard index {index} is outside 0..{shards}")]
    InvalidShardIndex { index: usize, shards: usize },
    #[error("lookup table index {0} is outside the producer table range")]
    InvalidTableIndex(usize),
    #[error("claim byte {0:#04x} has an invalid table selector")]
    InvalidClaim(u8),
    #[error("{0} size overflowed")]
    SizeOverflow(&'static str),
}

pub type Result<T> = std::result::Result<T, ProducerLayoutError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerGeometry {
    total_rows: usize,
    shard_count: usize,
}

impl ProducerGeometry {
    pub fn new(total_rows: usize) -> Result<Self> {
        if total_rows == 0 || !total_rows.is_power_of_two() || total_rows > MAX_TOTAL_ROWS {
            return Err(ProducerLayoutError::InvalidRowCount(total_rows));
        }
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
        Ok(ProducerShardPlan {
            total_rows: self.total_rows,
            shard_index: index,
            absolute_row_start,
            rows,
            chunks: rows.div_ceil(PRODUCER_CHUNK_ROWS),
        })
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
}

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
