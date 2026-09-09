use std::mem::size_of;

use super::Fp128;

pub const ADDRESS_RAF_LANES: usize = 6;
pub const ADDRESS_RAF_BINS: usize = 256;

const TABLE_INDEX_SHIFT: u32 = 56;
const TABLE_INDEX_MASK: u64 = 0x3f;
const RAF_FLAG_SHIFT: u32 = 62;

/// The 40-byte row ABI consumed by the address phase.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AddressRafScanRow {
    words: [u64; 5],
}

impl AddressRafScanRow {
    pub const fn new(lookup_index: u128, raf_flag: bool) -> Self {
        Self::new_with_table(lookup_index, None, raf_flag)
    }

    pub const fn new_with_table(
        lookup_index: u128,
        table_index: Option<usize>,
        raf_flag: bool,
    ) -> Self {
        let table_plus_one = match table_index {
            Some(index) => index as u64 + 1,
            None => 0,
        };
        Self {
            words: [
                lookup_index as u64,
                (lookup_index >> 64) as u64,
                0,
                0,
                (table_plus_one << TABLE_INDEX_SHIFT) | ((raf_flag as u64) << RAF_FLAG_SHIFT),
            ],
        }
    }

    pub const fn lookup_index(self) -> u128 {
        self.words[0] as u128 | ((self.words[1] as u128) << 64)
    }

    pub const fn raf_flag(self) -> bool {
        self.words[4] & (1 << RAF_FLAG_SHIFT) != 0
    }

    pub const fn table_index(self) -> Option<usize> {
        let table_plus_one = ((self.words[4] >> TABLE_INDEX_SHIFT) & TABLE_INDEX_MASK) as usize;
        table_plus_one.checked_sub(1)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressRafSums {
    values: Vec<Fp128>,
}

impl AddressRafSums {
    pub(super) fn from_values(values: Vec<Fp128>) -> Self {
        Self { values }
    }

    pub fn as_flat_slice(&self) -> &[Fp128] {
        &self.values
    }

    pub fn shift_half(&self) -> &[Fp128] {
        self.lane(0)
    }

    pub fn left(&self) -> &[Fp128] {
        self.lane(1)
    }

    pub fn right(&self) -> &[Fp128] {
        self.lane(2)
    }

    pub fn shift_full(&self) -> &[Fp128] {
        self.lane(3)
    }

    pub fn identity(&self) -> &[Fp128] {
        self.lane(4)
    }

    pub fn upper_all_ones(&self) -> &[Fp128] {
        self.lane(5)
    }

    fn lane(&self, lane: usize) -> &[Fp128] {
        &self.values[lane * ADDRESS_RAF_BINS..(lane + 1) * ADDRESS_RAF_BINS]
    }
}

const _: () = assert!(size_of::<AddressRafScanRow>() == 40);
