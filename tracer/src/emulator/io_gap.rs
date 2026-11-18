//! The I/O gap `[io_end, RAM_START_ADDRESS)`: zero-initialized read/write
//! guest memory between the I/O segments and RAM. Every tracer backend keeps
//! it outside its RAM backing and builds a run's final memory image with
//! [`final_memory_image`].

use std::collections::BTreeMap;

use common::constants::RAM_START_ADDRESS;
use jolt_program::execution::MemoryImage;

/// The contents of the I/O gap. Sparse because the gap can span half the I/O
/// region (tens of MiB for large advice capacities), guests rarely touch it,
/// and every checkpoint copies it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IoGapMemory {
    /// The doublewords holding a nonzero value, keyed by aligned address.
    /// Storing zero removes the entry, so equal contents compare equal.
    words: BTreeMap<u64, u64>,
}

impl IoGapMemory {
    pub fn load_doubleword(&self, aligned: u64) -> u64 {
        self.words.get(&aligned).copied().unwrap_or(0)
    }

    pub fn store_doubleword(&mut self, aligned: u64, value: u64) {
        if value == 0 {
            self.words.remove(&aligned);
        } else {
            self.words.insert(aligned, value);
        }
    }

    pub fn load_byte(&self, address: u64) -> u8 {
        (self.load_doubleword(address & !7) >> (8 * (address & 7))) as u8
    }

    pub fn store_byte(&mut self, address: u64, value: u8) {
        let aligned = address & !7;
        let shift = 8 * (address & 7);
        let word = (self.load_doubleword(aligned) & !(0xff << shift)) | (u64::from(value) << shift);
        self.store_doubleword(aligned, word);
    }

    /// Every nonzero byte, by absolute address in ascending order.
    fn nonzero_bytes(&self) -> impl Iterator<Item = (u64, u8)> + '_ {
        self.words.iter().flat_map(|(&aligned, word)| {
            (0..8)
                .zip(word.to_le_bytes())
                .filter(|&(_, byte)| byte != 0)
                .map(move |(offset, byte)| (aligned + offset, byte))
        })
    }
}

/// A run's final memory image: every nonzero guest byte outside the I/O
/// segments, by absolute address in ascending order. `ram` holds the RAM
/// backing's nonzero bytes in ascending order, keyed by offset from
/// `RAM_START_ADDRESS`.
pub fn final_memory_image(io_gap: &IoGapMemory, ram: Vec<(u64, u8)>) -> MemoryImage {
    let mut bytes = ram;
    for (address, _) in &mut bytes {
        *address += RAM_START_ADDRESS;
    }
    bytes.splice(0..0, io_gap.nonzero_bytes());
    MemoryImage { bytes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_stores_merge_into_doublewords_and_zero_words_vanish() {
        let gap = RAM_START_ADDRESS - 16;
        let mut memory = IoGapMemory::default();
        memory.store_doubleword(gap, 0x1122_3344_5566_7788);
        memory.store_byte(gap + 1, 0xab);
        assert_eq!(memory.load_doubleword(gap), 0x1122_3344_5566_ab88);
        assert_eq!(memory.load_byte(gap + 7), 0x11);
        assert_eq!(memory.load_doubleword(gap + 8), 0);

        memory.store_byte(gap + 8, 0xcd);
        memory.store_byte(gap + 8, 0);
        memory.store_doubleword(gap, 0);
        assert_eq!(memory, IoGapMemory::default());
    }
}
