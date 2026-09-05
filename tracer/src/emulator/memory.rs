use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Backing storage for guest memory, at doubleword granularity.
///
/// `Flat` is the execution backing: one contiguous zero-initialized array
/// covering the whole guest address range (uninitialized reads are 0, same as
/// the historical sparse map). `Sparse` backs replay from a checkpoint, whose
/// memory image is exactly the first-touch values recorded while the chunk
/// originally executed — materializing those as a flat array per checkpoint
/// would defeat the point of checkpoints.
#[derive(Clone, Debug)]
enum MemoryBacking {
    Flat(Vec<u64>),
    Sparse(HashMap<usize, u64>),
}

#[derive(Clone, Debug)]
pub struct MemoryData {
    /// The underlying representation of memory, at the granularity of doublewords.
    backing: MemoryBacking,
    /// The number of doublewords that can be stored in this memory.
    num_doublewords: usize,
    /// One past the highest doubleword index ever accessed through
    /// `access_u64` on a flat backing. Everything at or beyond this index is
    /// still zero, so memory snapshots only need to copy the prefix below it.
    high_water: usize,
    /// Checkpoint memory. If this is `Some`, the initial values of all memory accesses will be
    /// stored.
    checkpoint: Option<HashMap<usize, u64>>,
}

#[cold]
#[inline(never)]
fn out_of_bounds(index: usize, num_doublewords: usize) -> ! {
    panic!("Out of bounds memory access ({index} >= {num_doublewords})");
}

impl MemoryData {
    /// Create an empty memory structure with a capacity of 0.
    fn empty() -> Self {
        Self {
            backing: MemoryBacking::Flat(Vec::new()),
            num_doublewords: 0,
            high_water: 0,
            checkpoint: None,
        }
    }

    /// Set the capacity of the memory structure, allocating the flat backing.
    fn init_with_capacity(&mut self, capacity: u64) {
        self.num_doublewords = capacity.div_ceil(8) as usize;
        self.backing = MemoryBacking::Flat(vec![0; self.num_doublewords]);
        self.high_water = 0;
    }

    /// Get the number of entries in the doubleword-aligned memory storage backend.
    pub fn get_num_doublewords(&self) -> usize {
        self.num_doublewords
    }

    /// Access the values of the doubleword stored at `index` for reading/writing. If the memory is
    /// set up for checkpointing, this also records the access.
    // NOTE: This is mutable to support inserting into the checkpointing hashmap. Note that we need
    // to do this even when we're not writing.
    #[inline]
    fn access_u64(&mut self, index: usize) -> &mut u64 {
        match &mut self.backing {
            MemoryBacking::Flat(dwords) => {
                if index >= dwords.len() {
                    out_of_bounds(index, dwords.len());
                }
                if index >= self.high_water {
                    self.high_water = index + 1;
                }
                // We store only the initial value of each index accessed (read or written) over
                // the course of a chunk. If the access is a read, the value is the value read. If
                // the access is a write, the value is the value stored *prior* to the write. If
                // the index has already been accessed, we do not modify it.
                if let Some(checkpoint) = self.checkpoint.as_mut() {
                    checkpoint.entry(index).or_insert(dwords[index]);
                }
                &mut dwords[index]
            }
            MemoryBacking::Sparse(map) => {
                if index >= self.num_doublewords {
                    out_of_bounds(index, self.num_doublewords);
                }
                // Unset entries are assumed to be zero-initialized.
                let res = map.entry(index).or_insert(0);
                if let Some(checkpoint) = self.checkpoint.as_mut() {
                    checkpoint.entry(index).or_insert(*res);
                }
                res
            }
        }
    }

    /// Get read-only access to the doubleword stored at `index` *without* recording the access for
    /// checkpointing.
    #[inline]
    fn get_u64(&self, index: usize) -> u64 {
        match &self.backing {
            MemoryBacking::Flat(dwords) => {
                if index >= dwords.len() {
                    out_of_bounds(index, dwords.len());
                }
                dwords[index]
            }
            MemoryBacking::Sparse(map) => {
                if index >= self.num_doublewords {
                    out_of_bounds(index, self.num_doublewords);
                }
                *map.get(&index).unwrap_or(&0)
            }
        }
    }

    /// Retrieve the memory for the previously executed chunk as a replayable [`MemoryData`]. This
    /// also starts a new chunk by setting `self.checkpoint` to be an empty hashmap.
    #[expect(clippy::expect_used)]
    pub fn save_checkpoint(&mut self) -> Self {
        let memory = std::mem::take(
            self.checkpoint
                .as_mut()
                .expect("Tried to save checkpoint without calling start_saving_checkpoints first"),
        );
        Self {
            backing: MemoryBacking::Sparse(memory),
            num_doublewords: self.num_doublewords,
            high_water: self.num_doublewords,
            checkpoint: None,
        }
    }

    pub fn is_saving_checkpoints(&self) -> bool {
        self.checkpoint.is_some()
    }

    /// The flat backing and its touched prefix length (everything at or past
    /// the prefix is zero). Panics if the backing is sparse —
    /// checkpoint-replay memories are not snapshot sources.
    pub(crate) fn flat_parts(&self) -> (&[u64], usize) {
        match &self.backing {
            MemoryBacking::Flat(dwords) => (dwords, self.high_water.min(dwords.len())),
            MemoryBacking::Sparse(_) => {
                panic!("cannot snapshot sparse (checkpoint-replay) memory")
            }
        }
    }

    /// Replace the backing with a full flat image — the image *becomes* the
    /// working memory, no copy — returning the previous flat backing for
    /// buffer pooling. Panics if the previous backing was sparse.
    pub(crate) fn replace_flat(&mut self, image: Vec<u64>) -> Vec<u64> {
        self.num_doublewords = image.len();
        // Conservative: replay memories are never snapshot sources, so the
        // exact touched prefix is not tracked for them.
        self.high_water = image.len();
        match std::mem::replace(&mut self.backing, MemoryBacking::Flat(image)) {
            MemoryBacking::Flat(old) => old,
            MemoryBacking::Sparse(_) => {
                panic!("cannot pool sparse (checkpoint-replay) memory")
            }
        }
    }

    /// Enable checkpoint saving for this memory. If this is true, all memory accesses will have
    /// their initial values stored to `self.checkpoint`.
    /// NOTE: This is necessary because memory accesses used to store the bytecode in memory should
    /// *not* have their initial (zero) values saved.
    pub fn start_saving_checkpoints(&mut self) {
        if self.checkpoint.is_none() {
            self.checkpoint = Some(HashMap::new());
        }
    }
}

/// Emulates main memory.
#[derive(Clone, Debug)]
pub struct Memory {
    /// Memory content
    pub data: MemoryData,
}

impl Memory {
    /// Creates a new empty memory with a capacity of 0.
    pub(crate) fn empty() -> Self {
        Self {
            data: MemoryData::empty(),
        }
    }

    /// Initializes memory content.
    /// This method is expected to be called only once.
    ///
    /// # Arguments
    /// * `capacity`
    pub(crate) fn init(&mut self, capacity: u64) {
        self.data.init_with_capacity(capacity)
    }

    /// Reads a byte from memory.
    ///
    /// # Arguments
    /// * `address`
    #[inline]
    pub(crate) fn read_byte(&mut self, address: u64) -> u8 {
        let index = (address >> 3) as usize;
        let pos = (address % 8) * 8;
        (*self.data.access_u64(index) >> pos) as u8
    }

    /// Reads two bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    #[inline]
    pub(crate) fn read_halfword(&mut self, address: u64) -> u16 {
        if address.is_multiple_of(2) {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            (*self.data.access_u64(index) >> pos) as u16
        } else {
            self.read_bytes(address, 2) as u16
        }
    }

    /// Reads four bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    #[inline]
    pub(crate) fn read_word(&mut self, address: u64) -> u32 {
        if address.is_multiple_of(4) {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            (*self.data.access_u64(index) >> pos) as u32
        } else {
            self.read_bytes(address, 4) as u32
        }
    }

    /// Reads eight bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    #[inline]
    pub(crate) fn read_doubleword(&mut self, address: u64) -> u64 {
        if address.is_multiple_of(8) {
            let index = (address >> 3) as usize;
            *self.data.access_u64(index)
        } else if address.is_multiple_of(4) {
            (self.read_word(address) as u64)
                | ((self.read_word(address.wrapping_add(4)) as u64) << 32)
        } else {
            self.read_bytes(address, 8)
        }
    }

    /// Reads multiple bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `width` up to eight
    pub(crate) fn read_bytes(&mut self, address: u64, width: u64) -> u64 {
        let mut data = 0_u64;
        for i in 0..width {
            data |= (self.read_byte(address.wrapping_add(i)) as u64) << (i * 8);
        }
        data
    }

    /// Writes a byte to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    #[inline]
    pub(crate) fn write_byte(&mut self, address: u64, value: u8) {
        let index = (address >> 3) as usize;
        let pos = (address % 8) * 8;
        let slot = self.data.access_u64(index);
        *slot = (*slot & !(0xff << pos)) | ((value as u64) << pos);
    }

    /// Writes two bytes to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    #[inline]
    pub(crate) fn write_halfword(&mut self, address: u64, value: u16) {
        if address.is_multiple_of(2) {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            let slot = self.data.access_u64(index);
            *slot = (*slot & !(0xffff << pos)) | ((value as u64) << pos);
        } else {
            self.write_bytes(address, value as u64, 2);
        }
    }

    /// Writes four bytes to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    #[inline]
    pub(crate) fn write_word(&mut self, address: u64, value: u32) {
        if address.is_multiple_of(4) {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            let slot = self.data.access_u64(index);
            *slot = (*slot & !(0xffffffff << pos)) | ((value as u64) << pos);
        } else {
            self.write_bytes(address, value as u64, 4);
        }
    }

    /// Writes eight bytes to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    #[inline]
    pub(crate) fn write_doubleword(&mut self, address: u64, value: u64) {
        if address.is_multiple_of(8) {
            let index = (address >> 3) as usize;
            *self.data.access_u64(index) = value;
        } else if address.is_multiple_of(4) {
            self.write_word(address, (value & 0xffffffff) as u32);
            self.write_word(address.wrapping_add(4), (value >> 32) as u32);
        } else {
            self.write_bytes(address, value, 8);
        }
    }

    /// Write multiple bytes to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    /// * `width` up to eight
    pub(crate) fn write_bytes(&mut self, address: u64, value: u64, width: u64) {
        for i in 0..width {
            self.write_byte(address.wrapping_add(i), (value >> (i * 8)) as u8);
        }
    }

    /// Check if the address is valid memory address
    ///
    /// # Arguments
    /// * `address`
    pub(crate) fn validate_address(&self, address: u64) -> bool {
        let word_index = (address >> 3) as usize;
        word_index < self.data.get_num_doublewords()
    }

    /// Reads a byte from memory.
    ///
    /// # Arguments
    /// * `address`
    pub fn get_byte(&self, address: u64) -> u8 {
        let index = (address >> 3) as usize;
        let pos = (address % 8) * 8;
        (self.data.get_u64(index) >> pos) as u8
    }

    /// Reads four bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    pub fn get_word(&self, address: u64) -> u32 {
        if address.is_multiple_of(4) {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            (self.data.get_u64(index) >> pos) as u32
        } else {
            self.get_bytes(address, 4) as u32
        }
    }

    /// Reads eight bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    pub fn get_doubleword(&self, address: u64) -> u64 {
        if address.is_multiple_of(8) {
            let index = (address >> 3) as usize;
            self.data.get_u64(index)
        } else if address.is_multiple_of(4) {
            (self.get_word(address) as u64)
                | ((self.get_word(address.wrapping_add(4)) as u64) << 32)
        } else {
            self.get_bytes(address, 8)
        }
    }

    pub fn materialized_nonzero_bytes(&self) -> Vec<(u64, u8)> {
        let mut bytes = Vec::new();
        let push_nonzero = |index: usize, doubleword: u64, bytes: &mut Vec<(u64, u8)>| {
            if doubleword == 0 {
                return;
            }
            let base_address = (index as u64) * 8;
            for byte_offset in 0..8 {
                let byte = (doubleword >> (byte_offset * 8)) as u8;
                if byte != 0 {
                    bytes.push((base_address + byte_offset, byte));
                }
            }
        };
        match &self.data.backing {
            MemoryBacking::Flat(dwords) => {
                for (index, &doubleword) in dwords.iter().enumerate() {
                    push_nonzero(index, doubleword, &mut bytes);
                }
            }
            MemoryBacking::Sparse(map) => {
                for (&index, &doubleword) in map {
                    push_nonzero(index, doubleword, &mut bytes);
                }
            }
        }
        bytes.sort_by_key(|(address, _)| *address);
        bytes
    }

    /// Reads multiple bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `width` up to eight
    pub(crate) fn get_bytes(&self, address: u64, width: u64) -> u64 {
        let mut data = 0_u64;
        for i in 0..width {
            data |= (self.get_byte(address.wrapping_add(i)) as u64) << (i * 8);
        }
        data
    }

    /// Take the underlying collection of doublewords out of the memory structure, replacing it
    /// with an empty collection. We use this instead of `std::mem::take` in order to preserve the
    /// `num_doublewords` value in the returned memory while still taking the underlying data
    /// structure. The emptied `self` reports zero capacity, consistent with its empty backing.
    pub(crate) fn take_memory(&mut self) -> Self {
        let taken = Self {
            data: MemoryData {
                backing: std::mem::replace(&mut self.data.backing, MemoryBacking::Flat(Vec::new())),
                num_doublewords: self.data.num_doublewords,
                high_water: self.data.high_water,
                checkpoint: None,
            },
        };
        self.data.num_doublewords = 0;
        self.data.high_water = 0;
        taken
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test-only assertions")]
mod tests {
    use std::panic::AssertUnwindSafe;

    use super::*;

    fn memory(capacity: u64) -> Memory {
        let mut memory = Memory::empty();
        memory.init(capacity);
        memory
    }

    #[test]
    fn misaligned_accesses_agree_with_byte_wise_composition() {
        let mut memory = memory(64);
        // Write a doubleword at an odd address; every read width must
        // reassemble the same little-endian bytes.
        memory.write_doubleword(3, 0x1122_3344_5566_7788);
        assert_eq!(memory.read_doubleword(3), 0x1122_3344_5566_7788);
        assert_eq!(memory.read_word(3), 0x5566_7788);
        assert_eq!(memory.read_word(7), 0x1122_3344);
        assert_eq!(memory.read_halfword(3), 0x7788);
        assert_eq!(memory.read_halfword(5), 0x5566);
        assert_eq!(memory.read_byte(3), 0x88);
        assert_eq!(memory.read_byte(10), 0x11);

        // Word-aligned (but not doubleword-aligned) doubleword access
        memory.write_doubleword(4, 0xaabb_ccdd_eeff_0011);
        assert_eq!(memory.read_doubleword(4), 0xaabb_ccdd_eeff_0011);

        // Misaligned halfword/word writes decompose into byte writes
        memory.write_halfword(17, 0xbeef);
        assert_eq!(memory.read_byte(17), 0xef);
        assert_eq!(memory.read_byte(18), 0xbe);
        memory.write_word(21, 0xdead_beef);
        assert_eq!(memory.read_word(21), 0xdead_beef);
    }

    #[test]
    fn non_mutating_getters_match_written_state_without_materializing() {
        let mut memory = memory(64);
        memory.write_doubleword(8, 0x0102_0304_0506_0708);
        assert_eq!(memory.get_byte(8), 0x08);
        assert_eq!(memory.get_word(8), 0x0506_0708);
        assert_eq!(memory.get_word(10), 0x0304_0506); // misaligned getter
        assert_eq!(memory.get_doubleword(8), 0x0102_0304_0506_0708);
        assert_eq!(memory.get_doubleword(12), 0x0102_0304); // word-aligned getter
        assert_eq!(memory.get_doubleword(9), 0x0001_0203_0405_0607); // byte path
                                                                     // Untouched addresses read as zero
        assert_eq!(memory.get_doubleword(32), 0);
    }

    #[test]
    fn materialized_nonzero_bytes_lists_only_written_bytes_in_order() {
        let mut memory = memory(64);
        memory.write_byte(9, 0xAA);
        memory.write_byte(40, 0xBB);
        memory.write_byte(3, 0xCC);
        // A doubleword written then zeroed again is skipped entirely
        memory.write_doubleword(16, 0x1234);
        memory.write_doubleword(16, 0);

        assert_eq!(
            memory.materialized_nonzero_bytes(),
            vec![(3, 0xCC), (9, 0xAA), (40, 0xBB)]
        );
    }

    #[test]
    fn checkpoints_record_the_pre_chunk_value_of_each_touched_word() {
        let mut memory = memory(64);
        memory.write_doubleword(0, 111);
        memory.write_doubleword(8, 222);

        assert!(!memory.data.is_saving_checkpoints());
        memory.data.start_saving_checkpoints();
        assert!(memory.data.is_saving_checkpoints());

        // Touch index 0 (write) and index 1 (read); leave index 2 untouched
        memory.write_doubleword(0, 999);
        memory.write_doubleword(0, 1000); // second write must not overwrite the snapshot
        assert_eq!(memory.read_doubleword(8), 222);

        let checkpoint = memory.data.save_checkpoint();
        let MemoryBacking::Sparse(snapshot) = &checkpoint.backing else {
            panic!("checkpoint replay memory must have a sparse backing");
        };
        assert_eq!(snapshot.get(&0), Some(&111), "pre-write value");
        assert_eq!(snapshot.get(&1), Some(&222), "read snapshot");
        assert_eq!(snapshot.get(&2), None, "untouched word absent");
        assert_eq!(checkpoint.get_num_doublewords(), 8);

        // Saving started a fresh chunk: only new accesses are recorded
        memory.write_doubleword(16, 5);
        let next = memory.data.save_checkpoint();
        let MemoryBacking::Sparse(snapshot) = &next.backing else {
            panic!("checkpoint replay memory must have a sparse backing");
        };
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot.get(&2), Some(&0));
    }

    #[test]
    fn take_memory_moves_capacity_and_content_to_the_taken_memory() {
        let mut memory = memory(64);
        memory.write_doubleword(0, 42);
        let taken = memory.take_memory();
        assert_eq!(taken.data.get_num_doublewords(), 8);
        assert_eq!(taken.data.get_u64(0), 42);
        // The emptied source reports zero capacity, consistent with its
        // empty backing; every address is now out of bounds.
        assert_eq!(memory.data.get_num_doublewords(), 0);
        assert!(!memory.validate_address(0), "content moved out");
    }

    #[test]
    fn out_of_bounds_accesses_panic_with_the_capacity() {
        let mut memory = memory(16); // 2 doublewords
        assert!(memory.validate_address(15));
        assert!(!memory.validate_address(16));
        let err = std::panic::catch_unwind(AssertUnwindSafe(|| memory.read_byte(16)))
            .expect_err("read beyond capacity must panic");
        let message = err.downcast_ref::<String>().expect("panic message");
        assert!(message.contains("Out of bounds memory access (2 >= 2)"));

        let memory = self::memory(16);
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| { memory.get_byte(24) })).is_err());
    }
}
