use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[derive(Clone, Debug)]
pub struct MemoryData {
    /// The underlying representation of memory, at the granularity of doublewords.
    memory: HashMap<usize, u64>,
    /// Whether to error out or zero
    /// The number of doublewords that can be stored in this memory.
    num_doublewords: usize,
    /// Checkpoint memory. If this is `Some`, the initial values of all memory accesses will be
    /// stored.
    checkpoint: Option<HashMap<usize, u64>>,
}

impl MemoryData {
    /// Create an empty memory structure with a capacity of 0.
    fn empty() -> Self {
        Self {
            memory: HashMap::new(),
            num_doublewords: 0,
            checkpoint: None,
        }
    }

    /// Set the capacity of the memory structure.
    fn init_with_capacity(&mut self, capacity: u64) {
        self.num_doublewords = capacity.div_ceil(8) as usize;
    }

    /// Get the number of entries in the doubleword-aligned memory storage backend.
    pub fn get_num_doublewords(&self) -> usize {
        self.num_doublewords
    }

    /// Access the values of the doubleword stored at `index` for reading/writing. If the memory is
    /// set up for checkpointing, this also records the access.
    // NOTE: This is mutable to support inserting into the checkpointing hashmap. Note that we need
    // to do this even when we're not writing.
    fn access_u64(&mut self, index: usize) -> &mut u64 {
        if index >= self.num_doublewords {
            panic!(
                "Out of bounds memory access ({index} >= {})",
                self.num_doublewords
            );
        }

        // Return the value at the given index if it's been set. If it hasn't been set, assume it
        // to be zero-initialized.
        // NOTE: If
        let res = self.memory.entry(index).or_insert(0);
        // We store only the initial value of each index accessed (read or written) over the course
        // of a chunk. If the access is a read, the value is the value read. If the access is a
        // write, the value is the value stored *prior* to the write. If the index has already been
        // accessed, we do not modify it.
        if let Some(checkpoint) = self.checkpoint.as_mut() {
            checkpoint.entry(index).or_insert_with(|| *res);
        }

        res
    }

    /// Get read-only access to the doubleword stored at `index` *without* recording the access for
    /// checkpointing.
    fn get_u64(&self, index: usize) -> u64 {
        if index >= self.num_doublewords {
            panic!(
                "Out of bounds memory access ({index} >= {})",
                self.num_doublewords
            );
        }

        *self.memory.get(&index).unwrap_or(&0)
    }

    /// Retrieve the memory for the previously executed chunk as a [`ReplayableMemory`]. This
    /// also starts a new chunk by setting `self.checkpoint` to be an empty hashmap.
    pub fn save_checkpoint(&mut self) -> Self {
        let memory = std::mem::take(
            self.checkpoint
                .as_mut()
                .expect("Tried to save checkpoint without calling start_saving_checkpoints first"),
        );
        Self {
            memory,
            num_doublewords: self.num_doublewords,
            checkpoint: None,
        }
    }

    pub fn is_saving_checkpoints(&self) -> bool {
        self.checkpoint.is_some()
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
    pub(crate) fn read_byte(&mut self, address: u64) -> u8 {
        let index = (address >> 3) as usize;
        let pos = (address % 8) * 8;
        (*self.data.access_u64(index) >> pos) as u8
    }

    /// Reads two bytes from memory.
    ///
    /// # Arguments
    /// * `address`
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
    pub(crate) fn write_byte(&mut self, address: u64, value: u8) {
        let index = (address >> 3) as usize;
        let pos = (address % 8) * 8;
        *self.data.access_u64(index) =
            (*self.data.access_u64(index) & !(0xff << pos)) | ((value as u64) << pos);
    }

    /// Writes two bytes to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    pub(crate) fn write_halfword(&mut self, address: u64, value: u16) {
        if address.is_multiple_of(2) {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            *self.data.access_u64(index) =
                (*self.data.access_u64(index) & !(0xffff << pos)) | ((value as u64) << pos);
        } else {
            self.write_bytes(address, value as u64, 2);
        }
    }

    /// Writes four bytes to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    pub(crate) fn write_word(&mut self, address: u64, value: u32) {
        if address.is_multiple_of(4) {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            *self.data.access_u64(index) =
                (*self.data.access_u64(index) & !(0xffffffff << pos)) | ((value as u64) << pos);
        } else {
            self.write_bytes(address, value as u64, 4);
        }
    }

    /// Writes eight bytes to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
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
    /// `num_doublewords` value while still taking the underlying memory data structure.
    pub(crate) fn take_memory(&mut self) -> Self {
        Self {
            data: MemoryData {
                memory: std::mem::take(&mut self.data.memory),
                num_doublewords: self.data.num_doublewords,
                checkpoint: None,
            },
        }
    }
}
