use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

pub trait MemoryData: Clone + Default + std::fmt::Debug {
    fn init_with_capacity(&mut self, capacity: u64);

    fn get_num_doublewords(&self) -> usize;

    // NOTE: This is mutable to support inserting into the checkpointing hashmap. Note that we need
    // to do this even when we're not writing.
    fn get_u64(&mut self, index: usize) -> &mut u64;

    // NOTE: We use this for now to convert into a vector-based emulator, since many parts of the
    // code seem to assume this is possible. Perhaps we can eliminate that assumption? If so, we
    // can get rid of this function.
    fn into_vec_memory(self) -> Vec<u64>;
}

impl MemoryData for Vec<u64> {
    fn init_with_capacity(&mut self, capacity: u64) {
        for _i in 0..capacity.div_ceil(8) {
            self.push(0);
        }
    }

    fn get_num_doublewords(&self) -> usize {
        self.len()
    }

    fn get_u64(&mut self, index: usize) -> &mut u64 {
        &mut self[index]
    }

    fn into_vec_memory(self) -> Vec<u64> {
        self
    }
}

#[derive(Clone, Default, Debug)]
pub struct ReplayableMemory {
    num_doublewords: usize,
    memory: HashMap<usize, u64>,
}

impl MemoryData for ReplayableMemory {
    fn init_with_capacity(&mut self, capacity: u64) {
        *self = Self {
            num_doublewords: capacity.div_ceil(8) as usize,
            memory: HashMap::new(),
        };
    }

    fn get_num_doublewords(&self) -> usize {
        self.num_doublewords
    }

    fn get_u64(&mut self, index: usize) -> &mut u64 {
        if index > self.num_doublewords {
            panic!("Out of bounds memory access");
        }
        // Return the value at the given index if it's been set. If it hasn't been set, the
        // executing program should never access it withing the current chunk, so we error out.
        self.memory
            .get_mut(&index)
            .expect("Invalid memory access for chunk")
    }

    fn into_vec_memory(self) -> Vec<u64> {
        (0..self.num_doublewords)
            .map(|i| *self.memory.get(&i).unwrap_or(&0))
            .collect()
    }
}

/// This memory representation uses a standard `Vec<u64>` representation for execution, but saves
/// the initial value of each memory access, which can then be retrieved as a [`ReplayableMemory`]
#[derive(Clone, Default, Debug)]
pub struct CheckpointingMemory {
    memory: MemoryBackend<Vec<u64>>,
    checkpoint: HashMap<usize, u64>,
}

impl MemoryData for CheckpointingMemory {
    fn init_with_capacity(&mut self, capacity: u64) {
        self.memory.init(capacity);
        self.checkpoint = HashMap::new();
    }

    fn get_num_doublewords(&self) -> usize {
        self.memory.data.get_num_doublewords()
    }

    fn get_u64(&mut self, index: usize) -> &mut u64 {
        let res = &mut self.memory.data[index];
        // We store only the initial value of each index accessed (read or written) over the course
        // of a chunk. If the access is a read, the value is the value read. If the access is a
        // write, the value is the value stored *prior* to the write. If the index has already been
        // accessed, we do not modify it.
        self.checkpoint.entry(index).or_insert(*res);

        res
    }

    fn into_vec_memory(self) -> Vec<u64> {
        self.memory.data.into_vec_memory()
    }
}

impl CheckpointingMemory {
    /// Retrieve a the memory for the previously executed chunk as a [`ReplayableMemory`]. This
    /// also starts a new chunk by setting `self.checkpoint` to be an empty hashmap.
    pub fn save_checkpoint(&mut self) -> ReplayableMemory {
        ReplayableMemory {
            num_doublewords: self.memory.data.len(),
            memory: std::mem::take(&mut self.checkpoint),
        }
    }
}

/// Emulates main memory.
#[derive(Clone, Debug, Default)]
pub struct MemoryBackend<Data> {
    /// Memory content
    pub data: Data,
}

pub type Memory = MemoryBackend<Vec<u64>>;

impl<Data: MemoryData> MemoryBackend<Data> {
    /// Initializes memory content.
    /// This method is expected to be called only once.
    ///
    /// # Arguments
    /// * `capacity`
    pub fn init(&mut self, capacity: u64) {
        self.data.init_with_capacity(capacity)
    }

    /// Reads a byte from memory.
    ///
    /// # Arguments
    /// * `address`
    pub fn read_byte(&mut self, address: u64) -> u8 {
        let index = (address >> 3) as usize;
        let pos = (address % 8) * 8;
        (*self.data.get_u64(index) >> pos) as u8
    }

    /// Reads two bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    pub fn read_halfword(&mut self, address: u64) -> u16 {
        if address.is_multiple_of(2) {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            (*self.data.get_u64(index) >> pos) as u16
        } else {
            self.read_bytes(address, 2) as u16
        }
    }

    /// Reads four bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    pub fn read_word(&mut self, address: u64) -> u32 {
        if address.is_multiple_of(4) {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            (*self.data.get_u64(index) >> pos) as u32
        } else {
            self.read_bytes(address, 4) as u32
        }
    }

    /// Reads eight bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    pub fn read_doubleword(&mut self, address: u64) -> u64 {
        if address.is_multiple_of(8) {
            let index = (address >> 3) as usize;
            *self.data.get_u64(index)
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
    pub fn read_bytes(&mut self, address: u64, width: u64) -> u64 {
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
    pub fn write_byte(&mut self, address: u64, value: u8) {
        let index = (address >> 3) as usize;
        let pos = (address % 8) * 8;
        *self.data.get_u64(index) =
            (*self.data.get_u64(index) & !(0xff << pos)) | ((value as u64) << pos);
    }

    /// Writes two bytes to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    pub fn write_halfword(&mut self, address: u64, value: u16) {
        if address.is_multiple_of(2) {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            *self.data.get_u64(index) =
                (*self.data.get_u64(index) & !(0xffff << pos)) | ((value as u64) << pos);
        } else {
            self.write_bytes(address, value as u64, 2);
        }
    }

    /// Writes four bytes to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    pub fn write_word(&mut self, address: u64, value: u32) {
        if address.is_multiple_of(4) {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            *self.data.get_u64(index) =
                (*self.data.get_u64(index) & !(0xffffffff << pos)) | ((value as u64) << pos);
        } else {
            self.write_bytes(address, value as u64, 4);
        }
    }

    /// Writes eight bytes to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    pub fn write_doubleword(&mut self, address: u64, value: u64) {
        if address.is_multiple_of(8) {
            let index = (address >> 3) as usize;
            *self.data.get_u64(index) = value;
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
    pub fn write_bytes(&mut self, address: u64, value: u64, width: u64) {
        for i in 0..width {
            self.write_byte(address.wrapping_add(i), (value >> (i * 8)) as u8);
        }
    }

    /// Check if the address is valid memory address
    ///
    /// # Arguments
    /// * `address`
    pub fn validate_address(&self, address: u64) -> bool {
        let word_index = (address >> 3) as usize;
        word_index < self.data.get_num_doublewords()
    }

    pub fn into_vec_memory_backend(self) -> Memory {
        MemoryBackend {
            data: self.data.into_vec_memory(),
        }
    }
}
