use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[derive(Clone, Debug)]
pub enum MemoryData {
    Full {
        memory: Vec<u64>,
        // NOTE: This is just the length of `self.memory`, under normal circumstances. We store it
        // separately because we sometimes call [`std::mem::take`] on `self.memory`, but we still need
        // the length in that case.
        num_doublewords: usize,
        checkpoint: HashMap<usize, u64>,
        saving_checkpoints: bool,
    },
    Checkpoint {
        memory: HashMap<usize, u64>,
        num_doublewords: usize,
    },
}

impl MemoryData {
    /// Create an empty memory structure with a capacity of 0.
    fn empty() -> Self {
        Self::Full {
            memory: vec![],
            num_doublewords: 0,
            checkpoint: HashMap::default(),
            saving_checkpoints: false,
        }
    }

    /// Set the capacity of the memory structure.
    fn init_with_capacity(&mut self, capacity: u64) {
        match self {
            Self::Full { memory, num_doublewords, checkpoint, saving_checkpoints } => {
                *num_doublewords = capacity.div_ceil(8) as usize;

                *memory = vec![0; *num_doublewords];
                *checkpoint = HashMap::new();
                *saving_checkpoints = false;
            }
            Self::Checkpoint { memory, num_doublewords } => {
                *num_doublewords = capacity.div_ceil(8) as usize;
                *memory = HashMap::new();
            }
        }
    }

    /// Get the number of entries in the doubleword-aligned memory storage backend.
    pub fn get_num_doublewords(&self) -> usize {
        match self {
            Self::Full { memory: _, num_doublewords, checkpoint: _, saving_checkpoints: _ } => *num_doublewords,
            Self::Checkpoint { memory: _, num_doublewords } => *num_doublewords,
        }
    }

    /// Access the values of the doubleword stored at `index` for reading/writing. If the memory is
    /// set up for checkpointing, this also records the access.
    // NOTE: This is mutable to support inserting into the checkpointing hashmap. Note that we need
    // to do this even when we're not writing.
    fn access_u64(&mut self, index: usize) -> &mut u64 {
        match self {
            Self::Full { memory, num_doublewords: _, checkpoint, saving_checkpoints } => {
                let res = &mut memory[index];
                // We store only the initial value of each index accessed (read or written) over the course
                // of a chunk. If the access is a read, the value is the value read. If the access is a
                // write, the value is the value stored *prior* to the write. If the index has already been
                // accessed, we do not modify it.
                if *saving_checkpoints {
                    checkpoint.entry(index).or_insert_with(|| *res);
                }

                res
            }
            Self::Checkpoint { memory, num_doublewords } => {
                if index >= *num_doublewords {
                    panic!("Out of bounds memory access ({index} >= {num_doublewords})");
                }
                // Return the value at the given index if it's been set. If it hasn't been set, the
                // executing program should never access it within the current chunk, so we error out.
                memory.get_mut(&index).expect("Invalid memory access for chunk")
            }
        }
    }

    /// Take the underlying vector of doublewords out of the memory structure, replacing it with an
    /// empty collection.
    // NOTE: We use this for now to convert into a vector-based emulator, since many parts of the
    // code seem to assume this is possible. Perhaps we can eliminate that assumption? If so, we
    // can get rid of this function.
    fn take_as_vec_memory(&mut self) -> Self {
        match self {
            Self::Full { memory, num_doublewords, checkpoint: _, saving_checkpoints } => {
                Self::Full {
                    memory: std::mem::take(memory),
                    num_doublewords: *num_doublewords,
                    checkpoint: HashMap::default(),
                    saving_checkpoints: *saving_checkpoints,
                }
            }
            Self::Checkpoint { memory: _, num_doublewords: _ } => {
                unimplemented!("Can't take ReplayableMemory as a Vec<u64>");
            }
        }
    }

    /// Get read-only access to the doubleword stored at `index` *without* recording the access for
    /// checkpointing.
    fn get_u64(&self, index: usize) -> u64 {
        match self {
            Self::Full { memory, num_doublewords: _, checkpoint: _, saving_checkpoints: _ } => memory[index],
            Self::Checkpoint { memory, num_doublewords: _ } => memory[&index],
        }
    }

    /// Retrieve a the memory for the previously executed chunk as a [`ReplayableMemory`]. This
    /// also starts a new chunk by setting `self.checkpoint` to be an empty hashmap.
    pub fn save_checkpoint(&mut self) -> Self {
        match self {
            Self::Full { memory: _, num_doublewords, checkpoint, saving_checkpoints: _ } => {
                assert!(*num_doublewords != 0);
                Self::Checkpoint {
                    num_doublewords: *num_doublewords,
                    memory: std::mem::take(checkpoint),
                }
            }
            Self::Checkpoint { memory: _, num_doublewords: _ } => {
                unimplemented!("Can't save checkpoints from within a checkpoint")
            }
        }
    }

    pub fn is_saving_checkpoints(&self) -> bool {
        match self {
            Self::Full { memory: _, num_doublewords: _, checkpoint: _, saving_checkpoints } => *saving_checkpoints,
            Self::Checkpoint { memory: _, num_doublewords: _ } => false,
        }
    }

    /// Enable checkpoint saving for this memory. If this is true, all memory accesses will have
    /// their initial values stored to `self.checkpoint`.
    /// NOTE: This is necessary because memory accesses used to store the bytecode in memory should
    /// *not* have their initial (zero) values saved.
    pub fn start_saving_checkpoints(&mut self) {
        match self {
            Self::Full { memory: _, num_doublewords: _, checkpoint: _, saving_checkpoints } => {
                *saving_checkpoints = true;
            }
            Self::Checkpoint { memory: _, num_doublewords: _ } => {
                unimplemented!("Can't save checkpoints from within a checkpoint")
            }
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

    pub(crate) fn take_as_vec_memory_backend(&mut self) -> Memory {
        Memory {
            data: self.data.take_as_vec_memory(),
        }
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
}
