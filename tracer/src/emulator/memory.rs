/// Emulates main memory.
pub struct Memory {
    /// Memory content
    data: Vec<u64>,
}

impl Memory {
    /// Creates a new `Memory`
    pub fn new() -> Self {
        Memory { data: vec![] }
    }

    /// Initializes memory content.
    /// This method is expected to be called only once.
    ///
    /// # Arguments
    /// * `capacity`
    pub fn init(&mut self, capacity: u64) {
        for _i in 0..((capacity + 7) / 8) {
            self.data.push(0);
        }
    }

    /// Reads a byte from memory.
    ///
    /// # Arguments
    /// * `address`
    pub fn read_byte(&self, address: u64) -> u8 {
        let index = (address >> 3) as usize;
        let pos = (address % 8) * 8;
        (self.data[index] >> pos) as u8
    }

    /// Reads two bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    pub fn read_halfword(&self, address: u64) -> u16 {
        if (address % 2) == 0 {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            (self.data[index] >> pos) as u16
        } else {
            self.read_bytes(address, 2) as u16
        }
    }

    /// Reads four bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    pub fn read_word(&self, address: u64) -> u32 {
        if (address % 4) == 0 {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            (self.data[index] >> pos) as u32
        } else {
            self.read_bytes(address, 4) as u32
        }
    }

    /// Reads eight bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    pub fn read_doubleword(&self, address: u64) -> u64 {
        if (address % 8) == 0 {
            let index = (address >> 3) as usize;
            self.data[index]
        } else if (address % 4) == 0 {
            (self.read_word(address) as u64)
                | ((self.read_word(address.wrapping_add(4)) as u64) << 4)
        } else {
            self.read_bytes(address, 8)
        }
    }

    /// Reads multiple bytes from memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `width` up to eight
    pub fn read_bytes(&self, address: u64, width: u64) -> u64 {
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
        self.data[index] = (self.data[index] & !(0xff << pos)) | ((value as u64) << pos);
    }

    /// Writes two bytes to memory.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    pub fn write_halfword(&mut self, address: u64, value: u16) {
        if (address % 2) == 0 {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            self.data[index] = (self.data[index] & !(0xffff << pos)) | ((value as u64) << pos);
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
        if (address % 4) == 0 {
            let index = (address >> 3) as usize;
            let pos = (address % 8) * 8;
            self.data[index] = (self.data[index] & !(0xffffffff << pos)) | ((value as u64) << pos);
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
        if (address % 8) == 0 {
            let index = (address >> 3) as usize;
            self.data[index] = value;
        } else if (address % 4) == 0 {
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
        (address as usize) < self.data.len()
    }
}
