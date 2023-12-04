use crate::emulator::cpu::{MIP_MSIP, MIP_MTIP};

/// Emulates CLINT known as Timer. Refer to the [specification](https://sifive.cdn.prismic.io/sifive%2Fc89f6e5a-cf9e-44c3-a3db-04420702dcc1_sifive+e31+manual+v19.08.pdf)
/// for the detail.
pub struct Clint {
    clock: u64,
    msip: u32,
    mtimecmp: u64,
    mtime: u64,
}

impl Clint {
    /// Creates a new `Clint`
    pub fn new() -> Self {
        Clint {
            clock: 0,
            msip: 0,
            mtimecmp: 0,
            mtime: 0, // @TODO: Should be bound to csr time register
        }
    }

    /// Runs one cycle. `Clint` can raise interrupt. If it does it rises a certain bit
    /// depending on interrupt type of CPU `mip` register.
    ///
    /// # Arguments
    /// * `mip` CPU `mip` register. It can be updated if interrupt occurs.
    pub fn tick(&mut self, mip: &mut u64) {
        self.clock = self.clock.wrapping_add(1);
        self.mtime = self.mtime.wrapping_add(1);

        if (self.msip & 1) != 0 {
            *mip |= MIP_MSIP;
        }

        if self.mtimecmp > 0 && self.mtime >= self.mtimecmp {
            *mip |= MIP_MTIP;
        }
    }

    /// Loads register content.
    ///
    /// # Arguments
    /// * `address`
    pub fn load(&self, address: u64) -> u8 {
        //println!("CLINT Load AD:{:X}", address);
        match address {
            // MSIP register 4 bytes
            0x02000000 => (self.msip & 0xff) as u8,
            0x02000001 => ((self.msip >> 8) & 0xff) as u8,
            0x02000002 => ((self.msip >> 16) & 0xff) as u8,
            0x02000003 => ((self.msip >> 24) & 0xff) as u8,
            // MTIMECMP Registers 8 bytes
            0x02004000 => self.mtimecmp as u8,
            0x02004001 => (self.mtimecmp >> 8) as u8,
            0x02004002 => (self.mtimecmp >> 16) as u8,
            0x02004003 => (self.mtimecmp >> 24) as u8,
            0x02004004 => (self.mtimecmp >> 32) as u8,
            0x02004005 => (self.mtimecmp >> 40) as u8,
            0x02004006 => (self.mtimecmp >> 48) as u8,
            0x02004007 => (self.mtimecmp >> 56) as u8,
            0x0200bff8 => self.mtime as u8,
            0x0200bff9 => (self.mtime >> 8) as u8,
            0x0200bffa => (self.mtime >> 16) as u8,
            0x0200bffb => (self.mtime >> 24) as u8,
            0x0200bffc => (self.mtime >> 32) as u8,
            0x0200bffd => (self.mtime >> 40) as u8,
            0x0200bffe => (self.mtime >> 48) as u8,
            0x0200bfff => (self.mtime >> 56) as u8,
            _ => 0,
        }
    }

    /// Stores register content.
    ///
    /// # Arguments
    /// * `address`
    /// * `value`
    pub fn store(&mut self, address: u64, value: u8) {
        //println!("CLINT Store AD:{:X} VAL:{:X}", address, value);
        match address {
            // MSIP register 4 bytes. Upper 31 bits are hardwired to zero.
            0x02000000 => {
                self.msip = (self.msip & !0x1) | ((value & 1) as u32);
            }
            // MTIMECMP Registers 8 bytes
            0x02004000 => {
                self.mtimecmp = (self.mtimecmp & !0xff) | (value as u64);
            }
            0x02004001 => {
                self.mtimecmp = (self.mtimecmp & !(0xff << 8)) | ((value as u64) << 8);
            }
            0x02004002 => {
                self.mtimecmp = (self.mtimecmp & !(0xff << 16)) | ((value as u64) << 16);
            }
            0x02004003 => {
                self.mtimecmp = (self.mtimecmp & !(0xff << 24)) | ((value as u64) << 24);
            }
            0x02004004 => {
                self.mtimecmp = (self.mtimecmp & !(0xff << 32)) | ((value as u64) << 32);
            }
            0x02004005 => {
                self.mtimecmp = (self.mtimecmp & !(0xff << 40)) | ((value as u64) << 40);
            }
            0x02004006 => {
                self.mtimecmp = (self.mtimecmp & !(0xff << 48)) | ((value as u64) << 48);
            }
            0x02004007 => {
                self.mtimecmp = (self.mtimecmp & !(0xff << 56)) | ((value as u64) << 56);
            }
            // MTIME registers 8 bytes
            0x0200bff8 => {
                self.mtime = (self.mtime & !0xff) | (value as u64);
            }
            0x0200bff9 => {
                self.mtime = (self.mtime & !(0xff << 8)) | ((value as u64) << 8);
            }
            0x0200bffa => {
                self.mtime = (self.mtime & !(0xff << 16)) | ((value as u64) << 16);
            }
            0x0200bffb => {
                self.mtime = (self.mtime & !(0xff << 24)) | ((value as u64) << 24);
            }
            0x0200bffc => {
                self.mtime = (self.mtime & !(0xff << 32)) | ((value as u64) << 32);
            }
            0x0200bffd => {
                self.mtime = (self.mtime & !(0xff << 40)) | ((value as u64) << 40);
            }
            0x0200bffe => {
                self.mtime = (self.mtime & !(0xff << 48)) | ((value as u64) << 48);
            }
            0x0200bfff => {
                self.mtime = (self.mtime & !(0xff << 56)) | ((value as u64) << 56);
            }
            _ => {}
        };
    }

    /// Reads `mtime` register content
    pub fn read_mtime(&self) -> u64 {
        self.mtime
    }

    /// Writes to `mtime` register content
    pub fn write_mtime(&mut self, value: u64) {
        self.mtime = value;
    }
}
