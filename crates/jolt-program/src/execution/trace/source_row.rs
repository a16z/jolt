use super::{RamAccess, RamRead, RamWrite, RegisterRead, RegisterState, RegisterWrite};

/// One source instruction's architectural transition, with optional register
/// operands and an eight-byte-aligned RAM doubleword. The source backend pins
/// instruction indices, PC continuity, and ISA semantics; construction only
/// packs the supplied values, including accesses at address zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceTraceRow {
    pc: u64,
    next_pc: u64,
    rs1_value: u64,
    rs2_value: u64,
    rd_pre_value: u64,
    rd_post_value: u64,
    ram_address: u64,
    ram_pre_value: u64,
    ram_post_value: u64,
    instruction_index: u32,
    rs1: u8,
    rs2: u8,
    rd: u8,
    tags: u8,
}

const _: () = assert!(std::mem::size_of::<SourceTraceRow>() == 80);
const _: () = assert!(std::mem::align_of::<SourceTraceRow>() == 8);

impl SourceTraceRow {
    /// Packs the arguments without validating registers, addresses, or values.
    pub fn new(
        instruction_index: u32,
        pc: u64,
        next_pc: u64,
        registers: RegisterState,
        ram_access: RamAccess,
    ) -> Self {
        let (ram_address, ram_pre_value, ram_post_value, ram_tag) = match ram_access {
            RamAccess::NoOp => (0, 0, 0, 0),
            RamAccess::Read(read) => (read.address, read.value, read.value, 1),
            RamAccess::Write(write) => (write.address, write.pre_value, write.post_value, 2),
        };
        Self {
            pc,
            next_pc,
            rs1_value: registers.rs1.map_or(0, |read| read.value),
            rs2_value: registers.rs2.map_or(0, |read| read.value),
            rd_pre_value: registers.rd.map_or(0, |write| write.pre_value),
            rd_post_value: registers.rd.map_or(0, |write| write.post_value),
            ram_address,
            ram_pre_value,
            ram_post_value,
            instruction_index,
            rs1: registers.rs1.map_or(0, |read| read.register),
            rs2: registers.rs2.map_or(0, |read| read.register),
            rd: registers.rd.map_or(0, |write| write.register),
            tags: u8::from(registers.rs1.is_some())
                | (u8::from(registers.rs2.is_some()) << 1)
                | (u8::from(registers.rd.is_some()) << 2)
                | (ram_tag << 3),
        }
    }

    pub const fn instruction_index(&self) -> u32 {
        self.instruction_index
    }
    pub const fn pc(&self) -> u64 {
        self.pc
    }
    pub const fn next_pc(&self) -> u64 {
        self.next_pc
    }
    pub const fn rs1_value(&self) -> u64 {
        self.rs1_value
    }
    pub const fn rs2_value(&self) -> u64 {
        self.rs2_value
    }
    pub const fn rd_pre_value(&self) -> u64 {
        self.rd_pre_value
    }
    pub const fn rd_post_value(&self) -> u64 {
        self.rd_post_value
    }
    pub const fn ram_address(&self) -> u64 {
        self.ram_address
    }
    pub const fn ram_pre_value(&self) -> u64 {
        self.ram_pre_value
    }
    pub const fn ram_post_value(&self) -> u64 {
        self.ram_post_value
    }

    /// Reconstructs operand presence, distinguishing an absent operand from x0.
    pub fn registers(&self) -> RegisterState {
        RegisterState {
            rs1: (self.tags & 1 != 0).then_some(RegisterRead {
                register: self.rs1,
                value: self.rs1_value,
            }),
            rs2: (self.tags & 2 != 0).then_some(RegisterRead {
                register: self.rs2,
                value: self.rs2_value,
            }),
            rd: (self.tags & 4 != 0).then_some(RegisterWrite {
                register: self.rd,
                pre_value: self.rd_pre_value,
                post_value: self.rd_post_value,
            }),
        }
    }

    /// Reconstructs the RAM access, distinguishing no access from address zero.
    pub fn ram_access(&self) -> RamAccess {
        match self.tags >> 3 {
            1 => RamAccess::Read(RamRead {
                address: self.ram_address,
                value: self.ram_pre_value,
            }),
            2 => RamAccess::Write(RamWrite {
                address: self.ram_address,
                pre_value: self.ram_pre_value,
                post_value: self.ram_post_value,
            }),
            _ => RamAccess::NoOp,
        }
    }
}
