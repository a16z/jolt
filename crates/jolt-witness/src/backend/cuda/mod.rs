mod descriptors;
mod device;
mod packed;
mod tables;
#[cfg(test)]
mod tests;

pub use device::{DeviceTrace, COLD};
pub use packed::{PackedTrace, NO_SEQUENCE, RAM_NO_ACCESS};

pub struct DeviceAtomColumns {
    pub flags: cudarc::driver::CudaSlice<u32>,
    pub table_index: cudarc::driver::CudaSlice<u32>,
    pub bytecode_pc: cudarc::driver::CudaSlice<u64>,
    pub rd_pre_value: cudarc::driver::CudaSlice<u64>,
    pub rs1_address: cudarc::driver::CudaSlice<u32>,
    pub rs2_address: cudarc::driver::CudaSlice<u32>,
    pub rd_address: cudarc::driver::CudaSlice<u32>,
    pub rd_inc: cudarc::driver::CudaSlice<u64>,
    pub ram_inc: cudarc::driver::CudaSlice<u64>,
}

pub const FLAG_BIT_CIRCUIT_BASE: u32 = 0;

pub const FLAG_BIT_INSTRUCTION_BASE: u32 = 14;

pub const FLAG_BIT_RAF: u32 = 20;

pub const FLAG_BIT_NOOP_ROW: u32 = 21;

pub const FLAG_BIT_NEXT_IS_NOOP: u32 = 22;

pub const FLAG_BIT_RAM_HAMMING: u32 = 23;

pub const REGISTER_ADDRESS_ABSENT: u32 = u32::MAX;

pub const TABLE_INDEX_ABSENT: u32 = tables::TABLE_INDEX_ABSENT;

pub use tables::{PACK_CIRCUIT_ORDER, PACK_INSTRUCTION_ORDER};

pub fn circuit_flag_bit(flag: jolt_riscv::CircuitFlags) -> Option<u32> {
    let slot = PACK_CIRCUIT_ORDER.iter().position(|&entry| entry == flag)?;
    Some(FLAG_BIT_CIRCUIT_BASE + slot as u32)
}

pub fn instruction_flag_bit(flag: jolt_riscv::InstructionFlags) -> Option<u32> {
    let slot = PACK_INSTRUCTION_ORDER
        .iter()
        .position(|&entry| entry == flag)?;
    Some(FLAG_BIT_INSTRUCTION_BASE + slot as u32)
}

pub struct NarrowColumn {
    pub column: cudarc::driver::CudaSlice<u32>,
    pub span: usize,
    pub first: u64,
}

#[derive(Clone, Copy)]
pub enum HotSource<'a> {
    Interleaved(&'a cudarc::driver::CudaSlice<u64>),
    Word(&'a cudarc::driver::CudaSlice<u32>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceTraceColumn {
    MappedPcWord,
    RemappedRamWord { addresses: usize },
}
