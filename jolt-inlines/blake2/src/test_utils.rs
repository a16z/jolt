use jolt_inlines_common::constants::{blake2, INLINE_OPCODE};
use tracer::emulator::mmu::DRAM_BASE;
use tracer::instruction::format::format_inline::FormatInline;
use tracer::instruction::inline::INLINE;
use tracer::utils::test_harness::CpuTestHarness;

pub struct Blake2CpuHarness {
    pub harness: CpuTestHarness,
}

impl Blake2CpuHarness {
    /// Memory layout constants
    const STATE_ADDR: u64 = DRAM_BASE;
    const MESSAGE_ADDR: u64 = DRAM_BASE + (crate::STATE_VECTOR_LEN * 8) as u64;
    const COUNTER_ADDR: u64 = Self::MESSAGE_ADDR + (crate::MSG_BLOCK_LEN * 8) as u64;
    const FLAG_ADDR: u64 = Self::COUNTER_ADDR + 8;

    pub const RS1: u8 = 10; // Points to the state
    pub const RS2: u8 = 11; // Points to the message block + counter + final flag

    pub fn new() -> Self {
        Self {
            harness: CpuTestHarness::new(),
        }
    }

    pub fn load_blake2_data(
        &mut self,
        state: &[u64; crate::STATE_VECTOR_LEN],
        message: &[u64; crate::MSG_BLOCK_LEN],
        counter: u64,
        is_final: bool,
    ) {
        // Set up memory pointers in registers
        self.harness.cpu.x[Self::RS1 as usize] = Self::STATE_ADDR as i64;
        self.harness.cpu.x[Self::RS2 as usize] = Self::MESSAGE_ADDR as i64;

        // Load state into memory
        self.harness.set_memory(Self::STATE_ADDR, state);
        // Load message block into memory
        self.harness.set_memory(Self::MESSAGE_ADDR, message);
        // Load counter
        self.harness.set_memory(Self::COUNTER_ADDR, &[counter]);
        // Load final flag
        let flag_value = if is_final { 1u64 } else { 0u64 };
        self.harness.set_memory(Self::FLAG_ADDR, &[flag_value]);
    }

    pub fn read_state(&mut self) -> [u64; crate::STATE_VECTOR_LEN] {
        let mut state = [0u64; crate::STATE_VECTOR_LEN];
        self.harness.read_memory(Self::STATE_ADDR, &mut state);
        state
    }

    pub fn instruction() -> INLINE {
        INLINE {
            address: 0,
            operands: FormatInline {
                rs1: Self::RS1,
                rs2: Self::RS2,
                rs3: 0,
            },
            opcode: INLINE_OPCODE,
            funct3: blake2::FUNCT3,
            funct7: blake2::FUNCT7,
            inline_sequence_remaining: None,
            is_compressed: false,
        }
    }
}

impl Default for Blake2CpuHarness {
    fn default() -> Self {
        Self::new()
    }
}
