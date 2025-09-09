//! Generic test harness for inline instructions.
//!
//! Provides a unified testing framework for all inline instructions,
//! eliminating the need for inline-specific test harnesses.

use super::test_harness::TEST_MEMORY_CAPACITY;
use crate::emulator::cpu::{Cpu, Xlen};
use crate::emulator::default_terminal::DefaultTerminal;
use crate::emulator::mmu::DRAM_BASE;
use crate::instruction::format::format_inline::FormatInline;
use crate::instruction::inline::INLINE;
use crate::instruction::{RISCVTrace, RV32IMInstruction};

pub struct InlineMemoryLayout {
    pub input_base: u64,
    pub input_size: usize,
    pub input2_base: Option<u64>,
    pub input2_size: Option<usize>,
    pub output_base: u64,
    pub output_size: usize,
}

impl InlineMemoryLayout {
    /// Single input, single output (e.g., hash functions like SHA256, Keccak)
    pub fn single_input(input_size: usize, output_size: usize) -> Self {
        Self {
            input_base: DRAM_BASE,
            input_size,
            input2_base: None,
            input2_size: None,
            output_base: DRAM_BASE + input_size as u64,
            output_size,
        }
    }

    /// Two inputs, single output (e.g., BigInt multiplication, Blake2/3 with extra parameters)
    pub fn two_inputs(input_size: usize, input2_size: usize, output_size: usize) -> Self {
        Self {
            input_base: DRAM_BASE,
            input_size,
            input2_base: Some(DRAM_BASE + input_size as u64),
            input2_size: Some(input2_size),
            output_base: DRAM_BASE + (input_size + input2_size) as u64,
            output_size,
        }
    }
}

pub struct InlineTestHarness {
    pub cpu: Cpu,
    layout: InlineMemoryLayout,
    xlen: Xlen,
}

impl InlineTestHarness {
    pub fn new(layout: InlineMemoryLayout, xlen: Xlen) -> Self {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        if xlen == Xlen::Bit32 {
            cpu.update_xlen(Xlen::Bit32);
        }
        Self { cpu, layout, xlen }
    }

    pub fn load_input32(&mut self, data: &[u32]) {
        assert!(
            data.len() * 4 <= self.layout.input_size,
            "Input data exceeds allocated size"
        );
        for (i, &word) in data.iter().enumerate() {
            self.cpu
                .mmu
                .store_word(self.layout.input_base + (i * 4) as u64, word)
                .expect("Failed to store input word");
        }
    }

    pub fn load_input64(&mut self, data: &[u64]) {
        assert!(
            data.len() * 8 <= self.layout.input_size,
            "Input data exceeds allocated size"
        );
        for (i, &word) in data.iter().enumerate() {
            self.cpu
                .mmu
                .store_doubleword(self.layout.input_base + (i * 8) as u64, word)
                .expect("Failed to store input doubleword");
        }
    }

    pub fn load_state32(&mut self, data: &[u32]) {
        assert!(
            data.len() * 4 <= self.layout.output_size,
            "State data exceeds allocated size"
        );
        for (i, &word) in data.iter().enumerate() {
            self.cpu
                .mmu
                .store_word(self.layout.output_base + (i * 4) as u64, word)
                .expect("Failed to store state word");
        }
    }

    pub fn load_state64(&mut self, data: &[u64]) {
        assert!(
            data.len() * 8 <= self.layout.output_size,
            "State data exceeds allocated size"
        );
        for (i, &word) in data.iter().enumerate() {
            self.cpu
                .mmu
                .store_doubleword(self.layout.output_base + (i * 8) as u64, word)
                .expect("Failed to store state doubleword");
        }
    }

    pub fn load_input2_32(&mut self, data: &[u32]) {
        let base = self
            .layout
            .input2_base
            .expect("No second input memory region defined");
        let size = self
            .layout
            .input2_size
            .expect("No second input size defined");
        assert!(
            data.len() * 4 <= size,
            "Second input data exceeds allocated size"
        );

        for (i, &word) in data.iter().enumerate() {
            self.cpu
                .mmu
                .store_word(base + (i * 4) as u64, word)
                .expect("Failed to store second input word");
        }
    }

    pub fn load_input2_64(&mut self, data: &[u64]) {
        let base = self
            .layout
            .input2_base
            .expect("No second input memory region defined");
        let size = self
            .layout
            .input2_size
            .expect("No second input size defined");
        assert!(
            data.len() * 8 <= size,
            "Second input data exceeds allocated size"
        );

        for (i, &word) in data.iter().enumerate() {
            self.cpu
                .mmu
                .store_doubleword(base + (i * 8) as u64, word)
                .expect("Failed to store second input doubleword");
        }
    }

    pub fn read_output32(&mut self, count: usize) -> Vec<u32> {
        assert!(
            count * 4 <= self.layout.output_size,
            "Read exceeds output size"
        );
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let word = self
                .cpu
                .mmu
                .load_word(self.layout.output_base + (i * 4) as u64)
                .expect("Failed to load output word")
                .0;
            result.push(word);
        }
        result
    }

    pub fn read_output64(&mut self, count: usize) -> Vec<u64> {
        assert!(
            count * 8 <= self.layout.output_size,
            "Read exceeds output size"
        );
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let word = self
                .cpu
                .mmu
                .load_doubleword(self.layout.output_base + (i * 8) as u64)
                .expect("Failed to load output doubleword")
                .0;
            result.push(word);
        }
        result
    }

    pub fn setup_registers(&mut self, rs1: u8, rs2: u8, rs3: Option<u8>) {
        self.cpu.x[rs1 as usize] = self.layout.input_base as i64;
        self.cpu.x[rs2 as usize] = self.layout.output_base as i64;
        if let Some(rs3_reg) = rs3 {
            let input2_base = self.layout.input2_base.unwrap_or(self.layout.output_base);
            self.cpu.x[rs3_reg as usize] = input2_base as i64;
        }
    }

    pub fn execute_inline(&mut self, instruction: INLINE) {
        instruction.trace(&mut self.cpu, None);
    }

    pub fn execute_sequence(&mut self, sequence: &[RV32IMInstruction]) {
        for instr in sequence {
            instr.execute(&mut self.cpu);
        }
    }

    pub fn create_instruction(
        opcode: u32,
        funct3: u32,
        funct7: u32,
        rs1: u8,
        rs2: u8,
        rs3: u8,
    ) -> INLINE {
        INLINE {
            address: 0,
            operands: FormatInline { rs1, rs2, rs3 },
            opcode,
            funct3,
            funct7,
            inline_sequence_remaining: None,
            is_compressed: false,
        }
    }

    pub fn xlen(&self) -> Xlen {
        self.xlen
    }
}

pub mod hash_helpers {
    use super::*;

    pub fn sha256_harness(xlen: Xlen) -> InlineTestHarness {
        let layout = InlineMemoryLayout::single_input(64, 32); // 64-byte block, 32-byte state
        InlineTestHarness::new(layout, xlen)
    }

    pub fn blake2_harness(xlen: Xlen) -> InlineTestHarness {
        // Blake2 has message block and extra parameters (counter + flag)
        let layout = InlineMemoryLayout::two_inputs(128, 16, 64); // 128-byte block, 16-byte params, 64-byte state
        InlineTestHarness::new(layout, xlen)
    }

    pub fn blake3_harness(xlen: Xlen) -> InlineTestHarness {
        // Blake3 has message block and extra parameters (counter + block_len + flags)
        let layout = InlineMemoryLayout::two_inputs(64, 16, 32); // 64-byte block, 16-byte params, 32-byte state
        InlineTestHarness::new(layout, xlen)
    }

    pub fn keccak256_harness(xlen: Xlen) -> InlineTestHarness {
        let layout = InlineMemoryLayout::single_input(136, 200); // 136-byte block, 200-byte state
        InlineTestHarness::new(layout, xlen)
    }
}

pub mod bigint_helpers {
    use super::*;

    pub fn bigint256_mul_harness(xlen: Xlen) -> InlineTestHarness {
        let layout = InlineMemoryLayout::two_inputs(32, 32, 64); // Two 32-byte inputs, 64-byte output
        InlineTestHarness::new(layout, xlen)
    }
}
