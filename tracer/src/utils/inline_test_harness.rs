//! Generic test harness for inline instructions.
//!
//! Provides a unified testing framework for all inline instructions,
//! eliminating the need for inline-specific test harnesses.

use crate::emulator::cpu::Cpu;
use crate::emulator::default_terminal::DefaultTerminal;
use crate::emulator::mmu::DRAM_BASE;
use crate::instruction::format::format_inline::FormatInline;
use crate::instruction::inline::INLINE;
use crate::instruction::test::TEST_MEMORY_CAPACITY;
use crate::instruction::{Instruction, RISCVTrace};

#[derive(Clone, Copy)]
pub enum RegisterMapping {
    Input,
    Input2,
    Output,
}

pub struct InlineMemoryLayout {
    pub input_base: u64,
    pub input_size: usize,
    pub input2_base: Option<u64>,
    pub input2_size: Option<usize>,
    pub output_base: u64,
    pub output_size: usize,
    // Register mappings: which memory region does each register point to
    pub rs1_mapping: RegisterMapping,
    pub rs2_mapping: RegisterMapping,
    pub rs3_mapping: Option<RegisterMapping>,
}

impl InlineMemoryLayout {
    /// Single input, single output with default mapping (rs1=output, rs2=input)
    /// Used by Sha2, Blake2, Blake3, Keccak256
    pub fn single_input(input_size: usize, output_size: usize) -> Self {
        Self {
            input_base: DRAM_BASE,
            input_size,
            input2_base: None,
            input2_size: None,
            output_base: DRAM_BASE + input_size as u64,
            output_size,
            rs1_mapping: RegisterMapping::Output,
            rs2_mapping: RegisterMapping::Input,
            rs3_mapping: None,
        }
    }

    /// Two inputs, single output for BigInt (rs1=input, rs2=input2, rs3=output)
    pub fn two_inputs(input_size: usize, input2_size: usize, output_size: usize) -> Self {
        Self {
            input_base: DRAM_BASE,
            input_size,
            input2_base: Some(DRAM_BASE + input_size as u64),
            input2_size: Some(input2_size),
            output_base: DRAM_BASE + (input_size + input2_size) as u64,
            output_size,
            rs1_mapping: RegisterMapping::Input,
            rs2_mapping: RegisterMapping::Input2,
            rs3_mapping: Some(RegisterMapping::Output),
        }
    }
}

// Standard register indices used by inline instructions
pub const INLINE_RS1: u8 = 10;
pub const INLINE_RS2: u8 = 11;
pub const INLINE_RS3: u8 = 12;

pub struct InlineTestHarness {
    pub cpu: Cpu,
    layout: InlineMemoryLayout,
}

impl InlineTestHarness {
    pub fn new(layout: InlineMemoryLayout) -> Self {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        Self { cpu, layout }
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

    fn get_address_for_mapping(&self, mapping: RegisterMapping) -> u64 {
        match mapping {
            RegisterMapping::Input => self.layout.input_base,
            RegisterMapping::Input2 => self
                .layout
                .input2_base
                .expect("Input2 mapping requires input2_base"),
            RegisterMapping::Output => self.layout.output_base,
        }
    }

    pub fn setup_registers(&mut self) {
        // Set up registers based on the layout's mappings
        self.cpu.x[INLINE_RS1 as usize] =
            self.get_address_for_mapping(self.layout.rs1_mapping) as i64;
        self.cpu.x[INLINE_RS2 as usize] =
            self.get_address_for_mapping(self.layout.rs2_mapping) as i64;

        if let Some(rs3_mapping) = self.layout.rs3_mapping {
            self.cpu.x[INLINE_RS3 as usize] = self.get_address_for_mapping(rs3_mapping) as i64;
        }
    }

    pub fn execute_inline(&mut self, instruction: INLINE) {
        instruction.trace(&mut self.cpu, None);
    }

    pub fn execute_sequence(&mut self, sequence: &[Instruction]) {
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
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    pub fn create_default_instruction(opcode: u32, funct3: u32, funct7: u32) -> INLINE {
        Self::create_instruction(opcode, funct3, funct7, INLINE_RS1, INLINE_RS2, INLINE_RS3)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test-only assertions")]
mod tests {
    use super::*;

    #[test]
    fn single_input_layout_round_trips_data_through_disjoint_regions() {
        let mut harness = InlineTestHarness::new(InlineMemoryLayout::single_input(32, 16));

        harness.load_input32(&[0x1111_2222, 0x3333_4444]);
        harness.load_state32(&[0xaaaa_bbbb, 0xcccc_dddd]);
        // State (output region) reads back unclobbered by the input words
        assert_eq!(harness.read_output32(2), vec![0xaaaa_bbbb, 0xcccc_dddd]);

        harness.load_state64(&[0x0102_0304_0506_0708]);
        assert_eq!(harness.read_output64(1), vec![0x0102_0304_0506_0708]);

        // The input region is exactly input_size bytes below the output region
        harness.load_input64(&[0x9999_8888_7777_6666]);
        let (input_value, _) = harness.cpu.mmu.load_doubleword(DRAM_BASE).unwrap();
        assert_eq!(input_value, 0x9999_8888_7777_6666);

        // rs1 -> output, rs2 -> input for the single-input convention
        harness.setup_registers();
        assert_eq!(harness.cpu.x[INLINE_RS1 as usize] as u64, DRAM_BASE + 32);
        assert_eq!(harness.cpu.x[INLINE_RS2 as usize] as u64, DRAM_BASE);
    }

    #[test]
    fn two_input_layout_maps_rs3_to_the_output_region() {
        let mut harness = InlineTestHarness::new(InlineMemoryLayout::two_inputs(16, 16, 8));
        harness.load_input2_32(&[0x5555_6666]);
        harness.load_input2_64(&[0x0a0b_0c0d_0e0f_1011]);
        let (value, _) = harness.cpu.mmu.load_doubleword(DRAM_BASE + 16).unwrap();
        assert_eq!(value, 0x0a0b_0c0d_0e0f_1011);

        harness.setup_registers();
        assert_eq!(harness.cpu.x[INLINE_RS1 as usize] as u64, DRAM_BASE);
        assert_eq!(harness.cpu.x[INLINE_RS2 as usize] as u64, DRAM_BASE + 16);
        assert_eq!(harness.cpu.x[INLINE_RS3 as usize] as u64, DRAM_BASE + 32);
    }

    #[test]
    fn execute_sequence_runs_plain_instructions_on_the_harness_cpu() {
        let mut harness = InlineTestHarness::new(InlineMemoryLayout::single_input(16, 16));
        // add x5, x10, x11
        let word = (11 << 20) | (10 << 15) | (5 << 7) | 0x33;
        let add = crate::instruction::Instruction::decode(word, DRAM_BASE, false).unwrap();
        harness.cpu.write_register(10, 20);
        harness.cpu.write_register(11, 22);
        harness.execute_sequence(&[add]);
        assert_eq!(harness.cpu.x[5], 42);
    }

    #[test]
    fn execute_inline_expands_the_registered_test_inline() {
        // TEST_INLINE_PROFILE (registered in instruction::inline::tests) emits
        // a VirtualROTRIW x5, rs1 with a bitmask selecting a 32-bit rotation:
        // rd receives the zero-extended low word of rs1.
        let mut harness = InlineTestHarness::new(InlineMemoryLayout::single_input(16, 16));
        harness.setup_registers();
        harness
            .cpu
            .write_register(INLINE_RS1 as usize, 0xdead_beef_1234_5678_u64 as i64);

        let instruction = InlineTestHarness::create_default_instruction(0x2b, 0x6, 0x7e);
        assert_eq!(instruction.opcode, 0x2b);
        assert_eq!(instruction.operands.rs1, INLINE_RS1);
        harness.execute_inline(instruction);

        assert_eq!(harness.cpu.x[5] as u64, 0x1234_5678);
    }
}
