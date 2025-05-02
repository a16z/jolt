use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

use crate::constants::{RAM_START_ADDRESS, REGISTER_COUNT};

#[allow(clippy::too_long_first_doc_paragraph)]
/// Represented as a "peripheral device" in the RISC-V emulator, this captures
/// all reads from the reserved memory address space for program inputs and all writes
/// to the reserved memory address space for program outputs.
/// The inputs and outputs are part of the public inputs to the proof.
#[derive(
    Default,
    Debug,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct JoltDevice {
    pub inputs: Vec<u8>,
    pub outputs: Vec<u8>,
    pub panic: bool,
    pub memory_layout: MemoryLayout,
}

impl JoltDevice {
    pub fn new(max_input_size: u64, max_output_size: u64) -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            panic: false,
            memory_layout: MemoryLayout::new(max_input_size, max_output_size),
        }
    }

    pub fn load(&self, address: u64) -> u8 {
        if self.is_panic(address) {
            self.panic as u8
        } else if self.is_termination(address) {
            0 // Termination bit should never be loaded after it is set
        } else if self.is_input(address) {
            let internal_address = self.convert_read_address(address);
            if self.inputs.len() <= internal_address {
                0
            } else {
                self.inputs[internal_address]
            }
        } else if self.is_output(address) {
            let internal_address = self.convert_write_address(address);
            if self.outputs.len() <= internal_address {
                0
            } else {
                self.outputs[internal_address]
            }
        } else {
            assert!(address <= RAM_START_ADDRESS - 4);
            0 // zero-padding
        }
    }

    pub fn store(&mut self, address: u64, value: u8) {
        if address == self.memory_layout.panic {
            println!("GUEST PANIC");
            self.panic = true;
            return;
        }

        if address == self.memory_layout.termination {
            return;
        }

        let internal_address = self.convert_write_address(address);
        if self.outputs.len() <= internal_address {
            if value != 0 {
                self.outputs.resize(internal_address + 1, 0);
                self.outputs[internal_address] = value;
            }
        } else {
            self.outputs[internal_address] = value;
        }
    }

    pub fn size(&self) -> usize {
        self.inputs.len() + self.outputs.len()
    }

    pub fn is_input(&self, address: u64) -> bool {
        address >= self.memory_layout.input_start && address < self.memory_layout.input_end
    }

    pub fn is_output(&self, address: u64) -> bool {
        address >= self.memory_layout.output_start && address < self.memory_layout.termination
    }

    pub fn is_panic(&self, address: u64) -> bool {
        address == self.memory_layout.panic
    }

    pub fn is_termination(&self, address: u64) -> bool {
        address == self.memory_layout.termination
    }

    fn convert_read_address(&self, address: u64) -> usize {
        (address - self.memory_layout.input_start) as usize
    }

    fn convert_write_address(&self, address: u64) -> usize {
        (address - self.memory_layout.output_start) as usize
    }
}

#[derive(
    Default,
    Debug,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct MemoryLayout {
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub input_start: u64,
    pub input_end: u64,
    pub output_start: u64,
    pub output_end: u64,
    pub panic: u64,
    pub termination: u64,
}

impl MemoryLayout {
    pub fn new(mut max_input_size: u64, mut max_output_size: u64) -> Self {
        // Must be word-aligned
        max_input_size = max_input_size.next_multiple_of(4);
        max_output_size = max_output_size.next_multiple_of(4);

        // Adds 8 to account for panic bit and termination bit
        // (they each occupy one full 4-byte word)
        let io_region_num_bytes = max_input_size + max_output_size + 8;

        // Padded so that the witness index corresponding to `RAM_START_ADDRESS`
        // is a power of 2
        let io_region_num_words =
            (REGISTER_COUNT + io_region_num_bytes / 4).next_power_of_two() - REGISTER_COUNT;
        let input_start = RAM_START_ADDRESS - io_region_num_words * 4;
        let input_end = input_start + max_input_size;
        let output_start = input_end;
        let output_end = output_start + max_output_size;
        let panic = output_end;
        let termination = panic + 4;

        Self {
            max_input_size,
            max_output_size,
            input_start,
            input_end,
            output_start,
            output_end,
            panic,
            termination,
        }
    }
}
