use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

use crate::constants::{
    DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE, DEFAULT_MEMORY_SIZE, DEFAULT_STACK_SIZE,
    RAM_START_ADDRESS,
};

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
    pub fn new(memory_config: &MemoryConfig) -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            panic: false,
            memory_layout: MemoryLayout::new(memory_config),
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
        } else if self.is_panic(address) || self.is_termination(address) {
            return;
        }

        let internal_address = self.convert_write_address(address);
        if self.outputs.len() <= internal_address {
            self.outputs.resize(internal_address + 1, 0);
        }
        self.outputs[internal_address] = value;
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
        address >= self.memory_layout.panic && address < self.memory_layout.termination
    }

    pub fn is_termination(&self, address: u64) -> bool {
        address >= self.memory_layout.termination && address < self.memory_layout.io_end
    }

    fn convert_read_address(&self, address: u64) -> usize {
        (address - self.memory_layout.input_start) as usize
    }

    fn convert_write_address(&self, address: u64) -> usize {
        (address - self.memory_layout.output_start) as usize
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MemoryConfig {
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub stack_size: u64,
    pub memory_size: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_input_size: DEFAULT_MAX_INPUT_SIZE,
            max_output_size: DEFAULT_MAX_OUTPUT_SIZE,
            stack_size: DEFAULT_STACK_SIZE,
            memory_size: DEFAULT_MEMORY_SIZE,
        }
    }
}

#[derive(
    Default, Clone, PartialEq, Serialize, Deserialize, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct MemoryLayout {
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub input_start: u64,
    pub input_end: u64,
    pub output_start: u64,
    pub output_end: u64,
    pub stack_size: u64,
    /// Stack starts at the IO inputs and goes "down" from there by `stack_size` bytes.
    pub stack_end: u64,
    pub memory_size: u64,
    /// Heap starts at RAM_START_ADDRESS and is `memory_size` bytes.
    pub memory_end: u64,
    pub panic: u64,
    pub termination: u64,
    /// End of the memory region containing inputs, outputs, the panic bit,
    /// and the termination bit
    pub io_end: u64,
}

impl core::fmt::Debug for MemoryLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MemoryLayout")
            .field("max_input_size", &self.max_input_size)
            .field("max_output_size", &self.max_output_size)
            .field("input_start", &format_args!("{:#X}", self.input_start))
            .field("input_end", &format_args!("{:#X}", self.input_end))
            .field("output_start", &format_args!("{:#X}", self.output_start))
            .field("output_end", &format_args!("{:#X}", self.output_end))
            .field("stack_size", &format_args!("{:#X}", self.stack_size))
            .field("stack_end", &format_args!("{:#X}", self.stack_end))
            .field("memory_size", &format_args!("{:#X}", self.memory_size))
            .field("memory_end", &format_args!("{:#X}", self.memory_end))
            .field("panic", &format_args!("{:#X}", self.panic))
            .field("termination", &format_args!("{:#X}", self.termination))
            .finish()
    }
}

impl MemoryLayout {
    pub fn new(config: &MemoryConfig) -> Self {
        // helper to align ‘val’ *up* to a multiple of ‘align’, panicking on overflow
        #[inline]
        fn align_up(val: u64, align: u64) -> u64 {
            if align == 0 {
                val
            } else {
                match val % align {
                    0 => val,
                    rem => {
                        // panics if val + (align - rem) overflows
                        val.checked_add(align - rem).expect("alignment overflow")
                    }
                }
            }
        } // Must be word-aligned

        let max_input_size = align_up(config.max_input_size, 4);
        let max_output_size = align_up(config.max_output_size, 4);
        let stack_size = align_up(config.stack_size, 4);
        let memory_size = align_up(config.memory_size, 4);

        // Adds 8 to account for panic bit and termination bit
        // (they each occupy one full 4-byte word)
        let io_region_bytes = max_input_size
            .checked_add(max_output_size)
            .and_then(|s| s.checked_add(8))
            .expect("I/O region size overflow");

        // Padded so that the witness index corresponding to `input_start`
        // has the form 0b11...100...0
        let io_region_words = (io_region_bytes / 4).next_power_of_two();
        // let io_region_words = (io_region_bytes / 4 + 1).next_power_of_two() - 1;

        let io_bytes = io_region_words
            .checked_mul(4)
            .expect("I/O region byte count overflow");
        let input_start = RAM_START_ADDRESS
            .checked_sub(io_bytes)
            .expect("I/O region exceeds RAM_START_ADDRESS");
        let input_end = input_start
            .checked_add(max_input_size)
            .expect("input_end overflow");
        let output_start = input_end;
        let output_end = output_start
            .checked_add(max_output_size)
            .expect("output_end overflow");
        let panic = output_end;
        let termination = panic.checked_add(4).expect("termination overflow");
        let io_end = termination.checked_add(4).expect("io_end overflow");

        // stack grows *down* from input_start
        let stack_end = input_start
            .checked_sub(stack_size)
            .expect("stack region exceeds I/O region");

        // heap grows *up* from RAM_START_ADDRESS
        let memory_end = RAM_START_ADDRESS
            .checked_add(memory_size)
            .expect("memory_end overflow");

        Self {
            max_input_size,
            max_output_size,
            input_start,
            input_end,
            output_start,
            output_end,
            stack_size,
            stack_end,
            memory_size,
            memory_end,
            panic,
            termination,
            io_end,
        }
    }
}
