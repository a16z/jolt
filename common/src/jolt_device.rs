use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::constants::{
    DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE, DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
    DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE, DEFAULT_MEMORY_SIZE, DEFAULT_STACK_SIZE, RAM_START_ADDRESS,
    STACK_CANARY_SIZE,
};

#[allow(clippy::too_long_first_doc_paragraph)]
/// Represented as a "peripheral device" in the RISC-V emulator, this captures
/// all reads from the reserved memory address space for program inputs and all writes
/// to the reserved memory address space for program outputs.
/// The inputs and outputs are part of the public inputs to the proof.
#[derive(
    Allocative,
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
    pub trusted_advice: Vec<u8>,
    pub untrusted_advice: Vec<u8>,
    pub outputs: Vec<u8>,
    pub panic: bool,
    pub memory_layout: MemoryLayout,
}

impl JoltDevice {
    pub fn new(memory_config: &MemoryConfig) -> Self {
        Self {
            inputs: Vec::new(),
            trusted_advice: Vec::new(),
            untrusted_advice: Vec::new(),
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
        } else if self.is_trusted_advice(address) {
            let internal_address = self.convert_trusted_advice_read_address(address);
            if self.trusted_advice.len() <= internal_address {
                0
            } else {
                self.trusted_advice[internal_address]
            }
        } else if self.is_untrusted_advice(address) {
            let internal_address = self.convert_untrusted_advice_read_address(address);
            if self.untrusted_advice.len() <= internal_address {
                0
            } else {
                self.untrusted_advice[internal_address]
            }
        } else if self.is_output(address) {
            let internal_address = self.convert_write_address(address);
            if self.outputs.len() <= internal_address {
                0
            } else {
                self.outputs[internal_address]
            }
        } else {
            assert!(address <= RAM_START_ADDRESS - 8);
            0 // zero-padding
        }
    }

    pub fn store(&mut self, address: u64, value: u8) {
        if address == self.memory_layout.panic {
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

    pub fn is_trusted_advice(&self, address: u64) -> bool {
        address >= self.memory_layout.trusted_advice_start
            && address < self.memory_layout.trusted_advice_end
    }

    pub fn is_untrusted_advice(&self, address: u64) -> bool {
        address >= self.memory_layout.untrusted_advice_start
            && address < self.memory_layout.untrusted_advice_end
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

    fn convert_trusted_advice_read_address(&self, address: u64) -> usize {
        (address - self.memory_layout.trusted_advice_start) as usize
    }

    fn convert_untrusted_advice_read_address(&self, address: u64) -> usize {
        (address - self.memory_layout.untrusted_advice_start) as usize
    }

    fn convert_write_address(&self, address: u64) -> usize {
        (address - self.memory_layout.output_start) as usize
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MemoryConfig {
    pub max_input_size: u64,
    pub max_trusted_advice_size: u64,
    pub max_untrusted_advice_size: u64,
    pub max_output_size: u64,
    pub stack_size: u64,
    pub memory_size: u64,
    pub program_size: Option<u64>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_input_size: DEFAULT_MAX_INPUT_SIZE,
            max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
            max_output_size: DEFAULT_MAX_OUTPUT_SIZE,
            stack_size: DEFAULT_STACK_SIZE,
            memory_size: DEFAULT_MEMORY_SIZE,
            program_size: None,
        }
    }
}

#[derive(
    Allocative,
    Default,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct MemoryLayout {
    /// The total size of the elf's sections, including the .text, .data, .rodata, and .bss sections.
    pub program_size: u64,
    pub max_trusted_advice_size: u64,
    pub trusted_advice_start: u64,
    pub trusted_advice_end: u64,
    pub max_untrusted_advice_size: u64,
    pub untrusted_advice_start: u64,
    pub untrusted_advice_end: u64,
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub input_start: u64,
    pub input_end: u64,
    pub output_start: u64,
    pub output_end: u64,
    pub stack_size: u64,
    /// Stack starts from (RAM_START_ADDRESS + `program_size` + `stack_size`) and grows in descending addresses by `stack_size` bytes.
    pub stack_end: u64,
    pub memory_size: u64,
    /// Heap starts just after the start of the stack and is `memory_size` bytes.
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
            .field("program_size", &self.program_size)
            .field("max_input_size", &self.max_input_size)
            .field("max_trusted_advice_size", &self.max_trusted_advice_size)
            .field("max_untrusted_advice_size", &self.max_untrusted_advice_size)
            .field("max_output_size", &self.max_output_size)
            .field(
                "trusted_advice_start",
                &format_args!("{:#X}", self.trusted_advice_start),
            )
            .field(
                "trusted_advice_end",
                &format_args!("{:#X}", self.trusted_advice_end),
            )
            .field(
                "untrusted_advice_start",
                &format_args!("{:#X}", self.untrusted_advice_start),
            )
            .field(
                "untrusted_advice_end",
                &format_args!("{:#X}", self.untrusted_advice_end),
            )
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
            .field("io_end", &format_args!("{:#X}", self.io_end))
            .finish()
    }
}

impl MemoryLayout {
    pub fn new(config: &MemoryConfig) -> Self {
        assert!(
            config.program_size.is_some(),
            "MemoryLayout requires bytecode size to be set"
        );

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
        } // Must be 8-byte aligned

        let max_trusted_advice_size = align_up(config.max_trusted_advice_size, 8);
        let max_untrusted_advice_size = align_up(config.max_untrusted_advice_size, 8);
        let max_input_size = align_up(config.max_input_size, 8);
        let max_output_size = align_up(config.max_output_size, 8);
        let stack_size = align_up(config.stack_size, 8);
        let memory_size = align_up(config.memory_size, 8);

        // Critical for ValEvaluation and ValFinal sumchecks in RAM
        assert!(
            max_trusted_advice_size.is_power_of_two() || max_trusted_advice_size == 0,
            "Trusted advice size must be a power of two (got {max_trusted_advice_size})",
        );
        assert!(
            max_untrusted_advice_size.is_power_of_two() || max_untrusted_advice_size == 0,
            "Untrusted advice size must be a power of two (got {max_untrusted_advice_size})",
        );

        // Adds 16 to account for panic bit and termination bit
        // (they each occupy one full 8-byte word)
        let io_region_bytes = max_input_size
            .checked_add(max_trusted_advice_size)
            .and_then(|s| s.checked_add(max_untrusted_advice_size))
            .and_then(|s| s.checked_add(max_output_size))
            .and_then(|s| s.checked_add(16))
            .expect("I/O region size overflow");

        // Padded so that the witness index corresponding to `input_start`
        // has the form 0b11...100...0
        let io_region_words = (io_region_bytes / 8).next_power_of_two();
        // let io_region_words = (io_region_bytes / 8 + 1).next_power_of_two() - 1;

        let io_bytes = io_region_words
            .checked_mul(8)
            .expect("I/O region byte count overflow");

        // Place the larger or equal-sized advice region first in memory (at the lower address).
        let (
            trusted_advice_start,
            trusted_advice_end,
            untrusted_advice_start,
            untrusted_advice_end,
        ) = if max_trusted_advice_size >= max_untrusted_advice_size {
            // Trusted advice goes first
            let trusted_start = RAM_START_ADDRESS
                .checked_sub(io_bytes)
                .expect("I/O region exceeds RAM_START_ADDRESS");
            let trusted_end = trusted_start
                .checked_add(max_trusted_advice_size)
                .expect("trusted_advice_end overflow");
            let untrusted_start = trusted_end;
            let untrusted_end = untrusted_start
                .checked_add(max_untrusted_advice_size)
                .expect("untrusted_advice_end overflow");
            (trusted_start, trusted_end, untrusted_start, untrusted_end)
        } else {
            // Untrusted advice goes first
            let untrusted_start = RAM_START_ADDRESS
                .checked_sub(io_bytes)
                .expect("I/O region exceeds RAM_START_ADDRESS");
            let untrusted_end = untrusted_start
                .checked_add(max_untrusted_advice_size)
                .expect("untrusted_advice_end overflow");
            let trusted_start = untrusted_end;
            let trusted_end = trusted_start
                .checked_add(max_trusted_advice_size)
                .expect("trusted_advice_end overflow");
            (trusted_start, trusted_end, untrusted_start, untrusted_end)
        };

        let input_start = core::cmp::max(untrusted_advice_end, trusted_advice_end);
        let input_end = input_start
            .checked_add(max_input_size)
            .expect("input_end overflow");
        let output_start = input_end;
        let output_end = output_start
            .checked_add(max_output_size)
            .expect("output_end overflow");
        let panic = output_end;
        let termination = panic.checked_add(8).expect("termination overflow");
        let io_end = termination.checked_add(8).expect("io_end overflow");

        let program_size = config.program_size.unwrap();

        // stack grows downwards (decreasing addresses) from the bytecode_end + stack_size up to bytecode_end
        let stack_end = RAM_START_ADDRESS
            .checked_add(program_size)
            .expect("stack_end overflow");
        let stack_start = stack_end
            .checked_add(stack_size)
            .expect("stack_start overflow");

        // heap grows *up* (increasing addresses) from the stack of the stack
        let memory_end = stack_start
            .checked_add(memory_size)
            .expect("memory_end overflow");

        Self {
            program_size,
            max_trusted_advice_size,
            max_untrusted_advice_size,
            max_input_size,
            max_output_size,
            trusted_advice_start,
            trusted_advice_end,
            untrusted_advice_start,
            untrusted_advice_end,
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

    /// Returns the start address memory.
    pub fn get_lowest_address(&self) -> u64 {
        self.trusted_advice_start.min(self.untrusted_advice_start)
    }

    /// Returns the total emulator memory
    pub fn get_total_memory_size(&self) -> u64 {
        self.memory_size + self.stack_size + STACK_CANARY_SIZE
    }
}
