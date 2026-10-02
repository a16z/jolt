#[cfg(feature = "std")]
use allocative::Allocative;
#[cfg(feature = "std")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use core::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::constants::{
    DEFAULT_HEAP_SIZE, DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE,
    DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE, DEFAULT_STACK_SIZE,
    RAM_START_ADDRESS, STACK_CANARY_SIZE,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayoutError {
    ZeroAddress,
    AddressBelowLowest { address: u64, lowest_address: u64 },
    MissingProgramSize,
    SizeOverflow { region: &'static str },
    InvalidTrustedAdviceSize { size: u64 },
    InvalidUntrustedAdviceSize { size: u64 },
    IoRegionTooLarge { padded_bytes: u64 },
}

impl Display for MemoryLayoutError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::ZeroAddress => write!(f, "cannot remap the zero address"),
            Self::AddressBelowLowest {
                address,
                lowest_address,
            } => write!(
                f,
                "address {address} is below lowest mapped address {lowest_address}"
            ),
            Self::MissingProgramSize => write!(f, "MemoryLayout requires bytecode size to be set"),
            Self::SizeOverflow { region } => write!(f, "{region} size or address overflow"),
            Self::InvalidTrustedAdviceSize { size } => write!(
                f,
                "Trusted advice size must be a power of two (got {size})"
            ),
            Self::InvalidUntrustedAdviceSize { size } => write!(
                f,
                "Untrusted advice size must be a power of two (got {size})"
            ),
            Self::IoRegionTooLarge { padded_bytes } => write!(
                f,
                "padded I/O region ({padded_bytes} bytes) reaches the zero address or exceeds RAM_START_ADDRESS"
            ),
        }
    }
}

impl Error for MemoryLayoutError {}

#[expect(
    clippy::too_long_first_doc_paragraph,
    reason = "pre-existing doc paragraph exceeds the pedantic limit"
)]
/// Represented as a "peripheral device" in the RISC-V emulator, this captures
/// all reads from the reserved memory address space for program inputs and all writes
/// to the reserved memory address space for program outputs.
/// The inputs and outputs are part of the public inputs to the proof.
///
/// The advice fields are *not* public: they hold the prover's private inputs,
/// populated so the emulator can service loads from the advice memory regions.
/// The verifier never reads them (advice is bound through polynomial
/// commitments), so both serialization impls emit them as empty to keep
/// private bytes out of any serialized device (e.g. a host publishing
/// `program_io` alongside a proof). Deserialization still accepts populated
/// advice fields, so pre-existing blobs decode unchanged.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "std", derive(Allocative, CanonicalDeserialize))]
pub struct JoltDevice {
    pub inputs: Vec<u8>,
    /// Private prover input; serialized as empty (see struct docs).
    #[serde(serialize_with = "serialize_advice_stripped")]
    pub trusted_advice: Vec<u8>,
    /// Private prover input; serialized as empty (see struct docs).
    #[serde(serialize_with = "serialize_advice_stripped")]
    pub untrusted_advice: Vec<u8>,
    pub outputs: Vec<u8>,
    pub panic: bool,
    pub memory_layout: MemoryLayout,
}

/// Serializes an advice field as an empty byte vector, byte-compatible with
/// the derived impl for a device whose advice is empty.
fn serialize_advice_stripped<S: serde::Serializer>(
    _advice: &[u8],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    Vec::<u8>::new().serialize(serializer)
}

/// Mirrors the derived impl except that the advice fields are written as
/// empty vectors — advice bytes are private inputs and must not leave the
/// host in a serialized device.
#[cfg(feature = "std")]
impl CanonicalSerialize for JoltDevice {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        let empty_advice = Vec::<u8>::new();
        self.inputs.serialize_with_mode(&mut writer, compress)?;
        empty_advice.serialize_with_mode(&mut writer, compress)?;
        empty_advice.serialize_with_mode(&mut writer, compress)?;
        self.outputs.serialize_with_mode(&mut writer, compress)?;
        self.panic.serialize_with_mode(&mut writer, compress)?;
        self.memory_layout
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        let empty_advice = Vec::<u8>::new();
        self.inputs.serialized_size(compress)
            + 2 * empty_advice.serialized_size(compress)
            + self.outputs.serialized_size(compress)
            + self.panic.serialized_size(compress)
            + self.memory_layout.serialized_size(compress)
    }
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
            self.inputs.get(internal_address).copied().unwrap_or(0)
        } else if self.is_trusted_advice(address) {
            let internal_address = self.convert_trusted_advice_read_address(address);
            self.trusted_advice
                .get(internal_address)
                .copied()
                .unwrap_or(0)
        } else if self.is_untrusted_advice(address) {
            let internal_address = self.convert_untrusted_advice_read_address(address);
            self.untrusted_advice
                .get(internal_address)
                .copied()
                .unwrap_or(0)
        } else if self.is_output(address) {
            let internal_address = self.convert_write_address(address);
            self.outputs.get(internal_address).copied().unwrap_or(0)
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
        let max_output_size =
            (self.memory_layout.output_end - self.memory_layout.output_start) as usize;
        assert!(
            internal_address < max_output_size,
            "Output too long: guest wrote {} bytes, max is {} bytes (set by MemoryConfig.max_output_size).",
            internal_address + 1,
            max_output_size,
        );
        if self.outputs.len() <= internal_address {
            self.outputs.resize(internal_address + 1, 0);
        }
        #[expect(
            clippy::indexing_slicing,
            reason = "the resize above guarantees internal_address < outputs.len()"
        )]
        {
            self.outputs[internal_address] = value;
        }
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

    pub fn input_words_le(&self) -> Vec<u64> {
        bytes_to_words_le(&self.inputs)
    }

    pub fn output_words_le(&self) -> Vec<u64> {
        bytes_to_words_le(&self.outputs)
    }
}

pub fn bytes_to_words_le(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks(8)
        .map(|chunk| {
            let mut value = 0u64;
            for (index, byte) in chunk.iter().enumerate() {
                value |= u64::from(*byte) << (8 * index);
            }
            value
        })
        .collect()
}

#[derive(Debug, Copy, Clone)]
pub struct MemoryConfig {
    pub max_input_size: u64,
    pub max_trusted_advice_size: u64,
    pub max_untrusted_advice_size: u64,
    pub max_output_size: u64,
    pub stack_size: u64,
    pub heap_size: u64,
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
            heap_size: DEFAULT_HEAP_SIZE,
            program_size: None,
        }
    }
}

#[derive(Default, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "std",
    derive(Allocative, CanonicalSerialize, CanonicalDeserialize)
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
    pub stack_end: u64,
    pub heap_size: u64,
    pub heap_end: u64,
    pub panic: u64,
    pub termination: u64,
    /// End of the memory region containing inputs, outputs, the panic bit,
    /// and the termination bit.
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
            .field("heap_size", &format_args!("{:#X}", self.heap_size))
            .field("heap_end", &format_args!("{:#X}", self.heap_end))
            .field("panic", &format_args!("{:#X}", self.panic))
            .field("termination", &format_args!("{:#X}", self.termination))
            .field("io_end", &format_args!("{:#X}", self.io_end))
            .finish()
    }
}

impl MemoryLayout {
    /// Constructs a layout, panicking if the configuration is invalid.
    /// Host preprocessing should use [`Self::try_new`] to report configuration errors.
    #[expect(
        clippy::panic,
        reason = "the existing infallible constructor retains its documented panic contract; host preprocessing uses try_new"
    )]
    pub fn new(config: &MemoryConfig) -> Self {
        Self::try_new(config).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Constructs a checked layout without allocating guest memory.
    /// Advice capacities are zero or powers of two after eight-byte alignment.
    /// The padded I/O region must leave its lowest address above zero.
    pub fn try_new(config: &MemoryConfig) -> Result<Self, MemoryLayoutError> {
        let program_size = config
            .program_size
            .ok_or(MemoryLayoutError::MissingProgramSize)?;

        #[inline]
        fn align_up(val: u64, region: &'static str) -> Result<u64, MemoryLayoutError> {
            match val % 8 {
                0 => Ok(val),
                rem => val
                    .checked_add(8 - rem)
                    .ok_or(MemoryLayoutError::SizeOverflow { region }),
            }
        }

        let max_trusted_advice_size = align_up(config.max_trusted_advice_size, "trusted advice")?;
        let max_untrusted_advice_size =
            align_up(config.max_untrusted_advice_size, "untrusted advice")?;
        let max_input_size = align_up(config.max_input_size, "input")?;
        let max_output_size = align_up(config.max_output_size, "output")?;
        let stack_size = align_up(config.stack_size, "stack")?;
        let heap_size = align_up(config.heap_size, "heap")?;

        // Critical for ValEvaluation and ValFinal sumchecks in RAM
        if max_trusted_advice_size != 0 && !max_trusted_advice_size.is_power_of_two() {
            return Err(MemoryLayoutError::InvalidTrustedAdviceSize {
                size: max_trusted_advice_size,
            });
        }
        if max_untrusted_advice_size != 0 && !max_untrusted_advice_size.is_power_of_two() {
            return Err(MemoryLayoutError::InvalidUntrustedAdviceSize {
                size: max_untrusted_advice_size,
            });
        }

        // Adds 16 to account for panic bit and termination bit
        // (they each occupy one full 8-byte word)
        let io_region_bytes = max_input_size
            .checked_add(max_trusted_advice_size)
            .and_then(|s| s.checked_add(max_untrusted_advice_size))
            .and_then(|s| s.checked_add(max_output_size))
            .and_then(|s| s.checked_add(16))
            .ok_or(MemoryLayoutError::SizeOverflow { region: "I/O" })?;

        // Padded so that the witness index corresponding to `input_start`
        // has the form 0b11...100...0
        let io_region_words = (io_region_bytes / 8)
            .checked_next_power_of_two()
            .ok_or(MemoryLayoutError::SizeOverflow { region: "I/O" })?;

        let io_bytes = io_region_words
            .checked_mul(8)
            .ok_or(MemoryLayoutError::SizeOverflow { region: "I/O" })?;

        // Zero is the no-access sentinel. Power-of-two padding therefore keeps
        // admitted I/O in the upper half below RAM, above emulator peripherals.
        let io_start = RAM_START_ADDRESS
            .checked_sub(io_bytes)
            .filter(|start| *start != 0)
            .ok_or(MemoryLayoutError::IoRegionTooLarge {
                padded_bytes: io_bytes,
            })?;

        // Place the larger or equal-sized advice region first in memory (at the lower address).
        let (
            trusted_advice_start,
            trusted_advice_end,
            untrusted_advice_start,
            untrusted_advice_end,
        ) = if max_trusted_advice_size >= max_untrusted_advice_size {
            // Trusted advice goes first
            let trusted_start = io_start;
            let trusted_end = trusted_start.checked_add(max_trusted_advice_size).ok_or(
                MemoryLayoutError::SizeOverflow {
                    region: "trusted advice",
                },
            )?;
            let untrusted_start = trusted_end;
            let untrusted_end = untrusted_start
                .checked_add(max_untrusted_advice_size)
                .ok_or(MemoryLayoutError::SizeOverflow {
                    region: "untrusted advice",
                })?;
            (trusted_start, trusted_end, untrusted_start, untrusted_end)
        } else {
            // Untrusted advice goes first
            let untrusted_start = io_start;
            let untrusted_end = untrusted_start
                .checked_add(max_untrusted_advice_size)
                .ok_or(MemoryLayoutError::SizeOverflow {
                    region: "untrusted advice",
                })?;
            let trusted_start = untrusted_end;
            let trusted_end = trusted_start.checked_add(max_trusted_advice_size).ok_or(
                MemoryLayoutError::SizeOverflow {
                    region: "trusted advice",
                },
            )?;
            (trusted_start, trusted_end, untrusted_start, untrusted_end)
        };

        let input_start = core::cmp::max(untrusted_advice_end, trusted_advice_end);
        let input_end = input_start
            .checked_add(max_input_size)
            .ok_or(MemoryLayoutError::SizeOverflow { region: "input" })?;
        let output_start = input_end;
        let output_end = output_start
            .checked_add(max_output_size)
            .ok_or(MemoryLayoutError::SizeOverflow { region: "output" })?;
        let panic = output_end;
        let termination = panic
            .checked_add(8)
            .ok_or(MemoryLayoutError::SizeOverflow {
                region: "termination",
            })?;
        let io_end = termination
            .checked_add(8)
            .ok_or(MemoryLayoutError::SizeOverflow { region: "I/O" })?;

        // stack grows downwards (decreasing addresses) from the top of the stack down to stack_end
        let stack_end = RAM_START_ADDRESS
            .checked_add(program_size)
            .ok_or(MemoryLayoutError::SizeOverflow { region: "program" })?;
        let stack_start = stack_end
            .checked_add(STACK_CANARY_SIZE)
            .and_then(|s| s.checked_add(stack_size))
            .ok_or(MemoryLayoutError::SizeOverflow { region: "stack" })?;

        // heap grows *up* (increasing addresses) from the top of the stack
        let heap_end = stack_start
            .checked_add(heap_size)
            .ok_or(MemoryLayoutError::SizeOverflow { region: "heap" })?;

        Ok(Self {
            program_size,
            max_trusted_advice_size,
            trusted_advice_start,
            trusted_advice_end,
            max_untrusted_advice_size,
            untrusted_advice_start,
            untrusted_advice_end,
            max_input_size,
            max_output_size,
            input_start,
            input_end,
            output_start,
            output_end,
            stack_size,
            stack_end,
            heap_size,
            heap_end,
            panic,
            termination,
            io_end,
        })
    }

    /// Returns the start address memory.
    pub fn get_lowest_address(&self) -> u64 {
        self.trusted_advice_start.min(self.untrusted_advice_start)
    }

    pub fn remap_word_address(&self, address: u64) -> Result<Option<u64>, MemoryLayoutError> {
        if address == 0 {
            return Ok(None);
        }

        let lowest_address = self.get_lowest_address();
        if address >= lowest_address {
            Ok(Some((address - lowest_address) / 8))
        } else {
            Err(MemoryLayoutError::AddressBelowLowest {
                address,
                lowest_address,
            })
        }
    }

    pub fn remapped_word_address(&self, address: u64) -> Result<u64, MemoryLayoutError> {
        self.remap_word_address(address)?
            .ok_or(MemoryLayoutError::ZeroAddress)
    }

    /// Returns the total emulator memory (program + canary + stack + heap).
    pub fn get_total_memory_size(&self) -> u64 {
        self.heap_end - RAM_START_ADDRESS
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn checked_layout_admits_large_advice_and_rejects_zero_address_overlap() {
        let config = MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 1 << 29,
            max_untrusted_advice_size: 1 << 28,
            ..Default::default()
        };
        let layout = MemoryLayout::try_new(&config).unwrap();
        assert_eq!(layout.get_lowest_address(), 1 << 30);
        assert_eq!(layout.trusted_advice_end, (1 << 30) + (1 << 29));
        assert_eq!(layout.untrusted_advice_start, layout.trusted_advice_end);
        assert_eq!(layout.max_untrusted_advice_size, 1 << 28);
        assert!(layout.io_end <= RAM_START_ADDRESS);

        assert_eq!(
            MemoryLayout::try_new(&MemoryConfig {
                max_untrusted_advice_size: 1 << 29,
                ..config
            }),
            Err(MemoryLayoutError::IoRegionTooLarge {
                padded_bytes: 1 << 31
            })
        );
    }

    #[test]
    fn checked_layout_reports_arithmetic_overflow_without_allocation() {
        let config = MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        };
        for invalid in [
            MemoryConfig {
                max_input_size: u64::MAX,
                ..config
            },
            MemoryConfig {
                max_input_size: u64::MAX - 7,
                max_output_size: 0,
                max_trusted_advice_size: 0,
                max_untrusted_advice_size: 0,
                ..config
            },
            MemoryConfig {
                max_trusted_advice_size: 1 << 63,
                ..config
            },
            MemoryConfig {
                program_size: Some(u64::MAX),
                ..config
            },
            MemoryConfig {
                heap_size: u64::MAX - 7,
                ..config
            },
        ] {
            assert!(matches!(
                MemoryLayout::try_new(&invalid),
                Err(MemoryLayoutError::SizeOverflow { .. })
            ));
        }
    }

    #[test]
    #[should_panic(expected = "Output too long")]
    fn panics_when_output_exceeds_max() {
        let memory_config = MemoryConfig {
            program_size: Some(1024),
            max_output_size: 8,
            ..Default::default()
        };
        let mut device = JoltDevice::new(&memory_config);
        // Use io_end which bypasses panic/termination early returns
        // but still lands past the output region in convert_write_address
        let overflow_address = device.memory_layout.io_end;
        device.store(overflow_address, 0x42);
    }

    #[test]
    fn packs_public_io_bytes_into_little_endian_words() {
        let mut device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        });
        device.inputs = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        device.outputs = vec![0xaa, 0xbb];

        assert_eq!(device.input_words_le(), vec![0x0807_0605_0403_0201, 9]);
        assert_eq!(device.output_words_le(), vec![0xbbaa]);
    }

    fn device_with_advice() -> JoltDevice {
        let mut device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        });
        device.inputs = vec![1, 2, 3];
        device.trusted_advice = vec![0xde, 0xad];
        device.untrusted_advice = vec![0xbe, 0xef, 0x42];
        device.outputs = vec![7, 8];
        device.panic = true;
        device
    }

    #[cfg(feature = "std")]
    #[test]
    fn canonical_serialization_strips_advice() {
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

        let device = device_with_advice();
        let mut public_device = device.clone();
        public_device.trusted_advice.clear();
        public_device.untrusted_advice.clear();

        let mut bytes = Vec::new();
        device.serialize_compressed(&mut bytes).unwrap();
        assert_eq!(bytes.len(), device.compressed_size());

        let mut public_bytes = Vec::new();
        public_device
            .serialize_compressed(&mut public_bytes)
            .unwrap();
        assert_eq!(bytes, public_bytes);

        let roundtrip = JoltDevice::deserialize_compressed(bytes.as_slice()).unwrap();
        assert_eq!(roundtrip, public_device);
    }

    #[test]
    fn serde_serialization_strips_advice() {
        let device = device_with_advice();
        let mut public_device = device.clone();
        public_device.trusted_advice.clear();
        public_device.untrusted_advice.clear();

        let bytes = bincode::serde::encode_to_vec(&device, bincode::config::standard()).unwrap();
        let public_bytes =
            bincode::serde::encode_to_vec(&public_device, bincode::config::standard()).unwrap();
        assert_eq!(bytes, public_bytes);

        let (roundtrip, _): (JoltDevice, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(roundtrip, public_device);
    }

    #[test]
    fn layout_packs_io_regions_contiguously_below_ram_start() {
        // trusted (4096) < untrusted (8192) forces the untrusted-first branch.
        // io_region_bytes = 4096 + 8192 + 4096 + 4096 + 16 = 20496 bytes
        //   => 2562 words => padded to 4096 words => 32768 bytes below RAM_START.
        let layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 4096,
            max_untrusted_advice_size: 8192,
            max_input_size: 4096,
            max_output_size: 4096,
            ..Default::default()
        });

        let region_start = RAM_START_ADDRESS - 32768;
        assert_eq!(layout.untrusted_advice_start, region_start);
        assert_eq!(layout.untrusted_advice_end, region_start + 8192);
        assert_eq!(layout.trusted_advice_start, region_start + 8192);
        assert_eq!(layout.trusted_advice_end, region_start + 8192 + 4096);
        assert_eq!(layout.input_start, layout.trusted_advice_end);
        assert_eq!(layout.input_end, layout.input_start + 4096);
        assert_eq!(layout.output_start, layout.input_end);
        assert_eq!(layout.output_end, layout.output_start + 4096);
        assert_eq!(layout.panic, layout.output_end);
        assert_eq!(layout.termination, layout.panic + 8);
        assert_eq!(layout.io_end, layout.termination + 8);
        assert_eq!(layout.get_lowest_address(), region_start);
    }

    #[test]
    fn layout_places_trusted_advice_first_when_not_smaller() {
        let layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 4096,
            max_untrusted_advice_size: 4096,
            ..Default::default()
        });
        assert!(layout.trusted_advice_start < layout.untrusted_advice_start);
        assert_eq!(layout.trusted_advice_end, layout.untrusted_advice_start);
    }

    #[test]
    fn layout_aligns_sizes_up_to_eight_bytes_and_sizes_ram_regions() {
        let layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(1000),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 10,
            max_output_size: 9,
            stack_size: 100,
            heap_size: 12,
        });
        assert_eq!(layout.max_input_size, 16);
        assert_eq!(layout.max_output_size, 16);
        assert_eq!(layout.stack_size, 104);
        assert_eq!(layout.heap_size, 16);

        // Stack grows down from stack_start; the canary sits between the
        // program image and the stack.
        assert_eq!(layout.stack_end, RAM_START_ADDRESS + 1000);
        let stack_start = layout.stack_end + STACK_CANARY_SIZE + 104;
        assert_eq!(layout.heap_end, stack_start + 16);
        assert_eq!(
            layout.get_total_memory_size(),
            layout.heap_end - RAM_START_ADDRESS
        );
    }

    #[test]
    #[should_panic(expected = "Trusted advice size must be a power of two")]
    fn layout_rejects_non_power_of_two_trusted_advice() {
        let _ = MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 24,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "Untrusted advice size must be a power of two")]
    fn layout_rejects_non_power_of_two_untrusted_advice() {
        let _ = MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            max_untrusted_advice_size: 40,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "MemoryLayout requires bytecode size to be set")]
    fn layout_requires_program_size() {
        let _ = MemoryLayout::new(&MemoryConfig::default());
    }

    #[test]
    fn remaps_word_addresses_relative_to_lowest_reserved_address() {
        let device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        });
        let layout = &device.memory_layout;
        let lowest = layout.get_lowest_address();

        assert_eq!(layout.remap_word_address(0), Ok(None));
        assert_eq!(
            layout.remapped_word_address(0),
            Err(MemoryLayoutError::ZeroAddress)
        );
        assert_eq!(layout.remapped_word_address(lowest), Ok(0));
        assert_eq!(layout.remapped_word_address(lowest + 16), Ok(2));
        assert_eq!(
            layout.remapped_word_address(lowest - 8),
            Err(MemoryLayoutError::AddressBelowLowest {
                address: lowest - 8,
                lowest_address: lowest,
            })
        );
    }
}
