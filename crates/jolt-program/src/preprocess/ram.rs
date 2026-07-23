#[cfg(feature = "serialization")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS};
use common::jolt_device::{JoltDevice, MemoryLayout, MemoryLayoutError};

use super::public_io::PublicMemorySegment;

#[derive(Default, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(
        CanonicalSerialize,
        CanonicalDeserialize,
        serde::Serialize,
        serde::Deserialize
    )
)]
pub struct RAMPreprocessing {
    pub min_bytecode_address: u64,
    pub bytecode_words: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RamDomainError {
    #[error("memory layout error: {0}")]
    MemoryLayout(MemoryLayoutError),
    #[error("memory layout heap end {heap_end} is below lowest mapped address {lowest_address}")]
    HeapBelowLowest { heap_end: u64, lowest_address: u64 },
    #[error("RAM domain size exceeds this platform's addressable usize range")]
    DomainTooLarge,
}

impl From<MemoryLayoutError> for RamDomainError {
    fn from(error: MemoryLayoutError) -> Self {
        Self::MemoryLayout(error)
    }
}

impl RAMPreprocessing {
    pub fn preprocess(memory_init: Vec<(u64, u8)>) -> Self {
        let min_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .min()
            .unwrap_or(0);

        let max_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .max()
            .unwrap_or(0)
            + (BYTES_PER_INSTRUCTION as u64 - 1);

        let num_words = max_bytecode_address.div_ceil(8) - min_bytecode_address / 8 + 1;
        let mut bytecode_words = vec![0u64; num_words as usize];

        for chunk in
            memory_init.chunk_by(|(address_a, _), (address_b, _)| address_a / 8 == address_b / 8)
        {
            let mut word = [0u8; 8];
            for (address, byte) in chunk {
                word[(address % 8) as usize] = *byte;
            }
            let word = u64::from_le_bytes(word);
            let remapped_index = (chunk[0].0 / 8 - min_bytecode_address / 8) as usize;
            bytecode_words[remapped_index] = word;
        }

        Self {
            min_bytecode_address,
            bytecode_words,
        }
    }
}

pub fn compute_min_ram_k(
    min_bytecode_address: u64,
    program_image_len_words: usize,
    memory_layout: &MemoryLayout,
) -> Result<usize, RamDomainError> {
    let bytecode_start = match memory_layout.remap_word_address(min_bytecode_address)? {
        Some(address) => usize::try_from(address).map_err(|_| RamDomainError::DomainTooLarge)?,
        None => 0,
    };
    let bytecode_end = bytecode_start
        .checked_add(program_image_len_words)
        .ok_or(RamDomainError::DomainTooLarge)?;

    let io_end = usize::try_from(memory_layout.remapped_word_address(RAM_START_ADDRESS)?)
        .map_err(|_| RamDomainError::DomainTooLarge)?;

    bytecode_end
        .max(io_end)
        .checked_next_power_of_two()
        .ok_or(RamDomainError::DomainTooLarge)
}

pub fn compute_max_ram_k(memory_layout: &MemoryLayout) -> Result<usize, RamDomainError> {
    let lowest_address = memory_layout.get_lowest_address();
    let total_bytes = memory_layout.heap_end.checked_sub(lowest_address).ok_or(
        RamDomainError::HeapBelowLowest {
            heap_end: memory_layout.heap_end,
            lowest_address,
        },
    )?;
    let total_words =
        usize::try_from(total_bytes / 8).map_err(|_| RamDomainError::DomainTooLarge)?;
    total_words
        .checked_next_power_of_two()
        .ok_or(RamDomainError::DomainTooLarge)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicInitialRam {
    pub segments: Vec<PublicMemorySegment>,
}

impl PublicInitialRam {
    pub fn new(ram: &RAMPreprocessing, public_io: &JoltDevice) -> Result<Self, MemoryLayoutError> {
        let layout = &public_io.memory_layout;
        let mut segments = Vec::new();

        if !ram.bytecode_words.is_empty() {
            segments.push(PublicMemorySegment {
                start_index: layout.remapped_word_address(ram.min_bytecode_address)? as u128,
                words: ram.bytecode_words.clone(),
            });
        }

        segments.extend(Self::inputs_only(public_io)?.segments);

        Ok(Self { segments })
    }

    /// Public initial RAM without the program image, used in committed program
    /// mode where the image contribution is a staged opening instead.
    pub fn inputs_only(public_io: &JoltDevice) -> Result<Self, MemoryLayoutError> {
        let layout = &public_io.memory_layout;
        let mut segments = Vec::new();

        if !public_io.inputs.is_empty() {
            segments.push(PublicMemorySegment {
                start_index: layout.remapped_word_address(layout.input_start)? as u128,
                words: public_io.input_words_le(),
            });
        }

        Ok(Self { segments })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compute_max_ram_k, compute_min_ram_k, PublicInitialRam, RAMPreprocessing, RamDomainError,
    };
    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayoutError};

    /// Layout with io_end at word index 8 and lowest address 64 bytes below
    /// RAM start (16-byte input + 16-byte output + 16 control bytes, padded
    /// to 8 words).
    fn bounded_device() -> JoltDevice {
        JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 16,
            max_output_size: 16,
            stack_size: 8,
            heap_size: 8,
        })
    }

    #[test]
    fn preprocesses_memory_bytes_into_words() {
        let preprocessing = RAMPreprocessing::preprocess(vec![
            (0x8000_0000, 0x01),
            (0x8000_0001, 0x02),
            (0x8000_0008, 0x03),
        ]);

        assert_eq!(preprocessing.min_bytecode_address, 0x8000_0000);
        assert_eq!(preprocessing.bytecode_words[0], 0x0201);
        assert_eq!(preprocessing.bytecode_words[1], 0x03);
    }

    #[test]
    fn materializes_public_initial_ram_segments() {
        let mut device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 16,
            ..Default::default()
        });
        device.inputs = vec![0x2a, 0, 0, 0, 0, 0, 0, 0, 0x07];
        let preprocessing = RAMPreprocessing {
            min_bytecode_address: 0x8000_0000,
            bytecode_words: vec![0x0201, 0x03],
        };

        let initial = PublicInitialRam::new(&preprocessing, &device);
        assert!(
            initial.is_ok(),
            "initial RAM should materialize: {initial:?}"
        );
        let Ok(initial) = initial else {
            return;
        };

        assert_eq!(initial.segments.len(), 2);
        assert_eq!(initial.segments[0].words, vec![0x0201, 0x03]);
        assert_eq!(initial.segments[1].words, vec![0x2a, 0x07]);
    }

    #[test]
    fn computes_ram_domain_bounds_from_layout() {
        let device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 16,
            max_output_size: 16,
            stack_size: 8,
            heap_size: 8,
        });
        assert_eq!(
            compute_min_ram_k(0x8000_0000, 7, &device.memory_layout),
            Ok(16)
        );
        assert_eq!(compute_max_ram_k(&device.memory_layout), Ok(256));
    }

    #[test]
    fn compute_max_ram_k_rejects_heap_end_below_lowest_address() {
        let mut layout = bounded_device().memory_layout;
        let lowest_address = layout.get_lowest_address();
        layout.heap_end = lowest_address - 8;
        assert_eq!(
            compute_max_ram_k(&layout),
            Err(RamDomainError::HeapBelowLowest {
                heap_end: lowest_address - 8,
                lowest_address,
            })
        );
    }

    #[test]
    fn compute_max_ram_k_accepts_heap_end_equal_to_lowest_address() {
        let mut layout = bounded_device().memory_layout;
        layout.heap_end = layout.get_lowest_address();
        // zero words round up to a domain of size 1, not an error
        assert_eq!(compute_max_ram_k(&layout), Ok(1));
    }

    #[test]
    fn compute_min_ram_k_overflow_guards_return_domain_too_large() {
        let device = bounded_device();
        let layout = &device.memory_layout;
        let lowest_address = layout.get_lowest_address();
        // bytecode_start (word 1) + usize::MAX image words overflows the sum
        assert_eq!(
            compute_min_ram_k(lowest_address + 8, usize::MAX, layout),
            Err(RamDomainError::DomainTooLarge)
        );
        // 2^63 + 1 words has no representable next power of two
        assert_eq!(
            compute_min_ram_k(0, (usize::MAX >> 1) + 2, layout),
            Err(RamDomainError::DomainTooLarge)
        );
    }

    #[test]
    fn compute_min_ram_k_maps_zero_bytecode_address_to_word_zero() {
        let device = bounded_device();
        // remap_word_address(0) is None, so the image occupies words 0..len:
        // 9 image words beat the io_end word index of 8 and round up to 16
        assert_eq!(compute_min_ram_k(0, 9, &device.memory_layout), Ok(16));
        // with no image words, io_end alone determines the domain
        assert_eq!(compute_min_ram_k(0, 0, &device.memory_layout), Ok(8));
    }

    #[test]
    fn compute_min_ram_k_propagates_below_lowest_address_error() {
        let device = bounded_device();
        let lowest_address = device.memory_layout.get_lowest_address();
        assert_eq!(
            compute_min_ram_k(lowest_address - 8, 1, &device.memory_layout),
            Err(RamDomainError::MemoryLayout(
                MemoryLayoutError::AddressBelowLowest {
                    address: lowest_address - 8,
                    lowest_address,
                }
            ))
        );
    }
}
