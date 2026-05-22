#[cfg(feature = "serialization")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::BYTES_PER_INSTRUCTION;
use common::jolt_device::{JoltDevice, MemoryLayoutError};

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
                start_index: layout.remapped_word_address(ram.min_bytecode_address)? as usize,
                words: ram.bytecode_words.clone(),
            });
        }

        if !public_io.inputs.is_empty() {
            segments.push(PublicMemorySegment {
                start_index: layout.remapped_word_address(layout.input_start)? as usize,
                words: public_io.input_words_le(),
            });
        }

        Ok(Self { segments })
    }
}

#[cfg(test)]
mod tests {
    use super::{PublicInitialRam, RAMPreprocessing};
    use common::jolt_device::{JoltDevice, MemoryConfig};

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
}
