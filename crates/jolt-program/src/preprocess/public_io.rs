use common::{
    constants::RAM_START_ADDRESS,
    jolt_device::{JoltDevice, MemoryLayoutError},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicMemorySegment {
    pub start_index: u128,
    pub words: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicIoMemory {
    pub segments: Vec<PublicMemorySegment>,
    pub io_mask_start: u128,
    pub io_mask_end: u128,
    io_num_vars: usize,
}

impl PublicIoMemory {
    pub fn from_segments(
        segments: Vec<PublicMemorySegment>,
        io_mask_start: u128,
        io_mask_end: u128,
    ) -> Self {
        let io_num_vars = io_mask_end.next_power_of_two().max(1).ilog2() as usize;
        Self {
            segments,
            io_mask_start,
            io_mask_end,
            io_num_vars,
        }
    }

    pub fn new(public_io: &JoltDevice) -> Result<Self, MemoryLayoutError> {
        let layout = &public_io.memory_layout;
        let io_mask_start = layout.remapped_word_address(layout.input_start)? as u128;
        let io_mask_end = layout.remapped_word_address(RAM_START_ADDRESS)? as u128;
        let mut segments = Vec::new();

        if !public_io.inputs.is_empty() {
            segments.push(PublicMemorySegment {
                start_index: layout.remapped_word_address(layout.input_start)? as u128,
                words: public_io.input_words_le(),
            });
        }

        if !public_io.outputs.is_empty() {
            segments.push(PublicMemorySegment {
                start_index: layout.remapped_word_address(layout.output_start)? as u128,
                words: public_io.output_words_le(),
            });
        }

        segments.push(PublicMemorySegment {
            start_index: layout.remapped_word_address(layout.panic)? as u128,
            words: vec![public_io.panic as u64],
        });

        if !public_io.panic {
            segments.push(PublicMemorySegment {
                start_index: layout.remapped_word_address(layout.termination)? as u128,
                words: vec![1],
            });
        }

        Ok(Self::from_segments(segments, io_mask_start, io_mask_end))
    }

    pub fn io_num_vars(&self) -> usize {
        self.io_num_vars
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "tests should fail loudly")]

    use super::*;
    use common::jolt_device::{JoltDevice, MemoryConfig};

    #[test]
    fn materializes_public_io_segments() {
        let mut device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 16,
            max_output_size: 16,
            ..Default::default()
        });
        device.inputs = vec![42];
        device.outputs = vec![7];

        let memory = PublicIoMemory::new(&device).expect("public IO memory should materialize");

        assert_eq!(memory.segments.len(), 4);
        assert_eq!(memory.segments[0].words, vec![42]);
        assert_eq!(memory.segments[1].words, vec![7]);
        assert_eq!(memory.segments[2].words, vec![0]);
        assert_eq!(memory.segments[3].words, vec![1]);
        assert!(memory.io_mask_start < memory.io_mask_end);
    }

    #[test]
    fn omits_termination_word_on_panic() {
        let mut device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            ..Default::default()
        });
        device.panic = true;

        let memory = PublicIoMemory::new(&device).expect("public IO memory should materialize");

        assert_eq!(memory.segments.len(), 1);
        assert_eq!(memory.segments[0].words, vec![1]);
    }
}
