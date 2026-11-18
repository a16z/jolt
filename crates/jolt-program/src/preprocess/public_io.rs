use common::jolt_device::{JoltDevice, MemoryLayoutError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicMemorySegment {
    pub start_index: u128,
    pub words: Vec<u64>,
}

/// The public I/O words the RAM output check binds. Its mask
/// `[io_mask_start, io_mask_end)` covers the words of
/// `[input_start, output_end)`, the inputs and outputs, and holds every
/// segment; it is empty for a layout without input or output capacity. The
/// check leaves the final values of other words, the I/O gap among them, free.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicIoMemory {
    pub segments: Vec<PublicMemorySegment>,
    pub io_mask_start: u128,
    pub io_mask_end: u128,
    io_num_vars: usize,
}

impl PublicIoMemory {
    pub fn new(public_io: &JoltDevice) -> Result<Self, MemoryLayoutError> {
        let layout = &public_io.memory_layout;
        let io_mask_start = layout.remapped_word_address(layout.input_start)? as u128;
        let io_mask_end = layout.remapped_word_address(layout.output_end)? as u128;
        let io_num_vars = io_mask_end.next_power_of_two().max(1).ilog2() as usize;
        let mut segments = Vec::new();

        for (region, start, len, words) in [
            (
                "Input",
                layout.input_start,
                public_io.inputs.len(),
                public_io.input_words_le(),
            ),
            (
                "Output",
                layout.output_start,
                public_io.outputs.len(),
                public_io.output_words_le(),
            ),
        ] {
            if words.is_empty() {
                continue;
            }
            let start_index = layout.remapped_word_address(start)? as u128;
            // The output check binds no word outside the mask, and the
            // verifier's ValIo evaluation aliases a word at or past
            // `2^io_num_vars`.
            if start_index < io_mask_start || start_index + words.len() as u128 > io_mask_end {
                return Err(MemoryLayoutError::IoOutsideOutputCheck {
                    region,
                    start,
                    len: u64::try_from(len).unwrap_or(u64::MAX),
                    mask_start: layout.input_start,
                    mask_end: layout.output_end,
                });
            }
            segments.push(PublicMemorySegment { start_index, words });
        }

        Ok(Self {
            segments,
            io_mask_start,
            io_mask_end,
            io_num_vars,
        })
    }

    pub fn io_num_vars(&self) -> usize {
        self.io_num_vars
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "tests should fail loudly")]
    #![expect(clippy::indexing_slicing, reason = "tests index fixture data")]

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

        assert_eq!(memory.segments.len(), 2);
        assert_eq!(memory.segments[0].words, vec![42]);
        assert_eq!(memory.segments[1].words, vec![7]);
        // Inputs, outputs and one I/O gap word pad to an eight-word region:
        // inputs are words 0-1, outputs 2-3 and the I/O gap 4-7.
        assert_eq!(
            (
                memory.io_mask_start,
                memory.io_mask_end,
                memory.io_num_vars()
            ),
            (0, 4, 2)
        );
    }

    #[test]
    fn rejects_public_io_outside_the_mask() {
        let mut device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 8,
            max_untrusted_advice_size: 0,
            max_input_size: 16,
            max_output_size: 16,
            ..Default::default()
        });
        let layout = device.memory_layout.clone();
        let outside = |start, len| MemoryLayoutError::IoOutsideOutputCheck {
            region: "Output",
            start,
            len,
            mask_start: layout.input_start,
            mask_end: layout.output_end,
        };
        device.outputs = vec![1; 17];
        assert_eq!(
            PublicIoMemory::new(&device),
            Err(outside(layout.output_start, 17))
        );

        device.outputs = vec![1];
        device.memory_layout.output_start = layout.trusted_advice_start;
        assert_eq!(
            PublicIoMemory::new(&device),
            Err(outside(layout.trusted_advice_start, 1))
        );
    }

    /// The panic claim is not memory: product lane 3 binds it.
    #[test]
    fn panic_claim_adds_no_public_memory() {
        let mut device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        });
        device.inputs = vec![42];
        device.outputs = vec![7];
        let success = PublicIoMemory::new(&device).expect("public IO memory should materialize");
        device.panic = true;
        let panic = PublicIoMemory::new(&device).expect("public IO memory should materialize");

        assert_eq!(success, panic);
    }
}
