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
        // Packing starts each segment at lane 0 of a word.
        layout.validate_io_segments()?;
        let io_mask_start = layout.remapped_word_address(layout.input_start)? as u128;
        let io_mask_end = layout.remapped_word_address(layout.output_end)? as u128;
        let io_num_vars = io_mask_end.next_power_of_two().max(1).ilog2() as usize;
        let mut segments = Vec::new();

        for (region, start, max, len, words) in [
            (
                "Input",
                layout.input_start,
                layout.max_input_size,
                public_io.inputs.len(),
                public_io.input_words_le(),
            ),
            (
                "Output",
                layout.output_start,
                layout.max_output_size,
                public_io.outputs.len(),
                public_io.output_words_le(),
            ),
        ] {
            // An over-long input would overlap the outputs' words, and an
            // over-long output would reach past `io_mask_end`, where the output
            // check binds no word.
            let len = u64::try_from(len).unwrap_or(u64::MAX);
            if len > max {
                return Err(MemoryLayoutError::IoTooLong { region, len, max });
            }
            if words.is_empty() {
                continue;
            }
            segments.push(PublicMemorySegment {
                start_index: layout.remapped_word_address(start)? as u128,
                words,
            });
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
    fn rejects_public_io_outside_its_segment() {
        let mut device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 8,
            max_untrusted_advice_size: 0,
            max_input_size: 16,
            max_output_size: 16,
            ..Default::default()
        });
        let layout = device.memory_layout.clone();
        device.outputs = vec![1; 17];
        assert_eq!(
            PublicIoMemory::new(&device),
            Err(MemoryLayoutError::IoTooLong {
                region: "Output",
                len: 17,
                max: 16
            })
        );

        device.outputs = vec![1];
        device.memory_layout.output_start = layout.trusted_advice_start;
        assert_eq!(
            PublicIoMemory::new(&device),
            Err(MemoryLayoutError::IoSegmentMisplaced {
                region: "Output",
                start: layout.trusted_advice_start,
                end: layout.output_end
            })
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
