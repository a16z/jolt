//! Host-side implementation and registration.
pub use crate::sequence_builder;

use jolt_inlines_common::constants;
use tracer::register_inline;

use jolt_inlines_common::trace_writer::{
    write_inline_trace, AppendMode, InlineDescriptor, SequenceInputs,
};

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        constants::INLINE_OPCODE,
        constants::keccak256::FUNCT3,
        constants::keccak256::FUNCT7,
        constants::keccak256::NAME,
        std::boxed::Box::new(sequence_builder::keccak256_inline_sequence_builder),
    )?;

    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    let inline_info = InlineDescriptor::new(
        constants::keccak256::NAME.to_string(),
        constants::INLINE_OPCODE,
        constants::keccak256::FUNCT3,
        constants::keccak256::FUNCT7,
    );
    let sequence_inputs = SequenceInputs::default();
    let instructions = sequence_builder::keccak256_inline_sequence_builder(
        sequence_inputs.address,
        sequence_inputs.is_compressed,
        sequence_inputs.xlen,
        sequence_inputs.rs1,
        sequence_inputs.rs2,
        sequence_inputs.rs3,
    );
    write_inline_trace(
        "keccak256_trace.joltinline",
        &inline_info,
        &sequence_inputs,
        &instructions,
        AppendMode::Overwrite,
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        tracing::error!("Failed to register Keccak256 inlines: {e}");
    }

    if let Err(e) = store_inlines() {
        eprintln!("Failed to store Keccak256 inline traces: {e}");
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{BufRead, BufReader};
    use std::path::Path;

    use jolt_inlines_common::trace_writer::{
        DEFAULT_RAM_START_ADDRESS, DEFAULT_RS1, DEFAULT_RS2, DEFAULT_RS3, DEFAULT_XLEN,
    };

    use crate::sequence_builder::keccak256_inline_sequence_builder;

    /// Test that verifies the trace writer correctly generates a trace file with placeholders
    /// and that the placeholders can be replaced to match the original instructions.
    ///
    /// This test uses the keccak256 inline instruction as an example.
    #[test]
    fn test_keccak256_trace_file_matches_generated() {
        // Generate the instructions
        let generated_instructions = keccak256_inline_sequence_builder(
            DEFAULT_RAM_START_ADDRESS,
            false,
            DEFAULT_XLEN,
            DEFAULT_RS1,
            DEFAULT_RS2,
            DEFAULT_RS3,
        );

        // Path to the keccak256 trace file
        let trace_file_path = Path::new("keccak256_trace.joltinline");

        // Ensure the file is deleted at the end of the test
        let _cleanup = FileCleanup(trace_file_path);

        // Now read and validate the file
        let file = fs::File::open(trace_file_path).expect("Failed to open test trace file");
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read and validate header (line 1)
        let header = lines
            .next()
            .expect("Missing header line")
            .expect("Failed to read header");
        assert!(
            header.contains("KECCAK256_INLINE"),
            "Expected KECCAK256_INLINE in header, got: {header}"
        );

        // Read and validate sequence inputs (line 2)
        let inputs_line = lines
            .next()
            .expect("Missing inputs line")
            .expect("Failed to read inputs");
        assert!(inputs_line.contains("address: $ADDR"));

        // Now compare each instruction line by line
        for (i, generated_instr) in generated_instructions.iter().enumerate() {
            // Read next line from trace file
            let mut trace_line = lines.next().unwrap().unwrap();
            trace_line = trace_line.replace(
                "address: $ADDR",
                &format!("address: {DEFAULT_RAM_START_ADDRESS:#x}"),
            );
            trace_line = trace_line.replace("$RS1", &format!("{DEFAULT_RS1}"));
            trace_line = trace_line.replace("$RS2", &format!("{DEFAULT_RS2}"));
            trace_line = trace_line.replace("$RS3", &format!("{DEFAULT_RS3}"));

            // Compare
            assert_eq!(
                format!("{generated_instr:?}"),
                trace_line,
                "Instruction mismatch at line {} (index {}). Generated: {}, Trace: {}",
                i + 3,
                i,
                format!("{:?}", generated_instr),
                trace_line
            );
        }

        // Check if there are any extra lines in the trace file
        assert!(
            lines.next().is_none(),
            "Trace file has extra lines after all instructions"
        );
    }

    /// Helper struct to ensure test file cleanup on drop
    struct FileCleanup<'a>(&'a Path);

    impl<'a> Drop for FileCleanup<'a> {
        fn drop(&mut self) {
            if self.0.exists() {
                let _ = fs::remove_file(self.0);
            }
        }
    }
}
