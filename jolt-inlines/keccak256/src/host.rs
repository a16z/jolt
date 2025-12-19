//! Host-side implementation and registration.
pub use crate::sequence_builder;

use crate::{INLINE_OPCODE, KECCAK256_FUNCT3, KECCAK256_FUNCT7, KECCAK256_NAME};
use tracer::register_inline;

use tracer::utils::inline_sequence_writer::{
    write_inline_trace, AppendMode, InlineDescriptor, SequenceInputs,
};

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        INLINE_OPCODE,
        KECCAK256_FUNCT3,
        KECCAK256_FUNCT7,
        KECCAK256_NAME,
        std::boxed::Box::new(sequence_builder::keccak256_inline_sequence_builder),
        None,
    )?;

    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    let inline_info = InlineDescriptor::new(
        KECCAK256_NAME.to_string(),
        INLINE_OPCODE,
        KECCAK256_FUNCT3,
        KECCAK256_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions =
        sequence_builder::keccak256_inline_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "keccak256_trace.joltinline",
        &inline_info,
        &inputs,
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

    if std::env::var("STORE_INLINE").unwrap_or_default() == "true" {
        if let Err(e) = store_inlines() {
            eprintln!("Failed to store Keccak256 inline traces: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{BufRead, BufReader};
    use std::path::Path;

    use tracer::utils::inline_sequence_writer::{
        DEFAULT_RAM_START_ADDRESS, DEFAULT_RS1, DEFAULT_RS2, DEFAULT_RS3,
    };

    use crate::sequence_builder::keccak256_inline_sequence_builder;

    #[test]
    fn test_keccak256_trace_file_matches_generated() {
        use tracer::utils::inline_sequence_writer::{
            write_inline_trace, AppendMode, InlineDescriptor, SequenceInputs,
        };

        let inputs = SequenceInputs::default();
        // Generate the instructions
        let generated_instructions =
            keccak256_inline_sequence_builder((&inputs).into(), (&inputs).into());

        let test_file_name = format!("keccak256_trace_test_{}.joltinline", std::process::id());
        let trace_file_path = Path::new(&test_file_name);

        let _cleanup = FileCleanup(trace_file_path);

        let inline_info = InlineDescriptor::new(
            crate::KECCAK256_NAME.to_string(),
            crate::INLINE_OPCODE,
            crate::KECCAK256_FUNCT3,
            crate::KECCAK256_FUNCT7,
        );
        let sequence_inputs = SequenceInputs::default();
        write_inline_trace(
            &test_file_name,
            &inline_info,
            &sequence_inputs,
            &generated_instructions,
            AppendMode::Overwrite,
        )
        .expect("Failed to write test trace file");

        let file = fs::File::open(trace_file_path).expect("Failed to open test trace file");
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let header = lines
            .next()
            .expect("Missing header line")
            .expect("Failed to read header");
        assert!(
            header.contains("KECCAK256_INLINE"),
            "Expected KECCAK256_INLINE in header, got: {header}"
        );

        let inputs_line = lines
            .next()
            .expect("Missing inputs line")
            .expect("Failed to read inputs");
        assert!(inputs_line.contains("address: $ADDR"));

        for (i, generated_instr) in generated_instructions.iter().enumerate() {
            let mut trace_line = lines.next().unwrap().unwrap();
            trace_line = trace_line.replace(
                "address: $ADDR",
                &format!("address: {DEFAULT_RAM_START_ADDRESS:#x}"),
            );
            trace_line = trace_line.replace("$RS1", &format!("{DEFAULT_RS1}"));
            trace_line = trace_line.replace("$RS2", &format!("{DEFAULT_RS2}"));
            trace_line = trace_line.replace("$RS3", &format!("{DEFAULT_RS3}"));

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

        assert!(
            lines.next().is_none(),
            "Trace file has extra lines after all instructions"
        );
    }

    struct FileCleanup<'a>(&'a Path);

    impl<'a> Drop for FileCleanup<'a> {
        fn drop(&mut self) {
            if self.0.exists() {
                let _ = fs::remove_file(self.0);
            }
        }
    }
}
