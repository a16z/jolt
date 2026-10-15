use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
};

use common::jolt_device::JoltDevice;
use jolt_program::execution::TraceData;
use jolt_riscv::JoltInstructionRow;
use serde::Serialize;

/// Full execution analysis, retaining source rows and any field payloads.
#[derive(Debug, Serialize)]
pub struct ProgramSummary {
    pub trace: Arc<TraceData>,
    pub bytecode: Vec<JoltInstructionRow>,
    pub memory_init: Vec<(u64, u8)>,
    pub io_device: JoltDevice,
}

impl ProgramSummary {
    pub fn trace_len(&self) -> usize {
        self.trace.len()
    }

    pub fn analyze(&self) -> Vec<(&'static str, usize)> {
        let mut counts = HashMap::<&'static str, usize>::new();
        for row in self.trace.rows() {
            let instruction_name = row.instruction().instruction_kind.name();
            if let Some(count) = counts.get(instruction_name) {
                let _ = counts.insert(instruction_name, count + 1);
            } else {
                let _ = counts.insert(instruction_name, 1);
            }
        }

        let mut counts: Vec<_> = counts.into_iter().collect();
        counts.sort_by_key(|v| v.1);
        counts.reverse();

        counts
    }

    pub fn write_to_file(self, path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = BufWriter::new(File::create(path)?);
        let _ =
            bincode::serde::encode_into_std_write(&self, &mut file, bincode::config::standard())?;
        file.flush()?;
        Ok(())
    }
}
