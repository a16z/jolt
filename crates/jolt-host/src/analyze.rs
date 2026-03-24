//! Program trace analysis.

use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use tracer::instruction::{Cycle, Instruction};
use tracer::JoltDevice;

#[derive(Serialize, Deserialize)]
pub struct ProgramSummary {
    pub trace: Vec<Cycle>,
    pub bytecode: Vec<Instruction>,
    pub memory_init: Vec<(u64, u8)>,
    pub io_device: JoltDevice,
}

impl ProgramSummary {
    /// Returns the number of cycles in the execution trace.
    pub fn trace_len(&self) -> usize {
        self.trace.len()
    }

    /// Count instructions by type, sorted descending by frequency.
    pub fn analyze(&self) -> Vec<(&'static str, usize)> {
        let mut counts = BTreeMap::<&'static str, usize>::new();
        for cycle in &self.trace {
            let instruction_name: &'static str = cycle.into();
            *counts.entry(instruction_name).or_default() += 1;
        }

        let mut counts: Vec<_> = counts.into_iter().collect();
        counts.sort_by_key(|v| std::cmp::Reverse(v.1));
        counts
    }

    /// Serialize this summary to a file using bincode.
    ///
    /// Bincode encoding errors are mapped to [`io::Error`] since they indicate
    /// a serialization failure indistinguishable from an I/O fault at this level.
    pub fn write_to_file(&self, path: &Path) -> Result<(), io::Error> {
        let mut file = File::create(path)?;
        let data = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(io::Error::other)?;
        io::Write::write_all(&mut file, &data)?;
        Ok(())
    }
}
