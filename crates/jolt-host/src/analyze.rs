//! Program trace analysis.

use std::{collections::BTreeMap, fs::File, io, path::PathBuf};

use serde::{Deserialize, Serialize};
use tracer::{
    instruction::{Cycle, Instruction},
    JoltDevice,
};

#[derive(Serialize, Deserialize)]
pub struct ProgramSummary {
    pub trace: Vec<Cycle>,
    pub bytecode: Vec<Instruction>,
    pub memory_init: Vec<(u64, u8)>,
    pub io_device: JoltDevice,
}

impl ProgramSummary {
    pub fn trace_len(&self) -> usize {
        self.trace.len()
    }

    /// Count instructions by type, sorted descending by frequency.
    pub fn analyze(&self) -> Vec<(&'static str, usize)> {
        let mut counts = BTreeMap::<&'static str, usize>::new();
        for cycle in &self.trace {
            let instruction_name: &'static str = cycle.into();
            *counts.entry(instruction_name).or_insert(0) += 1;
        }

        let mut counts: Vec<_> = counts.into_iter().collect();
        counts.sort_by_key(|v| std::cmp::Reverse(v.1));
        counts
    }

    pub fn write_to_file(self, path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        let data = bincode::serialize(&self)?;
        io::Write::write_all(&mut file, &data)?;
        Ok(())
    }
}
