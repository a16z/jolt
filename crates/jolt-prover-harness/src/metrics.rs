use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::HarnessResult;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunMetrics {
    pub time_ms: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub proof_size_bytes: Option<u64>,
}

impl RunMetrics {
    pub const fn new(
        time_ms: Option<f64>,
        peak_rss_bytes: Option<u64>,
        proof_size_bytes: Option<u64>,
    ) -> Self {
        Self {
            time_ms,
            peak_rss_bytes,
            proof_size_bytes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrontierMetrics {
    pub frontier: String,
    pub features: Vec<String>,
    pub fixture: String,
    pub size: Option<u64>,
    pub samples: u32,
    pub core: RunMetrics,
    pub modular: RunMetrics,
    pub optimization_ids: Vec<String>,
}

impl FrontierMetrics {
    pub fn to_json_pretty(&self) -> HarnessResult<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn write_json(&self, path: &Path) -> HarnessResult<()> {
        std::fs::write(path, self.to_json_pretty()?)?;
        Ok(())
    }
}

#[cfg(feature = "perf")]
pub fn current_peak_rss_bytes() -> Option<u64> {
    memory_stats::memory_stats().map(|stats| stats.physical_mem as u64)
}

#[cfg(not(feature = "perf"))]
pub const fn current_peak_rss_bytes() -> Option<u64> {
    None
}
