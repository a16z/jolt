use std::sync::Arc;

use sysinfo::{Pid, System};

use super::{AbstractObjective, Direction, MeasurementError, ObjectiveEntry};
use crate::{ProverPreprocessing, TestCase};

inventory::submit! {
    ObjectiveEntry {
        name: "peak_rss",
        direction: Direction::Minimize,
        needs_guest: true,
        build: |s, inputs| { let setup = s.unwrap(); Box::new(PeakRssObjective::new(
            setup.test_case.clone(), setup.prover_preprocessing.clone(), inputs,
            )) },
    }
}

/// Measures peak resident set size (RSS) during proving.
///
/// Uses the `sysinfo` crate to sample memory before and after proving.
/// For more accurate results, run in an isolated process.
pub struct PeakRssObjective {
    pub test_case: Arc<TestCase>,
    pub prover_preprocessing: Arc<ProverPreprocessing>,
    pub inputs: Vec<u8>,
}

impl PeakRssObjective {
    pub fn new(
        test_case: Arc<TestCase>,
        prover_preprocessing: Arc<ProverPreprocessing>,
        inputs: Vec<u8>,
    ) -> Self {
        Self {
            test_case,
            prover_preprocessing,
            inputs,
        }
    }
}

impl AbstractObjective for PeakRssObjective {
    fn name(&self) -> &str {
        "peak_rss"
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let pid = Pid::from_u32(std::process::id());
        let mut sys = System::new();

        sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let rss_before = sys.process(pid).map(|p| p.memory()).unwrap_or(0);

        let (_proof, _io) = self
            .test_case
            .prove(&self.prover_preprocessing, &self.inputs);

        sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let rss_after = sys.process(pid).map(|p| p.memory()).unwrap_or(0);

        // Report peak RSS in megabytes
        let peak_mb = rss_after.max(rss_before) as f64 / (1024.0 * 1024.0);
        Ok(peak_mb)
    }

    fn recommended_samples(&self) -> usize {
        1
    }

    fn regression_threshold(&self) -> Option<f64> {
        Some(0.10)
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }
}
