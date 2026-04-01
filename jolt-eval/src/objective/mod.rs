pub mod guest_cycles;
pub mod inline_lengths;
pub mod optimize;
pub mod peak_rss;
pub mod proof_size;
pub mod prover_time;
pub mod verifier_time;
pub mod wrapping_cost;

use std::collections::HashMap;
use std::fmt;

use crate::SharedSetup;

/// Whether lower or higher values are better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Minimize,
    Maximize,
}

/// Error during objective measurement.
#[derive(Debug, Clone)]
pub struct MeasurementError {
    pub message: String,
}

impl fmt::Display for MeasurementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for MeasurementError {}

impl MeasurementError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Core objective trait for measurable properties.
pub trait AbstractObjective: Send + Sync {
    fn name(&self) -> &str;

    /// Take a single measurement and return its scalar value.
    fn collect_measurement(&self) -> Result<f64, MeasurementError>;

    /// How many samples to take for statistical significance.
    fn recommended_samples(&self) -> usize {
        1
    }

    /// What threshold is considered a regression (e.g. 0.05 = 5% slowdown).
    fn regression_threshold(&self) -> Option<f64> {
        None
    }

    fn direction(&self) -> Direction;
}

/// Registration entry for the [`inventory`] crate.
///
/// Each built-in objective module calls `inventory::submit!` with one of
/// these, so all objectives are discoverable at runtime.
pub struct ObjectiveEntry {
    pub name: &'static str,
    pub direction: Direction,
    pub build: fn(&SharedSetup, Vec<u8>) -> Box<dyn AbstractObjective>,
}
inventory::collect!(ObjectiveEntry);

/// Iterate all objective entries registered via `inventory`.
pub fn registered_objectives() -> impl Iterator<Item = &'static ObjectiveEntry> {
    inventory::iter::<ObjectiveEntry>()
}

/// Build all registered objectives from a [`SharedSetup`].
pub fn build_objectives_from_inventory(
    setup: &SharedSetup,
    inputs: Vec<u8>,
) -> Vec<Box<dyn AbstractObjective>> {
    inventory::iter::<ObjectiveEntry>()
        .map(|entry| (entry.build)(setup, inputs.clone()))
        .collect()
}

/// Measure a list of trait-object objectives.
pub fn measure_dyn(objectives: &[Box<dyn AbstractObjective>]) -> HashMap<String, f64> {
    objectives
        .iter()
        .filter_map(|obj| {
            let name = obj.name().to_string();
            obj.collect_measurement().ok().map(|v| (name, v))
        })
        .collect()
}

/// Centralized objective enum dispatching to concrete implementations.
pub enum Objective {
    PeakRss(peak_rss::PeakRssObjective),
    ProverTime(prover_time::ProverTimeObjective),
    ProofSize(proof_size::ProofSizeObjective),
    VerifierTime(verifier_time::VerifierTimeObjective),
    GuestCycleCount(guest_cycles::GuestCycleCountObjective),
    InlineLengths(inline_lengths::InlineLengthsObjective),
    WrappingCost(wrapping_cost::WrappingCostObjective),
}

impl Objective {
    pub fn name(&self) -> &str {
        match self {
            Self::PeakRss(o) => o.name(),
            Self::ProverTime(o) => o.name(),
            Self::ProofSize(o) => o.name(),
            Self::VerifierTime(o) => o.name(),
            Self::GuestCycleCount(o) => o.name(),
            Self::InlineLengths(o) => o.name(),
            Self::WrappingCost(o) => o.name(),
        }
    }

    pub fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        match self {
            Self::PeakRss(o) => o.collect_measurement(),
            Self::ProverTime(o) => o.collect_measurement(),
            Self::ProofSize(o) => o.collect_measurement(),
            Self::VerifierTime(o) => o.collect_measurement(),
            Self::GuestCycleCount(o) => o.collect_measurement(),
            Self::InlineLengths(o) => o.collect_measurement(),
            Self::WrappingCost(o) => o.collect_measurement(),
        }
    }

    pub fn direction(&self) -> Direction {
        match self {
            Self::PeakRss(o) => o.direction(),
            Self::ProverTime(o) => o.direction(),
            Self::ProofSize(o) => o.direction(),
            Self::VerifierTime(o) => o.direction(),
            Self::GuestCycleCount(o) => o.direction(),
            Self::InlineLengths(o) => o.direction(),
            Self::WrappingCost(o) => o.direction(),
        }
    }
}

/// Record of a single optimization attempt for post-hoc analysis.
pub struct OptimizationAttempt {
    pub description: String,
    pub diff: String,
    pub measurements: HashMap<String, f64>,
    pub invariants_passed: bool,
}

/// Measure all objectives and return a map of name -> value.
pub fn measure_objectives(objectives: &[Objective]) -> HashMap<String, f64> {
    objectives
        .iter()
        .filter_map(|obj| {
            let name = obj.name().to_string();
            obj.collect_measurement().ok().map(|v| (name, v))
        })
        .collect()
}
