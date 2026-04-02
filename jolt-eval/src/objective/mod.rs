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

pub struct ObjectiveEntry {
    pub name: &'static str,
    pub direction: Direction,
    /// Whether this objective requires a compiled guest program.
    pub needs_guest: bool,
    pub build: fn(Option<&SharedSetup>, Vec<u8>) -> Box<dyn AbstractObjective>,
}

/// All registered objective entries.
pub fn registered_objectives() -> impl Iterator<Item = ObjectiveEntry> {
    [
        ObjectiveEntry {
            name: "peak_rss",
            direction: Direction::Minimize,
            needs_guest: true,
            build: |s, inputs| {
                let setup = s.unwrap();
                Box::new(peak_rss::PeakRssObjective::new(
                    setup.test_case.clone(),
                    setup.prover_preprocessing.clone(),
                    inputs,
                ))
            },
        },
        ObjectiveEntry {
            name: "prover_time",
            direction: Direction::Minimize,
            needs_guest: true,
            build: |s, inputs| {
                let setup = s.unwrap();
                Box::new(prover_time::ProverTimeObjective::new(
                    setup.test_case.clone(),
                    setup.prover_preprocessing.clone(),
                    inputs,
                ))
            },
        },
        ObjectiveEntry {
            name: "proof_size",
            direction: Direction::Minimize,
            needs_guest: true,
            build: |s, inputs| {
                let setup = s.unwrap();
                Box::new(proof_size::ProofSizeObjective::new(
                    setup.test_case.clone(),
                    setup.prover_preprocessing.clone(),
                    inputs,
                ))
            },
        },
        ObjectiveEntry {
            name: "verifier_time",
            direction: Direction::Minimize,
            needs_guest: true,
            build: |s, inputs| {
                let setup = s.unwrap();
                Box::new(verifier_time::VerifierTimeObjective::new(
                    setup.test_case.clone(),
                    setup.prover_preprocessing.clone(),
                    setup.verifier_preprocessing.clone(),
                    inputs,
                ))
            },
        },
        ObjectiveEntry {
            name: "guest_cycle_count",
            direction: Direction::Minimize,
            needs_guest: true,
            build: |s, inputs| {
                let setup = s.unwrap();
                Box::new(guest_cycles::GuestCycleCountObjective::new(
                    setup.test_case.clone(),
                    inputs,
                ))
            },
        },
        ObjectiveEntry {
            name: "inline_lengths",
            direction: Direction::Maximize,
            needs_guest: true,
            build: |s, _inputs| {
                let setup = s.unwrap();
                Box::new(inline_lengths::InlineLengthsObjective::new(
                    setup.test_case.clone(),
                ))
            },
        },
        ObjectiveEntry {
            name: "wrapping_cost",
            direction: Direction::Minimize,
            needs_guest: true,
            build: |s, _inputs| {
                let setup = s.unwrap();
                Box::new(wrapping_cost::WrappingCostObjective::new(
                    setup.test_case.clone(),
                    setup.prover_preprocessing.clone(),
                ))
            },
        },
    ]
    .into_iter()
}

/// Build all registered objectives from a [`SharedSetup`].
///
/// Pass `None` to include only objectives that don't require a guest.
pub fn build_objectives_from_inventory(
    setup: Option<&SharedSetup>,
    inputs: Vec<u8>,
) -> Vec<Box<dyn AbstractObjective>> {
    registered_objectives()
        .filter(|entry| !entry.needs_guest || setup.is_some())
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
