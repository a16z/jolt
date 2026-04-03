pub mod bind_bench;
pub mod cognitive;
pub mod halstead_bugs;
pub mod lloc;
pub mod optimize;
pub mod synthesis;

use std::fmt;
use std::path::Path;

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
    fn collect_measurement(&self) -> Result<f64, MeasurementError>;
    fn direction(&self) -> Direction;
    fn units(&self) -> Option<&str> {
        None
    }
}

/// A performance objective suitable for Criterion benchmarking.
///
/// Separates setup (run once) from the hot path (run many times in
/// Criterion's `b.iter()` loop). Use the `bench_objective!` macro to
/// generate a Criterion benchmark harness from a `PerfObjective`.
pub trait PerfObjective: Default + Send + Sync {
    type Setup: Send;

    fn name(&self) -> &str;

    /// One-time setup (e.g. allocate polynomial, generate challenges).
    fn setup(&self) -> Self::Setup;

    /// The hot path to benchmark. Called repeatedly by Criterion.
    fn run(&self, setup: &mut Self::Setup);

    fn units(&self) -> &str {
        "s"
    }
}

/// Centralized enum for static-analysis objectives.
///
/// Performance objectives are handled separately via Criterion benchmarks
/// (see `PerfObjective` and `bench_objective!`).
pub enum Objective {
    Lloc(lloc::LlocObjective),
    CognitiveComplexity(cognitive::CognitiveComplexityObjective),
    HalsteadBugs(halstead_bugs::HalsteadBugsObjective),
}

impl Objective {
    pub fn all(root: &Path) -> Vec<Self> {
        vec![
            Self::Lloc(lloc::LlocObjective::new(root)),
            Self::CognitiveComplexity(cognitive::CognitiveComplexityObjective::new(root)),
            Self::HalsteadBugs(halstead_bugs::HalsteadBugsObjective::new(root)),
        ]
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Lloc(o) => o.name(),
            Self::CognitiveComplexity(o) => o.name(),
            Self::HalsteadBugs(o) => o.name(),
        }
    }

    pub fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        match self {
            Self::Lloc(o) => o.collect_measurement(),
            Self::CognitiveComplexity(o) => o.collect_measurement(),
            Self::HalsteadBugs(o) => o.collect_measurement(),
        }
    }

    pub fn units(&self) -> Option<&str> {
        match self {
            Self::Lloc(o) => o.units(),
            Self::CognitiveComplexity(o) => o.units(),
            Self::HalsteadBugs(o) => o.units(),
        }
    }

    pub fn direction(&self) -> Direction {
        match self {
            Self::Lloc(o) => o.direction(),
            Self::CognitiveComplexity(o) => o.direction(),
            Self::HalsteadBugs(o) => o.direction(),
        }
    }
}

/// Names of all registered `PerfObjective` benchmarks.
pub fn perf_objective_names() -> &'static [&'static str] {
    &[
        bind_bench::BindLowToHighObjective::NAME,
        bind_bench::BindHighToLowObjective::NAME,
    ]
}

/// Record of a single optimization attempt for post-hoc analysis.
pub struct OptimizationAttempt {
    pub description: String,
    pub diff: String,
    pub measurements: std::collections::HashMap<String, f64>,
    pub invariants_passed: bool,
}
