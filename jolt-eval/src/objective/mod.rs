pub mod comment_density;
pub mod lloc;
pub mod optimize;

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
}

/// Centralized objective enum dispatching to concrete implementations.
pub enum Objective {
    Lloc(lloc::LlocObjective),
    CommentDensity(comment_density::CommentDensityObjective),
}

impl Objective {
    pub fn all(root: &Path) -> Vec<Self> {
        vec![
            Self::Lloc(lloc::LlocObjective::new(root)),
            Self::CommentDensity(comment_density::CommentDensityObjective::new(root)),
        ]
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Lloc(o) => o.name(),
            Self::CommentDensity(o) => o.name(),
        }
    }

    pub fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        match self {
            Self::Lloc(o) => o.collect_measurement(),
            Self::CommentDensity(o) => o.collect_measurement(),
        }
    }

    pub fn direction(&self) -> Direction {
        match self {
            Self::Lloc(o) => o.direction(),
            Self::CommentDensity(o) => o.direction(),
        }
    }
}

/// Record of a single optimization attempt for post-hoc analysis.
pub struct OptimizationAttempt {
    pub description: String,
    pub diff: String,
    pub measurements: std::collections::HashMap<String, f64>,
    pub invariants_passed: bool,
}
