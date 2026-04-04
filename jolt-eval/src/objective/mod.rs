pub mod code_quality;
pub mod objective_fn;
pub mod optimize;
pub mod performance;
pub mod synthesis;

use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::Path;

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

/// Unified objective trait.
///
/// Static-analysis objectives override [`collect_measurement`](Objective::collect_measurement)
/// and set `Setup = ()`.
///
/// Performance objectives override [`setup`](Objective::setup) +
/// [`run`](Objective::run) and leave `collect_measurement` as the
/// default (returns an error).
pub trait Objective: Send + Sync {
    type Setup: Send;

    fn name(&self) -> &str;

    fn description(&self) -> &str {
        self.name()
    }

    fn units(&self) -> Option<&str> {
        None
    }

    /// Per-iteration setup for Criterion benchmarks.
    fn setup(&self) -> Self::Setup;

    /// Override for static-analysis objectives that produce a direct measurement.
    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        Err(MeasurementError::new("not directly measurable"))
    }

    /// Override for performance objectives benchmarked by Criterion.
    fn run(&self, _setup: Self::Setup) {}
}

// Data-containing enums — Hash/Eq based on discriminant only

/// Static-analysis objectives.
#[derive(Clone, Copy)]
pub enum StaticAnalysisObjective {
    Lloc(code_quality::lloc::LlocObjective),
    CognitiveComplexity(code_quality::cognitive::CognitiveComplexityObjective),
    HalsteadBugs(code_quality::halstead_bugs::HalsteadBugsObjective),
}

impl StaticAnalysisObjective {
    pub fn all(root: &Path) -> Vec<Self> {
        vec![
            Self::Lloc(code_quality::lloc::LlocObjective::new(root)),
            Self::CognitiveComplexity(code_quality::cognitive::CognitiveComplexityObjective::new(
                root,
            )),
            Self::HalsteadBugs(code_quality::halstead_bugs::HalsteadBugsObjective::new(
                root,
            )),
        ]
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Lloc(o) => o.name(),
            Self::CognitiveComplexity(o) => o.name(),
            Self::HalsteadBugs(o) => o.name(),
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Self::Lloc(o) => o.description(),
            Self::CognitiveComplexity(o) => o.description(),
            Self::HalsteadBugs(o) => o.description(),
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

    pub fn diff_paths(&self) -> &'static [&'static str] {
        &["jolt-core/"]
    }
}

impl PartialEq for StaticAnalysisObjective {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}
impl Eq for StaticAnalysisObjective {}
impl Hash for StaticAnalysisObjective {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
    }
}

/// Criterion-benchmarked performance objectives.
#[derive(Clone, Copy)]
pub enum PerformanceObjective {
    BindLowToHigh(performance::binding::BindLowToHighObjective),
    BindHighToLow(performance::binding::BindHighToLowObjective),
    NaiveSortTime(crate::sort_targets::NaiveSortObjective),
}

impl PerformanceObjective {
    pub fn all() -> Vec<Self> {
        vec![
            Self::BindLowToHigh(performance::binding::BindLowToHighObjective),
            Self::BindHighToLow(performance::binding::BindHighToLowObjective),
            Self::NaiveSortTime(crate::sort_targets::NaiveSortObjective),
        ]
    }

    pub fn name(&self) -> &str {
        match self {
            Self::BindLowToHigh(o) => o.name(),
            Self::BindHighToLow(o) => o.name(),
            Self::NaiveSortTime(o) => o.name(),
        }
    }

    pub fn units(&self) -> Option<&str> {
        match self {
            Self::BindLowToHigh(o) => o.units(),
            Self::BindHighToLow(o) => o.units(),
            Self::NaiveSortTime(o) => o.units(),
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Self::BindLowToHigh(o) => o.description(),
            Self::BindHighToLow(o) => o.description(),
            Self::NaiveSortTime(o) => o.description(),
        }
    }

    pub fn diff_paths(&self) -> &'static [&'static str] {
        match self {
            Self::BindLowToHigh(_) | Self::BindHighToLow(_) => &["jolt-core/"],
            Self::NaiveSortTime(_) => &["jolt-eval/src/sort_targets.rs"],
        }
    }
}

impl PartialEq for PerformanceObjective {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}
impl Eq for PerformanceObjective {}
impl Hash for PerformanceObjective {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
    }
}

/// Union of all known objectives — used as a type-safe HashMap key.
#[derive(Clone, Copy)]
pub enum OptimizationObjective {
    StaticAnalysis(StaticAnalysisObjective),
    Performance(PerformanceObjective),
}

// Re-export the const objective keys from their defining modules.
pub use code_quality::cognitive::COGNITIVE_COMPLEXITY;
pub use code_quality::halstead_bugs::HALSTEAD_BUGS;
pub use code_quality::lloc::LLOC;
pub use performance::binding::{BIND_HIGH_TO_LOW, BIND_LOW_TO_HIGH};
pub const NAIVE_SORT_TIME: OptimizationObjective = OptimizationObjective::Performance(
    PerformanceObjective::NaiveSortTime(crate::sort_targets::NaiveSortObjective),
);

impl OptimizationObjective {
    pub fn all(root: &Path) -> Vec<Self> {
        let mut all = Vec::new();
        for s in StaticAnalysisObjective::all(root) {
            all.push(Self::StaticAnalysis(s));
        }
        for p in PerformanceObjective::all() {
            all.push(Self::Performance(p));
        }
        all
    }

    pub fn name(&self) -> &str {
        match self {
            Self::StaticAnalysis(s) => s.name(),
            Self::Performance(p) => p.name(),
        }
    }

    pub fn units(&self) -> Option<&str> {
        match self {
            Self::StaticAnalysis(s) => s.units(),
            Self::Performance(p) => p.units(),
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Self::StaticAnalysis(s) => s.description(),
            Self::Performance(p) => p.description(),
        }
    }

    pub fn diff_paths(&self) -> &'static [&'static str] {
        match self {
            Self::StaticAnalysis(s) => s.diff_paths(),
            Self::Performance(p) => p.diff_paths(),
        }
    }

    pub fn is_perf(&self) -> bool {
        matches!(self, Self::Performance(_))
    }
}

/// Look up an objective's measurement and divide by its baseline value,
/// yielding a dimensionless ratio where 1.0 = the baseline.
///
/// `baselines` is typically the initial measurements captured at the
/// start of an optimization run (passed as the second argument to
/// [`ObjectiveFunction::evaluate`](objective_fn::ObjectiveFunction)).
///
/// ```ignore
/// use jolt_eval::objective::{normalized, LLOC, HALSTEAD_BUGS};
///
/// let evaluate = |m, b| 0.5 * normalized(&LLOC, m, b) + 0.5 * normalized(&HALSTEAD_BUGS, m, b);
/// ```
pub fn normalized(
    obj: &OptimizationObjective,
    measurements: &std::collections::HashMap<OptimizationObjective, f64>,
    baselines: &std::collections::HashMap<OptimizationObjective, f64>,
) -> f64 {
    let value = measurements.get(obj).copied().unwrap_or(f64::INFINITY);
    let baseline = baselines.get(obj).copied().unwrap_or(1.0);
    value / baseline
}

impl PartialEq for OptimizationObjective {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::StaticAnalysis(a), Self::StaticAnalysis(b)) => a == b,
            (Self::Performance(a), Self::Performance(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for OptimizationObjective {}
impl Hash for OptimizationObjective {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::StaticAnalysis(s) => s.hash(state),
            Self::Performance(p) => p.hash(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstantObjective {
        label: &'static str,
        value: f64,
    }

    impl Objective for ConstantObjective {
        type Setup = ();
        fn name(&self) -> &str {
            self.label
        }
        fn setup(&self) {}
        fn collect_measurement(&self) -> Result<f64, MeasurementError> {
            Ok(self.value)
        }
    }

    #[test]
    fn constant_objective() {
        let obj = ConstantObjective {
            label: "latency",
            value: 42.0,
        };
        assert_eq!(obj.name(), "latency");
        assert_eq!(obj.collect_measurement().unwrap(), 42.0);
    }

    #[test]
    fn static_analysis_all_measures() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        for sa in StaticAnalysisObjective::all(root) {
            let val = sa.collect_measurement().unwrap();
            assert!(val > 0.0, "{} should be > 0, got {val}", sa.name());
        }
    }

    #[test]
    fn optimization_objective_hashmap_key() {
        use std::collections::HashMap;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let lloc = OptimizationObjective::StaticAnalysis(StaticAnalysisObjective::Lloc(
            code_quality::lloc::LlocObjective::new(root),
        ));
        let bind = OptimizationObjective::Performance(PerformanceObjective::BindLowToHigh(
            performance::binding::BindLowToHighObjective,
        ));
        let mut m = HashMap::new();
        m.insert(lloc, 100.0);
        m.insert(bind, 0.5);

        // Look up with a freshly constructed key — works because Hash/Eq
        // is discriminant-based.
        let lloc2 = OptimizationObjective::StaticAnalysis(StaticAnalysisObjective::Lloc(
            code_quality::lloc::LlocObjective::new(Path::new("/other")),
        ));
        assert_eq!(m[&lloc2], 100.0);
    }

    #[test]
    fn optimization_objective_all() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let all = OptimizationObjective::all(root);
        assert_eq!(all.len(), 6); // 3 static + 3 perf
        assert!(all.iter().any(|o| o.is_perf()));
        assert!(all.iter().any(|o| !o.is_perf()));
    }
}
