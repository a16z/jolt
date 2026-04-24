pub mod code_quality;
pub mod objective_fn;
pub mod optimize;
pub mod performance;
pub mod synthesis;

use std::fmt;

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

    fn description(&self) -> String {
        self.name().to_string()
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

/// Static-analysis objectives.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum StaticAnalysisObjective {
    Lloc(code_quality::lloc::LlocObjective),
    CognitiveComplexity(code_quality::cognitive::CognitiveComplexityObjective),
    HalsteadBugs(code_quality::halstead_bugs::HalsteadBugsObjective),
}

impl StaticAnalysisObjective {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Lloc(code_quality::lloc::LlocObjective {
                target_dir: "jolt-core/src",
            }),
            Self::CognitiveComplexity(code_quality::cognitive::CognitiveComplexityObjective {
                target_dir: "jolt-core/src",
            }),
            Self::HalsteadBugs(code_quality::halstead_bugs::HalsteadBugsObjective {
                target_dir: "jolt-core/src",
            }),
        ]
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Lloc(o) => o.name(),
            Self::CognitiveComplexity(o) => o.name(),
            Self::HalsteadBugs(o) => o.name(),
        }
    }

    pub fn description(&self) -> String {
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

    pub fn collect_measurement_in(
        &self,
        repo_root: &std::path::Path,
    ) -> Result<f64, MeasurementError> {
        match self {
            Self::Lloc(o) => o.collect_measurement_in(repo_root),
            Self::CognitiveComplexity(o) => o.collect_measurement_in(repo_root),
            Self::HalsteadBugs(o) => o.collect_measurement_in(repo_root),
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

/// Criterion-benchmarked performance objectives.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PerformanceObjective {
    BindLowToHigh(performance::binding::BindLowToHighObjective),
    BindHighToLow(performance::binding::BindHighToLowObjective),
    NaiveSortTime(performance::naive_sort::NaiveSortObjective),
    JoltCryptoG1Msm1024(performance::jolt_crypto_g1_msm::JoltCryptoG1Msm1024Objective),
    JoltCryptoG1ScalarMul(performance::jolt_crypto_g1_scalar_mul::JoltCryptoG1ScalarMulObjective),
    JoltCryptoGtScalarMul(performance::jolt_crypto_gt_scalar_mul::JoltCryptoGtScalarMulObjective),
    JoltCryptoPedersenCommit1024(
        performance::jolt_crypto_pedersen_commit::JoltCryptoPedersenCommit1024Objective,
    ),
}

impl PerformanceObjective {
    pub fn all() -> Vec<Self> {
        vec![
            Self::BindLowToHigh(performance::binding::BindLowToHighObjective),
            Self::BindHighToLow(performance::binding::BindHighToLowObjective),
            Self::NaiveSortTime(performance::naive_sort::NaiveSortObjective),
            Self::JoltCryptoG1Msm1024(
                performance::jolt_crypto_g1_msm::JoltCryptoG1Msm1024Objective,
            ),
            Self::JoltCryptoG1ScalarMul(
                performance::jolt_crypto_g1_scalar_mul::JoltCryptoG1ScalarMulObjective,
            ),
            Self::JoltCryptoGtScalarMul(
                performance::jolt_crypto_gt_scalar_mul::JoltCryptoGtScalarMulObjective,
            ),
            Self::JoltCryptoPedersenCommit1024(
                performance::jolt_crypto_pedersen_commit::JoltCryptoPedersenCommit1024Objective,
            ),
        ]
    }

    pub fn name(&self) -> &str {
        match self {
            Self::BindLowToHigh(o) => o.name(),
            Self::BindHighToLow(o) => o.name(),
            Self::NaiveSortTime(o) => o.name(),
            Self::JoltCryptoG1Msm1024(o) => o.name(),
            Self::JoltCryptoG1ScalarMul(o) => o.name(),
            Self::JoltCryptoGtScalarMul(o) => o.name(),
            Self::JoltCryptoPedersenCommit1024(o) => o.name(),
        }
    }

    pub fn units(&self) -> Option<&str> {
        match self {
            Self::BindLowToHigh(o) => o.units(),
            Self::BindHighToLow(o) => o.units(),
            Self::NaiveSortTime(o) => o.units(),
            Self::JoltCryptoG1Msm1024(o) => o.units(),
            Self::JoltCryptoG1ScalarMul(o) => o.units(),
            Self::JoltCryptoGtScalarMul(o) => o.units(),
            Self::JoltCryptoPedersenCommit1024(o) => o.units(),
        }
    }

    pub fn description(&self) -> String {
        match self {
            Self::BindLowToHigh(o) => o.description(),
            Self::BindHighToLow(o) => o.description(),
            Self::NaiveSortTime(o) => o.description(),
            Self::JoltCryptoG1Msm1024(o) => o.description(),
            Self::JoltCryptoG1ScalarMul(o) => o.description(),
            Self::JoltCryptoGtScalarMul(o) => o.description(),
            Self::JoltCryptoPedersenCommit1024(o) => o.description(),
        }
    }

    pub fn diff_paths(&self) -> &'static [&'static str] {
        match self {
            Self::BindLowToHigh(_) | Self::BindHighToLow(_) => &["jolt-core/"],
            Self::NaiveSortTime(_) => &["jolt-eval/src/sort_targets.rs"],
            Self::JoltCryptoG1Msm1024(_)
            | Self::JoltCryptoG1ScalarMul(_)
            | Self::JoltCryptoGtScalarMul(_)
            | Self::JoltCryptoPedersenCommit1024(_) => &["crates/jolt-crypto/"],
        }
    }
}

/// Union of all known objectives — used as a type-safe HashMap key.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationObjective {
    StaticAnalysis(StaticAnalysisObjective),
    Performance(PerformanceObjective),
}

// Re-export the const objective keys from their defining modules.
pub use code_quality::cognitive::COGNITIVE_COMPLEXITY;
pub use code_quality::halstead_bugs::HALSTEAD_BUGS;
pub use code_quality::lloc::LLOC;
pub use performance::binding::{BIND_HIGH_TO_LOW, BIND_LOW_TO_HIGH};
pub use performance::jolt_crypto_g1_msm::JOLT_CRYPTO_G1_MSM_1024;
pub use performance::jolt_crypto_g1_scalar_mul::JOLT_CRYPTO_G1_SCALAR_MUL;
pub use performance::jolt_crypto_gt_scalar_mul::JOLT_CRYPTO_GT_SCALAR_MUL;
pub use performance::jolt_crypto_pedersen_commit::JOLT_CRYPTO_PEDERSEN_COMMIT_1024;
pub use performance::naive_sort::NAIVE_SORT_TIME;

impl OptimizationObjective {
    pub fn all() -> Vec<Self> {
        let mut all = Vec::new();
        for s in StaticAnalysisObjective::all() {
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

    pub fn description(&self) -> String {
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
        for sa in StaticAnalysisObjective::all() {
            let val = sa.collect_measurement().unwrap();
            assert!(val > 0.0, "{} should be > 0, got {val}", sa.name());
        }
    }

    #[test]
    fn optimization_objective_hashmap_key() {
        use std::collections::HashMap;
        let lloc = LLOC;
        let bind = BIND_LOW_TO_HIGH;
        let mut m = HashMap::new();
        m.insert(lloc, 100.0);
        m.insert(bind, 0.5);

        // Same variant with identical inner data looks up successfully.
        let lloc_same = OptimizationObjective::StaticAnalysis(StaticAnalysisObjective::Lloc(
            code_quality::lloc::LlocObjective {
                target_dir: "jolt-core/src",
            },
        ));
        assert_eq!(m[&lloc_same], 100.0);

        // Same variant with different inner data does NOT match.
        let lloc_other = OptimizationObjective::StaticAnalysis(StaticAnalysisObjective::Lloc(
            code_quality::lloc::LlocObjective {
                target_dir: "other/path",
            },
        ));
        assert!(!m.contains_key(&lloc_other));
    }
}
