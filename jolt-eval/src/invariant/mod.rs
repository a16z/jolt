pub mod completeness_prover;
pub mod completeness_verifier;
pub mod determinism;
pub mod serialization_roundtrip;
pub mod soundness;
pub mod synthesis;
pub mod zk_consistency;

use std::fmt;

use arbitrary::Arbitrary;
use enumset::{EnumSet, EnumSetType};
use rand::RngCore;

/// What to synthesize from an invariant definition.
#[derive(Debug, EnumSetType)]
pub enum SynthesisTarget {
    Test,
    Fuzz,
    RedTeam,
}

/// Error indicating an invariant was violated.
#[derive(Debug, Clone)]
pub struct InvariantViolation {
    pub message: String,
    pub details: Option<String>,
}

impl fmt::Display for InvariantViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(details) = &self.details {
            write!(f, ": {details}")?;
        }
        Ok(())
    }
}

impl std::error::Error for InvariantViolation {}

impl InvariantViolation {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            details: None,
        }
    }

    pub fn with_details(message: impl Into<String>, details: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            details: Some(details.into()),
        }
    }
}

/// Core invariant trait. Each invariant defines a setup phase (run once)
/// and a check phase (run per input). The `Input` type must support
/// `Arbitrary` for fuzzing and random testing.
pub trait Invariant: Send + Sync {
    type Setup;
    type Input: for<'a> Arbitrary<'a> + fmt::Debug + Clone;

    fn name(&self) -> &str;

    /// Human-readable description, also used as context for AI red-teaming.
    fn description(&self) -> String;

    fn targets(&self) -> EnumSet<SynthesisTarget>;

    /// One-time setup (e.g. preprocessing, generating an honest proof).
    fn setup(&self) -> Self::Setup;

    /// Check the invariant for a single input against the pre-computed setup.
    fn check(&self, setup: &Self::Setup, input: Self::Input) -> Result<(), InvariantViolation>;

    /// Known-interesting inputs for deterministic test generation.
    fn seed_corpus(&self) -> Vec<Self::Input> {
        vec![]
    }
}

/// A counterexample produced when an invariant is violated.
pub struct InvariantCounterexample<I: Invariant> {
    pub description: String,
    pub input: I::Input,
    pub error: InvariantViolation,
}

/// Record of a red-team attempt that failed to find a violation.
pub struct FailedAttempt {
    pub description: String,
    pub approach: String,
    pub failure_reason: String,
}

/// Object-safe wrapper for `Invariant`, enabling heterogeneous collections.
pub trait DynInvariant: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> String;
    fn targets(&self) -> EnumSet<SynthesisTarget>;

    /// Run seed corpus checks followed by `num_random` randomly-generated inputs.
    fn run_checks(&self, num_random: usize) -> Vec<Result<(), InvariantViolation>>;
}

impl<I: Invariant> DynInvariant for I {
    fn name(&self) -> &str {
        Invariant::name(self)
    }

    fn description(&self) -> String {
        Invariant::description(self)
    }

    fn targets(&self) -> EnumSet<SynthesisTarget> {
        Invariant::targets(self)
    }

    fn run_checks(&self, num_random: usize) -> Vec<Result<(), InvariantViolation>> {
        let setup = self.setup();
        let mut results = Vec::new();

        for input in self.seed_corpus() {
            results.push(self.check(&setup, input));
        }

        let mut rng = rand::thread_rng();
        for _ in 0..num_random {
            let mut raw = vec![0u8; 4096];
            rng.fill_bytes(&mut raw);
            let mut u = arbitrary::Unstructured::new(&raw);
            if let Ok(input) = I::Input::arbitrary(&mut u) {
                results.push(self.check(&setup, input));
            }
        }

        results
    }
}

/// Result of running an invariant check suite.
pub struct InvariantReport {
    pub name: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub violations: Vec<String>,
}

impl InvariantReport {
    pub fn from_results(name: &str, results: &[Result<(), InvariantViolation>]) -> Self {
        let total = results.len();
        let mut passed = 0;
        let mut failed = 0;
        let mut violations = Vec::new();
        for r in results {
            match r {
                Ok(()) => passed += 1,
                Err(e) => {
                    failed += 1;
                    violations.push(e.to_string());
                }
            }
        }
        Self {
            name: name.to_string(),
            total,
            passed,
            failed,
            violations,
        }
    }
}
