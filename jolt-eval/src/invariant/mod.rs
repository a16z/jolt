pub mod split_eq_bind;
pub mod synthesis;

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arbitrary::Arbitrary;
use enumset::{EnumSet, EnumSetType};
use rand::RngCore;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::TestCase;

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
/// `Arbitrary` for fuzzing, and `Serialize`/`DeserializeOwned` so an AI
/// agent can produce counterexamples as JSON.
pub trait Invariant: Send + Sync {
    type Setup: Send + Sync + 'static;
    type Input: for<'a> Arbitrary<'a>
        + fmt::Debug
        + Clone
        + Serialize
        + DeserializeOwned
        + JsonSchema;

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

/// Factory function type for constructing an invariant from an optional
/// test case and default inputs.
pub type InvariantBuildFn = fn(Option<Arc<TestCase>>, Vec<u8>) -> Box<dyn DynInvariant>;

pub struct InvariantEntry {
    pub name: &'static str,
    pub targets: fn() -> EnumSet<SynthesisTarget>,
    /// Whether this invariant requires a compiled guest program.
    pub needs_guest: bool,
    pub build: InvariantBuildFn,
}

/// All registered invariant entries.
pub fn registered_invariants() -> impl Iterator<Item = InvariantEntry> {
    [
        InvariantEntry {
            name: "split_eq_bind_low_high",
            targets: || SynthesisTarget::Test | SynthesisTarget::Fuzz,
            needs_guest: false,
            build: |_tc, _inputs| Box::new(split_eq_bind::SplitEqBindLowHighInvariant),
        },
        InvariantEntry {
            name: "split_eq_bind_high_low",
            targets: || SynthesisTarget::Test | SynthesisTarget::Fuzz,
            needs_guest: false,
            build: |_tc, _inputs| Box::new(split_eq_bind::SplitEqBindHighLowInvariant),
        },
    ]
    .into_iter()
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

/// Object-safe wrapper for `Invariant`, enabling heterogeneous collections
/// and JSON-based counterexample checking.
pub trait DynInvariant: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> String;
    fn targets(&self) -> EnumSet<SynthesisTarget>;

    /// Run seed corpus checks followed by `num_random` randomly-generated inputs.
    fn run_checks(&self, num_random: usize) -> Vec<Result<(), InvariantViolation>>;

    /// Return a JSON example of the `Input` type (from the seed corpus).
    fn input_json_example(&self) -> Option<String>;

    /// Return the JSON Schema for the `Input` type.
    fn input_json_schema(&self) -> serde_json::Value;

    /// Create the (type-erased) setup. Expensive — call once and reuse.
    fn dyn_setup(&self) -> Box<dyn Any + Send + Sync>;

    /// Deserialize a JSON-encoded `Input` and check it against a
    /// previously-created setup (from [`dyn_setup`]).
    fn check_json_input(&self, setup: &dyn Any, json: &str) -> CheckJsonResult;
}

/// Outcome of [`DynInvariant::check_json_input`].
pub enum CheckJsonResult {
    /// The input was valid and the invariant held.
    Pass,
    /// The input was valid and the invariant was violated.
    Violation(InvariantViolation),
    /// The JSON could not be deserialized into the expected `Input` type.
    BadInput(String),
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

    fn input_json_example(&self) -> Option<String> {
        self.seed_corpus()
            .into_iter()
            .next()
            .and_then(|input| serde_json::to_string_pretty(&input).ok())
    }

    fn input_json_schema(&self) -> serde_json::Value {
        let schema = schemars::schema_for!(I::Input);
        serde_json::to_value(schema).unwrap()
    }

    fn dyn_setup(&self) -> Box<dyn Any + Send + Sync> {
        Box::new(Invariant::setup(self))
    }

    fn check_json_input(&self, setup: &dyn Any, json: &str) -> CheckJsonResult {
        let setup = setup
            .downcast_ref::<I::Setup>()
            .expect("DynInvariant::check_json_input called with wrong setup type");
        let input: I::Input = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => return CheckJsonResult::BadInput(e.to_string()),
        };
        match self.check(setup, input) {
            Ok(()) => CheckJsonResult::Pass,
            Err(v) => CheckJsonResult::Violation(v),
        }
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

/// Try to extract a JSON object from free-form text. Looks for a
/// ````json` code block first, then falls back to the last `{…}` that
/// parses as valid JSON.
pub fn extract_json(text: &str) -> Option<String> {
    // 1. ```json ... ```
    if let Some(start) = text.find("```json") {
        let json_start = start + "```json".len();
        if let Some(end) = text[json_start..].find("```") {
            let candidate = text[json_start..json_start + end].trim();
            if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                return Some(candidate.to_string());
            }
        }
    }

    // 2. Last balanced {…} that is valid JSON
    let bytes = text.as_bytes();
    let mut i = bytes.len();
    while i > 0 {
        i -= 1;
        if bytes[i] == b'}' {
            let end = i;
            let mut depth: i32 = 0;
            let mut j = end + 1;
            while j > 0 {
                j -= 1;
                match bytes[j] {
                    b'}' => depth += 1,
                    b'{' => {
                        depth -= 1;
                        if depth == 0 {
                            let candidate = &text[j..=end];
                            if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                                return Some(candidate.to_string());
                            }
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    None
}
