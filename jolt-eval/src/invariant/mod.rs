pub mod split_eq_bind;
pub mod synthesis;

use std::any::Any;
use std::fmt;

use arbitrary::Arbitrary;
use enumset::{EnumSet, EnumSetType};
use rand::RngCore;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Serialize;

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

/// Enum collecting all Jolt invariants. Methods dispatch via match.
pub enum JoltInvariants {
    SplitEqBindLowHigh(split_eq_bind::SplitEqBindLowHighInvariant),
    SplitEqBindHighLow(split_eq_bind::SplitEqBindHighLowInvariant),
}

macro_rules! dispatch {
    ($self:expr, |$inv:ident| $body:expr) => {
        match $self {
            JoltInvariants::SplitEqBindLowHigh($inv) => $body,
            JoltInvariants::SplitEqBindHighLow($inv) => $body,
        }
    };
}

impl JoltInvariants {
    pub fn all() -> Vec<Self> {
        vec![
            Self::SplitEqBindLowHigh(split_eq_bind::SplitEqBindLowHighInvariant),
            Self::SplitEqBindHighLow(split_eq_bind::SplitEqBindHighLowInvariant),
        ]
    }

    pub fn name(&self) -> &str {
        dispatch!(self, |inv| inv.name())
    }

    pub fn description(&self) -> String {
        dispatch!(self, |inv| inv.description())
    }

    pub fn targets(&self) -> EnumSet<SynthesisTarget> {
        dispatch!(self, |inv| inv.targets())
    }

    pub fn run_checks(&self, num_random: usize) -> Vec<Result<(), InvariantViolation>> {
        dispatch!(self, |inv| run_checks_impl(inv, num_random))
    }

    pub fn dyn_setup(&self) -> Box<dyn Any + Send + Sync> {
        dispatch!(self, |inv| dyn_setup_impl(inv))
    }

    pub fn check_json_input(&self, setup: &dyn Any, json: &str) -> CheckJsonResult {
        dispatch!(self, |inv| check_json_input_impl(inv, setup, json))
    }
}

fn run_checks_impl<I: Invariant>(inv: &I, num_random: usize) -> Vec<Result<(), InvariantViolation>> {
    let setup = inv.setup();
    let mut results = Vec::new();

    for input in inv.seed_corpus() {
        results.push(inv.check(&setup, input));
    }

    let mut rng = rand::thread_rng();
    for _ in 0..num_random {
        let mut raw = vec![0u8; 4096];
        rng.fill_bytes(&mut raw);
        let mut u = arbitrary::Unstructured::new(&raw);
        if let Ok(input) = I::Input::arbitrary(&mut u) {
            results.push(inv.check(&setup, input));
        }
    }

    results
}

fn dyn_setup_impl<I: Invariant>(inv: &I) -> Box<dyn Any + Send + Sync> {
    Box::new(inv.setup())
}

fn check_json_input_impl<I: Invariant>(
    inv: &I,
    setup: &dyn Any,
    json: &str,
) -> CheckJsonResult {
    let setup = setup
        .downcast_ref::<I::Setup>()
        .expect("check_json_input called with wrong setup type");
    let input: I::Input = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(e) => return CheckJsonResult::BadInput(e.to_string()),
    };
    match inv.check(setup, input) {
        Ok(()) => CheckJsonResult::Pass,
        Err(v) => CheckJsonResult::Violation(v),
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

/// Outcome of [`JoltInvariants::check_json_input`].
pub enum CheckJsonResult {
    /// The input was valid and the invariant held.
    Pass,
    /// The input was valid and the invariant was violated.
    Violation(InvariantViolation),
    /// The JSON could not be deserialized into the expected `Input` type.
    BadInput(String),
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
