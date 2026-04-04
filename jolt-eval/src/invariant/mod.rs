#[cfg(test)]
mod macro_tests;
pub mod soundness;
pub mod split_eq_bind;
pub mod synthesis;

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

/// Result of checking an invariant against a single input.
#[derive(Debug)]
pub enum CheckError {
    /// The invariant was violated.
    Violation(InvariantViolation),
    /// The input is degenerate or uninteresting and should be skipped.
    InvalidInput(String),
}

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Violation(v) => write!(f, "violation: {v}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for CheckError {}

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

    /// One-time setup (e.g. preprocessing, generating an honest proof).
    fn setup(&self) -> Self::Setup;

    /// Check the invariant for a single input against the pre-computed setup.
    ///
    /// Returns `Ok(())` if the invariant holds, `Err(CheckError::Violation)`
    /// if it is violated, or `Err(CheckError::InvalidInput)` if the input
    /// is degenerate and should be skipped.
    fn check(&self, setup: &Self::Setup, input: Self::Input) -> Result<(), CheckError>;

    /// Known-interesting inputs for deterministic test generation.
    fn seed_corpus(&self) -> Vec<Self::Input> {
        vec![]
    }

    /// Post-process a deserialized input with context from the agent run.
    ///
    /// Called by the red-team harness after deserializing the agent's JSON
    /// counterexample. The `diff` is the unified diff captured from the
    /// agent's worktree (if any files were modified). The default is a no-op.
    fn enrich_input(&self, input: Self::Input, _diff: Option<&str>) -> Self::Input {
        input
    }
}

/// Declares which synthesis targets an invariant supports.
///
/// Defaults to an empty set. Use the `#[invariant(Test, Fuzz)]` macro
/// attribute to generate the implementation automatically.
pub trait InvariantTargets {
    fn targets(&self) -> EnumSet<SynthesisTarget> {
        EnumSet::empty()
    }
}

/// Enum collecting all Jolt invariants. Methods dispatch via match.
pub enum JoltInvariants {
    SplitEqBindLowHigh(split_eq_bind::SplitEqBindLowHighInvariant),
    SplitEqBindHighLow(split_eq_bind::SplitEqBindHighLowInvariant),
    Soundness(soundness::SoundnessInvariant),
}

macro_rules! dispatch {
    ($self:expr, |$inv:ident| $body:expr) => {
        match $self {
            JoltInvariants::SplitEqBindLowHigh($inv) => $body,
            JoltInvariants::SplitEqBindHighLow($inv) => $body,
            JoltInvariants::Soundness($inv) => $body,
        }
    };
}

impl JoltInvariants {
    pub fn all() -> Vec<Self> {
        vec![
            Self::SplitEqBindLowHigh(split_eq_bind::SplitEqBindLowHighInvariant),
            Self::SplitEqBindHighLow(split_eq_bind::SplitEqBindHighLowInvariant),
            Self::Soundness(soundness::SoundnessInvariant),
        ]
    }

    pub fn name(&self) -> &str {
        dispatch!(self, |inv| inv.name())
    }

    pub fn description(&self) -> String {
        dispatch!(self, |inv| inv.description())
    }

    pub fn targets(&self) -> EnumSet<SynthesisTarget> {
        dispatch!(self, |inv| InvariantTargets::targets(inv))
    }

    pub fn run_checks(&self, num_random: usize) -> Vec<Result<(), InvariantViolation>> {
        dispatch!(self, |inv| run_checks_impl(inv, num_random))
    }
}

fn run_checks_impl<I: Invariant>(
    inv: &I,
    num_random: usize,
) -> Vec<Result<(), InvariantViolation>> {
    let setup = inv.setup();
    let mut results = Vec::new();

    let mut record = |r: Result<(), CheckError>| match r {
        Ok(()) => results.push(Ok(())),
        Err(CheckError::Violation(v)) => results.push(Err(v)),
        Err(CheckError::InvalidInput(_)) => {}
    };

    for input in inv.seed_corpus() {
        record(inv.check(&setup, input));
    }

    let mut rng = rand::thread_rng();
    for _ in 0..num_random {
        let mut raw = vec![0u8; 4096];
        rng.fill_bytes(&mut raw);
        let mut u = arbitrary::Unstructured::new(&raw);
        if let Ok(input) = I::Input::arbitrary(&mut u) {
            record(inv.check(&setup, input));
        }
    }

    results
}

/// Record of a red-team attempt that failed to find a violation.
pub struct FailedAttempt {
    pub description: String,
    pub approach: String,
    pub failure_reason: String,
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

#[cfg(test)]
mod integration_tests {
    use super::*;

    struct TrivialInvariant;
    impl InvariantTargets for TrivialInvariant {}

    impl Invariant for TrivialInvariant {
        type Setup = ();
        type Input = u8;
        fn name(&self) -> &str {
            "trivial"
        }
        fn description(&self) -> String {
            "Always passes".into()
        }
        fn setup(&self) {}
        fn check(&self, _: &(), _: u8) -> Result<(), CheckError> {
            Ok(())
        }
        fn seed_corpus(&self) -> Vec<u8> {
            vec![0, 1, 255]
        }
    }

    struct FailingInvariant;
    impl InvariantTargets for FailingInvariant {}

    impl Invariant for FailingInvariant {
        type Setup = ();
        type Input = u8;
        fn name(&self) -> &str {
            "failing"
        }
        fn description(&self) -> String {
            "Always fails".into()
        }
        fn setup(&self) {}
        fn check(&self, _: &(), input: u8) -> Result<(), CheckError> {
            Err(CheckError::Violation(InvariantViolation::new(format!(
                "failed for input {input}"
            ))))
        }
        fn seed_corpus(&self) -> Vec<u8> {
            vec![42]
        }
    }

    #[test]
    fn trivial_invariant_passes() {
        let inv = TrivialInvariant;
        for input in inv.seed_corpus() {
            inv.check(&(), input).unwrap();
        }
    }

    #[test]
    fn failing_invariant_reports_violations() {
        let inv = FailingInvariant;
        for input in inv.seed_corpus() {
            assert!(inv.check(&(), input).is_err());
        }
    }
}
