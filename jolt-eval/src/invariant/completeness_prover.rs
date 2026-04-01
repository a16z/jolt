use std::sync::Arc;

use arbitrary::Arbitrary;
use enumset::EnumSet;

use super::{Invariant, InvariantEntry, InvariantViolation, SynthesisTarget};
use crate::{ProverPreprocessing, TestCase};

inventory::submit! {
    InvariantEntry {
        name: "prover_completeness",
        targets: || SynthesisTarget::Test | SynthesisTarget::Fuzz,
        build: |tc, _inputs| Box::new(ProverCompletenessInvariant::new(tc)),
    }
}

/// Prover completeness: for a fixed program, input, and valid size parameters,
/// the prover should produce a proof without panicking.
pub struct ProverCompletenessInvariant {
    pub test_case: Arc<TestCase>,
}

pub struct ProverCompletenessSetup {
    test_case: Arc<TestCase>,
    prover_preprocessing: ProverPreprocessing,
}

/// Program inputs for prover completeness testing.
#[derive(Debug, Clone, Arbitrary, serde::Serialize, serde::Deserialize)]
pub struct ProverInputs {
    pub data: Vec<u8>,
}

impl ProverCompletenessInvariant {
    pub fn new(test_case: Arc<TestCase>) -> Self {
        Self { test_case }
    }
}

impl Invariant for ProverCompletenessInvariant {
    type Setup = ProverCompletenessSetup;
    type Input = ProverInputs;

    fn name(&self) -> &str {
        "prover_completeness"
    }

    fn description(&self) -> String {
        "For a fixed program, input, and valid size parameters, \
         the prover should produce a proof (not panic)."
            .to_string()
    }

    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::Fuzz
    }

    fn setup(&self) -> Self::Setup {
        let prover_pp = self.test_case.prover_preprocessing();
        ProverCompletenessSetup {
            test_case: Arc::clone(&self.test_case),
            prover_preprocessing: prover_pp,
        }
    }

    fn check(&self, setup: &Self::Setup, input: ProverInputs) -> Result<(), InvariantViolation> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            setup
                .test_case
                .prove(&setup.prover_preprocessing, &input.data)
        }));

        match result {
            Ok((_proof, io_device)) => {
                // Guest panics are acceptable (the guest may reject bad input).
                // Prover panics are not -- those are caught by catch_unwind above.
                if io_device.panic {
                    // Guest panicked, but prover completed successfully
                    Ok(())
                } else {
                    Ok(())
                }
            }
            Err(panic_info) => {
                let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                Err(InvariantViolation::with_details(
                    "Prover panicked",
                    format!("inputs: {} bytes, panic: {msg}", input.data.len()),
                ))
            }
        }
    }

    fn seed_corpus(&self) -> Vec<Self::Input> {
        vec![
            ProverInputs { data: vec![] },
            ProverInputs {
                data: vec![0u8; 64],
            },
        ]
    }
}
