use std::sync::Arc;

use arbitrary::Arbitrary;
use enumset::EnumSet;

use super::{Invariant, InvariantViolation, SynthesisTarget};
use crate::{ProverPreprocessing, TestCase, VerifierPreprocessing};

/// Verifier completeness: for a fixed program and honest prover output/proof,
/// the verifier accepts the honest output/proof.
pub struct VerifierCompletenessInvariant {
    pub test_case: Arc<TestCase>,
}

/// Pre-computed preprocessing shared across checks.
pub struct VerifierCompletenessSetup {
    test_case: Arc<TestCase>,
    prover_preprocessing: ProverPreprocessing,
    verifier_preprocessing: VerifierPreprocessing,
}

/// Program inputs for completeness testing.
#[derive(Debug, Clone, Arbitrary)]
pub struct ProgramInputs {
    pub data: Vec<u8>,
}

impl VerifierCompletenessInvariant {
    pub fn new(test_case: Arc<TestCase>) -> Self {
        Self { test_case }
    }
}

impl Invariant for VerifierCompletenessInvariant {
    type Setup = VerifierCompletenessSetup;
    type Input = ProgramInputs;

    fn name(&self) -> &str {
        "verifier_completeness"
    }

    fn description(&self) -> String {
        "For a fixed program, input, and honest prover output/proof, \
         the verifier accepts the honest output/proof."
            .to_string()
    }

    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::Fuzz
    }

    fn setup(&self) -> Self::Setup {
        let prover_pp = self.test_case.prover_preprocessing();
        let verifier_pp = TestCase::verifier_preprocessing(&prover_pp);
        VerifierCompletenessSetup {
            test_case: Arc::clone(&self.test_case),
            prover_preprocessing: prover_pp,
            verifier_preprocessing: verifier_pp,
        }
    }

    fn check(&self, setup: &Self::Setup, input: ProgramInputs) -> Result<(), InvariantViolation> {
        let (proof, io_device) = setup
            .test_case
            .prove(&setup.prover_preprocessing, &input.data);

        // If the guest panicked, skip -- we only care about non-panicking executions
        if io_device.panic {
            return Ok(());
        }

        TestCase::verify(&setup.verifier_preprocessing, proof, &io_device).map_err(|e| {
            InvariantViolation::with_details(
                "Verifier rejected honest proof",
                format!("inputs: {} bytes, error: {e}", input.data.len()),
            )
        })
    }

    fn seed_corpus(&self) -> Vec<Self::Input> {
        vec![
            ProgramInputs { data: vec![] },
            ProgramInputs {
                data: vec![0u8; 32],
            },
        ]
    }
}
