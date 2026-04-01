use std::sync::Arc;

use arbitrary::Arbitrary;
use enumset::EnumSet;

use super::{Invariant, InvariantEntry, InvariantViolation, SynthesisTarget};
use crate::{serialize_proof, ProverPreprocessing, TestCase};

inventory::submit! {
    InvariantEntry {
        name: "determinism",
        targets: || SynthesisTarget::Test | SynthesisTarget::Fuzz,
        build: |tc, _inputs| Box::new(DeterminismInvariant::new(tc)),
    }
}

/// Determinism invariant: same program + input must produce byte-identical proofs.
pub struct DeterminismInvariant {
    pub test_case: Arc<TestCase>,
}

pub struct DeterminismSetup {
    test_case: Arc<TestCase>,
    prover_preprocessing: ProverPreprocessing,
}

/// Program inputs for determinism testing.
#[derive(Debug, Clone, Arbitrary, serde::Serialize, serde::Deserialize)]
pub struct DeterminismInputs {
    pub data: Vec<u8>,
}

impl DeterminismInvariant {
    pub fn new(test_case: Arc<TestCase>) -> Self {
        Self { test_case }
    }
}

impl Invariant for DeterminismInvariant {
    type Setup = DeterminismSetup;
    type Input = DeterminismInputs;

    fn name(&self) -> &str {
        "determinism"
    }

    fn description(&self) -> String {
        "Same program + input must produce the same proof (byte-identical).".to_string()
    }

    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::Fuzz
    }

    fn setup(&self) -> Self::Setup {
        let prover_pp = self.test_case.prover_preprocessing();
        DeterminismSetup {
            test_case: Arc::clone(&self.test_case),
            prover_preprocessing: prover_pp,
        }
    }

    fn check(
        &self,
        setup: &Self::Setup,
        input: DeterminismInputs,
    ) -> Result<(), InvariantViolation> {
        let (proof1, io1) = setup
            .test_case
            .prove(&setup.prover_preprocessing, &input.data);
        let (proof2, io2) = setup
            .test_case
            .prove(&setup.prover_preprocessing, &input.data);

        let bytes1 = serialize_proof(&proof1);
        let bytes2 = serialize_proof(&proof2);

        if bytes1 != bytes2 {
            // Find first differing byte
            let first_diff = bytes1
                .iter()
                .zip(bytes2.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(bytes1.len().min(bytes2.len()));

            return Err(InvariantViolation::with_details(
                "Non-deterministic proof generation",
                format!(
                    "proofs differ at byte {first_diff} (len1={}, len2={})",
                    bytes1.len(),
                    bytes2.len()
                ),
            ));
        }

        // Also check that I/O is deterministic
        if io1.outputs != io2.outputs {
            return Err(InvariantViolation::new("Non-deterministic program outputs"));
        }

        if io1.panic != io2.panic {
            return Err(InvariantViolation::new("Non-deterministic panic behavior"));
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<Self::Input> {
        vec![
            DeterminismInputs { data: vec![] },
            DeterminismInputs {
                data: vec![1, 2, 3, 4],
            },
        ]
    }
}
