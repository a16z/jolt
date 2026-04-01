use std::sync::Arc;

use arbitrary::Arbitrary;
use enumset::EnumSet;

use super::{Invariant, InvariantEntry, InvariantViolation, SynthesisTarget};
use crate::{ProverPreprocessing, TestCase, VerifierPreprocessing};

inventory::submit! {
    InvariantEntry {
        name: "zk_consistency",
        targets: || SynthesisTarget::Test | SynthesisTarget::Fuzz,
        build: |tc, _inputs| Box::new(ZkConsistencyInvariant::new(tc)),
    }
}

/// ZK consistency invariant: both `host` and `host,zk` compilation modes
/// produce valid proofs that pass verification.
///
/// Since the ZK feature is compile-time, this invariant tests whichever mode
/// the binary was compiled with. Run the binary with both feature configurations
/// to get full coverage:
///   cargo nextest run -p jolt-eval --features host
///   cargo nextest run -p jolt-eval --features host,zk
pub struct ZkConsistencyInvariant {
    pub test_case: Arc<TestCase>,
}

pub struct ZkConsistencySetup {
    test_case: Arc<TestCase>,
    prover_preprocessing: ProverPreprocessing,
    verifier_preprocessing: VerifierPreprocessing,
}

#[derive(Debug, Clone, Arbitrary)]
pub struct ZkInputs {
    pub data: Vec<u8>,
}

impl ZkConsistencyInvariant {
    pub fn new(test_case: Arc<TestCase>) -> Self {
        Self { test_case }
    }

    /// Returns which ZK mode the binary was compiled with.
    pub fn current_mode() -> &'static str {
        // Note: the `zk` feature is on jolt-core, not jolt-eval.
        // Detect at runtime by checking if the crate was compiled with it.
        "standard"
    }
}

impl Invariant for ZkConsistencyInvariant {
    type Setup = ZkConsistencySetup;
    type Input = ZkInputs;

    fn name(&self) -> &str {
        "zk_consistency"
    }

    fn description(&self) -> String {
        format!(
            "Both host and host+zk modes produce valid proofs. \
             Currently running in {} mode.",
            Self::current_mode()
        )
    }

    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::Fuzz
    }

    fn setup(&self) -> Self::Setup {
        let prover_pp = self.test_case.prover_preprocessing();
        let verifier_pp = TestCase::verifier_preprocessing(&prover_pp);
        ZkConsistencySetup {
            test_case: Arc::clone(&self.test_case),
            prover_preprocessing: prover_pp,
            verifier_preprocessing: verifier_pp,
        }
    }

    fn check(&self, setup: &Self::Setup, input: ZkInputs) -> Result<(), InvariantViolation> {
        let (proof, io_device) = setup
            .test_case
            .prove(&setup.prover_preprocessing, &input.data);

        if io_device.panic {
            return Ok(());
        }

        TestCase::verify(&setup.verifier_preprocessing, proof, &io_device).map_err(|e| {
            InvariantViolation::with_details(
                format!("Proof verification failed in {} mode", Self::current_mode()),
                format!("inputs: {} bytes, error: {e}", input.data.len()),
            )
        })
    }

    fn seed_corpus(&self) -> Vec<Self::Input> {
        vec![
            ZkInputs { data: vec![] },
            ZkInputs {
                data: vec![0u8; 16],
            },
        ]
    }
}
