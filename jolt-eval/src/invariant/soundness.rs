use std::sync::Arc;

use arbitrary::Arbitrary;
use enumset::EnumSet;

use super::{Invariant, InvariantEntry, InvariantViolation, SynthesisTarget};
use crate::{serialize_proof, JoltDevice, Proof, TestCase, VerifierPreprocessing};

inventory::submit! {
    InvariantEntry {
        name: "soundness",
        targets: || SynthesisTarget::Test | SynthesisTarget::Fuzz | SynthesisTarget::RedTeam,
        build: |tc, inputs| Box::new(SoundnessInvariant::new(tc, inputs)),
    }
}

/// Mutation applied to a serialized proof to test soundness.
#[derive(Debug, Clone, Arbitrary, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ProofMutation {
    pub byte_index: usize,
    pub new_value: u8,
}

/// Pre-computed honest proof and verification data.
pub struct SoundnessSetup {
    proof_bytes: Vec<u8>,
    io_device: JoltDevice,
    verifier_preprocessing: VerifierPreprocessing,
}

/// Soundness invariant: for a fixed program and honest prover output/proof,
/// the verifier must reject any mutated (different) proof.
pub struct SoundnessInvariant {
    pub test_case: Arc<TestCase>,
    pub default_inputs: Vec<u8>,
}

impl SoundnessInvariant {
    pub fn new(test_case: Arc<TestCase>, default_inputs: Vec<u8>) -> Self {
        Self {
            test_case,
            default_inputs,
        }
    }
}

impl Invariant for SoundnessInvariant {
    type Setup = SoundnessSetup;
    type Input = ProofMutation;

    fn name(&self) -> &str {
        "soundness"
    }

    fn description(&self) -> String {
        "For a fixed program, input, and honest prover output/proof, \
         the verifier does not accept for any other output/proof."
            .to_string()
    }

    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::Fuzz | SynthesisTarget::RedTeam
    }

    fn setup(&self) -> Self::Setup {
        let prover_pp = self.test_case.prover_preprocessing();
        let verifier_pp = TestCase::verifier_preprocessing(&prover_pp);
        let (proof, io_device) = self.test_case.prove(&prover_pp, &self.default_inputs);
        let proof_bytes = serialize_proof(&proof);
        SoundnessSetup {
            proof_bytes,
            io_device,
            verifier_preprocessing: verifier_pp,
        }
    }

    fn check(&self, setup: &Self::Setup, input: ProofMutation) -> Result<(), InvariantViolation> {
        if setup.proof_bytes.is_empty() {
            return Ok(());
        }

        let idx = input.byte_index % setup.proof_bytes.len();

        // Skip no-op mutations
        if setup.proof_bytes[idx] == input.new_value {
            return Ok(());
        }

        let mut mutated = setup.proof_bytes.clone();
        mutated[idx] = input.new_value;

        // If deserialization fails, the mutation was caught
        let mutated_proof: Proof = match crate::deserialize_proof(&mutated) {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        // Verification of a mutated proof must fail
        match TestCase::verify(
            &setup.verifier_preprocessing,
            mutated_proof,
            &setup.io_device,
        ) {
            Ok(()) => Err(InvariantViolation::with_details(
                "Verifier accepted mutated proof",
                format!(
                    "mutation at byte {idx}: 0x{:02x} -> 0x{:02x}",
                    setup.proof_bytes[idx], input.new_value
                ),
            )),
            Err(_) => Ok(()),
        }
    }

    fn seed_corpus(&self) -> Vec<Self::Input> {
        vec![
            // Mutate first byte
            ProofMutation {
                byte_index: 0,
                new_value: 0xFF,
            },
            // Mutate a byte in the middle
            ProofMutation {
                byte_index: 1000,
                new_value: 0x00,
            },
            // Flip a single bit
            ProofMutation {
                byte_index: 42,
                new_value: 0x01,
            },
        ]
    }
}
