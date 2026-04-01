use std::sync::Arc;

use arbitrary::Arbitrary;
use enumset::EnumSet;

use super::{Invariant, InvariantEntry, InvariantViolation, SynthesisTarget};
use crate::{deserialize_proof, serialize_proof, TestCase};

inventory::submit! {
    InvariantEntry {
        name: "serialization_roundtrip",
        targets: || { SynthesisTarget::Test.into() },
        needs_guest: true,
        build: |tc, inputs| Box::new(SerializationRoundtripInvariant::new(tc.unwrap(), inputs)),
    }
}

/// Serialization roundtrip invariant: `deserialize(serialize(proof)) == proof`,
/// verified by checking that re-serialization produces identical bytes.
pub struct SerializationRoundtripInvariant {
    pub test_case: Arc<TestCase>,
    pub default_inputs: Vec<u8>,
}

pub struct SerializationRoundtripSetup {
    proof_bytes: Vec<u8>,
}

/// Unit input -- the roundtrip check has no variable input beyond the
/// proof generated during setup.
#[derive(Debug, Clone, Arbitrary, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct RoundtripInput {
    _dummy: u8,
}

impl SerializationRoundtripInvariant {
    pub fn new(test_case: Arc<TestCase>, default_inputs: Vec<u8>) -> Self {
        Self {
            test_case,
            default_inputs,
        }
    }
}

impl Invariant for SerializationRoundtripInvariant {
    type Setup = SerializationRoundtripSetup;
    type Input = RoundtripInput;

    fn name(&self) -> &str {
        "serialization_roundtrip"
    }

    fn description(&self) -> String {
        "deserialize(serialize(proof)) == proof, verified via byte-identical \
         re-serialization."
            .to_string()
    }

    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test.into()
    }

    fn setup(&self) -> Self::Setup {
        let prover_pp = self.test_case.prover_preprocessing();
        let (proof, _io) = self.test_case.prove(&prover_pp, &self.default_inputs);
        let proof_bytes = serialize_proof(&proof);
        SerializationRoundtripSetup { proof_bytes }
    }

    fn check(&self, setup: &Self::Setup, _input: RoundtripInput) -> Result<(), InvariantViolation> {
        let deserialized = deserialize_proof(&setup.proof_bytes).map_err(|e| {
            InvariantViolation::with_details("Deserialization failed", e.to_string())
        })?;

        let reserialized = serialize_proof(&deserialized);

        if setup.proof_bytes != reserialized {
            let first_diff = setup
                .proof_bytes
                .iter()
                .zip(reserialized.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(setup.proof_bytes.len().min(reserialized.len()));

            Err(InvariantViolation::with_details(
                "Serialization roundtrip mismatch",
                format!(
                    "bytes differ at offset {first_diff} (original={}, roundtripped={})",
                    setup.proof_bytes.len(),
                    reserialized.len()
                ),
            ))
        } else {
            Ok(())
        }
    }

    fn seed_corpus(&self) -> Vec<Self::Input> {
        vec![RoundtripInput { _dummy: 0 }]
    }
}
