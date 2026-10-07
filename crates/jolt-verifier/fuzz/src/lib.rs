//! Shared fixture loading and argument-string mutation for the
//! `jolt-verifier` fuzz targets.
//!
//! Fixtures are the bundles `tests/generate_fuzz_fixture.rs` writes:
//! `(preprocessing, public_io, proof, trusted_advice_commitment)` in
//! bincode-serde encoding, with `proof` the NARG `JoltProof`.

use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::{DoryCommitment, DoryScheme};
use jolt_field::Fr;
use jolt_transcript::VerifierTranscript;
use jolt_verifier::{
    jolt_protocol_id, verify, JoltProof, JoltSponge, JoltVerifierPreprocessing, ProofHeader,
    VerifierError, ZkConfig, JOLT_SESSION,
};

pub type Preprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;

#[derive(Clone)]
pub struct Bundle {
    pub preprocessing: Preprocessing,
    pub public_io: JoltDevice,
    pub proof: JoltProof,
    pub trusted_advice_commitment: Option<DoryCommitment>,
}

impl Bundle {
    /// Decodes a checked-in fixture, rejecting trailing bytes.
    pub fn decode(bytes: &[u8]) -> Self {
        let ((preprocessing, public_io, proof, trusted_advice_commitment), consumed): (
            (Preprocessing, JoltDevice, JoltProof, Option<DoryCommitment>),
            usize,
        ) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .expect("fixture decodes");
        assert_eq!(consumed, bytes.len(), "fixture has trailing bytes");
        Self {
            preprocessing,
            public_io,
            proof,
            trusted_advice_commitment,
        }
    }

    /// Decodes a fixture and checks that the compiled verifier accepts it.
    pub fn decode_verified(bytes: &[u8]) -> Self {
        let bundle = Self::decode(bytes);
        bundle
            .verify(&bundle.proof)
            .expect("honest fixture proof must verify before tampering");
        bundle
    }

    pub fn verify(&self, proof: &JoltProof) -> Result<(), VerifierError> {
        verify::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &self.preprocessing,
            &self.public_io,
            proof,
            self.trusted_advice_commitment.as_ref(),
        )
    }

    /// The byte length of the proof header at the front of the argument
    /// string, as the verifier reads it.
    pub fn header_len(&self) -> usize {
        let narg = &self.proof.narg;
        let mut transcript = VerifierTranscript::<JoltSponge>::new(
            &jolt_protocol_id::<JoltSponge>(),
            JOLT_SESSION,
            narg,
        );
        ProofHeader::receive(&mut transcript).expect("honest fixture header decodes");
        narg.len() - transcript.remaining()
    }
}

const TAIL_WINDOW: usize = 4096;
const MAX_DELETE: usize = 64;

/// Applies one mutation chosen by `data` to an honest bundle and requires the
/// verifier to reject the result whenever it differs from the honest input.
///
/// Input layout: `data[0]` picks the bundle, `data[1]` the mutation,
/// `data[2..6]` a little-endian position, and `data[6..]` the payload. Most
/// mutations edit the argument string (flip, overwrite, insert, delete,
/// truncate, extend, with header- and tail-anchored flips so the short header
/// and the final opening are reached as often as the commitment block). The
/// rest edit the public statement the verifier binds outside the argument
/// string: public input, memory layout, trusted-advice commitment, program
/// preprocessing, and the declared protocol configuration.
pub fn tamper_must_reject(bundles: &[Bundle], data: &[u8], mode: &str) {
    if data.len() < 6 {
        return;
    }
    let bundle_index = data[0] as usize % bundles.len();
    let honest = &bundles[bundle_index];
    let mutation = data[1] % 12;
    let position = u32::from_le_bytes([data[2], data[3], data[4], data[5]]) as usize;
    let payload = &data[6..];
    let mask = payload.first().copied().unwrap_or(1).max(1);

    let mut statement = honest.clone();
    let narg = &mut statement.proof.narg;
    let len = narg.len();
    if len == 0 {
        return;
    }
    let at = position % len;
    match mutation {
        0 => narg[at] ^= mask,
        1 => narg[position % honest.header_len()] ^= mask,
        2 => narg[len - 1 - position % len.min(TAIL_WINDOW)] ^= mask,
        3 => {
            let end = len.min(at + payload.len());
            narg[at..end].copy_from_slice(&payload[..end - at]);
        }
        4 => {
            let inserted = if payload.is_empty() {
                &[0][..]
            } else {
                payload
            };
            narg.splice(at..at, inserted.iter().copied());
        }
        5 => {
            let count = 1 + mask as usize % MAX_DELETE;
            narg.drain(at..len.min(at + count));
        }
        6 => narg.truncate(at),
        7 => narg.extend_from_slice(if payload.is_empty() { &[0] } else { payload }),
        8 => {
            let inputs = &mut statement.public_io.inputs;
            if inputs.is_empty() {
                return;
            }
            let index = position % inputs.len();
            inputs[index] ^= mask;
        }
        9 => {
            let layout = &mut statement.public_io.memory_layout;
            layout.heap_size ^= 1;
        }
        10 => {
            statement.trusted_advice_commitment = match statement.trusted_advice_commitment {
                Some(_) => None,
                None => Some(DoryCommitment::default()),
            };
        }
        _ => {
            if position.is_multiple_of(2) {
                statement.preprocessing.program = bundles[(bundle_index + 1) % bundles.len()]
                    .preprocessing
                    .program
                    .clone();
            } else {
                let protocol = &mut statement.proof.protocol;
                protocol.zk = match protocol.zk {
                    ZkConfig::Transparent => ZkConfig::BlindFold,
                    ZkConfig::BlindFold => ZkConfig::Transparent,
                };
            }
        }
    }

    let unchanged = statement.proof == honest.proof
        && statement.public_io == honest.public_io
        && statement.trusted_advice_commitment == honest.trusted_advice_commitment
        && statement.preprocessing.program == honest.preprocessing.program;
    if unchanged {
        return;
    }
    let result = statement.verify(&statement.proof);
    assert!(
        result.is_err(),
        "verifier accepted a tampered {mode} proof (bundle {bundle_index}, mutation {mutation})"
    );
}
