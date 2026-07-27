//! The legacy [`Transcript`] vocabulary implemented directly over the
//! verifier-native [`jolt_transcript::LegacyBlake2bTranscript`], so the
//! packed (Akita) prover drives **one digest engine** end to end: the legacy
//! stage provers and the verifier-native helpers (`absorb_packed_commitment`,
//! `PackedBatch::prove_batch`) append to the very same transcript object,
//! with no state conversions anywhere (spec: jolt-verifier-akita, "Phase
//! B.1 — transcript unification").
//!
//! Byte compatibility with [`super::Blake2bTranscript`] holds because every
//! method below reproduces its framing verbatim over the shared raw
//! absorb/squeeze primitives, and both engines chain
//! `H(state ‖ 28 zero bytes ‖ n_rounds_be ‖ payload)` — pinned by the
//! parity test at the bottom of this file.

use jolt_field::TranscriptChallenge;
use jolt_transcript::{LegacyBlake2bTranscript, Transcript as VerifierTranscript};

use super::Transcript;
use crate::field::JoltField;

impl<Challenge: TranscriptChallenge> Transcript for LegacyBlake2bTranscript<Challenge> {
    fn new(label: &'static [u8]) -> Self {
        // Identical label framing on both engines: the label is right-padded
        // with zeros into one 32-byte block and hashed as the initial state.
        <Self as VerifierTranscript>::new(label)
    }

    #[cfg(test)]
    fn compare_to(&mut self, _other: Self) {
        unimplemented!(
            "state-history diagnostics live on the legacy Blake2bTranscript; \
             the verifier-native transcript does not record history"
        );
    }

    fn raw_append_label(&mut self, label: &'static [u8]) {
        // Labels must fit into one EVM word, right-padded with zeros
        // (matches Solidity's bytes32 string casting).
        assert!(label.len() < 33);
        let mut packed = [0u8; 32];
        packed[..label.len()].copy_from_slice(label);
        <Self as VerifierTranscript>::append_bytes(self, &packed);
    }

    fn raw_append_bytes(&mut self, bytes: &[u8]) {
        <Self as VerifierTranscript>::append_bytes(self, bytes);
    }

    fn raw_append_u64(&mut self, x: u64) {
        // 32-byte region, left-padded for EVM uint256 compatibility.
        let mut packed = [0u8; 32];
        packed[24..].copy_from_slice(&x.to_be_bytes());
        <Self as VerifierTranscript>::append_bytes(self, &packed);
    }

    fn raw_append_scalar<F: JoltField>(&mut self, scalar: &F) {
        let mut buf = vec![];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        // Uncompressed serialization is little-endian; reverse for the
        // EVM-compatible big-endian representation.
        buf.reverse();
        <Self as VerifierTranscript>::append_bytes(self, &buf);
    }

    fn challenge_u128(&mut self) -> u128 {
        let mut buf = [0u8; 16];
        self.raw_challenge_bytes(&mut buf);
        buf.reverse();
        u128::from_be_bytes(buf)
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        self.challenge_scalar_128_bits()
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        let mut buf = [0u8; 16];
        self.raw_challenge_bytes(&mut buf);
        buf.reverse();
        F::from_bytes(&buf)
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len).map(|_| self.challenge_scalar_128_bits()).collect()
    }

    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F = self.challenge_scalar_128_bits();
        let mut q_powers = vec![F::one(); len];
        for i in 1..len {
            q_powers[i] = q_powers[i - 1] * q;
        }
        q_powers
    }

    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
        F::Challenge::from(self.challenge_u128())
    }

    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
        (0..len)
            .map(|_| self.challenge_scalar_optimized::<F>())
            .collect()
    }

    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F::Challenge = self.challenge_scalar_optimized::<F>();
        let mut q_powers = vec![<F as ark_std::One>::one(); len];
        for i in 1..len {
            q_powers[i] = q * q_powers[i - 1];
        }
        q_powers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::Fr;

    type Native = LegacyBlake2bTranscript<jolt_field::Fr>;

    fn assert_states_match(legacy: &Blake2bTranscript, native: &Native, step: &str) {
        assert_eq!(
            legacy.state,
            VerifierTranscript::state(native),
            "engines diverged after {step}"
        );
    }

    /// The vocabulary-parity probe (spec: jolt-verifier-akita, phase B.1
    /// gate): one mixed absorb/squeeze sequence driven through the legacy
    /// `Blake2bTranscript` and through this adapter, asserting identical
    /// states after every operation — including a segment where the adapter
    /// is driven through the *verifier-native* vocabulary, the interop the
    /// packed prover relies on in place of state handoffs.
    #[test]
    fn legacy_and_verifier_native_engines_stay_byte_identical() {
        let mut legacy = <Blake2bTranscript as Transcript>::new(b"parity probe");
        let mut native = <Native as Transcript>::new(b"parity probe");
        assert_states_match(&legacy, &native, "new");

        legacy.append_label(b"label");
        Transcript::append_label(&mut native, b"label");
        assert_states_match(&legacy, &native, "append_label");

        legacy.append_bytes(b"bytes", &[1, 2, 3, 4, 5]);
        Transcript::append_bytes(&mut native, b"bytes", &[1, 2, 3, 4, 5]);
        assert_states_match(&legacy, &native, "append_bytes");

        legacy.append_u64(b"u64", 0xdead_beef);
        Transcript::append_u64(&mut native, b"u64", 0xdead_beef);
        assert_states_match(&legacy, &native, "append_u64");

        let scalar = Fr::from(1234567890123456789u64);
        legacy.append_scalar(b"scalar", &scalar);
        Transcript::append_scalar(&mut native, b"scalar", &scalar);
        assert_states_match(&legacy, &native, "append_scalar");

        let c1: Fr = legacy.challenge_scalar();
        let c2: Fr = Transcript::challenge_scalar(&mut native);
        assert_eq!(c1, c2, "challenge_scalar diverged");
        assert_states_match(&legacy, &native, "challenge_scalar");

        assert_eq!(
            legacy.challenge_u128(),
            Transcript::challenge_u128(&mut native),
            "challenge_u128 diverged"
        );

        let v1: Vec<Fr> = legacy.challenge_vector(3);
        let v2: Vec<Fr> = Transcript::challenge_vector(&mut native, 3);
        assert_eq!(v1, v2, "challenge_vector diverged");

        let p1: Vec<Fr> = legacy.challenge_scalar_powers(4);
        let p2: Vec<Fr> = Transcript::challenge_scalar_powers(&mut native, 4);
        assert_eq!(p1, p2, "challenge_scalar_powers diverged");

        let o1 = legacy.challenge_scalar_optimized::<Fr>();
        let o2 = Transcript::challenge_scalar_optimized::<Fr>(&mut native);
        assert_eq!(Fr::from(o1), Fr::from(o2), "optimized challenge diverged");
        assert_states_match(&legacy, &native, "challenge sequence");

        // The interop the packed prover relies on: mid-protocol, the SAME
        // adapter object is driven through the verifier-native vocabulary
        // (as `absorb_packed_commitment` / `PackedBatch::prove_batch` do)
        // while the legacy engine absorbs the equivalent raw bytes.
        legacy.raw_append_bytes(b"native vocabulary segment");
        VerifierTranscript::append_bytes(&mut native, b"native vocabulary segment");
        assert_states_match(&legacy, &native, "native-vocabulary absorb");

        let c1: Fr = legacy.challenge_scalar();
        let c2: Fr = VerifierTranscript::challenge_scalar(&mut native).into();
        assert_eq!(c1, c2, "post-interop challenge diverged");
        assert_states_match(&legacy, &native, "post-interop challenge");
    }
}
