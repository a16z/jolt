//! Simple toy Fiat–Shamir transcript for testing.

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use blake2::Blake2s256;
use digest::{Digest, Output};

/// Hash-based transcript using bn254 Fr field and Blake2s256 hasher.
#[derive(Clone)]
pub struct ToyTranscript {
    hasher: Blake2s256,
}

impl ToyTranscript {
    /// Constructor over some `domain`
    pub fn new(domain_label: &[u8]) -> Self {
        let mut hasher = Blake2s256::default();
        hasher.update(domain_label);
        Self { hasher }
    }

    /* ---------------- append helpers ---------------- */

    /// Append arbitrary bytes.
    pub fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.hasher.update(label);
        self.hasher.update(&(bytes.len() as u64).to_le_bytes());
        self.hasher.update(bytes);
    }

    /// Append a single field element (compressed as canonical little-endian).
    pub fn append_field(&mut self, label: &[u8], x: &Fr) {
        self.append_bytes(label, &x.into_bigint().to_bytes_le());
    }

    /// Append any `Group` element in compressed form
    pub fn append_group<G: CanonicalSerialize>(&mut self, label: &[u8], g: &G) {
        let mut bytes = Vec::new();
        g.serialize_compressed(&mut bytes) // ark-serialize helper
            .expect("serialization");
        self.append_bytes(label, &bytes);
    }

    /// Append any serde-serializable element
    pub fn append_serde<G: serde::Serialize>(&mut self, label: &[u8], g: &G) {
        match bincode::serialize(g) {
            Ok(bytes) => self.append_bytes(label, &bytes),
            Err(_) => panic!("bincode serialization failed"),
        }
    }

    /* ---------------- challenge helpers ---------- */

    /// Sample a **non-zero** field element deterministically from the current state.
    pub fn challenge_scalar(&mut self, label: &[u8]) -> Fr {
        let mut h = self.hasher.clone();
        h.update(label);
        let digest: Output<Blake2s256> = h.finalize();

        let repr = digest.as_slice().to_vec();

        let fe = Fr::from_le_bytes_mod_order(&repr);

        if fe.is_zero() {
            panic!("Challenge value cannot be identity")
        } else {
            fe
        }
    }
}

impl crate::transcript::Transcript for ToyTranscript {
    type Scalar = Fr;

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        ToyTranscript::append_bytes(self, label, bytes);
    }

    fn append_field(&mut self, label: &[u8], x: &Self::Scalar) {
        ToyTranscript::append_field(self, label, x);
    }

    fn append_group<G: CanonicalSerialize>(&mut self, label: &[u8], g: &G) {
        ToyTranscript::append_group(self, label, g);
    }

    fn append_serde<S: serde::Serialize>(&mut self, label: &[u8], s: &S) {
        ToyTranscript::append_serde(self, label, s);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> Self::Scalar {
        ToyTranscript::challenge_scalar(self, label)
    }

    fn reset(&mut self, domain_label: &[u8]) {
        let mut hasher = Blake2s256::default();
        hasher.update(domain_label);
        self.hasher = hasher;
    }
}

#[test]
fn transcript_consistency() {
    let mut t1 = ToyTranscript::new(b"demo");
    let mut t2 = ToyTranscript::new(b"demo");

    // same sequence of messages => same challenge
    t1.append_bytes(b"m", b"hello");
    t2.append_bytes(b"m", b"hello");
    assert_eq!(t1.challenge_scalar(b"x"), t2.challenge_scalar(b"x"));
}
