//! Simple toy Fiat–Shamir transcript for testing.

use ark_ff::{BigInteger, PrimeField};
use ark_serialize::CanonicalSerialize;
use digest::{Digest, Output};

/// Hash-based transcript generic over any `PrimeField` and any `Digest`.
#[derive(Clone)]
pub struct ToyTranscript<F: PrimeField, H: Digest> {
    hasher: H,
    _marker: core::marker::PhantomData<F>,
}

impl<F: PrimeField, H: Digest + Default + Clone> ToyTranscript<F, H> {
    /// Constructor over some `domain`
    pub fn new(domain_label: &[u8]) -> Self {
        let mut hasher = H::default();
        hasher.update(domain_label);
        Self {
            hasher,
            _marker: core::marker::PhantomData,
        }
    }

    /* ---------------- append helpers ---------------- */

    /// Append arbitrary bytes.
    pub fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.hasher.update(label);
        self.hasher.update(&(bytes.len() as u64).to_le_bytes());
        self.hasher.update(bytes);
    }

    /// Append a single field element (compressed as canonical little-endian).
    pub fn append_field(&mut self, label: &[u8], x: &F) {
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
    pub fn challenge_scalar(&mut self, label: &[u8]) -> F {
        let mut h = self.hasher.clone();
        h.update(label);
        let digest: Output<H> = h.finalize();

        let repr = digest.as_slice().to_vec();

        let fe = F::from_le_bytes_mod_order(&repr);

        if fe.is_zero() {
            panic!("Challenge value cannot be identity") //TODO(markosg04): this is low chance?
        } else {
            fe
        }

        // fe
    }
}

impl<F: PrimeField, H: Digest + Default + Clone> crate::transcript::Transcript
    for ToyTranscript<F, H>
where
    F: PrimeField + crate::arithmetic::Field,
    H: Digest + Default + Clone,
{
    type Scalar = F;

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
        let mut hasher = H::default();
        hasher.update(domain_label);
        self.hasher = hasher;
    }
}

#[test]
fn transcript_consistency() {
    use blake2::Blake2s256;
    type Fr = ark_bn254::Fr;

    let mut t1 = ToyTranscript::<Fr, Blake2s256>::new(b"demo");
    let mut t2 = ToyTranscript::<Fr, Blake2s256>::new(b"demo");

    // same sequence of messages => same challenge
    t1.append_bytes(b"m", b"hello");
    t2.append_bytes(b"m", b"hello");
    assert_eq!(t1.challenge_scalar(b"x"), t2.challenge_scalar(b"x"));
}
