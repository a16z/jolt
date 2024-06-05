use crate::field::JoltField;
use ark_ec::CurveGroup;
use merlin::Transcript;

#[derive(Clone)]
pub struct ProofTranscript {
    inner: Transcript,
}

impl ProofTranscript {
    pub fn new(label: &'static [u8]) -> Self {
        Self {
            inner: Transcript::new(label),
        }
    }

    pub fn append_message(&mut self, label: &'static [u8], msg: &'static [u8]) {
        self.inner.append_message(label, msg);
    }

    pub fn append_bytes(&mut self, label: &'static [u8], bytes: &[u8]) {
        self.inner.append_message(label, bytes);
    }

    pub fn append_u64(&mut self, label: &'static [u8], x: u64) {
        self.inner.append_u64(label, x);
    }

    pub fn append_protocol_name(&mut self, protocol_name: &'static [u8]) {
        self.append_message(b"protocol-name", protocol_name);
    }

    pub fn append_scalar<F: JoltField>(&mut self, label: &'static [u8], scalar: &F) {
        let mut buf = vec![];
        scalar.serialize_compressed(&mut buf).unwrap();
        self.inner.append_message(label, &buf);
    }

    pub fn append_scalars<F: JoltField>(&mut self, label: &'static [u8], scalars: &[F]) {
        self.append_message(label, b"begin_append_vector");
        for item in scalars.iter() {
            self.append_scalar(label, item);
        }
        self.inner.append_message(label, b"end_append_vector");
    }

    pub fn append_point<G: CurveGroup>(&mut self, label: &'static [u8], point: &G) {
        let mut buf = vec![];
        point.serialize_compressed(&mut buf).unwrap();
        self.inner.append_message(label, &buf);
    }

    pub fn append_points<G: CurveGroup>(&mut self, label: &'static [u8], points: &[G]) {
        self.append_message(label, b"begin_append_vector");
        for item in points.iter() {
            self.append_point(label, item);
        }
        self.inner.append_message(label, b"end_append_vector");
    }

    pub fn challenge_scalar<F: JoltField>(&mut self, label: &'static [u8]) -> F {
        let mut buf = vec![0u8; F::NUM_BYTES];
        self.inner.challenge_bytes(label, &mut buf);
        F::from_bytes(&buf)
    }

    pub fn challenge_vector<F: JoltField>(&mut self, label: &'static [u8], len: usize) -> Vec<F> {
        (0..len)
            .map(|_i| self.challenge_scalar(label))
            .collect::<Vec<F>>()
    }

    // Compute powers of scalar q : (1, q, q^2, ..., q^(len-1))
    pub fn challenge_scalar_powers<F: JoltField>(
        &mut self,
        label: &'static [u8],
        len: usize,
    ) -> Vec<F> {
        let q: F = self.challenge_scalar(label);
        let mut q_powers = vec![F::one(); len];
        for i in 1..len {
            q_powers[i] = q_powers[i - 1] * q;
        }
        q_powers
    }
}

pub trait AppendToTranscript {
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript);
}
