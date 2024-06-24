use crate::field::JoltField;
use ark_ec::CurveGroup;
use sha3::{Digest, Keccak256};

#[derive(Clone)]
pub struct ProofTranscript {
    // Ethereum compatible 256 bit running state
    state: [u8; 32],
    // We append an ordinal to each invoke of the hash
    n_rounds: u32,
}

impl ProofTranscript {
    pub fn new(label: &'static [u8]) -> Self {
        // Hash in the label
        let mut hasher = Keccak256::new();
        hasher.update(label);
        let out = hasher.finalize();

        Self {
            state: out.into(),
            n_rounds: 0,
        }
    }

    /// Gives the hasher object with the running seed and index added
    /// To load hash you must call finalize, after appending u8 vectors
    fn hasher(&self) -> Keccak256 {
        // Note we add the extra memory here to improve the ease of eth integrations
        Keccak256::new()
            .chain_update(self.state)
            .chain_update([0_u8; 24])
            .chain_update(self.n_rounds.to_le_bytes())
    }

    pub fn append_message(&mut self, label: &'static [u8], msg: &'static [u8]) {
        // Instantiate hasher add our seed, position and msg
        let hasher = self.hasher().chain_update(label).chain_update(msg);
        self.state = hasher.finalize().into();
        self.n_rounds += 1;
    }

    pub fn append_bytes(&mut self, label: &'static [u8], bytes: &[u8]) {
        // Add the message and label
        let hasher = self.hasher().chain_update(label).chain_update(bytes);
        self.state = hasher.finalize().into();
        self.n_rounds += 1;
    }

    pub fn append_u64(&mut self, label: &'static [u8], x: u64) {
        let hasher = self
            .hasher()
            .chain_update(label)
            .chain_update(x.to_le_bytes());
        self.state = hasher.finalize().into();
        self.n_rounds += 1;
    }

    pub fn append_protocol_name(&mut self, protocol_name: &'static [u8]) {
        self.append_message(b"protocol-name", protocol_name);
    }

    pub fn append_scalar<F: JoltField>(&mut self, label: &'static [u8], scalar: &F) {
        let mut buf = vec![];
        scalar.serialize_compressed(&mut buf).unwrap();
        self.append_bytes(label, &buf);
    }

    pub fn append_scalars<F: JoltField>(&mut self, label: &'static [u8], scalars: &[F]) {
        self.append_message(label, b"begin_append_vector");
        for item in scalars.iter() {
            self.append_scalar(label, item);
        }
        self.append_message(label, b"end_append_vector");
    }

    pub fn append_point<G: CurveGroup>(&mut self, label: &'static [u8], point: &G) {
        let mut buf = vec![];
        point.serialize_compressed(&mut buf).unwrap();
        self.append_bytes(label, &buf);
    }

    pub fn append_points<G: CurveGroup>(&mut self, label: &'static [u8], points: &[G]) {
        self.append_message(label, b"begin_append_vector");
        for item in points.iter() {
            self.append_point(label, item);
        }
        self.append_message(label, b"end_append_vector");
    }

    pub fn challenge_scalar<F: JoltField>(&mut self, label: &'static [u8]) -> F {
        let mut buf = vec![0u8; F::NUM_BYTES];
        self.challenge_bytes(label, &mut buf);
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

    // Loads arbitrary byte lengths using ceil(out/32) invocations of 32 byte randoms
    // Discards top bits when the size is less than 32 bytes
    fn challenge_bytes(&mut self, label: &'static [u8], out: &mut [u8]) {
        let mut remaining_len = out.len();
        let mut start = 0;
        while remaining_len > 32 {
            self.challenge_bytes32(label, &mut out[start..start + 32]);
            start += 32;
            remaining_len -= 32;
        }
        // We load a full 32 byte random region
        let mut full_rand = vec![0_u8; 32];
        self.challenge_bytes32(label, &mut full_rand);
        // Then only clone the first bits of this random region to perfectly fill out
        out[start..start + remaining_len].clone_from_slice(&full_rand[0..remaining_len]);
    }

    // Loads exactly 32 bytes from the transcript by hashing the seed with the round constant
    fn challenge_bytes32(&mut self, label: &'static [u8], out: &mut [u8]) {
        assert_eq!(32, out.len());
        let rand: [u8; 32] = self.hasher().chain_update(label).finalize().into();
        out.clone_from_slice(rand.as_slice());
        self.state = rand;
        self.n_rounds += 1;
    }
}

pub trait AppendToTranscript {
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript);
}
