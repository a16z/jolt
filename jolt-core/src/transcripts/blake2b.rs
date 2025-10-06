use super::transcript::Transcript;
use crate::field::JoltField;
use ark_ec::{AffineRepr, CurveGroup};
use ark_serialize::CanonicalSerialize;
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};

type Blake2b256 = Blake2b<U32>;
use std::borrow::Borrow;

/// Represents the current state of the protocol's Fiat-Shamir transcript using Blake2b.
#[derive(Default, Clone)]
pub struct Blake2bTranscript {
    /// 256-bit running state
    pub state: [u8; 32],
    /// We append an ordinal to each invocation of the hash
    n_rounds: u32,
    #[cfg(test)]
    /// A complete history of the transcript's `state`; used for testing.
    state_history: Vec<[u8; 32]>,
    #[cfg(test)]
    /// For a proof to be valid, the verifier's `state_history` should always match
    /// the prover's. In testing, the Jolt verifier may be provided the prover's
    /// `state_history` so that we can detect any deviations and the backtrace can
    /// tell us where it happened.
    expected_state_history: Option<Vec<[u8; 32]>>,
}

impl Blake2bTranscript {
    /// Gives the hasher object with the running seed and index added
    /// To load hash you must call finalize, after appending u8 vectors
    fn hasher(&self) -> Blake2b256 {
        let mut packed = [0_u8; 28].to_vec();
        packed.append(&mut self.n_rounds.to_be_bytes().to_vec());
        Blake2b256::new()
            .chain_update(self.state)
            .chain_update(&packed)
    }

    // Loads arbitrary byte lengths using ceil(out/32) invocations of 32 byte randoms
    // Discards top bits when the size is less than 32 bytes
    fn challenge_bytes(&mut self, out: &mut [u8]) {
        let mut remaining_len = out.len();
        let mut start = 0;
        while remaining_len > 32 {
            self.challenge_bytes32(&mut out[start..start + 32]);
            start += 32;
            remaining_len -= 32;
        }
        // We load a full 32 byte random region
        let mut full_rand = vec![0_u8; 32];
        self.challenge_bytes32(&mut full_rand);
        // Then only clone the first bits of this random region to perfectly fill out
        out[start..start + remaining_len].clone_from_slice(&full_rand[0..remaining_len]);
    }

    // Loads exactly 32 bytes from the transcript by hashing the seed with the round constant
    fn challenge_bytes32(&mut self, out: &mut [u8]) {
        assert_eq!(32, out.len());
        let rand: [u8; 32] = self.hasher().finalize().into();
        out.clone_from_slice(rand.as_slice());
        self.update_state(rand);
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds += 1;
        #[cfg(test)]
        {
            if let Some(expected_state_history) = &self.expected_state_history {
                assert!(
                    new_state == expected_state_history[self.n_rounds as usize],
                    "Fiat-Shamir transcript mismatch"
                );
            }
            self.state_history.push(new_state);
        }
    }
}

impl Transcript for Blake2bTranscript {
    fn new(label: &'static [u8]) -> Self {
        // Hash in the label
        assert!(label.len() < 33);
        let hasher = if label.len() == 32 {
            Blake2b256::new().chain_update(label)
        } else {
            let zeros = vec![0_u8; 32 - label.len()];
            Blake2b256::new().chain_update(label).chain_update(zeros)
        };
        let out = hasher.finalize();

        Self {
            state: out.into(),
            n_rounds: 0,
            #[cfg(test)]
            state_history: vec![out.into()],
            #[cfg(test)]
            expected_state_history: None,
        }
    }

    #[cfg(test)]
    /// Compare this transcript to `other` and panic if/when they deviate.
    /// Typically used to compare the verifier's transcript to the prover's.
    fn compare_to(&mut self, other: Self) {
        self.expected_state_history = Some(other.state_history);
    }

    fn append_message(&mut self, msg: &'static [u8]) {
        // We require all messages to fit into one evm word and then right pad them
        // right padding matches the format of the strings when cast to bytes 32 in solidity
        assert!(msg.len() < 33);
        let hasher = if msg.len() == 32 {
            self.hasher().chain_update(msg)
        } else {
            let mut packed = msg.to_vec();
            packed.append(&mut vec![0_u8; 32 - msg.len()]);
            self.hasher().chain_update(packed)
        };
        // Instantiate hasher add our seed, position and msg
        self.update_state(hasher.finalize().into());
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        // Add the message and label
        let hasher = self.hasher().chain_update(bytes);
        self.update_state(hasher.finalize().into());
    }

    fn append_u64(&mut self, x: u64) {
        // Allocate into a 32 byte region
        let mut packed = [0_u8; 24].to_vec();
        packed.append(&mut x.to_be_bytes().to_vec());
        let hasher = self.hasher().chain_update(packed.clone());
        self.update_state(hasher.finalize().into());
    }

    fn append_scalar<F: JoltField>(&mut self, scalar: &F) {
        let mut buf = vec![];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        // Serialize uncompressed gives the scalar in LE byte order which is not
        // a natural representation in the EVM for scalar math so we reverse
        // to get an EVM compatible version.
        buf = buf.into_iter().rev().collect();
        self.append_bytes(&buf);
    }

    fn append_serializable<F: CanonicalSerialize>(&mut self, scalar: &F) {
        let mut buf = vec![];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        // Serialize uncompressed gives the scalar in LE byte order which is not
        // a natural representation in the EVM for scalar math so we reverse
        // to get an EVM compatible version.
        buf = buf.into_iter().rev().collect();
        self.append_bytes(&buf);
    }

    fn append_scalars<F: JoltField>(&mut self, scalars: &[impl Borrow<F>]) {
        self.append_message(b"begin_append_vector");
        for item in scalars.iter() {
            self.append_scalar(item.borrow());
        }
        self.append_message(b"end_append_vector");
    }

    fn append_point<G: CurveGroup>(&mut self, point: &G) {
        // If we add the point at infinity then we hash over a region of zeros
        if point.is_zero() {
            self.append_bytes(&[0_u8; 64]);
            return;
        }

        let aff = point.into_affine();
        let mut x_bytes = vec![];
        let mut y_bytes = vec![];
        // The native serialize for the points are le encoded in x,y format and simply reversing
        // can lead to errors so we extract the affine coordinates and the encode them be before writing
        let x = aff.x().unwrap();
        x.serialize_compressed(&mut x_bytes).unwrap();
        x_bytes = x_bytes.into_iter().rev().collect();
        let y = aff.y().unwrap();
        y.serialize_compressed(&mut y_bytes).unwrap();
        y_bytes = y_bytes.into_iter().rev().collect();

        let hasher = self.hasher().chain_update(x_bytes).chain_update(y_bytes);
        self.update_state(hasher.finalize().into());
    }

    fn append_points<G: CurveGroup>(&mut self, points: &[G]) {
        self.append_message(b"begin_append_vector");
        for item in points.iter() {
            self.append_point(item);
        }
        self.append_message(b"end_append_vector");
    }

    fn challenge_u128(&mut self) -> u128 {
        let mut buf = vec![0u8; 16];
        self.challenge_bytes(&mut buf);
        buf = buf.into_iter().rev().collect();
        u128::from_be_bytes(buf.try_into().unwrap())
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        // Under the hood all Fr are 128 bits for performance
        self.challenge_scalar_128_bits()
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        let mut buf = vec![0u8; 16];
        self.challenge_bytes(&mut buf);

        buf = buf.into_iter().rev().collect();
        F::from_bytes(&buf)
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len)
            .map(|_i| self.challenge_scalar())
            .collect::<Vec<F>>()
    }

    // Compute powers of scalar q : (1, q, q^2, ..., q^(len-1))
    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F = self.challenge_scalar();
        let mut q_powers = vec![F::one(); len];
        for i in 1..len {
            q_powers[i] = q_powers[i - 1] * q;
        }
        q_powers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use std::collections::HashSet;

    #[test]
    fn test_challenge_scalar_128_bits() {
        let mut transcript = Blake2bTranscript::new(b"test_128_bit_scalar");
        let mut scalars = HashSet::new();

        for i in 0..10000 {
            let scalar: Fr = transcript.challenge_scalar_128_bits();

            let num_bits = scalar.num_bits();
            assert!(
                num_bits <= 128,
                "Scalar at iteration {i} has {num_bits} bits, expected <= 128",
            );

            assert!(
                scalars.insert(scalar),
                "Duplicate scalar found at iteration {i}",
            );
        }
    }
}
