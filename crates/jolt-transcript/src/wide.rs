//! Opt-in standalone BN254 policy; shared Jolt samplers are unchanged.

use jolt_field::{Fr, Ring};

use crate::{Blake2bTranscript, Label, Transcript};

/// BN254 challenges from three Blake2b-512 spongefish scalar draws.
///
/// The fixed session identifies the backend and 384-bit reduction policy. The
/// application label is absorbed before any challenge. Use the same instance
/// throughout Spartan and its PCS opening, and authenticate the application's
/// relation/setup policy separately. There is no constructor from another
/// transcript backend: the byte-distribution premise does not cover Poseidon.
///
/// Conditional on three independent uniform 128-bit scalar draws, each output
/// has probability at most `1/p + 2^-384`. This is not an extraction or
/// Fiat–Shamir composition theorem, nor a concrete security-level claim.
pub struct Bn254WideBlake2bTranscript {
    inner: Blake2bTranscript<Fr>,
}

impl Bn254WideBlake2bTranscript {
    fn reduce([a, b, c]: [Fr; 3]) -> Fr {
        let radix = Fr::from_u128(u128::MAX) + Fr::from_u64(1);
        (a * radix + b) * radix + c
    }
}

impl Default for Bn254WideBlake2bTranscript {
    fn default() -> Self {
        Self::new(b"default")
    }
}

impl Transcript for Bn254WideBlake2bTranscript {
    type Challenge = Fr;

    fn new(label: &'static [u8]) -> Self {
        let mut inner = Blake2bTranscript::new(b"bn254-blake2b-wide384-v1");
        inner.append_bytes(&(label.len() as u64).to_be_bytes());
        inner.append(&Label(label));
        Self { inner }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.inner.append_bytes(bytes);
    }

    fn challenge(&mut self) -> Fr {
        Self::reduce([
            self.inner.challenge_scalar(),
            self.inner.challenge_scalar(),
            self.inner.challenge_scalar(),
        ])
    }

    fn challenge_scalar(&mut self) -> Fr {
        self.challenge()
    }

    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::{CanonicalBytes, CanonicalEncoding, Zero};

    use super::*;

    #[test]
    fn reduction_matches_independent_integer_vectors() {
        // Python integer oracle: N % p, encoded as 32 little-endian bytes.
        let vectors: [([u128; 3], [u8; 32]); 3] = [
            (
                [1, 2, 3],
                [
                    254, 255, 255, 79, 28, 52, 150, 172, 41, 205, 96, 159, 149, 118, 252, 54, 48,
                    70, 121, 120, 111, 163, 110, 102, 47, 223, 7, 154, 193, 119, 10, 14,
                ],
            ),
            (
                [
                    340_282_366_920_938_463_463_374_607_431_768_211_455,
                    340_282_366_920_938_463_463_374_607_431_768_211_455,
                    340_282_366_920_938_463_463_374_607_431_768_211_455,
                ],
                [
                    184, 254, 140, 239, 129, 218, 117, 176, 140, 205, 182, 165, 204, 42, 241, 167,
                    123, 191, 87, 121, 4, 117, 196, 50, 94, 162, 255, 72, 215, 129, 213, 3,
                ],
            ),
            (
                [
                    1_512_366_075_204_170_929_049_582_354_406_559_215,
                    338_770_000_845_734_292_534_325_025_077_361_652_240,
                    703_710,
                ],
                [
                    82, 202, 177, 115, 118, 66, 223, 38, 106, 43, 10, 235, 68, 145, 157, 114, 103,
                    153, 13, 16, 84, 95, 214, 54, 113, 45, 22, 87, 34, 145, 65, 47,
                ],
            ),
        ];
        for (limbs, expected) in vectors {
            let actual = Bn254WideBlake2bTranscript::reduce(limbs.map(Fr::from_u128));
            let mut bytes = [0; 32];
            actual.to_bytes_le(&mut bytes);
            assert_eq!(bytes, expected);
        }
        assert_eq!(
            Bn254WideBlake2bTranscript::reduce([Fr::zero(); 3]),
            Fr::zero()
        );
    }

    #[test]
    fn policy_binding_and_all_draw_entry_points_match_scalar_byte_oracle() {
        let mut actual = Bn254WideBlake2bTranscript::new(b"vector");
        let mut raw = Blake2bTranscript::<Fr>::new(b"bn254-blake2b-wide384-v1");
        // Application domain is its big-endian u64 length and padded 32-byte word.
        let mut label = [0; 32];
        for (dst, src) in label.iter_mut().zip(b"vector") {
            *dst = *src;
        }
        raw.append_bytes(&6u64.to_be_bytes());
        raw.append_bytes(&label);
        actual.append_bytes(b"statement");
        raw.append_bytes(b"statement");
        let samples = [actual.challenge(), actual.challenge_scalar()]
            .into_iter()
            .chain(actual.challenge_vector(3));
        for sample in samples {
            let mut little_endian = Vec::with_capacity(48);
            let draws: [Fr; 3] = core::array::from_fn(|_| raw.challenge_scalar());
            for draw in draws.into_iter().rev() {
                let mut bytes = [0; 32];
                draw.to_bytes_le(&mut bytes);
                little_endian.extend(bytes.into_iter().take(16));
            }
            assert_eq!(sample, Fr::from_bytes_le_reduced(&little_endian));
        }
        assert_eq!(actual.state(), raw.state());
        assert_ne!(
            actual.state(),
            Blake2bTranscript::<Fr>::new(b"vector").state()
        );
        assert_ne!(
            Bn254WideBlake2bTranscript::new(b"vector").state(),
            Bn254WideBlake2bTranscript::new(b"vector\0").state()
        );
        assert_ne!(
            Bn254WideBlake2bTranscript::new(b"vector").state(),
            Bn254WideBlake2bTranscript::new(b"other").state()
        );
    }
}
