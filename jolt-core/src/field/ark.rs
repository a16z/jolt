use ark_ff::{PrimeField, UniformRand};
use ark_std::Zero;

use super::{FieldOps, JoltField};

impl FieldOps for ark_bn254::Fr {}
impl<'a, 'b> FieldOps<&'b ark_bn254::Fr, ark_bn254::Fr> for &'a ark_bn254::Fr {}
impl<'b> FieldOps<&'b ark_bn254::Fr, ark_bn254::Fr> for ark_bn254::Fr {}

impl JoltField for ark_bn254::Fr {
    const NUM_BYTES: usize = 32;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        <Self as UniformRand>::rand(rng)
    }

    fn from_u64(n: u64) -> Option<Self> {
        <Self as ark_ff::PrimeField>::from_u64(n)
    }

    fn from_i64(val: i64) -> Self {
        if val > 0 {
            <Self as JoltField>::from_u64(val as u64).unwrap()
        } else {
            Self::zero() - <Self as JoltField>::from_u64(-(val) as u64).unwrap()
        }
    }

    fn to_u64(&self) -> Option<u64> {
        let bigint = self.into_bigint();
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        match <Self as JoltField>::from_u64(result) {
            None => None,
            Some(x) => {
                if x == *self {
                    Some(result)
                } else {
                    None
                }
            }
        }
    }

    fn square(&self) -> Self {
        <Self as ark_ff::Field>::square(self)
    }

    fn inverse(&self) -> Option<Self> {
        <Self as ark_ff::Field>::inverse(self)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::NUM_BYTES);
        ark_bn254::Fr::from_le_bytes_mod_order(bytes)
    }
}
