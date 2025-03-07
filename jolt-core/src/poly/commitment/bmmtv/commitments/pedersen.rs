use ark_ec::CurveGroup;
use ark_std::rand::Rng;
use std::marker::PhantomData;

use super::{random_generators, Dhc, Error};

use super::super::inner_products::{InnerProduct, MultiexponentiationInnerProduct};

#[derive(Clone)]
pub struct PedersenCommitment<G: CurveGroup> {
    _group: PhantomData<G>,
}

impl<G: CurveGroup> Dhc for PedersenCommitment<G> {
    type Scalar = G::ScalarField;
    type Message = G::ScalarField;
    type Param = G;
    type Output = G;

    fn setup<R: Rng>(rng: &mut R, size: usize) -> Result<Vec<Self::Param>, Error> {
        Ok(random_generators(rng, size))
    }

    fn commit(k: &[Self::Param], m: &[Self::Message]) -> std::result::Result<G, Error> {
        Ok(MultiexponentiationInnerProduct::<G>::inner_product(k, m)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Fr, G1Projective};
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    type Pedersen = PedersenCommitment<G1Projective>;
    const TEST_SIZE: usize = 8;

    #[test]
    fn pedersen_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let commit_keys = Pedersen::setup(&mut rng, TEST_SIZE).unwrap();
        let mut message = Vec::new();
        let mut wrong_message = Vec::new();
        for _ in 0..TEST_SIZE {
            message.push(Fr::rand(&mut rng));
            wrong_message.push(Fr::rand(&mut rng));
        }
        let com = Pedersen::commit(&commit_keys, &message).unwrap();
        assert!(Pedersen::verify(&commit_keys, &message, &com).unwrap());
        assert!(!Pedersen::verify(&commit_keys, &wrong_message, &com).unwrap());
        message.push(Fr::rand(&mut rng));
        assert!(Pedersen::verify(&commit_keys, &message, &com).is_err());
    }
}
