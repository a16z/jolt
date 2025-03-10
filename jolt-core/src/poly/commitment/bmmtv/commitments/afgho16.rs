use ark_ec::pairing::{Pairing, PairingOutput};
use ark_std::rand::Rng;
use std::marker::PhantomData;

use super::{random_generators, Dhc, Error};

use super::super::inner_products::{InnerProduct, PairingInnerProduct};

/// Afgho commitment is simply an inner pairing product commitment
#[derive(Clone)]
pub struct AfghoCommitment<P: Pairing> {
    _pair: PhantomData<P>,
}

impl<P: Pairing> Dhc for AfghoCommitment<P> {
    type Scalar = P::ScalarField;
    type Message = P::G1;
    type Param = P::G2;
    type Output = PairingOutput<P>;

    fn setup<R: Rng>(rng: &mut R, size: usize) -> Result<Vec<Self::Param>, Error> {
        Ok(random_generators(rng, size))
    }

    fn commit(k: &[Self::Param], m: &[Self::Message]) -> Result<Self::Output, Error> {
        Ok(PairingInnerProduct::<P>::inner_product(m, k)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Bn254;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    type BnG1 = <Bn254 as Pairing>::G1;

    type CommitG1 = AfghoCommitment<Bn254>;
    const TEST_SIZE: usize = 8;

    #[test]
    fn afgho_g1_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let params = CommitG1::setup(&mut rng, TEST_SIZE).unwrap();

        // random message
        let mut message = (0..TEST_SIZE)
            .map(|_| BnG1::rand(&mut rng))
            .collect::<Vec<_>>();

        let wrong_message = (0..TEST_SIZE)
            .map(|_| BnG1::rand(&mut rng))
            .collect::<Vec<_>>();

        // commitment to message
        let com = CommitG1::commit(&params, &message).unwrap();
        assert!(CommitG1::verify(&params, &message, &com).unwrap());

        // should return false with wrong message
        assert!(!CommitG1::verify(&params, &wrong_message, &com).unwrap());

        // should throw error if size is not the same
        message.push(BnG1::rand(&mut rng));
        assert!(CommitG1::verify(&params, &message, &com).is_err());
    }
}
