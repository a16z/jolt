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

/// Represents an [`AfghoCommitment`] in G1
#[derive(Clone)]
pub struct AfghoCommitmentG1<P: Pairing>(AfghoCommitment<P>);

/// Represents an [`AfghoCommitment`] in G2
#[derive(Clone)]
pub struct AfghoCommitmentG2<P: Pairing>(AfghoCommitment<P>);

impl<P: Pairing> Dhc for AfghoCommitmentG1<P> {
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

impl<P: Pairing> Dhc for AfghoCommitmentG2<P> {
    type Scalar = P::ScalarField;
    type Message = P::G2;
    type Param = P::G1;
    type Output = PairingOutput<P>;

    fn setup<R: Rng>(rng: &mut R, size: usize) -> Result<Vec<Self::Param>, Error> {
        Ok(random_generators(rng, size))
    }

    fn commit(k: &[Self::Param], m: &[Self::Message]) -> Result<PairingOutput<P>, Error> {
        Ok(PairingInnerProduct::<P>::inner_product(k, m)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Bn254;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    type BnG1 = <Bn254 as Pairing>::G1;
    type BnG2 = <Bn254 as Pairing>::G2;

    type CommitG1 = AfghoCommitmentG1<Bn254>;
    type CommitG2 = AfghoCommitmentG2<Bn254>;
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

    #[test]
    fn afgho_g2_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let commit_keys = CommitG2::setup(&mut rng, TEST_SIZE).unwrap();
        // random message
        let mut message = (0..TEST_SIZE)
            .map(|_| BnG2::rand(&mut rng))
            .collect::<Vec<_>>();

        let wrong_message = (0..TEST_SIZE)
            .map(|_| BnG2::rand(&mut rng))
            .collect::<Vec<_>>();

        let com = CommitG2::commit(&commit_keys, &message).unwrap();
        assert!(CommitG2::verify(&commit_keys, &message, &com).unwrap());
        assert!(!CommitG2::verify(&commit_keys, &wrong_message, &com).unwrap());

        message.push(BnG2::rand(&mut rng));

        assert!(CommitG2::verify(&commit_keys, &message, &com).is_err());
    }
}
