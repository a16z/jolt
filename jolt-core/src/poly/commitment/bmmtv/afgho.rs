use ark_ec::{
    pairing::{Pairing, PairingOutput},
    Group,
};
use ark_std::rand::Rng;
use eyre::Error;
use std::marker::PhantomData;

use super::inner_product::inner_product;

pub fn random_generators<R: Rng, G: Group>(rng: &mut R, num: usize) -> Vec<G> {
    (0..num).map(|_| G::rand(rng)).collect()
}

#[derive(Clone)]
pub struct AFGHOCommitment<P: Pairing> {
    _pair: PhantomData<P>,
}

#[derive(Clone)]
pub struct AFGHOCommitmentG1<P: Pairing>(AFGHOCommitment<P>);

#[derive(Clone)]
pub struct AFGHOCommitmentG2<P: Pairing>(AFGHOCommitment<P>);

impl<P: Pairing> AFGHOCommitmentG1<P> {
    fn setup<R: Rng>(rng: &mut R, size: usize) -> Result<Vec<P::G2>, Error> {
        Ok(random_generators(rng, size))
    }

    pub fn commit(k: &[P::G2], m: &[P::G1]) -> Result<PairingOutput<P>, Error> {
        inner_product(m, k)
    }

    fn verify(k: &[P::G2], m: &[P::G1], com: &PairingOutput<P>) -> Result<bool, Error> {
        Ok(Self::commit(k, m)? == *com)
    }
}

impl<P: Pairing> AFGHOCommitmentG2<P> {
    fn setup<R: Rng>(rng: &mut R, size: usize) -> Result<Vec<P::G1>, Error> {
        Ok(random_generators(rng, size))
    }

    fn commit(k: &[P::G1], m: &[P::G2]) -> Result<PairingOutput<P>, Error> {
        inner_product(k, m)
    }

    fn verify(k: &[P::G1], m: &[P::G2], com: &PairingOutput<P>) -> Result<bool, Error> {
        Ok(Self::commit(k, m)? == *com)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Bn254;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    type C1 = AFGHOCommitmentG1<Bn254>;
    type C2 = AFGHOCommitmentG2<Bn254>;
    const TEST_SIZE: usize = 8;

    #[test]
    fn afgho_g1_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let commit_keys = C1::setup(&mut rng, TEST_SIZE).unwrap();
        let mut message = Vec::new();
        let mut wrong_message = Vec::new();
        for _ in 0..TEST_SIZE {
            message.push(<Bn254 as Pairing>::G1::rand(&mut rng));
            wrong_message.push(<Bn254 as Pairing>::G1::rand(&mut rng));
        }
        let com = C1::commit(&commit_keys, &message).unwrap();
        assert!(C1::verify(&commit_keys, &message, &com).unwrap());
        assert!(!C1::verify(&commit_keys, &wrong_message, &com).unwrap());
        message.push(<Bn254 as Pairing>::G1::rand(&mut rng));
        assert!(C1::verify(&commit_keys, &message, &com).is_err());
    }

    #[test]
    fn afgho_g2_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let commit_keys = C2::setup(&mut rng, TEST_SIZE).unwrap();
        let mut message = Vec::new();
        let mut wrong_message = Vec::new();
        for _ in 0..TEST_SIZE {
            message.push(<Bn254 as Pairing>::G2::rand(&mut rng));
            wrong_message.push(<Bn254 as Pairing>::G2::rand(&mut rng));
        }
        let com = C2::commit(&commit_keys, &message).unwrap();
        assert!(C2::verify(&commit_keys, &message, &com).unwrap());
        assert!(!C2::verify(&commit_keys, &wrong_message, &com).unwrap());
        message.push(<Bn254 as Pairing>::G2::rand(&mut rng));
        assert!(C2::verify(&commit_keys, &message, &com).is_err());
    }
}
