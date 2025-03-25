//! Afgho commitment
//!
//! This module implements an inner pairing commitment implementation

use ark_ec::{
    pairing::{Pairing, PairingOutput},
    CurveGroup,
};
use ark_std::rand::Rng;

use super::inner_products::PairingInnerProduct;

pub type Error = anyhow::Error;

/// Helpers for generator commitment keys used by Pedersen and AFGHO16
pub fn random_generators<R: Rng, G: CurveGroup>(rng: &mut R, num: usize) -> Vec<G> {
    (0..num).map(|_| G::rand(rng)).collect()
}

/// Afgho commitment is simply an inner pairing product commitment
#[derive(Clone)]
pub struct AfghoCommitment<P: Pairing> {
    _pairing_output: PairingOutput<P>,
}

impl<P: Pairing> AfghoCommitment<P> {
    /// Generates a setup for commitments with `size`
    ///
    /// Takes an `Rng` for parameter generator (if needed)
    ///
    /// Output [`Vec<Self::Param>`]
    pub fn setup<R: Rng>(rng: &mut R, size: usize) -> Vec<P::G2> {
        random_generators(rng, size)
    }

    /// Commits to some message `msg` taking the parameters `params` from [`Self::setup`]
    /// and outputting [`Self::Output`]
    pub fn commit(k: &[P::G2], m: &[P::G1]) -> Result<PairingOutput<P>, Error> {
        Ok(PairingInnerProduct::<P>::inner_product(m, k)?)
    }
    /// Verifies if the commitment is really the output of `msg` and params `k`
    pub fn verify(
        params: &[P::G2],
        msg: &[P::G1],
        commitment: &PairingOutput<P>,
    ) -> Result<bool, Error> {
        Ok(Self::commit(params, msg)? == *commitment)
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
        let params = CommitG1::setup(&mut rng, TEST_SIZE);

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
