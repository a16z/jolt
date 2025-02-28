use ark_ec::{
    pairing::{MillerLoopOutput, Pairing, PairingOutput},
    AffineRepr, Group,
};
use ark_std::{cfg_iter, rand::Rng};
use eyre::{bail, Error};
use std::marker::PhantomData;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

fn inner_product<P: Pairing>(
    left: &[P::G1Affine],
    right: &[P::G2Affine],
) -> Result<PairingOutput<P>, Error> {
    if left.len() != right.len() {
        bail!(
            "invalid message lenght left length, right length: {}, {}",
            left.len(),
            right.len(),
        )
    };

    Ok(cfg_multi_pairing(left, right).unwrap())
}

/// Equivalent to `P::multi_pairing`, but with more parallelism (if enabled)
pub fn cfg_multi_pairing<P: Pairing>(
    left: &[P::G1Affine],
    right: &[P::G2Affine],
) -> Option<PairingOutput<P>> {
    // We make the input affine, then convert to prepared. We do this for speed, since the
    // conversion from projective to prepared always goes through affine.

    let left = cfg_iter!(left).map(P::G1Prepared::from).collect::<Vec<_>>();
    let right = cfg_iter!(right)
        .map(P::G2Prepared::from)
        .collect::<Vec<_>>();

    // We want to process N chunks in parallel where N is the number of threads available
    #[cfg(feature = "rayon")]
    let num_chunks = rayon::current_num_threads();
    #[cfg(not(feature = "rayon"))]
    let num_chunks = 1;

    let chunk_size = if num_chunks <= left.len() {
        left.len() / num_chunks
    } else {
        // More threads than elements. Just do it all in parallel
        1
    };

    #[cfg(feature = "rayon")]
    let (left_chunks, right_chunks) = (left.par_chunks(chunk_size), right.par_chunks(chunk_size));
    #[cfg(not(feature = "rayon"))]
    let (left_chunks, right_chunks) = (left.chunks(chunk_size), right.chunks(chunk_size));

    // Compute all the (partial) pairings and take the product. We have to take the product over
    // P::TargetField because MillerLoopOutput doesn't impl Product
    let ml_result = left_chunks
        .zip(right_chunks)
        .map(|(aa, bb)| P::multi_miller_loop(aa.iter().cloned(), bb.iter().cloned()).0)
        .product();

    P::final_exponentiation(MillerLoopOutput(ml_result))
}

pub fn random_generators<R: Rng, G: AffineRepr>(rng: &mut R, num: usize) -> Vec<G> {
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
    fn setup<R: Rng>(rng: &mut R, size: usize) -> Result<Vec<P::G2Affine>, Error> {
        Ok(random_generators(rng, size))
    }

    pub fn commit(k: &[P::G2Affine], m: &[P::G1Affine]) -> Result<PairingOutput<P>, Error> {
        inner_product(m, k)
    }

    fn verify(k: &[P::G2Affine], m: &[P::G1Affine], com: &PairingOutput<P>) -> Result<bool, Error> {
        Ok(Self::commit(k, m)? == *com)
    }
}

impl<P: Pairing> AFGHOCommitmentG2<P> {
    fn setup<R: Rng>(rng: &mut R, size: usize) -> Result<Vec<P::G1Affine>, Error> {
        Ok(random_generators(rng, size))
    }

    fn commit(k: &[P::G1Affine], m: &[P::G2Affine]) -> Result<PairingOutput<P>, Error> {
        inner_product(k, m)
    }

    fn verify(k: &[P::G1Affine], m: &[P::G2Affine], com: &PairingOutput<P>) -> Result<bool, Error> {
        Ok(Self::commit(k, m)? == *com)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Bn254;
    use ark_ec::CurveGroup;
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
            message.push(<Bn254 as Pairing>::G1::rand(&mut rng).into_affine());
            wrong_message.push(<Bn254 as Pairing>::G1::rand(&mut rng).into_affine());
        }
        let com = C1::commit(&commit_keys, &message).unwrap();
        assert!(C1::verify(&commit_keys, &message, &com).unwrap());
        assert!(!C1::verify(&commit_keys, &wrong_message, &com).unwrap());
        message.push(<Bn254 as Pairing>::G1::rand(&mut rng).into_affine());
        assert!(C1::verify(&commit_keys, &message, &com).is_err());
    }

    #[test]
    fn afgho_g2_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let commit_keys = C2::setup(&mut rng, TEST_SIZE).unwrap();
        let mut message = Vec::new();
        let mut wrong_message = Vec::new();
        for _ in 0..TEST_SIZE {
            message.push(<Bn254 as Pairing>::G2::rand(&mut rng).into_affine());
            wrong_message.push(<Bn254 as Pairing>::G2::rand(&mut rng).into_affine());
        }
        let com = C2::commit(&commit_keys, &message).unwrap();
        assert!(C2::verify(&commit_keys, &message, &com).unwrap());
        assert!(!C2::verify(&commit_keys, &wrong_message, &com).unwrap());
        message.push(<Bn254 as Pairing>::G2::rand(&mut rng).into_affine());
        assert!(C2::verify(&commit_keys, &message, &com).is_err());
    }
}
