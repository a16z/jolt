use ark_ec::pairing::{MillerLoopOutput, Pairing, PairingOutput};
use ark_std::cfg_iter;
use eyre::{bail, Error};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub fn inner_product<P: Pairing>(
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
fn cfg_multi_pairing<P: Pairing>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Bn254;
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};

    const TEST_SIZE: usize = 8;

    pub fn random_generators<R: Rng, G: AffineRepr>(rng: &mut R, num: usize) -> Vec<G> {
        (0..num).map(|_| G::rand(rng)).collect()
    }

    #[test]
    fn afgho_g1_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let commit_keys = random_generators::<_, <Bn254 as Pairing>::G2Affine>(&mut rng, TEST_SIZE);
        let mut message = Vec::new();
        let mut wrong_message = Vec::new();
        for _ in 0..TEST_SIZE {
            message.push(<Bn254 as Pairing>::G1::rand(&mut rng).into_affine());
            wrong_message.push(<Bn254 as Pairing>::G1::rand(&mut rng).into_affine());
        }
        let com = inner_product::<Bn254>(&message, &commit_keys).unwrap();
        assert_eq!(
            com,
            <Bn254 as Pairing>::multi_pairing(&message, &commit_keys)
        );
    }
}
