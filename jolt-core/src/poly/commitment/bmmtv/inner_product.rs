use ark_ec::pairing::{Pairing, PairingOutput};
use ark_std::cfg_iter;
use eyre::{bail, Error};

use ark_ec::{pairing::MillerLoopOutput, CurveGroup};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub fn inner_product<P: Pairing>(
    left: &[P::G1],
    right: &[P::G2],
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
pub fn cfg_multi_pairing<P: Pairing>(left: &[P::G1], right: &[P::G2]) -> Option<PairingOutput<P>> {
    // We make the input affine, then convert to prepared. We do this for speed, since the
    // conversion from projective to prepared always goes through affine.
    let aff_left = P::G1::normalize_batch(left);
    let aff_right = P::G2::normalize_batch(right);

    let left = cfg_iter!(aff_left)
        .map(P::G1Prepared::from)
        .collect::<Vec<_>>();
    let right = cfg_iter!(aff_right)
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
