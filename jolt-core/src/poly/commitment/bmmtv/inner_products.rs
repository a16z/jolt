use ark_ec::{
    pairing::{MillerLoopOutput, Pairing, PairingOutput},
    CurveGroup,
};
use ark_ff::Field;
use ark_std::cfg_iter;
use std::marker::PhantomData;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum InnerProductError {
    #[error("left length, right length: {0}, {1}")]
    MessageLengthInvalid(usize, usize),
}

/// Represents an inner product operation
pub trait InnerProduct {
    /// Left side of the message
    type LeftMessage;
    /// Right side of the message
    type RightMessage;
    /// Output of the inner product
    type Output;

    fn inner_product(
        left: &[Self::LeftMessage],
        right: &[Self::RightMessage],
    ) -> Result<Self::Output, InnerProductError>;
}

/// Inner pairing product representation
///
/// Simple Afgho
pub struct PairingInnerProduct<P: Pairing> {
    _pair: PhantomData<P>,
}

impl<P: Pairing> InnerProduct for PairingInnerProduct<P> {
    type LeftMessage = P::G1;
    type RightMessage = P::G2;
    type Output = PairingOutput<P>;

    fn inner_product(
        left: &[Self::LeftMessage],
        right: &[Self::RightMessage],
    ) -> Result<Self::Output, InnerProductError> {
        if left.len() != right.len() {
            return Err(InnerProductError::MessageLengthInvalid(
                left.len(),
                right.len(),
            ));
        };

        Ok(cfg_multi_pairing(left, right).unwrap())
    }
}

/// Equivalent to `P::multi_pairing`, but with more parallelism (if enabled)
fn cfg_multi_pairing<P: Pairing>(left: &[P::G1], right: &[P::G2]) -> Option<PairingOutput<P>> {
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

/// Simple pedersen
#[derive(Copy, Clone)]
pub struct MultiexponentiationInnerProduct<G: CurveGroup> {
    _projective: PhantomData<G>,
}

impl<G: CurveGroup> InnerProduct for MultiexponentiationInnerProduct<G> {
    type LeftMessage = G;
    type RightMessage = G::ScalarField;
    type Output = G;

    fn inner_product(
        left: &[Self::LeftMessage],
        right: &[Self::RightMessage],
    ) -> Result<Self::Output, InnerProductError> {
        if left.len() != right.len() {
            return Err(InnerProductError::MessageLengthInvalid(
                left.len(),
                right.len(),
            ));
        };

        // Can unwrap because we did the length check above
        Ok(G::msm(&G::normalize_batch(left), right).unwrap())
    }
}

#[derive(Copy, Clone)]
pub struct ScalarInnerProduct<F: Field> {
    _field: PhantomData<F>,
}

impl<F: Field> InnerProduct for ScalarInnerProduct<F> {
    type LeftMessage = F;
    type RightMessage = F;
    type Output = F;

    fn inner_product(
        left: &[Self::LeftMessage],
        right: &[Self::RightMessage],
    ) -> Result<Self::Output, InnerProductError> {
        if left.len() != right.len() {
            return Err(InnerProductError::MessageLengthInvalid(
                left.len(),
                right.len(),
            ));
        };
        Ok(cfg_iter!(left).zip(right).map(|(x, y)| *x * y).sum())
    }
}
