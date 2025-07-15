use crate::field::JoltField;
use std::marker::PhantomData;

use ark_ec::{
    pairing::{MillerLoopOutput, Pairing, PairingOutput},
    CurveGroup,
};

use crate::msm::VariableBaseMSM;
use rayon::prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum InnerProductError {
    #[error("left length, right length: {0}, {1}")]
    MessageLengthInvalid(usize, usize),
}

/// Inner pairing product representation
///
/// Simple Afgho
pub struct PairingInnerProduct<P: Pairing> {
    _pair: PhantomData<P>,
}

impl<P: Pairing> PairingInnerProduct<P> {
    pub fn inner_product(
        left: &[P::G1],
        right: &[P::G2],
    ) -> Result<PairingOutput<P>, InnerProductError> {
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

    let left = aff_left
        .par_iter()
        .map(P::G1Prepared::from)
        .collect::<Vec<_>>();
    let right = aff_right
        .par_iter()
        .map(P::G2Prepared::from)
        .collect::<Vec<_>>();

    // We want to process N chunks in parallel where N is the number of threads available
    let num_chunks = rayon::current_num_threads();

    let chunk_size = if num_chunks <= left.len() {
        left.len() / num_chunks
    } else {
        // More threads than elements. Just do it all in parallel
        1
    };

    let (left_chunks, right_chunks) = (left.par_chunks(chunk_size), right.par_chunks(chunk_size));

    // Compute all the (partial) pairings and take the product. We have to take the product over
    // P::TargetField because MillerLoopOutput doesn't impl Product
    let ml_result = left_chunks
        .zip(right_chunks)
        .map(|(aa, bb)| P::multi_miller_loop(aa.iter().cloned(), bb.iter().cloned()).0)
        .product();

    P::final_exponentiation(MillerLoopOutput(ml_result))
}

pub enum MultiexponentiationInnerProduct {}

impl MultiexponentiationInnerProduct {
    pub fn inner_product<G>(left: &[G], right: &[G::ScalarField]) -> Result<G, InnerProductError>
    where
        G: CurveGroup,
        G::ScalarField: JoltField,
    {
        if left.len() != right.len() {
            return Err(InnerProductError::MessageLengthInvalid(
                left.len(),
                right.len(),
            ));
        };

        // Can unwrap because we did the length check above
        Ok(
            <G as VariableBaseMSM>::msm_field_elements(&G::normalize_batch(left), right, None)
                .unwrap(),
        )
    }
}
