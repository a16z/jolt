use super::DensePoly;
use crate::backend::poly_helpers::{
    build_decompose_fold_witness, decompose_ring_interleaved, DecomposeParams,
};
use crate::backend::tensor_fold::{
    integer_mul_acc_i64, materialize_tensor_challenges, narrow_tensor_accum_to_i32,
};
use crate::DecomposeFoldWitness;
use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_algebra::CyclotomicRing;
use akita_challenges::{IntegerChallenge, TensorChallenges as TensorChallengeSet};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore};

pub(super) fn decompose_fold_batched_tensor_dense<F, const D: usize>(
    polys: &[&DensePoly<F, D>],
    tensor: &TensorChallengeSet,
    block_len: usize,
    num_digits: usize,
    log_basis: u32,
) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if polys.is_empty() {
        return Ok(None);
    }

    let q = (-F::one()).to_canonical_u128() + 1;
    let (tensor_challenges, blocks_per_claim) = materialize_tensor_challenges::<D>(tensor)?;
    let accum_i64 = if let Some(digit_planes) = polys
        .iter()
        .map(|poly| poly.digit_planes_for(num_digits, log_basis))
        .collect::<Option<Vec<_>>>()
    {
        let _span = tracing::info_span!("dense_tensor_cached_digit_accumulate").entered();
        accumulate_cached_digit_planes_tensor::<D>(
            &digit_planes,
            &tensor_challenges,
            blocks_per_claim,
            block_len,
            num_digits,
        )?
    } else {
        let threshold = decompose_centering_threshold(num_digits, log_basis, q);
        let params = DecomposeParams {
            threshold,
            q,
            mask: (1i128 << log_basis) - 1,
            half_b: 1i128 << (log_basis - 1),
            b_val: 1i128 << log_basis,
            log_basis,
            overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
        };
        let coeff_slices = polys
            .iter()
            .map(|poly| poly.coeffs.as_slice())
            .collect::<Vec<_>>();
        let _span = tracing::info_span!("dense_tensor_accumulate").entered();
        balanced_ring_decompose_fold_tensor_partitioned::<F, D>(
            &coeff_slices,
            &tensor_challenges,
            blocks_per_claim,
            block_len,
            num_digits,
            &params,
        )?
    };

    let _span = tracing::info_span!("dense_tensor_convert").entered();
    let centered_coeffs = narrow_tensor_accum_to_i32::<D>(accum_i64)?;
    Ok(Some(build_decompose_fold_witness::<F, D>(
        centered_coeffs,
        q,
    )))
}

fn accumulate_cached_digit_planes_tensor<const D: usize>(
    digit_planes_by_poly: &[&[[i8; D]]],
    tensor_challenges: &[IntegerChallenge],
    blocks_per_claim: usize,
    block_len: usize,
    num_digits: usize,
) -> Result<Vec<[i64; D]>, AkitaError> {
    if block_len == 0 || num_digits == 0 {
        return Err(AkitaError::InvalidInput(
            "dense cached tensor decompose-fold requires non-zero block_len and num_digits"
                .to_string(),
        ));
    }
    let expected_blocks = digit_planes_by_poly
        .len()
        .checked_mul(blocks_per_claim)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))?;
    if tensor_challenges.len() != expected_blocks {
        return Err(AkitaError::InvalidSize {
            expected: expected_blocks,
            actual: tensor_challenges.len(),
        });
    }

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len.max(1)).max(1);
    let elem_chunk = block_len.div_ceil(actual_threads);
    let chunks = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let elem_start = tid * elem_chunk;
            if elem_start >= block_len {
                return Ok(Vec::new());
            }
            let elem_end = (elem_start + elem_chunk).min(block_len);
            let mut acc = vec![[0i64; D]; (elem_end - elem_start) * num_digits];

            for (block_idx, challenge) in tensor_challenges.iter().enumerate() {
                let claim_idx = block_idx / blocks_per_claim;
                let local_block_idx = block_idx % blocks_per_claim;
                let digit_planes = digit_planes_by_poly[claim_idx];

                for elem_idx in elem_start..elem_end {
                    let ring_idx = local_block_idx * block_len + elem_idx;
                    let plane_base = ring_idx * num_digits;
                    if plane_base >= digit_planes.len() {
                        continue;
                    }
                    let out_base = (elem_idx - elem_start) * num_digits;
                    for digit_idx in 0..num_digits {
                        let Some(digit_plane) = digit_planes.get(plane_base + digit_idx) else {
                            continue;
                        };
                        integer_mul_acc_i64::<D>(
                            digit_plane,
                            challenge,
                            &mut acc[out_base + digit_idx],
                        );
                    }
                }
            }

            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}

fn balanced_ring_decompose_fold_tensor_partitioned<F: CanonicalField, const D: usize>(
    poly_coeffs: &[&[CyclotomicRing<F, D>]],
    tensor_challenges: &[IntegerChallenge],
    blocks_per_claim: usize,
    block_len: usize,
    num_digits: usize,
    p: &DecomposeParams,
) -> Result<Vec<[i64; D]>, AkitaError> {
    if block_len == 0 || num_digits == 0 {
        return Err(AkitaError::InvalidInput(
            "dense tensor decompose-fold requires non-zero block_len and num_digits".to_string(),
        ));
    }
    let expected_blocks = poly_coeffs
        .len()
        .checked_mul(blocks_per_claim)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))?;
    if tensor_challenges.len() != expected_blocks {
        return Err(AkitaError::InvalidSize {
            expected: expected_blocks,
            actual: tensor_challenges.len(),
        });
    }

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len.max(1)).max(1);
    let elem_chunk = block_len.div_ceil(actual_threads);
    let chunks = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let elem_start = tid * elem_chunk;
            if elem_start >= block_len {
                return Ok(Vec::new());
            }
            let elem_end = (elem_start + elem_chunk).min(block_len);
            let mut acc = vec![[0i64; D]; (elem_end - elem_start) * num_digits];
            let mut digit_buf = vec![[0i8; D]; num_digits];

            for (block_idx, challenge) in tensor_challenges.iter().enumerate() {
                let claim_idx = block_idx / blocks_per_claim;
                let local_block_idx = block_idx % blocks_per_claim;
                let coeff_start = local_block_idx * block_len + elem_start;
                let coeffs = poly_coeffs[claim_idx];
                if coeff_start >= coeffs.len() {
                    continue;
                }
                let coeff_end = (local_block_idx * block_len + elem_end).min(coeffs.len());
                if coeff_start >= coeff_end {
                    continue;
                }

                for (local_elem_idx, ring) in coeffs[coeff_start..coeff_end].iter().enumerate() {
                    decompose_ring_interleaved::<F, D>(ring, &mut digit_buf, num_digits, p);
                    let base = local_elem_idx * num_digits;
                    for digit_idx in 0..num_digits {
                        integer_mul_acc_i64::<D>(
                            &digit_buf[digit_idx],
                            challenge,
                            &mut acc[base + digit_idx],
                        );
                    }
                }
            }

            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}
