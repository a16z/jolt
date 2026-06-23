use super::accumulate::{onehot_accumulate, onehot_accumulate_tensor};
use super::*;

fn expand_onehot_accum<const D: usize>(
    compressed: Vec<[i32; D]>,
    num_digits: usize,
) -> Vec<[i32; D]> {
    if num_digits == 1 {
        return compressed;
    }

    let mut expanded = Vec::with_capacity(compressed.len().saturating_mul(num_digits));
    for coeffs in compressed {
        expanded.push(coeffs);
        for _ in 1..num_digits {
            expanded.push([0i32; D]);
        }
    }
    expanded
}

fn finish_decompose_fold<F: CanonicalField, const D: usize>(
    compressed_accum: Vec<[i32; D]>,
    num_digits: usize,
) -> DecomposeFoldWitness<F, D> {
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let coeff_accum = {
        let _span = tracing::info_span!("onehot_expand_accum").entered();
        expand_onehot_accum(compressed_accum, num_digits)
    };
    let _span = tracing::info_span!("onehot_convert").entered();
    build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
}

fn decompose_fold_from_views<E, F, const D: usize>(
    block_views: &[&[E]],
    challenges: &[SparseChallenge],
    num_blocks: usize,
    block_len: usize,
    num_digits: usize,
) -> DecomposeFoldWitness<F, D>
where
    E: OneHotEntry,
    F: CanonicalField,
{
    let compressed_accum = {
        let _span = tracing::info_span!("onehot_accumulate").entered();
        onehot_accumulate::<E, D>(block_views, challenges, num_blocks, block_len)
    };
    finish_decompose_fold(compressed_accum, num_digits)
}

impl<F: FieldCore, const D: usize, I: OneHotIndex> OneHotPoly<F, D, I> {
    pub(super) fn decompose_fold_onehot<E>(
        &self,
        blocks: &FlatBlocks<E>,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F, D>
    where
        E: OneHotEntry,
        F: CanonicalField,
    {
        let num_blocks = challenges.len().min(blocks.num_blocks());
        let block_views: Vec<&[E]> = (0..blocks.num_blocks()).map(|i| blocks.block(i)).collect();
        decompose_fold_from_views(&block_views, challenges, num_blocks, block_len, num_digits)
    }

    pub(super) fn decompose_fold_batched_single_chunk_onehot(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F, D>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let mut flat_blocks: Vec<&[SingleChunkEntry]> = Vec::with_capacity(total_blocks);
        for poly in polys {
            let (_, cached) = poly.block_cache.get()?;
            let OneHotBlocks::SingleChunk(blocks) = cached else {
                return None;
            };
            for i in 0..blocks.num_blocks() {
                flat_blocks.push(blocks.block(i));
            }
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        Some(decompose_fold_from_views(
            &flat_blocks,
            challenges,
            active_blocks,
            block_len,
            num_digits,
        ))
    }

    pub(super) fn decompose_fold_batched_multi_chunk_onehot(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F, D>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let mut flat_blocks: Vec<&[MultiChunkEntry]> = Vec::with_capacity(total_blocks);
        for poly in polys {
            let (_, cached) = poly.block_cache.get()?;
            let OneHotBlocks::MultiChunk(blocks) = cached else {
                return None;
            };
            for i in 0..blocks.num_blocks() {
                flat_blocks.push(blocks.block(i));
            }
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        Some(decompose_fold_from_views(
            &flat_blocks,
            challenges,
            active_blocks,
            block_len,
            num_digits,
        ))
    }

    /// Tensor-shaped batched decompose-fold for one-hot polynomials.
    pub(super) fn decompose_fold_batched_tensor_onehot(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        block_len: usize,
        num_digits: usize,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError>
    where
        F: CanonicalField,
    {
        for poly in polys {
            poly.blocks_for(block_len).expect(
                "OneHotPoly::decompose_fold_batched_tensor_onehot: invalid block_len for one polynomial",
            );
        }
        let Some(first) = polys.first() else {
            return Ok(None);
        };
        let (_, first_blocks) = first
            .block_cache
            .get()
            .expect("block cache was just built above");
        let expected_blocks = tensor
            .left_len
            .checked_mul(tensor.right_len)
            .and_then(|blocks| blocks.checked_mul(tensor.num_claims))
            .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".into()))?;
        let modulus = (-F::one()).to_canonical_u128() + 1;

        let witness = match first_blocks {
            OneHotBlocks::SingleChunk(_) => {
                let mut flat_blocks: Vec<&[SingleChunkEntry]> = Vec::with_capacity(expected_blocks);
                for poly in polys {
                    let (_, cached) = poly.block_cache.get().expect("block cache exists");
                    let OneHotBlocks::SingleChunk(blocks) = cached else {
                        return Ok(None);
                    };
                    for i in 0..blocks.num_blocks() {
                        flat_blocks.push(blocks.block(i));
                    }
                }
                if flat_blocks.len() != expected_blocks {
                    return Err(AkitaError::InvalidSize {
                        expected: expected_blocks,
                        actual: flat_blocks.len(),
                    });
                }
                let coeff_accum_i64 = {
                    let _span = tracing::info_span!("onehot_accumulate_tensor").entered();
                    onehot_accumulate_tensor::<SingleChunkEntry, D>(
                        &flat_blocks,
                        tensor,
                        expected_blocks,
                        block_len,
                    )?
                };
                let compressed_accum = narrow_tensor_accum_to_i32::<D>(coeff_accum_i64)?;
                let coeff_accum = expand_onehot_accum(compressed_accum, num_digits);
                build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
            }
            OneHotBlocks::MultiChunk(_) => {
                let mut flat_blocks: Vec<&[MultiChunkEntry]> = Vec::with_capacity(expected_blocks);
                for poly in polys {
                    let (_, cached) = poly.block_cache.get().expect("block cache exists");
                    let OneHotBlocks::MultiChunk(blocks) = cached else {
                        return Ok(None);
                    };
                    for i in 0..blocks.num_blocks() {
                        flat_blocks.push(blocks.block(i));
                    }
                }
                if flat_blocks.len() != expected_blocks {
                    return Err(AkitaError::InvalidSize {
                        expected: expected_blocks,
                        actual: flat_blocks.len(),
                    });
                }
                let coeff_accum_i64 = {
                    let _span = tracing::info_span!("onehot_accumulate_tensor").entered();
                    onehot_accumulate_tensor::<MultiChunkEntry, D>(
                        &flat_blocks,
                        tensor,
                        expected_blocks,
                        block_len,
                    )?
                };
                let compressed_accum = narrow_tensor_accum_to_i32::<D>(coeff_accum_i64)?;
                let coeff_accum = expand_onehot_accum(compressed_accum, num_digits);
                build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
            }
        };
        Ok(Some(witness))
    }
}
