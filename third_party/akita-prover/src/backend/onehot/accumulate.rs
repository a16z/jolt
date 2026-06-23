use super::*;

/// Accumulates one-hot decompose-fold rows in compressed position order.
///
/// The returned vector has `block_len` rows. Callers expand each row across
/// `num_digits` later, inserting zero rows for higher digit planes.
///
/// `blocks` is a slice-of-slices view over per-block entries. Both
/// single-polynomial callers (which collect once via `FlatBlocks::block`)
/// and batched callers (which concatenate slices across polynomials) feed
/// through the same signature.
pub(super) fn onehot_accumulate<E, const D: usize>(
    blocks: &[&[E]],
    challenges: &[SparseChallenge],
    num_blocks: usize,
    block_len: usize,
) -> Vec<[i32; D]>
where
    E: OneHotEntry,
{
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len).max(1);
    let pos_chunk = block_len.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= block_len {
                return Vec::new();
            }
            let pos_end = (pos_start + pos_chunk).min(block_len);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];
            let mut rotated = vec![[0i16; D]; D];

            for (block_idx, challenge) in challenges.iter().enumerate().take(num_blocks) {
                let entries = blocks[block_idx];
                let lo = entries.partition_point(|entry| entry.pos_in_block() < pos_start);
                let hi = entries.partition_point(|entry| entry.pos_in_block() < pos_end);
                if lo >= hi {
                    continue;
                }

                fill_rotated_challenge::<D>(&mut rotated, challenge);

                for entry in &entries[lo..hi] {
                    let dst = &mut acc[entry.pos_in_block() - pos_start];
                    for &ci in entry.coeffs() {
                        let rot = &rotated[ci as usize];
                        for k in 0..D {
                            dst[k] += rot[k] as i32;
                        }
                    }
                }
            }

            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}

// Tensor accumulators use `[i64; D]` because each per-block challenge is a
// product of two sparse samples. The witness boundary narrows back to
// `[i32; D]` after checking the selected schedule's coefficient envelope.

pub(super) fn onehot_accumulate_tensor<E, const D: usize>(
    blocks: &[&[E]],
    tensor: &TensorChallengeSet,
    num_blocks: usize,
    block_len: usize,
) -> Result<Vec<[i64; D]>, AkitaError>
where
    E: OneHotEntry,
{
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len).max(1);
    let pos_chunk = block_len.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i64; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= block_len {
                return Ok(Vec::new());
            }
            let pos_end = (pos_start + pos_chunk).min(block_len);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i64; D]; len];
            let mut rotated = vec![[0i64; D]; D];

            for (block_idx, entries) in blocks.iter().enumerate().take(num_blocks) {
                let lo = entries.partition_point(|entry| entry.pos_in_block() < pos_start);
                let hi = entries.partition_point(|entry| entry.pos_in_block() < pos_end);
                if lo >= hi {
                    continue;
                }

                let (_, _, left, right) = tensor.factors_for_logical_block(block_idx)?;
                fill_rotated_tensor_challenge::<D>(&mut rotated, left, right)?;

                for entry in &entries[lo..hi] {
                    let dst = &mut acc[entry.pos_in_block() - pos_start];
                    for &ci in entry.coeffs() {
                        let rot = &rotated[ci as usize];
                        for k in 0..D {
                            dst[k] += rot[k];
                        }
                    }
                }
            }

            Ok(acc)
        })
        .collect::<Result<_, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}
