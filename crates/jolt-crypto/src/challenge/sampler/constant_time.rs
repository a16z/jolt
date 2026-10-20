//! Prover-side constant-time sampling of one selective-L2 challenge.
//!
//! Whiteout's Σ-protocol draws a challenge for every attempt, and a rejected
//! attempt's challenge is correlated with the witness (Thm `wo:thm:zk`,
//! hybrid H2), so its value must not reach timing or memory addresses. This
//! sampler returns exactly what [`sample_indexed_challenges_from_seed`]
//! returns for one D = 64 selective-L2 coordinate, from the same XOF bytes,
//! but processes every candidate without a branch, table index or address
//! that depends on its positions or signs:
//!
//! - the partial Fisher–Yates shuffle swaps through a full masked scan of
//!   the permutation instead of indexing it at the drawn position;
//! - signs become coefficients by arithmetic;
//! - the operator-norm predicate runs on the dense coefficient vector, built
//!   by a masked scatter, and evaluates every frequency.
//!
//! What remains data dependent is independent of the accepted challenge: the
//! number of bytes the uniform index draws reject, and the number of
//! candidates the norm filter rejects. Rejection sampling makes both
//! independent of the value it finally accepts.
//!
//! [`sample_indexed_challenges_from_seed`]: super::sample_indexed_challenges_from_seed

use super::{XofCursor, D64_SELECTIVE_L2_OP_NORM_TABLE, MAX_OP_NORM_ATTEMPTS};
use crate::challenge::sampler::IndexedXofPrefix;
use crate::challenge::sparse::{eq_mask, scatter_ct};
use crate::challenge::{
    OperatorNormRejection, SparseChallenge, D64_L2_OP_NORM_PM1_COUNT, D64_L2_OP_NORM_PM2_COUNT,
    D64_SELECTIVE_L2_CHALLENGE_CONFIG,
};
use crate::lattice::CHALLENGE_RING_DIM;
use crate::CryptoError;

/// Nonzero coefficients of every attempt challenge.
const WEIGHT: usize = D64_L2_OP_NORM_PM1_COUNT + D64_L2_OP_NORM_PM2_COUNT;

/// The D = 64 selective-L2 challenge that
/// `sample_indexed_challenges_from_seed(seed, 64, 1, &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
/// Some(OperatorNormRejection::D64_SELECTIVE_L2))` returns, sampled in
/// constant time with respect to the challenge.
///
/// Fails, like the verifier's sampler, if every one of the bounded number of
/// candidates fails the norm filter; that happens with probability below
/// `2^{−1500}`.
pub fn sample_challenge_ct(seed: &[u8]) -> Result<SparseChallenge, CryptoError> {
    sample_counted(seed).map(|(challenge, _)| challenge)
}

/// [`sample_challenge_ct`] and the number of candidates it drew.
pub(super) fn sample_counted(seed: &[u8]) -> Result<(SparseChallenge, usize), CryptoError> {
    let config = D64_SELECTIVE_L2_CHALLENGE_CONFIG;
    let threshold = OperatorNormRejection::D64_SELECTIVE_L2.threshold;
    let table = D64_SELECTIVE_L2_OP_NORM_TABLE
        .as_ref()
        .map_err(|message| CryptoError::InvalidSetup((*message).into()))?;
    let prefix =
        IndexedXofPrefix::new(seed).map_err(|message| CryptoError::InvalidSetup(message.into()))?;
    let mut cursor = XofCursor::new();
    cursor.reset_indexed_prefix(&prefix, 0);
    // Fixed-size stack buffers, and the returned challenge fits its inline
    // storage: no allocation depends on the draw.
    let mut positions = [0u32; WEIGHT];
    let mut coeffs = [0i8; WEIGHT];
    let mut sign_bytes = [0u8; WEIGHT];
    for attempt in 1..=MAX_OP_NORM_ATTEMPTS {
        draw_positions(&mut cursor, &mut positions);
        cursor.fill_bytes(&mut sign_bytes);
        for (index, (coeff, &byte)) in coeffs.iter_mut().zip(&sign_bytes).enumerate() {
            let magnitude = if index < config.count_pm1 { 1 } else { 2 };
            // `(1 − 2·(byte & 1)) · magnitude`: the sign bit selects by arithmetic.
            *coeff = (1 - 2 * (byte & 1) as i8) * magnitude;
        }
        let dense = scatter_ct(&positions, &coeffs);
        if table.accept_strict_dense_ct(&dense, config.l1_norm() as u64, u64::from(threshold))? {
            let challenge = SparseChallenge {
                positions: positions.iter().copied().collect(),
                coeffs: coeffs.iter().copied().collect(),
            };
            return Ok((challenge, attempt));
        }
    }
    Err(CryptoError::InvalidInput(format!(
        "operator-norm rejection exceeded {MAX_OP_NORM_ATTEMPTS} attempts"
    )))
}

/// The verifier's partial Fisher–Yates shuffle over `0..64`, reading and
/// writing the permutation through masked scans of all 64 entries.
fn draw_positions(cursor: &mut XofCursor, out: &mut [u32]) {
    let mut permutation: [u8; CHALLENGE_RING_DIM] = std::array::from_fn(|index| index as u8);
    for (i, dst) in out.iter_mut().enumerate() {
        let j = i + cursor.next_usize_mod(CHALLENGE_RING_DIM - i);
        let at_j = select_ct(&permutation, j);
        let at_i = select_ct(&permutation, i);
        for (index, slot) in permutation.iter_mut().enumerate() {
            let is_i = eq_mask(index, i);
            let is_j = eq_mask(index, j);
            // Position i takes the value at j; position j (when j ≠ i) takes
            // the value at i. When j = i both writes leave the slot unchanged.
            *slot = (*slot & !(is_i | is_j)) | (at_j & is_i) | (at_i & is_j & !is_i);
        }
        *dst = u32::from(at_j);
    }
}

/// `values[index]`, read through a masked scan of every entry.
fn select_ct(values: &[u8; CHALLENGE_RING_DIM], index: usize) -> u8 {
    values
        .iter()
        .enumerate()
        .fold(0, |acc, (position, &value)| {
            acc | (value & eq_mask(position, index))
        })
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code")]
mod tests {
    use super::*;
    use crate::challenge::sample_indexed_challenges_from_seed;

    /// Byte identity with the verifier's sampler, over seeds that include
    /// norm-filter rejections.
    #[test]
    fn matches_the_verifier_sampler() {
        let mut rejected = 0;
        for index in 0u64..100_000 {
            let mut seed = [0u8; 32];
            seed[..8].copy_from_slice(&index.to_le_bytes());
            seed[8..16].copy_from_slice(&index.wrapping_mul(0x9e37_79b9_7f4a_7c15).to_le_bytes());
            let expected = sample_indexed_challenges_from_seed(
                &seed,
                CHALLENGE_RING_DIM,
                1,
                &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
                Some(OperatorNormRejection::D64_SELECTIVE_L2),
            )
            .unwrap();
            let (actual, attempts) = sample_counted(&seed).unwrap();
            assert_eq!([actual], expected.as_slice(), "seed {index}");
            rejected += usize::from(attempts > 1);
        }
        assert!(
            rejected > 10_000,
            "only {rejected} seeds hit a norm rejection"
        );
    }
}
