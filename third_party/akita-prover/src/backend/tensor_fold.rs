use akita_challenges::{IntegerChallenge, SparseChallenge, TensorChallenges as TensorChallengeSet};
use akita_field::AkitaError;

pub(crate) fn materialize_tensor_challenges<const D: usize>(
    tensor: &TensorChallengeSet,
) -> Result<(Vec<IntegerChallenge>, usize), AkitaError> {
    let blocks_per_claim = tensor
        .left_len
        .checked_mul(tensor.right_len)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))?;
    let expected_blocks = tensor
        .num_claims
        .checked_mul(blocks_per_claim)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))?;
    let challenges = tensor.expand_integer::<D>()?;
    if challenges.len() != expected_blocks {
        return Err(AkitaError::InvalidSize {
            expected: expected_blocks,
            actual: challenges.len(),
        });
    }
    Ok((challenges, blocks_per_claim))
}

pub(crate) fn integer_mul_acc_i64<const D: usize>(
    digit_plane: &[i8; D],
    challenge: &IntegerChallenge,
    acc: &mut [i64; D],
) {
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let p = pos as usize;
        let split = D - p;
        let coeff = i64::from(coeff);
        for i in 0..split {
            acc[i + p] += coeff * i64::from(digit_plane[i]);
        }
        for i in split..D {
            acc[i - split] -= coeff * i64::from(digit_plane[i]);
        }
    }
}

pub(crate) fn fill_rotated_integer_challenge<const D: usize>(
    table: &mut [[i64; D]],
    challenge: &IntegerChallenge,
) {
    debug_assert!(D.is_power_of_two());
    debug_assert!(table.len() >= D);

    let mut dense = [0i64; D];
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        dense[pos as usize] = i64::from(coeff);
    }

    for (shift, row) in table.iter_mut().enumerate().take(D) {
        row[shift..D].copy_from_slice(&dense[..D - shift]);
        for (dst, src) in row[..shift].iter_mut().zip(dense[D - shift..].iter()) {
            *dst = -*src;
        }
    }
}

pub(crate) fn fill_rotated_tensor_challenge<const D: usize>(
    table: &mut [[i64; D]],
    left: &SparseChallenge,
    right: &SparseChallenge,
) -> Result<(), AkitaError> {
    let challenge = IntegerChallenge::tensor_product::<D>(left, right)?;
    fill_rotated_integer_challenge::<D>(table, &challenge);
    Ok(())
}

pub(crate) fn narrow_tensor_accum_to_i32<const D: usize>(
    accum_i64: Vec<[i64; D]>,
) -> Result<Vec<[i32; D]>, AkitaError> {
    let mut out = Vec::with_capacity(accum_i64.len());
    for row in accum_i64 {
        let mut narrowed = [0i32; D];
        for (dst, src) in narrowed.iter_mut().zip(row.iter()) {
            *dst = i32::try_from(*src).map_err(|_| {
                AkitaError::InvalidSetup(format!(
                    "tensor fold accumulator overflowed i32 envelope (value = {src})"
                ))
            })?;
        }
        out.push(narrowed);
    }
    Ok(out)
}
