//! Prover-side sampling for commitment masking.

#[cfg(test)]
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField};
#[cfg(test)]
use akita_types::FlatMatrix;
use akita_types::{zk, FlatDigitBlocks};
use rand_core::{OsRng, RngCore};

fn sample_balanced_pow2_digit<R: RngCore>(rng: &mut R, log_basis: u32) -> i8 {
    // The alphabet size is a power of two, so masking low bits is uniform.
    let raw = (rng.next_u32() & ((1u32 << log_basis) - 1)) as i16;
    let half_basis = 1i16 << (log_basis - 1);
    let basis = half_basis << 1;
    let balanced = if raw >= half_basis { raw - basis } else { raw };
    balanced as i8
}

/// Sample a fresh digit-source LHL blinding vector.
///
/// # Errors
///
/// Returns an error if digit block sizing overflows.
pub(crate) fn sample_blinding_digits<F, const D: usize>(
    output_ring_len: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks<D>, AkitaError>
where
    F: CanonicalField,
{
    if !(1..=8).contains(&log_basis) {
        return Err(AkitaError::InvalidInput(
            "ZK digit blinding log_basis must be in 1..=8".to_string(),
        ));
    }

    let blinding_planes = zk::blinding_digit_plane_count::<F>(output_ring_len, D, log_basis);
    if blinding_planes == 0 {
        return Ok(FlatDigitBlocks::empty());
    }

    let block_sizes = vec![blinding_planes];
    let mut out = FlatDigitBlocks::zeroed(block_sizes)?;
    let mut rng = OsRng;
    for plane in out.flat_digits_mut() {
        for coeff in plane {
            *coeff = sample_balanced_pow2_digit(&mut rng, log_basis);
        }
    }
    Ok(out)
}

#[cfg(test)]
fn digit_ring<F: CanonicalField, const D: usize>(digits: &[i8; D]) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| F::from_i64(digits[idx] as i64)))
}

/// Multiply stored ZK setup rows by cyclic digit planes.
#[cfg(test)]
pub(crate) fn zk_matrix_cyclic_digit_rows<F: CanonicalField, const D: usize>(
    matrix: &FlatMatrix<F>,
    row_len: usize,
    col_offset: usize,
    row_width: usize,
    digits: &[[i8; D]],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if digits.is_empty() {
        return Ok(vec![CyclotomicRing::zero(); row_len]);
    }
    let col_end = col_offset
        .checked_add(digits.len())
        .ok_or_else(|| AkitaError::InvalidSetup("ZK matrix digit column overflow".to_string()))?;
    if col_end > row_width {
        return Err(AkitaError::InvalidSetup(
            "ZK matrix digit columns exceed row width".to_string(),
        ));
    }
    let view = matrix.ring_view::<D>(row_len, row_width)?;
    let matrix_rows = view.as_slice();
    let stride = view.num_cols();
    let digit_rings = digits.iter().map(digit_ring::<F, D>).collect::<Vec<_>>();
    let rows = (0..row_len)
        .map(|row_idx| {
            let row_start = row_idx * stride + col_offset;
            let mut acc = [F::zero(); D];
            for (entry, digit) in matrix_rows[row_start..row_start + digit_rings.len()]
                .iter()
                .zip(digit_rings.iter())
            {
                add_cyclic_product(&mut acc, entry, digit);
            }
            CyclotomicRing::from_coefficients(acc)
        })
        .collect();
    Ok(rows)
}

#[cfg(test)]
fn add_cyclic_product<F: CanonicalField, const D: usize>(
    acc: &mut [F; D],
    lhs: &CyclotomicRing<F, D>,
    rhs: &CyclotomicRing<F, D>,
) {
    for (i, &a) in lhs.coefficients().iter().enumerate() {
        if a.is_zero() {
            continue;
        }
        for (j, &b) in rhs.coefficients().iter().enumerate() {
            if !b.is_zero() {
                acc[(i + j) % D] += a * b;
            }
        }
    }
}
