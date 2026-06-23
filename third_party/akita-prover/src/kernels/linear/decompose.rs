use super::*;

/// Convert a field element to a centered signed byte when it fits.
#[inline(always)]
pub fn try_centered_i8<F: CanonicalField>(coeff: F, q: u128, half_q: u128) -> Option<i8> {
    let canonical = coeff.to_canonical_u128();
    let centered = if canonical > half_q {
        -((q - canonical) as i128)
    } else {
        canonical as i128
    };
    if (i8::MIN as i128..=i8::MAX as i128).contains(&centered) {
        Some(centered as i8)
    } else {
        None
    }
}

/// Basis-decompose a block of ring elements into `block.len() * num_digits` gadget components.
pub fn decompose_block<F: FieldCore + CanonicalField, const D: usize>(
    block: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = vec![CyclotomicRing::<F, D>::zero(); block.len() * num_digits];
    for (i, coeff_vec) in block.iter().enumerate() {
        coeff_vec.balanced_decompose_pow2_into(
            &mut out[i * num_digits..(i + 1) * num_digits],
            log_basis,
        );
    }
    out
}

/// Like [`decompose_block`] but outputs `[i8; D]` digit planes instead of ring elements.
pub fn decompose_block_i8<F: FieldCore + CanonicalField, const D: usize>(
    block: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i8; D]> {
    let mut out = vec![[0i8; D]; block.len() * num_digits];
    decompose_rows_i8_into(block, &mut out, num_digits, log_basis);
    out
}

/// Decompose each ring element in `rows` into `[i8; D]` digit planes.
pub fn decompose_rows_i8<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i8; D]> {
    let mut out = vec![[0i8; D]; rows.len() * num_digits];
    decompose_rows_i8_into(rows, &mut out, num_digits, log_basis);
    out
}

/// Decompose each ring element in `rows` into a preallocated flat digit buffer.
///
/// # Panics
///
/// Panics if `out.len() != rows.len() * num_digits`.
pub fn decompose_rows_i8_into<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    out: &mut [[i8; D]],
    num_digits: usize,
    log_basis: u32,
) {
    assert_eq!(
        out.len(),
        rows.len() * num_digits,
        "flat digit output length must match rows * num_digits",
    );
    if num_digits == 0 {
        return;
    }
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(num_digits)
        .zip(rows.par_iter())
        .for_each(|(dst_chunk, row)| {
            row.balanced_decompose_pow2_i8_into_with_params(dst_chunk, &decompose_params)
        });

    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(num_digits)
        .zip(rows.iter())
        .for_each(|(dst_chunk, row)| {
            row.balanced_decompose_pow2_i8_into_with_params(dst_chunk, &decompose_params)
        });
}
