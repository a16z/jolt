use super::*;

pub(super) fn fold_onehot_block<E, F, const D: usize>(
    entries: &[E],
    scalars: &[F],
    block_len: usize,
) -> CyclotomicRing<F, D>
where
    E: OneHotEntry,
    F: FieldCore,
{
    let mut coeffs_acc = [F::zero(); D];

    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            let s = scalars[pos];
            for &ci in entry.coeffs() {
                coeffs_acc[ci as usize] += s;
            }
        }
    }

    CyclotomicRing::from_coefficients(coeffs_acc)
}

pub(super) fn fold_onehot_block_ring<E, F, const D: usize>(
    entries: &[E],
    scalars: &[CyclotomicRing<F, D>],
    block_len: usize,
) -> CyclotomicRing<F, D>
where
    E: OneHotEntry,
    F: FieldCore,
{
    let mut acc = CyclotomicRing::<F, D>::zero();

    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            for &ci in entry.coeffs() {
                scalars[pos].shift_accumulate_into(&mut acc, ci as usize);
            }
        }
    }

    acc
}
