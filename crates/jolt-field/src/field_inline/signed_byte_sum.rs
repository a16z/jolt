/// One operand ingress per signed byte, even for magnitude two. The iterator
/// may invoke field operations: keep the same accumulator lease as signed_sum.
#[inline(always)]
pub fn sum_signed_bytes<'a, const N: usize>(
    terms: impl IntoIterator<Item = (&'a [u64; N], i8)>,
) -> [u64; N] {
    debug_assert!(accumulators_free());
    ACC_LENT.store(true, Ordering::Relaxed);
    emit::acc_zero();
    for (term, coefficient) in terms {
        if coefficient == 0 {
            continue;
        }
        load(REG_A, term);
        // No iterator callback runs between these operations, so REG_A still
        // contains term. unsigned_abs also covers the i8::MIN boundary.
        if coefficient < 0 {
            for _ in 0..coefficient.unsigned_abs() {
                emit::acc_sub_a();
            }
        } else {
            for _ in 0..coefficient.unsigned_abs() {
                emit::acc_add_a();
            }
        }
    }
    let sum = read_out(REG_ACC);
    ACC_LENT.store(false, Ordering::Relaxed);
    sum
}
