use crate::{Field, Unreduced};

/// Reduce separated positive and negative unreduced accumulators into one field
/// element. Each accumulator must stay within its type's documented headroom.
/// The negative sum may exceed the positive sum because subtraction follows reduction.
#[inline]
pub fn reduce_signed_accum<E: Field + Unreduced>(pos: E::SmallProduct, neg: E::SmallProduct) -> E {
    E::reduce_small_product(pos) - E::reduce_small_product(neg)
}

#[cfg(all(test, feature = "solinas"))]
mod tests {
    use super::reduce_signed_accum;
    use crate::{Prime64Offset59, Ring, Unreduced};

    #[test]
    fn signed_accum_matches_field_difference() {
        type F = Prime64Offset59;
        for (pos, neg) in [(7, 3), (3, 7), (0, 0), (7, 7), (0, 7), (u64::MAX, 1)] {
            let pos = F::from_u64(pos);
            let neg = F::from_u64(neg);
            let pos_accum = pos.mul_u64_unreduced(u64::MAX) + pos.mul_u64_unreduced(2);
            let neg_accum = neg.mul_u64_unreduced(u64::MAX) + neg.mul_u64_unreduced(2);
            let scale = F::from_u64(u64::MAX) + F::from_u64(2);
            assert_eq!(
                reduce_signed_accum::<F>(pos_accum, neg_accum),
                pos * scale - neg * scale
            );
        }
    }
}
