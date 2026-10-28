use jolt_field::{Accumulator, JoltField};

use super::{extend, EXTENDED_NODE_COUNT};

const MAX_GROUPS: usize = 32;
const MAP_BITS: usize = 7;

struct GuardGroup<F: JoltField, const M: usize, const LIMBS: usize> {
    key: u64,
    guards: [i64; M],
    sums: [[F::SignedProductAccumulator; M]; LIMBS],
}

/// Rows with the same A interpolant share a linear B accumulation. The cap
/// bounds scratch space and lets unusual guard distributions use the ordinary
/// per-row calculation instead of growing a table with the trace.
pub(super) struct GuardGroups<F: JoltField, const M: usize, const LIMBS: usize> {
    slots: [Option<usize>; 1 << MAP_BITS],
    groups: Vec<GuardGroup<F, M, LIMBS>>,
}

impl<F: JoltField, const M: usize, const LIMBS: usize> GuardGroups<F, M, LIMBS> {
    pub(super) fn new() -> Self {
        Self {
            slots: [None; 1 << MAP_BITS],
            groups: Vec::with_capacity(MAX_GROUPS),
        }
    }

    #[inline]
    pub(super) fn try_add(
        &mut self,
        guards: &[i64; M],
        values: [&[i128; M]; LIMBS],
        weight: F,
    ) -> bool {
        if M > 21 {
            return false;
        }
        let mut key = 0u64;
        let mut valid = true;
        for (index, &guard) in guards.iter().enumerate() {
            let digit = guard.wrapping_add(3) as u64;
            valid &= digit <= 6;
            key |= (digit & 7) << (3 * index);
        }
        if !valid {
            return false;
        }
        let mut slot = (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - MAP_BITS)) as usize;
        let group_index = loop {
            if let Some(index) = self.slots[slot] {
                if self.groups[index].key == key {
                    break index;
                }
                slot = (slot + 1) & ((1 << MAP_BITS) - 1);
            } else {
                if self.groups.len() == MAX_GROUPS {
                    return false;
                }
                let index = self.groups.len();
                self.groups.push(GuardGroup {
                    key,
                    guards: *guards,
                    sums: [[F::SignedProductAccumulator::default(); M]; LIMBS],
                });
                self.slots[slot] = Some(index);
                break index;
            }
        };
        for (sums, values) in self.groups[group_index].sums.iter_mut().zip(values) {
            for (sum, &value) in sums.iter_mut().zip(values) {
                if value != 0 {
                    sum.fmadd_i128(weight, value);
                }
            }
        }
        true
    }

    pub(super) fn finish(self, radix: F) -> [F; EXTENDED_NODE_COUNT] {
        let mut result = [F::SignedProductAccumulator::default(); EXTENDED_NODE_COUNT];
        for group in self.groups {
            let mut values = [F::zero(); M];
            for (limb, sums) in group.sums.into_iter().rev().enumerate() {
                for (value, sum) in values.iter_mut().zip(sums) {
                    if limb == 0 {
                        *value = sum.reduce();
                    } else {
                        *value = radix * *value + sum.reduce();
                    }
                }
            }
            let a = extend(&group.guards);
            let b = extend(&values);
            for ((sum, guard), value) in result.iter_mut().zip(a).zip(b) {
                sum.fmadd_i64(value, guard);
            }
        }
        result.map(Accumulator::reduce)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;
    use crate::optimized::spartan_outer::{
        DOMAIN, DOMAIN_START, EXTENDED_POSITIONS, EXTENDED_START,
    };
    #[cfg(feature = "akita")]
    use jolt_field::Fp128;
    use jolt_field::Fr;
    use jolt_poly::lagrange::centered_lagrange_evals;

    fn check<F: JoltField, const M: usize, const LIMBS: usize>() {
        let mut groups = GuardGroups::<F, M, LIMBS>::new();
        let scales: [F; LIMBS] = std::array::from_fn(|limb| F::from_u128(1u128 << (64 * limb)));
        let mut expected = [F::zero(); EXTENDED_NODE_COUNT];
        let bases: Vec<_> = EXTENDED_POSITIONS
            .into_iter()
            .map(|position| {
                let point = F::from_i128(i128::from(EXTENDED_START) + position as i128);
                centered_lagrange_evals::<F>(DOMAIN, point).unwrap()
            })
            .collect();
        let mut accepted = 0;
        let mut rejected = 0;
        for row in 0..MAX_GROUPS * 12 {
            let mut code = row % (MAX_GROUPS * 2);
            let guards = std::array::from_fn(|_| {
                let digit = (code % 7) as i64 - 3;
                code /= 7;
                digit
            });
            let values: [[i128; M]; LIMBS] = std::array::from_fn(|limb| {
                std::array::from_fn(|coordinate| match (row + coordinate + limb) % 5 {
                    0 => 0,
                    1 => -1,
                    2 => (1i128 << 65) - row as i128,
                    3 => -(1i128 << 65) + coordinate as i128,
                    _ => row as i128 * 1_000_003 - coordinate as i128 * 901,
                })
            });
            let weight = F::from_u128((1u128 << 120) + row as u128 * 101);
            if !groups.try_add(&guards, std::array::from_fn(|limb| &values[limb]), weight) {
                rejected += 1;
                continue;
            }
            accepted += 1;
            for (expected, basis) in expected.iter_mut().zip(&bases) {
                let a: F = guards
                    .iter()
                    .zip(basis)
                    .map(|(&value, &basis)| F::from_i128(i128::from(value)) * basis)
                    .sum();
                let b: F = values
                    .iter()
                    .zip(scales)
                    .map(|(values, scale)| {
                        scale
                            * values
                                .iter()
                                .zip(basis)
                                .map(|(&value, &basis)| F::from_i128(value) * basis)
                                .sum::<F>()
                    })
                    .sum();
                *expected += weight * a * b;
            }
        }
        assert!(accepted > 0 && rejected > 0);
        assert_eq!(groups.groups.len(), MAX_GROUPS);
        let mut invalid = [0; M];
        invalid[0] = 4;
        assert!(!groups.try_add(&invalid, [&[0; M]; LIMBS], F::one()));
        assert_eq!(
            groups.finish(F::from_u128(1u128 << 64)),
            expected,
            "base starts at {DOMAIN_START}"
        );
    }

    #[test]
    fn grouped_rows_match_lagrange_products_with_bounded_fallback() {
        check::<Fr, DOMAIN, 1>();
        check::<Fr, { DOMAIN - 1 }, 2>();
        #[cfg(feature = "akita")]
        {
            check::<Fp128<{ u128::MAX - 274 }>, DOMAIN, 1>();
            check::<Fp128<{ u128::MAX - 274 }>, { DOMAIN - 1 }, 2>();
        }
    }
}
