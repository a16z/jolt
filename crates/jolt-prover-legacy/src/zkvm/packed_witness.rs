//! Prover-side Akita witness assembly. `OneHotTrace` contains the uniform
//! row-major one-hot columns derived from the execution trace; program objects
//! retain sparse prefix-packed representations.

pub use jolt_claims::protocols::jolt::lattice::FUSED_INC_BITS;
use jolt_poly::OneHotPolynomial;

use crate::zkvm::instruction::{CircuitFlags, Flags, JoltTraceCycle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DigitZeroRow {
    Committed,
    Virtualized,
}

/// Packs equal-length semantic columns into prefix slots and pads unused
/// slots with zero rows. A virtualized digit-zero row is omitted from the
/// physical polynomial; a committed column retains row zero.
pub fn pack_one_hot_columns(
    k: usize,
    slot_capacity: usize,
    columns: Vec<(Vec<Option<u8>>, DigitZeroRow)>,
) -> OneHotPolynomial {
    assert!(slot_capacity.is_power_of_two());
    assert!(!columns.is_empty() && columns.len() <= slot_capacity);
    let rows = columns[0].0.len();
    assert!(columns.iter().all(|(column, _)| column.len() == rows));

    let mut indices = Vec::with_capacity(slot_capacity * rows);
    for (column, digit_zero_row) in columns {
        indices.extend(column.into_iter().map(|row| match row {
            Some(0) if digit_zero_row == DigitZeroRow::Virtualized => None,
            None => None,
            Some(row) => {
                assert!((row as usize) < k);
                Some(row)
            }
        }));
    }
    indices.resize(slot_capacity * rows, None);
    OneHotPolynomial::new(k, indices)
}

/// The per-cycle fused increment stream: the RAM delta on store cycles, the
/// rd delta otherwise. Padding cycles carry `delta = 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedIncValue {
    pub delta: i128,
}

impl FusedIncValue {
    /// The per-cycle fused delta: the RAM write delta on store cycles, the
    /// rd write delta otherwise.
    pub fn from_cycle(cycle: &tracer::instruction::Cycle) -> Self {
        Self::from_cycle_with_store(cycle).0
    }

    /// [`from_cycle`](Self::from_cycle) plus the store selector itself, so
    /// witness generation and the read-raf fused stages read one
    /// predicate: the same `OpFlags(Store)` circuit flag the sumcheck
    /// selector opens.
    pub fn from_cycle_with_store(cycle: &tracer::instruction::Cycle) -> (Self, bool) {
        let store = JoltTraceCycle::try_new(cycle)
            .expect("OneHotTrace cycles must be final Jolt instruction rows")
            .circuit_flags()[CircuitFlags::Store];
        let ram_delta = match cycle.ram_access() {
            tracer::instruction::RAMAccess::Write(write) => {
                write.post_value as i128 - write.pre_value as i128
            }
            _ => 0,
        };
        let (_, rd_pre_value, rd_post_value) = cycle.rd_write().unwrap_or_default();
        let rd_delta = rd_post_value as i128 - rd_pre_value as i128;
        // One fused column can serve both inc consumers only because no
        // cycle increments RAM and rd at once (every RMW instruction lowers
        // into a sequence whose RAM-writing step is a plain store). A
        // violation means an instruction shape the fused encoding cannot
        // represent — fail here, not with an opaque sumcheck mismatch.
        debug_assert_eq!(
            store,
            matches!(cycle.ram_access(), tracer::instruction::RAMAccess::Write(_)),
            "Store circuit flag disagrees with the cycle's RAM-write access: {cycle:?}"
        );
        debug_assert!(
            if store { rd_delta == 0 } else { ram_delta == 0 },
            "cycle increments both RAM and rd; the fused inc encoding cannot represent it: {cycle:?}"
        );
        let delta = if store { ram_delta } else { rd_delta };
        (Self { delta }, store)
    }

    fn balanced_bias(width: usize) -> i128 {
        debug_assert!(width > 0 && FUSED_INC_BITS.is_multiple_of(width));
        let radix = 1i128 << width;
        (radix / 2) * (((1i128 << FUSED_INC_BITS) - 1) / (radix - 1))
    }

    fn biased_for_balanced_digits(self, width: usize) -> i128 {
        debug_assert!(self.delta.unsigned_abs() < 1u128 << FUSED_INC_BITS);
        self.delta + Self::balanced_bias(width)
    }

    /// The centered radix-`2^width` digit encoded modulo the radix.
    pub fn balanced_digit_row(self, width: usize, index: usize) -> usize {
        let radix = 1i128 << width;
        let mask = radix - 1;
        let standard_digit = (self.biased_for_balanced_digits(width) >> (width * index)) & mask;
        ((standard_digit + radix / 2) & mask) as usize
    }

    /// The signed carry above bit 63, encoded modulo the chunk radix.
    pub fn balanced_carry_row(self, width: usize) -> usize {
        let radix = 1i128 << width;
        let carry = self.biased_for_balanced_digits(width) >> FUSED_INC_BITS;
        debug_assert!((-1..=1).contains(&carry));
        carry.rem_euclid(radix) as usize
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::lattice::BalancedIncChunking;

    #[test]
    fn physical_one_hot_prefix_order_matches_selector_reduction() {
        use jolt_field::{Fr, Ring};
        use jolt_poly::{eq_index_msb, MultilinearPoly};

        let columns = vec![
            (vec![Some(0), Some(2)], DigitZeroRow::Virtualized),
            (vec![Some(0), None], DigitZeroRow::Committed),
        ];
        let packed = pack_one_hot_columns(4, 4, columns);
        assert_eq!(
            packed.indices(),
            [None, Some(2), Some(0), None, None, None, None, None]
        );

        let selector = [Fr::from_u64(3), Fr::from_u64(5)];
        let logical = [Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)];
        let first = OneHotPolynomial::new(4, vec![None, Some(2)]).evaluate(&logical);
        let second = OneHotPolynomial::new(4, vec![Some(0), None]).evaluate(&logical);
        let expected = eq_index_msb(&selector, 0) * first + eq_index_msb(&selector, 1) * second;
        let point = [selector.as_slice(), logical.as_slice()].concat();
        assert_eq!(packed.evaluate(&point), expected);
    }

    #[test]
    fn balanced_chunks_and_carry_reconstruct_the_fused_increment() {
        let values = [
            -(1i128 << 64) + 1,
            -(1i128 << 63),
            -129,
            -128,
            -127,
            -1,
            0,
            1,
            127,
            128,
            129,
            (1i128 << 63) - 1,
            (1i128 << 64) - 1,
        ];
        for width in [4, 8] {
            let chunking = BalancedIncChunking::new(width).unwrap();
            let radix = 1i128 << width;
            for delta in values {
                let inc = FusedIncValue { delta };
                let digits = (0..chunking.chunk_count()).map(|index| {
                    let row = inc.balanced_digit_row(width, index) as i128;
                    if row < radix / 2 {
                        row
                    } else {
                        row - radix
                    }
                });
                let carry_row = inc.balanced_carry_row(width) as i128;
                let carry = if carry_row < radix / 2 {
                    carry_row
                } else {
                    carry_row - radix
                };
                let reconstructed = digits
                    .enumerate()
                    .fold(0, |sum, (index, digit)| sum + (digit << (width * index)))
                    + (carry << FUSED_INC_BITS);
                assert_eq!(reconstructed, delta, "width={width}, delta={delta}");
            }
        }
    }

    #[test]
    fn zero_fused_increment_uses_digit_zero_for_every_column() {
        let inc = FusedIncValue { delta: 0 };
        for width in [4, 8, 16, 32] {
            let chunking = BalancedIncChunking::new(width).unwrap();
            for index in 0..chunking.chunk_count() {
                assert_eq!(inc.balanced_digit_row(width, index), 0);
            }
            assert_eq!(inc.balanced_carry_row(width), 0);
        }
    }
}
