use jolt_claims::protocols::jolt::lattice::FUSED_INC_BITS;
use jolt_field::Field;
use jolt_riscv::CircuitFlags;
use jolt_riscv::JoltTraceRow as TraceRow;

use super::{Extract, ExtractIndexed, ToField, WitnessEnv};
use crate::WitnessError;

/// Signed delta written to rd this cycle; 0 when the instruction has no rd
/// operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RdInc(pub i128);

/// Signed delta written to RAM this cycle; 0 for reads and no-ops.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamInc(pub i128);

impl ToField for RdInc {
    fn to_field<F: Field>(self) -> F {
        F::from_i128(self.0)
    }
}

impl Extract for RdInc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(if row.rd_index().is_some() {
            row.rd_write_value() as i128 - row.rd_pre_value() as i128
        } else {
            0
        }))
    }
}

impl ToField for RamInc {
    fn to_field<F: Field>(self) -> F {
        F::from_i128(self.0)
    }
}

impl Extract for RamInc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(if row.is_store() {
            row.ram_write_value() as i128 - row.ram_read_value() as i128
        } else {
            0
        }))
    }
}

/// The per-cycle fused increment of the packed (lattice) witness: the RAM
/// delta on store cycles (the `OpFlags(Store)` circuit flag), the rd delta
/// otherwise. One fused column serves both inc consumers because no cycle
/// increments RAM and rd at once — ISA stores carry no rd, and every
/// read-modify-write instruction lowers into a sequence whose RAM-writing
/// step is a plain store. Padding rows carry delta 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FusedInc(pub i128);

impl FusedInc {
    /// `(radix/2) · (2^FUSED_INC_BITS − 1)/(radix − 1)` per digit width
    /// dividing [`FUSED_INC_BITS`] (0 elsewhere). Precomputed because the
    /// closed form's i128 division lowers to a `__udivti3` libcall, and
    /// [`Self::selected_row`] reads the bias per cycle per balanced column.
    const BALANCED_BIASES: [i128; FUSED_INC_BITS + 1] = {
        let mut table = [0i128; FUSED_INC_BITS + 1];
        let mut width = 1;
        while width <= FUSED_INC_BITS {
            if FUSED_INC_BITS.is_multiple_of(width) {
                let radix = 1i128 << width;
                table[width] = (radix / 2) * (((1i128 << FUSED_INC_BITS) - 1) / (radix - 1));
            }
            width += 1;
        }
        table
    };

    fn balanced_bias(width: usize) -> i128 {
        debug_assert!(width > 0 && FUSED_INC_BITS.is_multiple_of(width));
        Self::BALANCED_BIASES[width]
    }

    fn biased_for_balanced_digits(self, width: usize) -> i128 {
        debug_assert!(self.0.unsigned_abs() < 1u128 << FUSED_INC_BITS);
        self.0 + Self::balanced_bias(width)
    }

    /// The selected row of one centered digit or its signed carry.
    pub fn selected_row(self, column: BalancedIncColumn) -> usize {
        match column {
            BalancedIncColumn::Digit { width, index } => {
                let radix = 1i128 << width;
                let standard =
                    (self.biased_for_balanced_digits(width) >> (width * index)) & (radix - 1);
                ((standard + radix / 2) & (radix - 1)) as usize
            }
            BalancedIncColumn::Carry { width } => {
                let radix = 1i128 << width;
                let carry = self.biased_for_balanced_digits(width) >> FUSED_INC_BITS;
                debug_assert!((-1..=1).contains(&carry));
                carry.rem_euclid(radix) as usize
            }
        }
    }
}

impl ToField for FusedInc {
    fn to_field<F: Field>(self) -> F {
        F::from_i128(self.0)
    }
}

impl Extract for FusedInc {
    fn extract(
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let store = row.circuit_flags()[CircuitFlags::Store];
        debug_assert_eq!(
            store,
            row.is_store(),
            "Store circuit flag disagrees with the cycle's RAM-write access"
        );
        let ram_delta = RamInc::extract(row, next, env)?.0;
        let rd_delta = RdInc::extract(row, next, env)?.0;
        // One fused column serves both inc consumers only because no cycle
        // increments RAM and rd at once (every read-modify-write instruction
        // lowers into a sequence whose RAM-writing step is a plain store). A
        // violation means an instruction shape the fused encoding cannot
        // represent — fail here, not with an opaque sumcheck mismatch.
        debug_assert!(
            if store { rd_delta == 0 } else { ram_delta == 0 },
            "cycle increments both RAM and rd; the fused inc encoding cannot represent it"
        );
        Ok(Self(if store { ram_delta } else { rd_delta }))
    }
}

/// Selects one centered radix digit or the signed carry above bit 63.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BalancedIncColumn {
    Digit { width: usize, index: usize },
    Carry { width: usize },
}

/// The row selected by one `BalancedIncDigit`/`BalancedIncCarry` column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BalancedIncRow(pub usize);

impl From<BalancedIncRow> for Option<usize> {
    fn from(row: BalancedIncRow) -> Self {
        Some(row.0)
    }
}

impl ExtractIndexed<BalancedIncColumn> for BalancedIncRow {
    fn extract_indexed(
        column: BalancedIncColumn,
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            FusedInc::extract(row, next, env)?.selected_row(column),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG_K_CHUNK: usize = 8;
    const DIGITS: usize = FUSED_INC_BITS / LOG_K_CHUNK;

    fn fused_trace() -> Vec<FusedInc> {
        [
            7i128,
            -3,
            0,
            (1 << 40) + 5,
            -(1 << 63),
            (1i128 << 64) - 1,
            -((1i128 << 64) - 1),
            1,
            -1,
        ]
        .into_iter()
        .map(FusedInc)
        .collect()
    }

    fn centered(row: usize, radix: i128) -> i128 {
        if (row as i128) < radix / 2 {
            row as i128
        } else {
            row as i128 - radix
        }
    }

    #[test]
    fn balanced_digits_and_carry_reconstruct_the_fused_increment() {
        let radix = 1i128 << LOG_K_CHUNK;
        for (cycle, inc) in fused_trace().iter().enumerate() {
            let mut reconstructed = 0i128;
            for index in 0..DIGITS {
                let selected_row = inc.selected_row(BalancedIncColumn::Digit {
                    width: LOG_K_CHUNK,
                    index,
                });
                assert!(selected_row < 1 << LOG_K_CHUNK, "cycle {cycle}");
                reconstructed += centered(selected_row, radix) << (LOG_K_CHUNK * index);
            }
            let carry = inc.selected_row(BalancedIncColumn::Carry { width: LOG_K_CHUNK });
            reconstructed += centered(carry, radix) << FUSED_INC_BITS;
            assert_eq!(reconstructed, inc.0, "cycle {cycle}");
        }
    }

    #[test]
    fn zero_delta_uses_balanced_zero_digits_and_carry() {
        let padding = FusedInc(0);
        assert_eq!(
            padding.selected_row(BalancedIncColumn::Carry { width: LOG_K_CHUNK }),
            0
        );
        for index in 0..DIGITS {
            assert_eq!(
                padding.selected_row(BalancedIncColumn::Digit {
                    width: LOG_K_CHUNK,
                    index,
                }),
                0
            );
        }
    }
}
