use jolt_claims::protocols::jolt::lattice::FUSED_INC_BITS;
use jolt_field::Field;
use jolt_program::execution::{RamAccess, TraceRow};
use jolt_riscv::CircuitFlags;

use super::{row_circuit_flags, Extract, ExtractIndexed, ToField, WitnessEnv};
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
        Ok(Self(match row.registers.rd {
            Some(write) => write.post_value as i128 - write.pre_value as i128,
            None => 0,
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
        Ok(Self(match row.ram_access {
            RamAccess::Write(write) => write.post_value as i128 - write.pre_value as i128,
            RamAccess::Read(_) | RamAccess::NoOp => 0,
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
    /// The encoder bias `(K/2)·(2^64 − 1)/(K − 1)` for radix `K = 2^width`:
    /// every centered digit of the biased zero is exactly `K/2`, so
    /// `delta = 0` lands every digit (and the carry) on lane 0.
    fn balanced_bias(width: usize) -> i128 {
        debug_assert!(width > 0 && FUSED_INC_BITS.is_multiple_of(width));
        let radix = 1i128 << width;
        (radix / 2) * (((1i128 << FUSED_INC_BITS) - 1) / (radix - 1))
    }

    fn biased_for_balanced_digits(self, width: usize) -> i128 {
        debug_assert!(self.0.unsigned_abs() < 1u128 << FUSED_INC_BITS);
        self.0 + Self::balanced_bias(width)
    }

    /// The hot address of one lane of the balanced one-hot decomposition:
    /// the centered radix-`2^width` digit (or the signed carry above bit 63)
    /// encoded modulo the radix. Lane `j` decodes to `j` if `j < K/2`, else
    /// `j − K`.
    pub fn hot_lane(self, lane: BalancedIncLane) -> usize {
        match lane {
            BalancedIncLane::Digit { width, index } => {
                let radix = 1i128 << width;
                let mask = radix - 1;
                let standard_digit =
                    (self.biased_for_balanced_digits(width) >> (width * index)) & mask;
                ((standard_digit + radix / 2) & mask) as usize
            }
            BalancedIncLane::Carry { width } => {
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
        let store = row_circuit_flags(row)?[CircuitFlags::Store];
        debug_assert_eq!(
            store,
            matches!(row.ram_access, RamAccess::Write(_)),
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

/// Selects one lane of the fused increment's balanced one-hot decomposition:
/// a centered `width`-bit digit (indexed from the least significant digit),
/// or the signed carry above bit 63.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BalancedIncLane {
    Digit { width: usize, index: usize },
    Carry { width: usize },
}

/// The per-cycle hot address of one `BalancedIncDigit`/`BalancedIncCarry`
/// column; every cycle is hot (padding rows land on lane 0 of every digit
/// and of the carry).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BalancedIncHot(pub usize);

impl From<BalancedIncHot> for Option<usize> {
    fn from(hot: BalancedIncHot) -> Self {
        Some(hot.0)
    }
}

impl ExtractIndexed<BalancedIncLane> for BalancedIncHot {
    fn extract_indexed(
        lane: BalancedIncLane,
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(FusedInc::extract(row, next, env)?.hot_lane(lane)))
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
                let hot = inc.hot_lane(BalancedIncLane::Digit {
                    width: LOG_K_CHUNK,
                    index,
                });
                assert!(hot < 1 << LOG_K_CHUNK, "cycle {cycle}");
                reconstructed += centered(hot, radix) << (LOG_K_CHUNK * index);
            }
            let carry = inc.hot_lane(BalancedIncLane::Carry { width: LOG_K_CHUNK });
            reconstructed += centered(carry, radix) << FUSED_INC_BITS;
            assert_eq!(reconstructed, inc.0, "cycle {cycle}");
        }
    }

    #[test]
    fn padding_cycles_encode_every_lane_at_digit_zero() {
        let padding = FusedInc(0);
        assert_eq!(
            padding.hot_lane(BalancedIncLane::Carry { width: LOG_K_CHUNK }),
            0
        );
        for index in 0..DIGITS {
            assert_eq!(
                padding.hot_lane(BalancedIncLane::Digit {
                    width: LOG_K_CHUNK,
                    index,
                }),
                0
            );
        }
    }
}
