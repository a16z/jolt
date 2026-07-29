use jolt_claims::protocols::jolt::lattice::UNSIGNED_INC_BITS;
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
    /// The shifted unsigned encoding `2^64 + delta`: the MSB and low-64-bit
    /// chunks. Padding (`delta = 0`) encodes as MSB hot with every chunk at
    /// hot lane zero.
    fn shifted(self) -> u128 {
        debug_assert!(self.0.unsigned_abs() < 1u128 << UNSIGNED_INC_BITS);
        (self.0 + (1i128 << UNSIGNED_INC_BITS)) as u128
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
        Ok(Self(if store {
            RamInc::extract(row, next, env)?.0
        } else {
            RdInc::extract(row, next, env)?.0
        }))
    }
}

/// Selects one lane of the fused increment's one-hot decomposition: a
/// `width`-bit chunk of the shifted encoding's low 64 bits (indexed from the
/// least significant chunk), or the MSB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsignedIncLane {
    Chunk { width: usize, index: usize },
    Msb,
}

/// The per-cycle hot address of one `UnsignedIncChunk`/`UnsignedIncMsb`
/// column; every cycle is hot (padding rows land on lane 0 of each chunk and
/// lane 1 of the msb).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnsignedIncHot(pub usize);

impl From<UnsignedIncHot> for Option<usize> {
    fn from(hot: UnsignedIncHot) -> Self {
        Some(hot.0)
    }
}

impl ExtractIndexed<UnsignedIncLane> for UnsignedIncHot {
    fn extract_indexed(
        lane: UnsignedIncLane,
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let shifted = FusedInc::extract(row, next, env)?.shifted();
        Ok(Self(match lane {
            UnsignedIncLane::Chunk { width, index } => {
                let low = shifted & ((1u128 << UNSIGNED_INC_BITS) - 1);
                ((low >> (width * index)) & ((1u128 << width) - 1)) as usize
            }
            UnsignedIncLane::Msb => (shifted >> UNSIGNED_INC_BITS) as usize,
        }))
    }
}
