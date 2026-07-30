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

    /// The hot address of one lane of the shifted encoding's one-hot
    /// decomposition.
    pub fn hot_lane(self, lane: UnsignedIncLane) -> usize {
        let shifted = self.shifted();
        match lane {
            UnsignedIncLane::Chunk { width, index } => {
                let low = shifted & ((1u128 << UNSIGNED_INC_BITS) - 1);
                ((low >> (width * index)) & ((1u128 << width) - 1)) as usize
            }
            UnsignedIncLane::Msb => (shifted >> UNSIGNED_INC_BITS) as usize,
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
        Ok(Self(FusedInc::extract(row, next, env)?.hot_lane(lane)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG_K_CHUNK: usize = 8;
    const CHUNKS: usize = UNSIGNED_INC_BITS / LOG_K_CHUNK;

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

    #[test]
    fn chunks_and_msb_reconstruct_the_shifted_fused_increment() {
        for (cycle, inc) in fused_trace().iter().enumerate() {
            let mut reconstructed = 0u128;
            for index in 0..CHUNKS {
                let hot = inc.hot_lane(UnsignedIncLane::Chunk {
                    width: LOG_K_CHUNK,
                    index,
                });
                assert!(hot < 1 << LOG_K_CHUNK, "cycle {cycle}");
                reconstructed |= (hot as u128) << (LOG_K_CHUNK * index);
            }
            reconstructed |= (inc.hot_lane(UnsignedIncLane::Msb) as u128) << UNSIGNED_INC_BITS;
            assert_eq!(
                reconstructed as i128 - (1i128 << UNSIGNED_INC_BITS),
                inc.0,
                "cycle {cycle}"
            );
        }
    }

    #[test]
    fn padding_cycles_encode_msb_hot_and_zero_digits() {
        let padding = FusedInc(0);
        assert_eq!(padding.hot_lane(UnsignedIncLane::Msb), 1);
        for index in 0..CHUNKS {
            assert_eq!(
                padding.hot_lane(UnsignedIncLane::Chunk {
                    width: LOG_K_CHUNK,
                    index,
                }),
                0
            );
        }
    }
}
