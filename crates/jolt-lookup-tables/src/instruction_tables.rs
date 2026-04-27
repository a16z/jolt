//! Mapping from Jolt instruction structs to their lookup table decomposition.
//!
//! Per-instruction `InstructionLookupTable` impls live in
//! [`crate::instructions`], one per file. Each invokes the
//! [`impl_lookup_table!`] macro defined here.

use crate::tables::LookupTableKind;

/// Maps an instruction to the lookup table it decomposes into for the proving system.
///
/// Returns `None` for instructions that don't use lookup tables (loads, stores,
/// system instructions). The prover uses this to route instruction evaluations
/// to the correct table during the instruction sumcheck.
///
/// Generic over `XLEN` so the same instruction can be used at production word
/// size (XLEN=64) and at test sizes (XLEN=8).
pub trait InstructionLookupTable<const XLEN: usize> {
    fn lookup_table(&self) -> Option<LookupTableKind<XLEN>>;
}

macro_rules! impl_lookup_table {
    ($instr:ty, Some($table:ident)) => {
        impl<const XLEN: usize> $crate::InstructionLookupTable<XLEN> for $instr {
            #[inline]
            fn lookup_table(&self) -> Option<$crate::LookupTableKind<XLEN>> {
                Some($crate::LookupTableKind::$table(
                    ::core::default::Default::default(),
                ))
            }
        }
    };
    ($instr:ty, None) => {
        impl<const XLEN: usize> $crate::InstructionLookupTable<XLEN> for $instr {
            #[inline]
            fn lookup_table(&self) -> Option<$crate::LookupTableKind<XLEN>> {
                None
            }
        }
    };
}

pub(crate) use impl_lookup_table;
