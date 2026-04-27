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
pub trait InstructionLookupTable {
    fn lookup_table(&self) -> Option<LookupTableKind>;
}

macro_rules! impl_lookup_table {
    ($instr:ty, Some($table:ident)) => {
        impl $crate::InstructionLookupTable for $instr {
            #[inline]
            fn lookup_table(&self) -> Option<$crate::LookupTableKind> {
                Some($crate::LookupTableKind::$table)
            }
        }
    };
    ($instr:ty, None) => {
        impl $crate::InstructionLookupTable for $instr {
            #[inline]
            fn lookup_table(&self) -> Option<$crate::LookupTableKind> {
                None
            }
        }
    };
}

pub(crate) use impl_lookup_table;
