use crate::traits::{impl_lookup_table, LookupQuery};
use jolt_riscv::instructions::{PcallCall, PcallRead, PcallReturn, PcallWrite};
use jolt_riscv::JoltCycle;

// Proving rejects PCALL rows before witness generation.
macro_rules! unsupported_lookup {
    ($kind:ident) => {
        impl_lookup_table!($kind, None);
        impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for $kind<C> {
            fn to_instruction_inputs(&self) -> (u64, i128) {
                (0, 0)
            }
            fn to_lookup_output(&self) -> u64 {
                0
            }
        }
    };
}
unsupported_lookup!(PcallCall);
unsupported_lookup!(PcallRead);
unsupported_lookup!(PcallReturn);
unsupported_lookup!(PcallWrite);
