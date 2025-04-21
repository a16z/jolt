use tracer::instruction::{virtual_assert_valid_div0::VirtualAssertValidDiv0, RISCVCycle};

use crate::jolt::lookup_table::{valid_div0::ValidDiv0Table, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<VirtualAssertValidDiv0> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(ValidDiv0Table.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, self.register_state.rs2)
    }

    fn to_lookup_output(&self) -> u64 {
        let (divisor, quotient) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        if divisor == 0 {
            match WORD_SIZE {
                32 => (quotient == u32::MAX as u64).into(),
                64 => (quotient == u64::MAX).into(),
                _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
            }
        } else {
            1
        }
    }
}
