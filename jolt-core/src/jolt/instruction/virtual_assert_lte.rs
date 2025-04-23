use tracer::instruction::{virtual_assert_lte::VirtualAssertLTE, RISCVCycle};

use crate::jolt::lookup_table::{
    unsigned_less_than_equal::UnsignedLessThanEqualTable, LookupTables,
};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualAssertLTE {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(UnsignedLessThanEqualTable.into())
    }
}

impl InstructionFlags for VirtualAssertLTE {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Assert as usize] = true;
        flags[CircuitFlags::Virtual as usize] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdatePC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualAssertLTE> {
    fn to_lookup_query(&self) -> (u64, u64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.register_state.rs2 as u8 as u64,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.register_state.rs2 as u32 as u64,
            ),
            64 => (self.register_state.rs1, self.register_state.rs2),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    #[cfg(test)]
    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_lookup_query(self);
        (x <= y).into()
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualAssertLTE>();
    }
}
