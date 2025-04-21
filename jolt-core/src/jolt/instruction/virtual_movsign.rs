use tracer::instruction::{virtual_movsign::VirtualMovsign, RISCVCycle};

use crate::jolt::lookup_table::{movsign::MovsignTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<VirtualMovsign> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(MovsignTable.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, _) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => {
                if x & (1 << 7) != 0 {
                    0xFF
                } else {
                    0
                }
            }
            32 => {
                if x & (1 << 31) != 0 {
                    0xFFFFFFFF
                } else {
                    0
                }
            }
            64 => {
                if x & (1 << 63) != 0 {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            }
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualMovsign>();
    }
}
