use tracer::instruction::{virtual_srl::VirtualSRL, RISCVCycle};

use crate::{
    jolt::lookup_table::{virtual_srl::VirtualSRLTable, LookupTables},
    subprotocols::sparse_dense_shout::LookupBits,
};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<VirtualSRL> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(VirtualSRLTable.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, self.register_state.rs2)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        let mut x = LookupBits::new(x as u64, WORD_SIZE);
        let mut y = LookupBits::new(y as u64, WORD_SIZE);

        let mut entry = 0;
        for _ in 0..WORD_SIZE {
            let x_i = x.pop_msb();
            let y_i = y.pop_msb();
            entry *= 1 + y_i as u64;
            entry += (x_i * y_i) as u64;
        }
        entry
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualSRL>();
    }
}
