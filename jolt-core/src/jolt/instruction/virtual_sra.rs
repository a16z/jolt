use tracer::instruction::{virtual_sra::VirtualSRA, RISCVCycle};

use crate::{
    jolt::lookup_table::{virtual_sra::VirtualSRATable, LookupTables},
    subprotocols::sparse_dense_shout::LookupBits,
};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<VirtualSRA> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(VirtualSRATable.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, self.register_state.rs2)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        let mut x = LookupBits::new(x as u64, WORD_SIZE);
        let mut y = LookupBits::new(y as u64, WORD_SIZE);

        let sign_bit = if x.leading_ones() == 0 { 0 } else { 1 };
        let mut entry = 0;
        let mut sign_extension = 0;
        for i in 0..WORD_SIZE {
            let x_i = x.pop_msb() as u64;
            let y_i = y.pop_msb() as u64;
            entry *= 1 + y_i;
            entry += x_i * y_i;
            if i != 0 {
                sign_extension += (1 << i) * (1 - y_i);
            }
        }
        entry + sign_bit * sign_extension
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualSRA>();
    }
}
