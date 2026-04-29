use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Pow2IW;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(Pow2IW, Some(Pow2W));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Pow2IW<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, self.0.instruction().imm())
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 + y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let y = LookupQuery::<XLEN>::to_lookup_index(self);
        1u64 << (y & ((XLEN as u128 / 2) - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::test::materialize_entry_test;
    use tracer::instruction::RISCVCycle;

    #[test]
    fn materialize_entry_virtualpow2iw() {
        materialize_entry_test::<
            Pow2IW<RISCVCycle<tracer::instruction::virtual_pow2i_w::VirtualPow2IW>>,
            RISCVCycle<tracer::instruction::virtual_pow2i_w::VirtualPow2IW>,
        >();
    }
}
