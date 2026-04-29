use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualAdviceLen;
use jolt_trace::JoltCycle;

impl_lookup_table!(VirtualAdviceLen, Some(RangeCheck));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualAdviceLen<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (0, (self.0.rd_vals().map_or(0, |(_, p)| p) & mask) as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        self.0.rd_vals().map_or(0, |(_, p)| p) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize_entry_test;

    #[test]
    fn materialize_entry_virtualadvicelen() {
        materialize_entry_test!(VirtualAdviceLen, tracer::instruction::virtual_advice_len::VirtualAdviceLen);
    }
}
