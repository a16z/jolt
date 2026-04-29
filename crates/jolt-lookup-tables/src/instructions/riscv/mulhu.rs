use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::MulHU;
use jolt_trace::JoltCycle;

impl_lookup_table!(MulHU, Some(UpperWord));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for MulHU<C> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 * y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        ((x as u128 * y as u64 as u128) >> XLEN) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize_entry_test;

    #[test]
    fn materialize_entry_mulhu() {
        materialize_entry_test!(MulHU, tracer::instruction::mulhu::MULHU);
    }
}
