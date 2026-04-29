use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::And;
use jolt_trace::JoltCycle;

impl_lookup_table!(And, Some(And));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for And<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        x & y as u64 & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize_entry_test;

    #[test]
    fn materialize_entry_and() {
        materialize_entry_test!(And, tracer::instruction::and::AND);
    }
}
