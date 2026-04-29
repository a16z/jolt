use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::SltU;
use jolt_trace::JoltCycle;

impl_lookup_table!(SltU, Some(UnsignedLessThan));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for SltU<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (x < y as u64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize_entry_test;

    #[test]
    fn materialize_entry_sltu() {
        materialize_entry_test!(SltU, tracer::instruction::sltu::SLTU);
    }
}
