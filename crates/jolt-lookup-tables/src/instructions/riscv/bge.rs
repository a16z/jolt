use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Bge;
use jolt_trace::JoltCycle;

impl_lookup_table!(Bge, Some(SignedGreaterThanEqual));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Bge<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let rs1 = self.0.rs1_val().unwrap_or(0) & mask;
        let rs2 = self.0.rs2_val().unwrap_or(0) & mask;
        let shift = 64 - XLEN as u32;
        let x = ((rs1 as i64) << shift >> shift) as u64;
        let y = ((rs2 as i64) << shift >> shift) as i128;
        (x, y)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        ((x as i64) >= (y as i64)).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize_entry_test;

    #[test]
    fn materialize_entry_bge() {
        materialize_entry_test!(Bge, tracer::instruction::bge::BGE);
    }
}
