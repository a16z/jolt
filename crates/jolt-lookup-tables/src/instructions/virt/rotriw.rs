use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualRotriw;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(VirtualRotriw, Some(VirtualROTRW));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualRotriw<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.0.rs1_val().unwrap_or(0), self.0.instruction().imm())
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let half = XLEN / 2;
        let mask = (1u128 << half).wrapping_sub(1) as u64;
        // Cap the rotation amount at `half` (matches `.min(half)` in jolt-core).
        // Using `% half` would map shifts of [half, 2*half) onto [0, half),
        // which produces a different rotation than the table expects.
        let r = ((imm as u64).trailing_zeros() as usize).min(half);
        let v = (rs1 & mask) as u128;
        if r == 0 || r == half {
            rs1 & mask
        } else {
            (((v >> r) | (v << (half - r))) as u64) & mask
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::test::materialize_entry_test;
    use tracer::instruction::RISCVCycle;

    #[test]
    fn materialize_entry_virtualrotriw() {
        materialize_entry_test::<
            VirtualRotriw<RISCVCycle<tracer::instruction::virtual_rotriw::VirtualROTRIW>>,
            RISCVCycle<tracer::instruction::virtual_rotriw::VirtualROTRIW>,
        >();
    }
}
