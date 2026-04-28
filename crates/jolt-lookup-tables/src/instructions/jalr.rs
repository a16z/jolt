use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Jalr;
use tracer::instruction::{jalr::JALR, RISCVCycle};

impl_lookup_table!(Jalr, Some(RangeCheckAligned));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<JALR> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, (x as i128 + y) as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.register_state.rs1 & mask,
            (self.instruction.operands.imm & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        x.wrapping_add(y as u64) & mask & !1
    }
}
