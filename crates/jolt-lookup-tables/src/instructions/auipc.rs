use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Auipc;
use tracer::instruction::{auipc::AUIPC, RISCVCycle};

impl_lookup_table!(Auipc, Some(RangeCheck));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<AUIPC> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (pc, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, (pc as i128 + imm) as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.instruction.address & mask,
            self.instruction.operands.imm as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (pc, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        pc.wrapping_add(imm as u64) & mask
    }
}
