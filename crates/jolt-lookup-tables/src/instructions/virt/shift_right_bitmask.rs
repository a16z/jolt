use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualShiftRightBitmask;
use tracer::instruction::{
    virtual_shift_right_bitmask::VirtualShiftRightBitmask as TracerVirtualShiftRightBitmask,
    RISCVCycle,
};

impl_lookup_table!(VirtualShiftRightBitmask, Some(ShiftRightBitmask));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<TracerVirtualShiftRightBitmask> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (self.register_state.rs1 & mask, 0)
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
        let shift = (y & ((XLEN as u128) - 1)) as usize;
        (((1u128 << (XLEN - shift)) - 1) << shift) as u64
    }
}
