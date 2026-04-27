use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::VirtualShiftRightBitmask;

impl_lookup_table!(VirtualShiftRightBitmask, Some(ShiftRightBitmask));
