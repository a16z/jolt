use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::VirtualZeroExtendWord;

impl_lookup_table!(VirtualZeroExtendWord, Some(RangeCheck));
