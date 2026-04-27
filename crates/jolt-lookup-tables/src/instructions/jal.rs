use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Jal;

impl_lookup_table!(Jal, Some(RangeCheck));
