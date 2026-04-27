use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Add;

impl_lookup_table!(Add, Some(RangeCheck));
