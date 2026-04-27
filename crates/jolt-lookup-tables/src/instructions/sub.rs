use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Sub;

impl_lookup_table!(Sub, Some(RangeCheck));
