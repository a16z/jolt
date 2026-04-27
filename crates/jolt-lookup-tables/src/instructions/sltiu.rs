use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::SltIU;

impl_lookup_table!(SltIU, Some(UnsignedLessThan));
