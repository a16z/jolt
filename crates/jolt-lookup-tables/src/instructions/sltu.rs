use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::SltU;

impl_lookup_table!(SltU, Some(UnsignedLessThan));
