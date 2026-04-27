use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::BltU;

impl_lookup_table!(BltU, Some(UnsignedLessThan));
