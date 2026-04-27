use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::MulI;

impl_lookup_table!(MulI, Some(RangeCheck));
