use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Slt;

impl_lookup_table!(Slt, Some(SignedLessThan));
