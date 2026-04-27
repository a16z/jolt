use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Blt;

impl_lookup_table!(Blt, Some(SignedLessThan));
