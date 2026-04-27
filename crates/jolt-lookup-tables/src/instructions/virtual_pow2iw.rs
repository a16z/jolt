use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Pow2IW;

impl_lookup_table!(Pow2IW, Some(Pow2W));
