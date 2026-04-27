use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::SltI;

impl_lookup_table!(SltI, Some(SignedLessThan));
