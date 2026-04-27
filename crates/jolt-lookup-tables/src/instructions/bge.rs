use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Bge;

impl_lookup_table!(Bge, Some(SignedGreaterThanEqual));
