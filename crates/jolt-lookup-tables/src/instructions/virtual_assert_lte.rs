use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::AssertLte;

impl_lookup_table!(AssertLte, Some(UnsignedLessThanEqual));
