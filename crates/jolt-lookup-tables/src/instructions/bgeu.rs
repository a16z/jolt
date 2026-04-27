use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::BgeU;

impl_lookup_table!(BgeU, Some(UnsignedGreaterThanEqual));
