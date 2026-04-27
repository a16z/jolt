use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::AssertValidDiv0;

impl_lookup_table!(AssertValidDiv0, Some(ValidDiv0));
