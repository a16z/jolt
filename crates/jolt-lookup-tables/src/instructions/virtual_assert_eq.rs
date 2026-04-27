use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::AssertEq;

impl_lookup_table!(AssertEq, Some(Equal));
