use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::AssertHalfwordAlignment;

impl_lookup_table!(AssertHalfwordAlignment, Some(HalfwordAlignment));
