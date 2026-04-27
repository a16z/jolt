use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::VirtualAdvice;

impl_lookup_table!(VirtualAdvice, Some(RangeCheck));
