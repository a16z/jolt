use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::VirtualAdviceLen;

impl_lookup_table!(VirtualAdviceLen, Some(RangeCheck));
