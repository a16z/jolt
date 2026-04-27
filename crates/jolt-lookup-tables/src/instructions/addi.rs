use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Addi;

impl_lookup_table!(Addi, Some(RangeCheck));
