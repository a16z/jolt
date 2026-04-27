use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Mul;

impl_lookup_table!(Mul, Some(RangeCheck));
