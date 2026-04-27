use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Lui;

impl_lookup_table!(Lui, Some(RangeCheck));
