use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Auipc;

impl_lookup_table!(Auipc, Some(RangeCheck));
