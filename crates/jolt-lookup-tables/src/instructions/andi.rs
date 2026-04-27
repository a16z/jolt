use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::AndI;

impl_lookup_table!(AndI, Some(And));
