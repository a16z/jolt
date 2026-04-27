use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Beq;

impl_lookup_table!(Beq, Some(Equal));
