use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Bne;

impl_lookup_table!(Bne, Some(NotEqual));
