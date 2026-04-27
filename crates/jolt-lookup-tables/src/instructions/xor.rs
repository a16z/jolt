use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Xor;

impl_lookup_table!(Xor, Some(Xor));
