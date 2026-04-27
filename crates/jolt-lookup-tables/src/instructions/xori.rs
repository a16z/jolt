use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::XorI;

impl_lookup_table!(XorI, Some(Xor));
