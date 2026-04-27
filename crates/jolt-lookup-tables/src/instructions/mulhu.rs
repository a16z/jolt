use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::MulHU;

impl_lookup_table!(MulHU, Some(UpperWord));
