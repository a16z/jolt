use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::MovSign;

impl_lookup_table!(MovSign, Some(SignMask));
