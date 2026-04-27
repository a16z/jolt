use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::Andn;

impl_lookup_table!(Andn, Some(Andn));
