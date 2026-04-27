use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::VirtualSra;

impl_lookup_table!(VirtualSra, Some(VirtualSRA));
