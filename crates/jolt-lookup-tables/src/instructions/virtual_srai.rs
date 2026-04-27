use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::VirtualSrai;

impl_lookup_table!(VirtualSrai, Some(VirtualSRA));
