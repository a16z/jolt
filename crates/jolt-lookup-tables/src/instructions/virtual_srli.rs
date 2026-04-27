use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::VirtualSrli;

impl_lookup_table!(VirtualSrli, Some(VirtualSRL));
