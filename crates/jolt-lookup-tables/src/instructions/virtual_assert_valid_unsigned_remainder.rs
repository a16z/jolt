use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::AssertValidUnsignedRemainder;

impl_lookup_table!(AssertValidUnsignedRemainder, Some(ValidUnsignedRemainder));
