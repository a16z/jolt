use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::AssertWordAlignment;

impl_lookup_table!(AssertWordAlignment, Some(WordAlignment));
