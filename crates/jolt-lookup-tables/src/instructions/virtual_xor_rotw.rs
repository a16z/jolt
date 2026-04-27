use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::{
    VirtualXorRotW12, VirtualXorRotW16, VirtualXorRotW7, VirtualXorRotW8,
};

impl_lookup_table!(VirtualXorRotW16, Some(VirtualXORROTW16));
impl_lookup_table!(VirtualXorRotW12, Some(VirtualXORROTW12));
impl_lookup_table!(VirtualXorRotW8, Some(VirtualXORROTW8));
impl_lookup_table!(VirtualXorRotW7, Some(VirtualXORROTW7));
