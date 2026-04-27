use crate::instruction_tables::impl_lookup_table;
use jolt_trace::instructions::{
    VirtualXorRot16, VirtualXorRot24, VirtualXorRot32, VirtualXorRot63,
};

impl_lookup_table!(VirtualXorRot32, Some(VirtualXORROT32));
impl_lookup_table!(VirtualXorRot24, Some(VirtualXORROT24));
impl_lookup_table!(VirtualXorRot16, Some(VirtualXORROT16));
impl_lookup_table!(VirtualXorRot63, Some(VirtualXORROT63));
