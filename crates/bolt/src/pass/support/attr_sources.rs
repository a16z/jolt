mod descriptor;
mod lowering;

pub(in crate::pass) use descriptor::LoweredAttr;
pub(in crate::pass) use lowering::lower_attr_sources;
