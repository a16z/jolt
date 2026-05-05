mod copy;
mod readers;

pub(in crate::pass) use copy::copy_attrs;
pub(crate) use readers::string_attr;
pub(in crate::pass) use readers::{bool_attr, symbol_attr};
