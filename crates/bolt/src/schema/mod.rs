use crate::ir::{BoltModule, Compute, Concrete, Cpu, Party, Protocol};
use crate::pass::verify_concrete_transcript;

mod constraints;
mod error;
mod kernels;
mod ops;
mod phase;
mod verify;

pub(crate) use constraints::{
    find_symbol, int_attr, missing_module_op, missing_symbol, operation_name, require_attrs,
    require_symbol_attr_eq, symbol_array_attr, symbol_attr,
};
pub use error::SchemaError;

use phase::ModulePhase;
use verify::verify_schema;

pub fn verify_protocol_schema(module: &BoltModule<'_, Protocol>) -> Result<(), SchemaError> {
    verify_schema(module, ModulePhase::Protocol)
}

pub fn verify_concrete_schema(module: &BoltModule<'_, Concrete>) -> Result<(), SchemaError> {
    verify_schema(module, ModulePhase::Concrete)?;
    verify_concrete_transcript(module)?;
    Ok(())
}

pub fn verify_party_schema(module: &BoltModule<'_, Party>) -> Result<(), SchemaError> {
    verify_schema(module, ModulePhase::Party)?;
    verify_concrete_transcript(module)?;
    Ok(())
}

pub fn verify_compute_schema(module: &BoltModule<'_, Compute>) -> Result<(), SchemaError> {
    verify_schema(module, ModulePhase::Compute)
}

pub fn verify_cpu_schema(module: &BoltModule<'_, Cpu>) -> Result<(), SchemaError> {
    verify_schema(module, ModulePhase::Cpu)
}
