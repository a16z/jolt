#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use syn::{Lit, Meta, MetaNameValue, NestedMeta};

#[cfg(feature = "std")]
use crate::constants::{
    DEFAULT_MAX_BYTECODE_SIZE, DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE,
    DEFAULT_MAX_TRACE_LENGTH, DEFAULT_MEMORY_SIZE, DEFAULT_STACK_SIZE,
};

pub struct Attributes {
    pub wasm: bool,
    pub memory_size: u64,
    pub stack_size: u64,
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub max_bytecode_size: u64,
    pub max_trace_length: u64,
}

#[cfg(feature = "std")]
pub fn parse_attributes(attr: &Vec<NestedMeta>) -> Attributes {
    let mut attributes = HashMap::<_, u64>::new();
    let mut wasm = false;

    for attr in attr {
        match attr {
            NestedMeta::Meta(Meta::NameValue(MetaNameValue { path, lit, .. })) => {
                let value: u64 = match lit {
                    Lit::Int(lit) => lit.base10_parse().unwrap(),
                    _ => panic!("expected integer literal"),
                };
                let ident = &path.get_ident().expect("Expected identifier");
                match ident.to_string().as_str() {
                    "memory_size" => attributes.insert("memory_size", value),
                    "stack_size" => attributes.insert("stack_size", value),
                    "max_input_size" => attributes.insert("max_input_size", value),
                    "max_output_size" => attributes.insert("max_output_size", value),
                    "max_bytecode_size" => attributes.insert("max_bytecode_size", value),
                    "max_trace_length" => attributes.insert("max_trace_length", value),
                    _ => panic!("invalid attribute"),
                };
            }
            NestedMeta::Meta(Meta::Path(path)) if path.is_ident("wasm") => {
                wasm = true;
            }
            _ => panic!("expected integer literal"),
        }
    }

    let memory_size = *attributes
        .get("memory_size")
        .unwrap_or(&DEFAULT_MEMORY_SIZE);
    let stack_size = *attributes.get("stack_size").unwrap_or(&DEFAULT_STACK_SIZE);
    let max_input_size = *attributes
        .get("max_input_size")
        .unwrap_or(&DEFAULT_MAX_INPUT_SIZE);
    let max_output_size = *attributes
        .get("max_output_size")
        .unwrap_or(&DEFAULT_MAX_OUTPUT_SIZE);
    let max_bytecode_size = *attributes
        .get("max_bytecode_size")
        .unwrap_or(&DEFAULT_MAX_BYTECODE_SIZE);
    let max_trace_length = *attributes
        .get("max_trace_length")
        .unwrap_or(&DEFAULT_MAX_TRACE_LENGTH);

    Attributes {
        wasm,
        memory_size,
        stack_size,
        max_input_size,
        max_output_size,
        max_bytecode_size,
        max_trace_length,
    }
}
