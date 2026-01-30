#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use syn::{Lit, Meta, MetaNameValue, NestedMeta};

#[cfg(feature = "std")]
use crate::constants::{
    DEFAULT_HEAP_SIZE, DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE, DEFAULT_MAX_TRACE_LENGTH,
    DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE, DEFAULT_STACK_SIZE,
};

pub struct Attributes {
    pub wasm: bool,
    pub nightly: bool,
    pub guest_only: bool,
    pub heap_size: u64,
    pub stack_size: u64,
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub max_trusted_advice_size: u64,
    pub max_untrusted_advice_size: u64,
    pub max_trace_length: u64,
}

#[cfg(feature = "std")]
pub fn parse_attributes(attr: &Vec<NestedMeta>) -> Attributes {
    let mut attributes = HashMap::<_, u64>::new();
    let mut wasm = false;
    let mut guest_only = false;
    let mut nightly = false;

    for attr in attr {
        match attr {
            NestedMeta::Meta(Meta::NameValue(MetaNameValue { path, lit, .. })) => {
                let value: u64 = match lit {
                    Lit::Int(lit) => lit.base10_parse().unwrap(),
                    _ => panic!("expected integer literal"),
                };
                let ident = &path.get_ident().expect("Expected identifier");
                match ident.to_string().as_str() {
                    "heap_size" => attributes.insert("heap_size", value),
                    "stack_size" => attributes.insert("stack_size", value),
                    "max_input_size" => attributes.insert("max_input_size", value),
                    "max_output_size" => attributes.insert("max_output_size", value),
                    "max_trusted_advice_size" => {
                        attributes.insert("max_trusted_advice_size", value)
                    }
                    "max_untrusted_advice_size" => {
                        attributes.insert("max_untrusted_advice_size", value)
                    }
                    "max_trace_length" => attributes.insert("max_trace_length", value),
                    _ => panic!("invalid attribute"),
                };
            }
            NestedMeta::Meta(Meta::Path(path)) if path.is_ident("wasm") => {
                wasm = true;
            }
            NestedMeta::Meta(Meta::Path(path)) if path.is_ident("guest_only") => {
                guest_only = true;
            }
            NestedMeta::Meta(Meta::Path(path)) if path.is_ident("nightly") => {
                nightly = true;
            }
            _ => panic!("expected integer literal"),
        }
    }

    let heap_size = *attributes
        .get("heap_size")
        .unwrap_or(&DEFAULT_HEAP_SIZE);
    let stack_size = *attributes.get("stack_size").unwrap_or(&DEFAULT_STACK_SIZE);
    let max_input_size = *attributes
        .get("max_input_size")
        .unwrap_or(&DEFAULT_MAX_INPUT_SIZE);
    let max_output_size = *attributes
        .get("max_output_size")
        .unwrap_or(&DEFAULT_MAX_OUTPUT_SIZE);
    let max_trusted_advice_size = *attributes
        .get("max_trusted_advice_size")
        .unwrap_or(&DEFAULT_MAX_TRUSTED_ADVICE_SIZE);
    let max_untrusted_advice_size = *attributes
        .get("max_untrusted_advice_size")
        .unwrap_or(&DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE);
    let max_trace_length = *attributes
        .get("max_trace_length")
        .unwrap_or(&DEFAULT_MAX_TRACE_LENGTH);

    Attributes {
        wasm,
        nightly,
        guest_only,
        heap_size,
        stack_size,
        max_input_size,
        max_output_size,
        max_trusted_advice_size,
        max_untrusted_advice_size,
        max_trace_length,
    }
}
