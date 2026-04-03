use std::collections::HashMap;
use syn::{punctuated::Punctuated, token::Comma, Expr, ExprLit, Lit, Meta, MetaNameValue};

use crate::constants::{
    DEFAULT_HEAP_SIZE, DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE, DEFAULT_MAX_TRACE_LENGTH,
    DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE, DEFAULT_STACK_SIZE,
};

pub struct Attributes {
    pub wasm: bool,
    pub nightly: bool,
    pub guest_only: bool,
    /// Optional cargo profile name to use for guest builds (e.g. "guest", "release").
    pub profile: Option<String>,
    pub heap_size: u64,
    pub stack_size: u64,
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub max_trusted_advice_size: u64,
    pub max_untrusted_advice_size: u64,
    pub max_trace_length: u64,
    pub backtrace: Option<String>,
}

pub fn parse_attributes(attr: &Punctuated<Meta, Comma>) -> Attributes {
    let mut attributes = HashMap::<_, u64>::new();
    let mut wasm = false;
    let mut guest_only = false;
    let mut nightly = false;
    let mut profile: Option<String> = None;
    let mut backtrace: Option<String> = None;

    for meta in attr {
        match meta {
            Meta::NameValue(MetaNameValue { path, value, .. }) => {
                let ident = &path.get_ident().expect("Expected identifier");
                let lit = match value {
                    Expr::Lit(ExprLit { lit, .. }) => lit,
                    _ => panic!("expected literal expression"),
                };
                match ident.to_string().as_str() {
                    "backtrace" => {
                        let value = match lit {
                            Lit::Str(lit) => lit.value(),
                            _ => panic!("backtrace attribute expects a string literal"),
                        };
                        backtrace = Some(value);
                    }
                    "profile" => {
                        let value = match lit {
                            Lit::Str(lit) => lit.value(),
                            _ => panic!("profile attribute expects a string literal"),
                        };
                        profile = Some(value);
                    }
                    _ => {
                        let value: u64 = match lit {
                            Lit::Int(lit) => lit.base10_parse().unwrap(),
                            _ => panic!("expected integer literal"),
                        };
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
                }
            }
            Meta::Path(path) if path.is_ident("wasm") => {
                wasm = true;
            }
            Meta::Path(path) if path.is_ident("guest_only") => {
                guest_only = true;
            }
            Meta::Path(path) if path.is_ident("nightly") => {
                nightly = true;
            }
            Meta::Path(path) if path.is_ident("backtrace") => {
                backtrace = Some("auto".to_string());
            }
            _ => panic!("expected integer literal"),
        }
    }

    let heap_size = *attributes.get("heap_size").unwrap_or(&DEFAULT_HEAP_SIZE);
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

    validate_attributes(
        max_trace_length,
        stack_size,
        heap_size,
        max_input_size,
        max_output_size,
        max_trusted_advice_size,
        max_untrusted_advice_size,
    );

    Attributes {
        wasm,
        nightly,
        guest_only,
        profile,
        heap_size,
        stack_size,
        max_input_size,
        max_output_size,
        max_trusted_advice_size,
        max_untrusted_advice_size,
        max_trace_length,
        backtrace,
    }
}

fn validate_attributes(
    max_trace_length: u64,
    stack_size: u64,
    heap_size: u64,
    _max_input_size: u64,
    _max_output_size: u64,
    _max_trusted_advice_size: u64,
    _max_untrusted_advice_size: u64,
) {
    if max_trace_length == 0 {
        panic!("max_trace_length must be greater than 0");
    }

    if stack_size == 0 {
        panic!("stack_size must be greater than 0");
    }

    if heap_size == 0 {
        panic!("heap_size must be greater than 0");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_defaults() {
        validate_attributes(
            DEFAULT_MAX_TRACE_LENGTH,
            DEFAULT_STACK_SIZE,
            DEFAULT_HEAP_SIZE,
            DEFAULT_MAX_INPUT_SIZE,
            DEFAULT_MAX_OUTPUT_SIZE,
            DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
        );
    }

    #[test]
    fn accepts_non_power_of_two_trace_length() {
        validate_attributes(50_000_000, 4096, 4096, 4096, 4096, 4096, 4096);
    }

    #[test]
    #[should_panic(expected = "max_trace_length must be greater than 0")]
    fn rejects_zero_trace_length() {
        validate_attributes(0, 4096, 4096, 4096, 4096, 4096, 4096);
    }

    #[test]
    #[should_panic(expected = "stack_size must be greater than 0")]
    fn rejects_zero_stack_size() {
        validate_attributes(1 << 20, 0, 4096, 4096, 4096, 4096, 4096);
    }

    #[test]
    #[should_panic(expected = "heap_size must be greater than 0")]
    fn rejects_zero_heap_size() {
        validate_attributes(1 << 20, 4096, 0, 4096, 4096, 4096, 4096);
    }

    #[test]
    fn accepts_zero_io_and_advice_sizes() {
        validate_attributes(1 << 20, 8, 8, 0, 0, 0, 0);
    }

    #[test]
    fn accepts_unaligned_sizes() {
        validate_attributes(1 << 20, 100_000, 100_000, 4096, 4096, 4096, 4096);
    }
}
