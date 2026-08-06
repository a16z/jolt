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

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test-only parsing helpers")]
mod tests {
    use super::*;
    use syn::parse::Parser;
    use syn::Token;

    fn parse(input: &str) -> Attributes {
        let punctuated = Punctuated::<Meta, Token![,]>::parse_terminated
            .parse_str(input)
            .expect("attribute list should parse");
        parse_attributes(&punctuated)
    }

    #[test]
    fn empty_attribute_list_yields_documented_defaults() {
        let attributes = parse("");
        assert_eq!(attributes.heap_size, DEFAULT_HEAP_SIZE);
        assert_eq!(attributes.stack_size, DEFAULT_STACK_SIZE);
        assert_eq!(attributes.max_input_size, DEFAULT_MAX_INPUT_SIZE);
        assert_eq!(attributes.max_output_size, DEFAULT_MAX_OUTPUT_SIZE);
        assert_eq!(
            attributes.max_trusted_advice_size,
            DEFAULT_MAX_TRUSTED_ADVICE_SIZE
        );
        assert_eq!(
            attributes.max_untrusted_advice_size,
            DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE
        );
        assert_eq!(attributes.max_trace_length, DEFAULT_MAX_TRACE_LENGTH);
        assert!(!attributes.wasm);
        assert!(!attributes.nightly);
        assert!(!attributes.guest_only);
        assert!(attributes.profile.is_none());
        assert!(attributes.backtrace.is_none());
    }

    #[test]
    fn explicit_sizes_override_defaults_per_key() {
        let attributes = parse(
            "heap_size = 65536, stack_size = 8192, max_input_size = 1024, \
             max_output_size = 2048, max_trusted_advice_size = 16, \
             max_untrusted_advice_size = 32, max_trace_length = 4096",
        );
        assert_eq!(attributes.heap_size, 65536);
        assert_eq!(attributes.stack_size, 8192);
        assert_eq!(attributes.max_input_size, 1024);
        assert_eq!(attributes.max_output_size, 2048);
        assert_eq!(attributes.max_trusted_advice_size, 16);
        assert_eq!(attributes.max_untrusted_advice_size, 32);
        assert_eq!(attributes.max_trace_length, 4096);
    }

    #[test]
    fn path_flags_set_booleans_and_bare_backtrace_means_auto() {
        let attributes = parse("wasm, guest_only, nightly, backtrace");
        assert!(attributes.wasm);
        assert!(attributes.guest_only);
        assert!(attributes.nightly);
        assert_eq!(attributes.backtrace.as_deref(), Some("auto"));
        // Flags do not disturb the size defaults.
        assert_eq!(attributes.heap_size, DEFAULT_HEAP_SIZE);
    }

    #[test]
    fn string_valued_keys_capture_their_literals() {
        let attributes = parse(r#"profile = "guest", backtrace = "full""#);
        assert_eq!(attributes.profile.as_deref(), Some("guest"));
        assert_eq!(attributes.backtrace.as_deref(), Some("full"));
    }

    #[test]
    #[should_panic(expected = "invalid attribute")]
    fn unknown_integer_key_is_rejected() {
        let _ = parse("bogus_key = 3");
    }

    #[test]
    #[should_panic(expected = "expected integer literal")]
    fn string_literal_for_integer_key_is_rejected() {
        let _ = parse(r#"heap_size = "big""#);
    }

    #[test]
    #[should_panic(expected = "expected literal expression")]
    fn non_literal_expression_is_rejected() {
        let _ = parse("heap_size = 1 + 2");
    }

    #[test]
    #[should_panic(expected = "expected integer literal")]
    fn unknown_bare_path_is_rejected() {
        let _ = parse("unknown_flag");
    }

    #[test]
    #[should_panic(expected = "profile attribute expects a string literal")]
    fn integer_literal_for_profile_is_rejected() {
        let _ = parse("profile = 3");
    }

    #[test]
    #[should_panic(expected = "backtrace attribute expects a string literal")]
    fn integer_literal_for_backtrace_is_rejected() {
        let _ = parse("backtrace = 3");
    }
}
