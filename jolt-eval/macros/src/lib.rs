extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Ident};

/// Attribute macro for invariant structs.
///
/// Generates test and fuzz harness functions based on the specified targets.
///
/// # Usage
///
/// ```ignore
/// #[jolt_eval_macros::invariant(targets = [Test, Fuzz, RedTeam])]
/// #[derive(Default)]
/// pub struct MySoundnessInvariant { ... }
/// ```
///
/// Generates:
/// - For `Test`: A `#[cfg(test)]` module with seed corpus and random tests
/// - For `Fuzz`: A `fuzz_check` function suitable for `libfuzzer_sys`
/// - For `RedTeam`: A `redteam_description` function returning the invariant's description
///
/// The struct must implement `Invariant + Default`.
#[proc_macro_attribute]
pub fn invariant(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let struct_name = &input.ident;
    let snake_name = to_snake_case(&struct_name.to_string());
    let test_mod_name = Ident::new(&format!("{snake_name}_synthesized"), struct_name.span());

    let targets = parse_targets(attr);
    let has_test = targets.contains(&"Test".to_string());
    let has_fuzz = targets.contains(&"Fuzz".to_string());
    let has_redteam = targets.contains(&"RedTeam".to_string());

    let test_block = if has_test {
        quote! {
            #[cfg(test)]
            mod #test_mod_name {
                use super::*;
                use jolt_eval::Invariant;

                #[test]
                fn seed_corpus() {
                    let invariant = #struct_name::default();
                    let setup = invariant.setup();
                    for (i, input) in invariant.seed_corpus().into_iter().enumerate() {
                        invariant.check(&setup, input).unwrap_or_else(|e| {
                            panic!(
                                "Invariant '{}' violated on seed {}: {}",
                                invariant.name(), i, e
                            );
                        });
                    }
                }

                #[test]
                fn random_inputs() {
                    use jolt_eval::rand::RngCore;
                    let invariant = #struct_name::default();
                    let setup = invariant.setup();
                    let mut rng = jolt_eval::rand::thread_rng();
                    for _ in 0..10 {
                        let mut raw = vec![0u8; 4096];
                        rng.fill_bytes(&mut raw);
                        let mut u = jolt_eval::arbitrary::Unstructured::new(&raw);
                        if let Ok(input) = <
                            <#struct_name as jolt_eval::Invariant>::Input
                            as jolt_eval::arbitrary::Arbitrary
                        >::arbitrary(&mut u) {
                            invariant.check(&setup, input).unwrap_or_else(|e| {
                                panic!(
                                    "Invariant '{}' violated: {}",
                                    invariant.name(), e
                                );
                            });
                        }
                    }
                }
            }
        }
    } else {
        quote! {}
    };

    let fuzz_fn_name = Ident::new(&format!("{snake_name}_fuzz_check"), struct_name.span());
    let fuzz_block = if has_fuzz {
        quote! {
            pub fn #fuzz_fn_name(data: &[u8]) {
                use jolt_eval::Invariant;
                use std::sync::LazyLock;

                static SETUP: LazyLock<(
                    #struct_name,
                    <#struct_name as jolt_eval::Invariant>::Setup,
                )> = LazyLock::new(|| {
                    let invariant = #struct_name::default();
                    let setup = invariant.setup();
                    (invariant, setup)
                });

                let mut u = jolt_eval::arbitrary::Unstructured::new(data);
                if let Ok(input) = <
                    <#struct_name as jolt_eval::Invariant>::Input
                    as jolt_eval::arbitrary::Arbitrary
                >::arbitrary(&mut u) {
                    let (invariant, setup) = &*SETUP;
                    if let Err(e) = invariant.check(setup, input) {
                        panic!("Invariant '{}' violated: {}", invariant.name(), e);
                    }
                }
            }
        }
    } else {
        quote! {}
    };

    let redteam_fn_name = Ident::new(
        &format!("{snake_name}_redteam_description"),
        struct_name.span(),
    );
    let redteam_block = if has_redteam {
        quote! {
            pub fn #redteam_fn_name() -> String {
                use jolt_eval::Invariant;
                let invariant = #struct_name::default();
                invariant.description()
            }
        }
    } else {
        quote! {}
    };

    let expanded = quote! {
        #input

        #test_block
        #fuzz_block
        #redteam_block
    };

    expanded.into()
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(c);
        }
    }
    result
}

fn parse_targets(attr: TokenStream) -> Vec<String> {
    let attr_str = attr.to_string();
    // Parse: targets = [Test, Fuzz, RedTeam]
    if let Some(bracket_start) = attr_str.find('[') {
        if let Some(bracket_end) = attr_str.find(']') {
            let inner = &attr_str[bracket_start + 1..bracket_end];
            return inner
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    vec![]
}
