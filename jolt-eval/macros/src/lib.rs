extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Ident};

/// Attribute macro for invariant structs.
///
/// Generates a `#[cfg(test)]` module with two tests:
/// - `seed_corpus`: runs all seed corpus inputs
/// - `random_inputs`: runs randomly-generated inputs via `Arbitrary`
///
/// The number of random iterations defaults to 10 and can be overridden
/// with the `JOLT_RANDOM_ITERS` environment variable.
///
/// The struct must implement `Invariant + Default`.
///
/// # Usage
///
/// ```ignore
/// #[jolt_eval_macros::invariant]
/// #[derive(Default)]
/// pub struct MySoundnessInvariant { ... }
/// ```
#[proc_macro_attribute]
pub fn invariant(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let struct_name = &input.ident;
    let snake_name = to_snake_case(&struct_name.to_string());
    let test_mod_name = Ident::new(&format!("{snake_name}_synthesized"), struct_name.span());

    let expanded = quote! {
        #input

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
                let num_iters: usize = std::env::var("JOLT_RANDOM_ITERS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10);
                let invariant = #struct_name::default();
                let setup = invariant.setup();
                let mut rng = jolt_eval::rand::thread_rng();
                for _ in 0..num_iters {
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
