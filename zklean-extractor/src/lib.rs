extern crate proc_macro;
use proc_macro::TokenStream;
use syn::{parse::{Parse, ParseStream}, parse_macro_input, punctuated::Punctuated, Token};
use proc_macro2::{Ident, Literal};
use quote::{format_ident, quote};

/// Parser for a proc-macro enum declaration. Parses an enum declaration, consisting of an
/// identifier (for the resulting enum type), followed by a comma, and then a comma-separated
/// sequence of types, each of which will become a variant, named after the type, and containing an
/// item of that type.
#[derive(Clone)]
struct EnumParser {
    id: Ident,
    entries: Punctuated<VariantParser, Token![,]>,
}

impl Parse for EnumParser {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let id = Ident::parse(input)?;
        let _ = <Token![,]>::parse(input)?;
        let entries: Punctuated<VariantParser, Token![,]> = Punctuated::parse_terminated(input)?;
        Ok(Self { id, entries })
    }
}

/// Parser for a variant within a proc-macro enum declaration. Contains a type, possibly with a
/// module path and a collection of const generics in angle brackets.
#[derive(Clone)]
struct VariantParser {
    path: Punctuated<Ident, Token![::]>,
    id: Ident,
    const_generics: Punctuated<Literal, Token![,]>,
}

impl VariantParser {
    fn to_ident(&self) -> Ident {
        let mut res = self.id.clone();
        for c in &self.const_generics {
            res = format_ident!("{res}_{}", c.to_string());
        }
        res
    }
}

impl Parse for VariantParser {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut path = Punctuated::<Ident, Token![::]>::new();
        let mut id = Ident::parse(input)?;
        loop {
            if input.peek(Token![::]) {
                path.push_value(id.clone());
                path.push_punct(<Token![::]>::parse(input)?);
                id = Ident::parse(input)?;
            } else {
                break;
            }
        }
        let const_generics = if input.peek(Token![<]) {
            let _ = <Token![<]>::parse(input)?;
            let gs = Punctuated::<Literal, Token![,]>::parse_separated_nonempty(input)?;
            let _ = <Token![>]>::parse(input)?;
            gs
        } else {
            Punctuated::<Literal, Token![,]>::new()
        };
        Ok(Self { path, id, const_generics })
    }
}

/// Declare an enum of subtable types.
#[proc_macro]
pub fn declare_subtables_enum(input: TokenStream) -> TokenStream {
    let EnumParser { id: enum_id, entries } = parse_macro_input!(input as EnumParser);
    let mut variants = vec![];
    let mut name_cases = vec![];
    let mut eval_cases = vec![];
    let mut iter_cases = vec![];
    let mut conv_conditions = vec![];
    let mut tests = vec![];

    for entry in entries {
        let path = entry.path.clone();
        let id = entry.id.clone();
        let name = entry.to_ident();
        let name_str = name.to_string();
        let const_generics = entry.const_generics;

        variants.push(quote! {
            #[allow(non_camel_case_types)]
            #name(#path #id<F, #const_generics>)
        });
        name_cases.push(quote! {
            Self::#name(_) => #name_str
        });
        eval_cases.push(quote! {
            Self::#name(s) => s.evaluate_mle(&vars)
        });
        iter_cases.push(quote! {
            Self::#name(#path #id::new())
        });
        conv_conditions.push(quote! {
            if t == #path #id::<F, #const_generics>::new().subtable_id() {
                return Self::#name(#path #id::<F, #const_generics>::new());
            }
        });
        tests.push(quote! {
            /// Test that extracting the subtable as an `crate::mle_ast::MleAst` and evaluating it
            /// results in the same value as simply evaluating it would.
            #[test]
            #[allow(non_snake_case)]
            fn #name(values_u64 in proptest::collection::vec(proptest::num::u64::ANY, 8)) {
                type RefField = jolt_core::field::binius::BiniusField<binius_field::BinaryField128b>;
                type AstField = crate::mle_ast::MleAst<2048>;
                let (actual, expected, mle) = crate::util::test_evaluate_fn(
                    &values_u64,
                    #path #id::<RefField, #const_generics>::new(),
                    #path #id::<AstField, #const_generics>::new(),
                );
                prop_assert_eq!(actual, expected, "\n   mle: {}:", mle);
            }
        });
    }

    quote! {
        pub enum #enum_id<F: crate::util::ZkLeanReprField, const REG_SIZE: usize> {
            #(#variants),*
        }

        impl<F: crate::util::ZkLeanReprField, const REG_SIZE: usize> #enum_id<F, REG_SIZE> {
            /// Name of this subtable variant, incorporating the type and any const-generics.
            pub fn name(&self) -> &'static str {
                match self {
                    #(#name_cases),*
                }
            }

            /// Call the `evaluate_mle` method on the contained subtable.
            pub fn evaluate_mle(&self, reg_name: char) -> F {
                use jolt_core::jolt::subtable::LassoSubtable;
                use crate::util::ZkLeanReprField;
                let vars = F::register(reg_name, REG_SIZE);
                match self {
                    #(#eval_cases),*
                }
            }

            /// Enumerate all variants.
            pub fn variants() -> Vec<Self> {
                vec![
                    #(#iter_cases),*
                ]
            }

            /// Construct a new object of the appropriate type given a subtable's
            /// [`std::any::TypeId`].
            pub fn from_subtable_id(t: std::any::TypeId) -> Self {
                use jolt_core::jolt::subtable::LassoSubtable;
                #(#conv_conditions)*
                panic!("Unimplemented conversion from {t:?}")
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use proptest::prelude::*;
            proptest! {
                #(#tests)*
            }
        }
    }.into()
}

/// Declare an enum of instruction types.
#[proc_macro]
pub fn declare_instructions_enum(input: TokenStream) -> TokenStream {
    let EnumParser { id: enum_id, entries } = parse_macro_input!(input as EnumParser);
    let mut variants = vec![];
    let mut name_cases = vec![];
    let mut combine_cases = vec![];
    let mut subtables_cases = vec![];
    let mut iter_cases = vec![];
    let mut to_instruction_set_cases = vec![];
    //let mut tests = vec![];

    for entry in entries {
        let path = entry.path.clone();
        let id = entry.id.clone();
        let name = entry.to_ident();
        let name_str = name.to_string();
        let const_generics = entry.const_generics;

        variants.push(quote! {
            #[allow(non_camel_case_types)]
            #name(#path #id<WORD_SIZE, #const_generics>)
        });
        name_cases.push(quote! {
            Self::#name(_) => #name_str
        });
        combine_cases.push(quote! {
            Self::#name(i) => i.combine_lookups(&vars, C, 1 << LOG_M)
        });
        subtables_cases.push(quote! {
            Self::#name(i) => i.subtables(C, 1 << LOG_M)
        });
        iter_cases.push(quote! {
            Self::#name(#path #id::default())
        });
        to_instruction_set_cases.push(quote! {
            Self::#name(i) => jolt_core::jolt::vm::rv32i_vm::RV32I::from(i.clone())
        });
        // TODO: tests
        //tests.push(quote! {
        //    #[test]
        //    #[allow(non_snake_case)]
        //    fn #name(values_u64 in proptest::collection::vec(proptest::num::u64::ANY, 8)) {
        //        type RefField = jolt_core::field::binius::BiniusField<binius_field::BinaryField128b>;
        //        type AstField = crate::mle_ast::MleAst<2048>;
        //        let (actual, expected, mle) = crate::util::test_evaluate_fn(
        //            &values_u64,
        //            #path #id::<RefField, #const_generics>::new(),
        //            #path #id::<AstField, #const_generics>::new(),
        //        );
        //        prop_assert_eq!(actual, expected, "\n   mle: {}:", mle);
        //    }
        //});
    }

    quote! {
        pub enum #enum_id<const WORD_SIZE: usize, const C: usize, const LOG_M: usize> {
            #(#variants),*
        }

        impl<const WORD_SIZE: usize, const C: usize, const LOG_M: usize> #enum_id<WORD_SIZE, C, LOG_M> {
            /// Name of this instruction variant, incorporating the type and any const generics.
            pub fn name(&self) -> &'static str {
                match self {
                    #(#name_cases),*
                }
            }

            /// Call the `combine_lookups` method on the underlying instruction.
            pub fn combine_lookups<F: ZkLeanReprField>(&self, reg_name: char) -> F {
                use jolt_core::jolt::instruction::JoltInstruction;

                // Count total subtable evaluations required
                let reg_size = self.subtables::<F>().len();
                let vars = F::register(reg_name, reg_size);

                match self {
                    #(#combine_cases),*
                }
            }

            /// Call the `subtables` method on the underlying instruction.
            pub fn subtables<F: ZkLeanReprField>(&self) -> Vec<(String, usize)> {
                use jolt_core::jolt::{instruction::{JoltInstruction, SubtableIndices}, subtable::LassoSubtable};
                use crate::subtable::NamedSubtable;

                let subtables: Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> = match self {
                    #(#subtables_cases),*
                };

                let mut res = vec![];
                for (subtable, indices) in subtables {
                    // FIXME: Referencing `NamedSubtable` here directly isn't great.
                    let subtable = NamedSubtable::<F, LOG_M>::from_subtable_id(subtable.subtable_id());
                    let subtable_name = subtable.name().to_string();
                    for i in indices.iter() {
                        res.push((subtable_name.clone(), i));
                    }
                }
                res
            }

            /// Enumerate all variants.
            pub fn variants() -> Vec<Self> {
                vec![
                    #(#iter_cases),*
                ]
            }
        }

        impl #enum_id<32, 4, 16> {
            pub fn to_instruction_set(&self) -> jolt_core::jolt::vm::rv32i_vm::RV32I {
                match self {
                    #(#to_instruction_set_cases),*
                }
            }
        }

        //#[cfg(test)]
        //mod tests {
        //    use super::*;
        //    use proptest::prelude::*;
        //    proptest! {
        //        #(#tests)*
        //    }
        //}
    }.into()
}
