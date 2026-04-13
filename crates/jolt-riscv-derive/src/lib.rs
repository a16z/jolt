use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/// Derives the `Flags` trait for an instruction struct.
///
/// Use `#[circuit(...)]` and `#[instruction(...)]` attributes to declare flags.
/// Both are optional — omitting either produces an empty flag set.
///
/// ```ignore
/// #[derive(Flags)]
/// #[circuit(AddOperands, WriteLookupOutputToRD)]
/// #[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
/// pub struct Add;
/// ```
#[proc_macro_derive(Flags, attributes(circuit, instruction))]
pub fn derive_flags(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_flags_impl(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn derive_flags_impl(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;

    let circuit_flags = extract_flags(&input.attrs, "circuit")?;
    let instruction_flags = extract_flags(&input.attrs, "instruction")?;

    let circuit_body = if circuit_flags.is_empty() {
        quote! { crate::CircuitFlagSet::default() }
    } else {
        let sets = circuit_flags
            .iter()
            .map(|f| quote! { .set(crate::CircuitFlags::#f) });
        quote! { crate::CircuitFlagSet::default()#(#sets)* }
    };

    let instruction_body = if instruction_flags.is_empty() {
        quote! { crate::InstructionFlagSet::default() }
    } else {
        let sets = instruction_flags
            .iter()
            .map(|f| quote! { .set(crate::InstructionFlags::#f) });
        quote! { crate::InstructionFlagSet::default()#(#sets)* }
    };

    Ok(quote! {
        impl crate::Flags for #name {
            #[inline]
            fn circuit_flags(&self) -> crate::CircuitFlagSet {
                #circuit_body
            }

            #[inline]
            fn instruction_flags(&self) -> crate::InstructionFlagSet {
                #instruction_body
            }
        }
    })
}

fn extract_flags(attrs: &[syn::Attribute], name: &str) -> syn::Result<Vec<syn::Ident>> {
    let mut flags = Vec::new();
    for attr in attrs {
        if attr.path().is_ident(name) {
            attr.parse_nested_meta(|meta| {
                if let Some(ident) = meta.path.get_ident() {
                    flags.push(ident.clone());
                }
                Ok(())
            })?;
        }
    }
    Ok(flags)
}
