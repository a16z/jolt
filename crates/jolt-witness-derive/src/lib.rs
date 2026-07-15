//! Derive macro for jolt-witness consumer bundles.
//!
//! `#[derive(WitnessBundle)]` turns a struct of atomic witnesses — a
//! consumer's data flow, stated as a type — into a [`WitnessBundle`]:
//!
//! - `from_row` composes the fields' `Extract`/`ExtractIndexed` impls, so the
//!   derivation of every field stays single-sourced in `jolt-witness`;
//! - `annotated_ids`/`annotated_columns` expose the `#[opening(..)]`-annotated
//!   fields' jolt-claims ids and field-element columns;
//! - a `#[cfg(test)]` consistency test per annotated field pins the bundle
//!   column against the backend's `oracle_table` for the same id, so the
//!   typed path and the id path cannot drift.
//!
//! ## `#[opening(..)]` field grammar (the `OutputClaims` style)
//!
//! - `#[opening(VirtualVariant)]` — a virtual-polynomial id; the field type's
//!   `Extract` impl derives the value.
//! - `#[opening(VirtualVariant(Payload::PATH))]` — a payload-carrying virtual
//!   variant, e.g. `#[opening(OpFlags(CircuitFlags::Jump))]`; the payload is
//!   also the `ExtractIndexed` index binding the family member.
//! - `#[opening(committed = CommittedVariant)]` — a committed-polynomial id.
//! - No annotation — a *fact* field: extracted like the others but tied to no
//!   protocol id (it is input data of the consumer, not an opening).

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Fields, Ident};

enum OpeningId {
    Virtual {
        variant: syn::Path,
        payload: Option<TokenStream2>,
    },
    Committed {
        variant: Ident,
    },
}

struct BundleField {
    name: Ident,
    ty: syn::Type,
    opening: Option<OpeningId>,
}

#[proc_macro_derive(WitnessBundle, attributes(opening))]
pub fn derive_witness_bundle(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "#[derive(WitnessBundle)] does not support generic bundles",
        ));
    }
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "#[derive(WitnessBundle)] requires a struct",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            input,
            "#[derive(WitnessBundle)] requires named fields",
        ));
    };

    let fields = fields
        .named
        .iter()
        .map(parse_field)
        .collect::<syn::Result<Vec<_>>>()?;

    let name = &input.ident;
    let from_row_fields = fields.iter().map(from_row_field);
    let annotated: Vec<&BundleField> = fields
        .iter()
        .filter(|field| field.opening.is_some())
        .collect();
    let id_exprs: Vec<TokenStream2> = annotated
        .iter()
        .filter_map(|field| field.opening.as_ref())
        .map(id_expr)
        .collect();
    let column_entries = annotated.iter().zip(&id_exprs).map(|(field, id)| {
        let field_name = &field.name;
        quote! {
            (
                #id,
                rows.iter()
                    .map(|bundle| ::jolt_witness::witnesses::ToField::to_field(bundle.#field_name))
                    .collect(),
            )
        }
    });

    let consistency_tests = consistency_tests(name, &annotated);

    Ok(quote! {
        impl ::jolt_witness::WitnessBundle for #name {
            fn from_row(
                row: &::jolt_witness::__private::TraceRow,
                next: ::core::option::Option<&::jolt_witness::__private::TraceRow>,
                env: &::jolt_witness::witnesses::WitnessEnv<'_>,
            ) -> ::core::result::Result<Self, ::jolt_witness::WitnessError> {
                ::core::result::Result::Ok(Self {
                    #(#from_row_fields,)*
                })
            }

            fn annotated_ids() -> ::std::vec::Vec<::jolt_witness::__private::JoltPolynomialId> {
                ::std::vec![#(#id_exprs),*]
            }

            fn annotated_columns<FieldElement: ::jolt_witness::__private::Field>(
                rows: &[Self],
            ) -> ::std::vec::Vec<(
                ::jolt_witness::__private::JoltPolynomialId,
                ::std::vec::Vec<FieldElement>,
            )> {
                ::std::vec![#(#column_entries),*]
            }
        }

        #consistency_tests
    })
}

fn parse_field(field: &syn::Field) -> syn::Result<BundleField> {
    let name = field
        .ident
        .clone()
        .ok_or_else(|| syn::Error::new_spanned(field, "bundle fields must be named"))?;
    let mut opening = None;
    for attr in &field.attrs {
        if attr.path().is_ident("opening") {
            if opening.is_some() {
                return Err(syn::Error::new_spanned(
                    attr,
                    "at most one #[opening(..)] per field",
                ));
            }
            opening = Some(parse_opening(attr)?);
        }
    }
    Ok(BundleField {
        name,
        ty: field.ty.clone(),
        opening,
    })
}

fn parse_opening(attr: &Attribute) -> syn::Result<OpeningId> {
    let mut virtual_variant: Option<(syn::Path, Option<TokenStream2>)> = None;
    let mut committed: Option<Ident> = None;

    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("committed") {
            committed = Some(meta.value()?.parse()?);
        } else {
            // A virtual-polynomial variant: a bare `Variant`, or a
            // payload-carrying `Variant(payload::PATH)`. Consume the optional
            // payload group here so `parse_nested_meta` does not choke on it.
            let variant = meta.path.clone();
            let payload = if meta.input.peek(syn::token::Paren) {
                let content;
                syn::parenthesized!(content in meta.input);
                Some(content.parse::<TokenStream2>()?)
            } else {
                None
            };
            virtual_variant = Some((variant, payload));
        }
        Ok(())
    })?;

    match (virtual_variant, committed) {
        (Some((variant, payload)), None) => Ok(OpeningId::Virtual { variant, payload }),
        (None, Some(variant)) => Ok(OpeningId::Committed { variant }),
        (None, None) => Err(syn::Error::new_spanned(
            attr,
            "#[opening(..)] must name one id",
        )),
        (Some(_), Some(_)) => Err(syn::Error::new_spanned(
            attr,
            "#[opening(..)] must name exactly one id",
        )),
    }
}

/// The field initializer in `from_row`: a payload-carrying annotation binds
/// the family member through `ExtractIndexed`; everything else goes through
/// `Extract`.
fn from_row_field(field: &BundleField) -> TokenStream2 {
    let name = &field.name;
    let ty = &field.ty;
    if let Some(OpeningId::Virtual {
        payload: Some(payload),
        ..
    }) = &field.opening
    {
        quote! {
            #name: <#ty as ::jolt_witness::witnesses::ExtractIndexed<_>>::extract_indexed(
                #payload, row, next, env,
            )?
        }
    } else {
        quote! {
            #name: <#ty as ::jolt_witness::witnesses::Extract>::extract(row, next, env)?
        }
    }
}

fn id_expr(opening: &OpeningId) -> TokenStream2 {
    match opening {
        OpeningId::Virtual { variant, payload } => {
            let payload = payload.as_ref().map(|payload| quote!((#payload)));
            quote! {
                ::jolt_witness::__private::JoltPolynomialId::Virtual(
                    ::jolt_witness::__private::JoltVirtualPolynomial::#variant #payload
                )
            }
        }
        OpeningId::Committed { variant } => quote! {
            ::jolt_witness::__private::JoltPolynomialId::Committed(
                ::jolt_witness::__private::JoltCommittedPolynomial::#variant
            )
        },
    }
}

/// One `#[cfg(test)]` test per annotated field: the bundle column must equal
/// the backend's `oracle_table` for the same id on the sample trace.
fn consistency_tests(name: &Ident, annotated: &[&BundleField]) -> TokenStream2 {
    if annotated.is_empty() {
        return TokenStream2::new();
    }
    let module = format_ident!(
        "{}_witness_bundle_consistency",
        snake_case(&name.to_string())
    );
    let tests = annotated.iter().enumerate().map(|(position, field)| {
        let test_name = format_ident!("{}_column_matches_oracle_table", field.name);
        quote! {
            #[test]
            fn #test_name() {
                ::jolt_witness::testing::assert_annotated_column_matches::<super::#name>(#position);
            }
        }
    });
    quote! {
        #[cfg(test)]
        mod #module {
            #(#tests)*
        }
    }
}

fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (index, character) in name.chars().enumerate() {
        if character.is_ascii_uppercase() {
            if index != 0 {
                out.push('_');
            }
            out.push(character.to_ascii_lowercase());
        } else {
            out.push(character);
        }
    }
    out
}
