//! Derive macros generating the opening-claim plumbing for `jolt-verifier`.
//!
//! These derives operate on a single relation's claim struct. They emit the
//! per-struct encode/resolve impls so that the canonical opening **order** and
//! **count** are single-sourced from the struct's field declaration order
//! (rather than hand-written three times, where they drift). They generate impls
//! of the `OutputClaims` / `InputClaims` traits defined in
//! `jolt_verifier::stages::relations`; the generated code references those traits
//! through `crate::stages::relations::*`, so the derives are for use *within*
//! `jolt-verifier`.
//!
//! ## `#[derive(OutputClaims)]`
//!
//! For a relation's *produced*-claim struct. Requires a struct-level
//! `#[relation(RelationVariant)]` (the owning `JoltRelationId`) when the struct
//! has leaf opening fields. Each field is either a leaf opening (annotated with
//! `#[opening(..)]`) or a nested aggregate (no annotation; its type must also
//! implement `OutputClaims`).
//!
//! ## `#[derive(InputClaims)]`
//!
//! For a relation's *consumed*-claim struct. Each leaf field carries its own
//! `from = ProducingRelation`, because consumed openings originate in several
//! upstream relations. `Option<F>` leaf fields resolve to the stored option;
//! plain `F` fields resolve to `Some(value)`.
//!
//! ## `#[opening(..)]` field grammar (both derives)
//!
//! - `#[opening(VirtualVariant)]` — a virtual-polynomial opening.
//! - `#[opening(committed = CommittedVariant)]` — a committed opening.
//! - `#[opening(trusted_advice)]` / `#[opening(untrusted_advice)]` — an advice
//!   opening.
//!
//! Arity is read from the field type, not the annotation: a `Vec<F>` field is an
//! indexed family (element `i` maps to `Variant(i)`, so `Variant` must be a
//! tuple variant taking the index), while an `F` or `Option<F>` field is a
//! single opening (`Variant` must be a unit variant).
//!
//! `InputClaims` leaves additionally take `, from = ProducingRelation`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Fields, GenericParam, Ident, Type};

/// Owning relation comes from the struct-level `#[relation(..)]`.
#[proc_macro_derive(OutputClaims, attributes(relation, opening))]
pub fn derive_output_claims(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_output(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Owning relation comes from each leaf field's `from = ..`.
#[proc_macro_derive(InputClaims, attributes(opening))]
pub fn derive_input_claims(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_input(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

enum LeafKind {
    Virtual(Ident),
    Committed(Ident),
    TrustedAdvice,
    UntrustedAdvice,
}

struct OpeningSpec {
    kind: LeafKind,
    from: Option<Ident>,
}

/// A struct field, classified as either a leaf opening or a nested aggregate.
enum FieldPlan {
    Leaf {
        ident: Ident,
        is_option: bool,
        is_many: bool,
        kind: LeafKind,
        relation: Ident,
    },
    Nested {
        ident: Ident,
    },
}

fn named_fields(data: &Data, span: proc_macro2::Span) -> syn::Result<Vec<syn::Field>> {
    match data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => Ok(fields.named.iter().cloned().collect()),
            _ => Err(syn::Error::new(
                span,
                "OutputClaims/InputClaims require a struct with named fields",
            )),
        },
        _ => Err(syn::Error::new(
            span,
            "OutputClaims/InputClaims can only be derived for structs",
        )),
    }
}

/// The first type generic parameter (the field type, conventionally `F`).
fn field_type_param(generics: &syn::Generics) -> syn::Result<Ident> {
    generics
        .params
        .iter()
        .find_map(|param| match param {
            GenericParam::Type(param) => Some(param.ident.clone()),
            _ => None,
        })
        .ok_or_else(|| {
            syn::Error::new_spanned(
                generics,
                "expected a field type generic parameter (e.g. `<F: Field>`)",
            )
        })
}

fn parse_struct_relation(attrs: &[Attribute]) -> syn::Result<Option<Ident>> {
    let mut relation = None;
    for attr in attrs {
        if attr.path().is_ident("relation") {
            if relation.is_some() {
                return Err(syn::Error::new_spanned(
                    attr,
                    "duplicate #[relation(..)] attribute",
                ));
            }
            relation = Some(attr.parse_args::<Ident>()?);
        }
    }
    Ok(relation)
}

fn opening_attr(field: &syn::Field) -> Option<&Attribute> {
    field.attrs.iter().find(|attr| attr.path().is_ident("opening"))
}

fn parse_opening(attr: &Attribute) -> syn::Result<OpeningSpec> {
    let mut variant: Option<Ident> = None;
    let mut committed: Option<Ident> = None;
    let mut trusted_advice = false;
    let mut untrusted_advice = false;
    let mut from: Option<Ident> = None;

    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("committed") {
            committed = Some(meta.value()?.parse()?);
        } else if meta.path.is_ident("from") {
            from = Some(meta.value()?.parse()?);
        } else if meta.path.is_ident("trusted_advice") {
            trusted_advice = true;
        } else if meta.path.is_ident("untrusted_advice") {
            untrusted_advice = true;
        } else {
            let ident = meta
                .path
                .get_ident()
                .cloned()
                .ok_or_else(|| meta.error("expected a polynomial variant identifier"))?;
            variant = Some(ident);
        }
        Ok(())
    })?;

    let kinds = [
        variant.map(LeafKind::Virtual),
        committed.map(LeafKind::Committed),
        trusted_advice.then_some(LeafKind::TrustedAdvice),
        untrusted_advice.then_some(LeafKind::UntrustedAdvice),
    ];
    let mut selected = kinds.into_iter().flatten();
    let kind = selected
        .next()
        .ok_or_else(|| syn::Error::new_spanned(attr, "#[opening(..)] must name one opening"))?;
    if selected.next().is_some() {
        return Err(syn::Error::new_spanned(
            attr,
            "#[opening(..)] must name exactly one opening",
        ));
    }

    Ok(OpeningSpec { kind, from })
}

/// `true` if the field type's last path segment is `ident`.
fn type_named(ty: &Type, ident: &str) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == ident)
}

/// `true` if the field type is syntactically `Option<..>` (a single optional
/// opening).
fn is_option_type(ty: &Type) -> bool {
    type_named(ty, "Option")
}

/// `true` if the field type is syntactically `Vec<..>` (an indexed opening
/// family). Arity is read from the type rather than the annotation.
fn is_vec_type(ty: &Type) -> bool {
    type_named(ty, "Vec")
}

fn plan_field(field: &syn::Field, struct_relation: Option<&Ident>) -> syn::Result<FieldPlan> {
    let ident = field
        .ident
        .clone()
        .ok_or_else(|| syn::Error::new_spanned(field, "fields must be named"))?;
    let Some(attr) = opening_attr(field) else {
        return Ok(FieldPlan::Nested { ident });
    };
    let spec = parse_opening(attr)?;
    let relation = match (struct_relation, spec.from) {
        // OutputClaims: relation is struct-level; `from` is not allowed.
        (Some(relation), None) => relation.clone(),
        (Some(_), Some(from)) => {
            return Err(syn::Error::new_spanned(
                from,
                "`from = ..` is only used by InputClaims; OutputClaims uses #[relation(..)]",
            ));
        }
        // InputClaims: relation is the per-field `from`.
        (None, Some(from)) => from,
        (None, None) => {
            return Err(syn::Error::new_spanned(
                attr,
                "missing `from = ProducingRelation` on an input opening (or missing struct-level #[relation(..)] for an output opening)",
            ));
        }
    };
    let is_many = is_vec_type(&field.ty);
    if is_many && matches!(spec.kind, LeafKind::TrustedAdvice | LeafKind::UntrustedAdvice) {
        return Err(syn::Error::new_spanned(
            &field.ty,
            "advice openings are scalar; a `Vec` advice field has no indexed id",
        ));
    }
    Ok(FieldPlan::Leaf {
        ident,
        is_option: is_option_type(&field.ty),
        is_many,
        kind: spec.kind,
        relation,
    })
}

/// `JoltOpeningId` constructor for a leaf, with an optional index expression for
/// indexed (`many`) families.
fn id_expr(kind: &LeafKind, relation: &Ident, index: Option<TokenStream2>) -> TokenStream2 {
    let jolt = quote!(::jolt_claims::protocols::jolt);
    let rel = quote!(#jolt::JoltRelationId::#relation);
    match kind {
        LeafKind::Virtual(variant) => {
            let polynomial = if let Some(index) = index {
                quote!(#jolt::JoltVirtualPolynomial::#variant(#index))
            } else {
                quote!(#jolt::JoltVirtualPolynomial::#variant)
            };
            quote!(#jolt::JoltOpeningId::virtual_polynomial(#polynomial, #rel))
        }
        LeafKind::Committed(variant) => {
            let polynomial = if let Some(index) = index {
                quote!(#jolt::JoltCommittedPolynomial::#variant(#index))
            } else {
                quote!(#jolt::JoltCommittedPolynomial::#variant)
            };
            quote!(#jolt::JoltOpeningId::committed(#polynomial, #rel))
        }
        LeafKind::TrustedAdvice => quote!(#jolt::JoltOpeningId::trusted_advice(#rel)),
        LeafKind::UntrustedAdvice => quote!(#jolt::JoltOpeningId::untrusted_advice(#rel)),
    }
}

fn expand_output(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let field_param = field_type_param(&input.generics)?;
    let struct_relation = parse_struct_relation(&input.attrs)?;
    let fields = named_fields(&input.data, name.span())?;
    let plans = fields
        .iter()
        .map(|field| plan_field(field, struct_relation.as_ref()))
        .collect::<syn::Result<Vec<_>>>()?;

    let mut value_chains = Vec::new();
    let mut count_terms = Vec::new();
    let mut append_stmts = Vec::new();
    let mut resolve_arms = Vec::new();

    for plan in &plans {
        match plan {
            FieldPlan::Leaf {
                ident,
                is_option,
                is_many,
                kind,
                relation,
            } => {
                if *is_option {
                    return Err(syn::Error::new_spanned(
                        ident,
                        "Option fields are not yet supported by OutputClaims (produced claims are concrete)",
                    ));
                }
                if *is_many {
                    let id = id_expr(kind, relation, Some(quote!(index)));
                    value_chains.push(quote!(.chain(self.#ident.iter().copied())));
                    count_terms.push(quote!(self.#ident.len()));
                    append_stmts.push(quote! {
                        for value in &self.#ident {
                            transcript.append_labeled(b"opening_claim", value);
                        }
                    });
                    resolve_arms.push(quote! {
                        for (index, value) in self.#ident.iter().enumerate() {
                            if *id == #id {
                                return ::core::option::Option::Some(*value);
                            }
                        }
                    });
                } else {
                    let id = id_expr(kind, relation, None);
                    value_chains.push(quote!(.chain(::core::iter::once(self.#ident))));
                    count_terms.push(quote!(1usize));
                    append_stmts.push(quote! {
                        transcript.append_labeled(b"opening_claim", &self.#ident);
                    });
                    resolve_arms.push(quote! {
                        if *id == #id {
                            return ::core::option::Option::Some(self.#ident);
                        }
                    });
                }
            }
            FieldPlan::Nested { ident } => {
                value_chains.push(quote! {
                    .chain(crate::stages::relations::OutputClaims::opening_values(&self.#ident).into_iter())
                });
                count_terms.push(quote! {
                    crate::stages::relations::OutputClaims::opening_count(&self.#ident)
                });
                append_stmts.push(quote! {
                    crate::stages::relations::OutputClaims::append_openings(&self.#ident, transcript);
                });
                resolve_arms.push(quote! {
                    if let ::core::option::Option::Some(value) =
                        crate::stages::relations::OutputClaims::resolve_output(&self.#ident, id)
                    {
                        return ::core::option::Option::Some(value);
                    }
                });
            }
        }
    }

    let count_body = if count_terms.is_empty() {
        quote!(0usize)
    } else {
        quote!(#(#count_terms)+*)
    };
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics crate::stages::relations::OutputClaims<#field_param>
            for #name #ty_generics #where_clause
        {
            fn opening_values(&self) -> ::std::vec::Vec<#field_param> {
                ::core::iter::empty::<#field_param>()
                    #(#value_chains)*
                    .collect()
            }

            fn opening_count(&self) -> usize {
                #count_body
            }

            fn append_openings<T: ::jolt_transcript::Transcript<Challenge = #field_param>>(
                &self,
                transcript: &mut T,
            ) {
                #(#append_stmts)*
            }

            fn resolve_output(
                &self,
                id: &::jolt_claims::protocols::jolt::JoltOpeningId,
            ) -> ::core::option::Option<#field_param> {
                #(#resolve_arms)*
                ::core::option::Option::None
            }
        }
    })
}

fn expand_input(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let field_param = field_type_param(&input.generics)?;
    let fields = named_fields(&input.data, name.span())?;
    let plans = fields
        .iter()
        .map(|field| plan_field(field, None))
        .collect::<syn::Result<Vec<_>>>()?;

    let mut resolve_arms = Vec::new();
    for plan in &plans {
        match plan {
            FieldPlan::Leaf {
                ident,
                is_option,
                is_many,
                kind,
                relation,
            } => {
                if *is_many {
                    let id = id_expr(kind, relation, Some(quote!(index)));
                    resolve_arms.push(quote! {
                        for (index, value) in self.#ident.iter().enumerate() {
                            if *id == #id {
                                return ::core::option::Option::Some(*value);
                            }
                        }
                    });
                } else {
                    let id = id_expr(kind, relation, None);
                    let hit = if *is_option {
                        // The field is already `Option<F>`; surface it directly.
                        quote!(return self.#ident;)
                    } else {
                        quote!(return ::core::option::Option::Some(self.#ident);)
                    };
                    resolve_arms.push(quote! {
                        if *id == #id {
                            #hit
                        }
                    });
                }
            }
            FieldPlan::Nested { ident } => {
                resolve_arms.push(quote! {
                    if let ::core::option::Option::Some(value) =
                        crate::stages::relations::InputClaims::resolve_input(&self.#ident, id)
                    {
                        return ::core::option::Option::Some(value);
                    }
                });
            }
        }
    }

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics crate::stages::relations::InputClaims<#field_param>
            for #name #ty_generics #where_clause
        {
            fn resolve_input(
                &self,
                id: &::jolt_claims::protocols::jolt::JoltOpeningId,
            ) -> ::core::option::Option<#field_param> {
                #(#resolve_arms)*
                ::core::option::Option::None
            }
        }
    })
}
