//! Derive macros generating the opening-claim plumbing for Jolt relations.
//!
//! These derives operate on a single relation's claim struct. They emit the
//! per-struct encode/resolve impls so that the canonical opening **order** is
//! single-sourced from the struct's field declaration order (rather than
//! hand-written, where copies drift). They generate impls
//! of the `OutputClaims` / `InputClaims` traits defined in `jolt_claims`; the
//! generated code references those traits through `::jolt_claims::*` (absolute
//! paths), so the derives can be applied to structs in any crate that depends on
//! `jolt-claims`.
//!
//! The claim struct is generic over an opening *cell* (`OpeningClaim<F>` on the
//! clear path, `Vec<F>` for ZK points, `F` for the serialized wire form). The
//! generated impls read each field's value through the `GetValue` cell trait, so
//! one struct definition serves all three forms.
//!
//! ## `#[derive(OutputClaims)]`
//!
//! For a relation's *produced*-claim struct. Requires a struct-level
//! `#[relation(RelationVariant)]` (the owning `JoltRelationId`) when the struct
//! has leaf opening fields. Each field is either a leaf opening (annotated with
//! `#[opening(..)]`) or a nested aggregate (no annotation; its type must also
//! implement `OutputClaims`). An `Option<C>` leaf is a *conditional* opening: it
//! contributes to `opening_values` / `canonical_order` and resolves by id only when
//! `Some` (used for advice / committed-program openings that are present only in
//! some proof configurations).
//!
//! ## `#[derive(InputClaims)]`
//!
//! For a relation's *consumed*-claim struct. Each leaf field carries its own
//! `from = ProducingRelation`, because consumed openings originate in several
//! upstream relations. `Option<C>` leaf fields resolve to the located value if
//! present; plain `C` fields resolve to `Some(value)`.
//!
//! ## `#[opening(..)]` field grammar (both derives)
//!
//! - `#[opening(VirtualVariant)]` — a virtual-polynomial opening.
//! - `#[opening(VirtualVariant(Payload::PATH))]` — a payload-carrying virtual
//!   variant, e.g. `#[opening(OpFlags(CircuitFlags::VirtualInstruction))]`. The
//!   payload tokens are emitted verbatim, so they must resolve at the derive site.
//! - `#[opening(committed = CommittedVariant)]` — a committed opening.
//! - `#[opening(trusted_advice)]` / `#[opening(untrusted_advice)]` — an advice
//!   opening.
//!
//! Arity is read from the field type, not the annotation. A `C` or `Option<C>`
//! field is a single opening: `Variant` must be a unit variant or a
//! payload-carrying variant (`OpFlags(CircuitFlags::VirtualInstruction)`). A
//! `Vec<C>` field is an indexed family; element `i` maps to:
//! - `Variant(i)` for a `usize`-indexed variant, e.g. `#[opening(LookupTableFlag)]`
//!   → `LookupTableFlag(i)`; or
//! - `Variant(ARRAY[i])` when the annotation supplies a per-element payload array,
//!   e.g. `#[opening(OpFlags(CIRCUIT_FLAGS))]` → `OpFlags(CIRCUIT_FLAGS[i])`, for
//!   families keyed by an enum rather than a contiguous index.
//!
//! `InputClaims` leaves additionally take `, from = ProducingRelation`.
//!
//! ## `#[derive(SumcheckChallenges)]`
//!
//! For a relation's drawn Fiat-Shamir challenges. The struct is generic over the
//! field `F` directly (challenges carry no opening point, so there is no opening
//! *cell* / `GetValue` indirection — field values are read directly). Each field
//! carries `#[challenge(SubEnum::Variant)]` naming a challenge sub-enum *unit*
//! variant; the resolved id is `JoltChallengeId::from(SubEnum::Variant)` (relying
//! on the `From<SubEnum> for JoltChallengeId` impls). Every field is a scalar `F`
//! (one drawn Fiat-Shamir scalar). A `Vec<F>` field is rejected (challenge sub-enum
//! variants are unit, so there is no indexed id), and an `Option<F>` field is
//! rejected (no relation draws a conditional challenge, and the `draw_challenges`
//! default treats every field as one unconditional `challenge_scalar`).
//!
//! Generates both halves of [`SumcheckChallenges`]: `resolve_challenge` (id →
//! value) and `from_transcript_values` (consume one drawn scalar per field in
//! declaration order, erroring if the stream runs dry).

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

/// Each field names its challenge via `#[challenge(SubEnum::Variant)]`; the id is
/// `JoltChallengeId::from(SubEnum::Variant)`.
#[proc_macro_derive(SumcheckChallenges, attributes(challenge))]
pub fn derive_sumcheck_challenges(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_challenges(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

enum LeafKind {
    /// A virtual-polynomial variant: its variant path plus an optional payload
    /// (`OpFlags(CircuitFlags::VirtualInstruction)` carries the `CircuitFlags::..`
    /// path as `payload`). A payload variant is always scalar — never indexed.
    Virtual {
        variant: syn::Path,
        payload: Option<TokenStream2>,
    },
    Committed(Ident),
    TrustedAdvice,
    UntrustedAdvice,
}

struct OpeningSpec {
    kind: LeafKind,
    from: Option<Ident>,
}

/// A leaf opening field: its identifier, arity, kind, and owning relation. Every
/// field of a claim struct must be a leaf `#[opening(..)]` — nested aggregates are
/// not supported (aggregate structs hand-write their encoders).
struct FieldPlan {
    ident: Ident,
    is_option: bool,
    is_many: bool,
    kind: LeafKind,
    relation: Ident,
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

/// The first type generic parameter (the opening *cell*, conventionally `C`).
fn cell_type_param(generics: &syn::Generics) -> syn::Result<Ident> {
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
                "expected an opening-cell generic parameter (e.g. `<C>`)",
            )
        })
}

/// Build the impl header for a cell-generic claim struct: introduce a fresh
/// field-value type parameter, require the struct's cell parameter to expose it
/// via `GetValue`, and return `(value_param, impl_generics, ty_generics,
/// where_clause)`.
fn cell_impl_pieces(
    generics: &syn::Generics,
) -> syn::Result<(Ident, TokenStream2, TokenStream2, TokenStream2)> {
    let cell = cell_type_param(generics)?;
    let value = Ident::new("__JoltCellValue", proc_macro2::Span::call_site());
    let params = &generics.params;
    let (_, ty_generics, _) = generics.split_for_impl();
    let impl_generics = quote!(<#value: ::jolt_field::Field, #params>);
    let orig_predicates = generics
        .where_clause
        .as_ref()
        .map(|where_clause| &where_clause.predicates);
    let where_clause = quote! {
        where #cell: ::jolt_claims::GetValue<#value>, #orig_predicates
    };
    Ok((value, impl_generics, quote!(#ty_generics), where_clause))
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
    field
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("opening"))
}

fn parse_opening(attr: &Attribute) -> syn::Result<OpeningSpec> {
    let mut virtual_variant: Option<(syn::Path, Option<TokenStream2>)> = None;
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
            // A virtual-polynomial variant: a bare `Variant`, or a payload-carrying
            // `Variant(payload::PATH)` such as
            // `OpFlags(CircuitFlags::VirtualInstruction)`. Consume the optional
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

    let kinds = [
        virtual_variant.map(|(variant, payload)| LeafKind::Virtual { variant, payload }),
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
    let attr = opening_attr(field).ok_or_else(|| {
        syn::Error::new_spanned(
            field,
            "every field needs an #[opening(..)] annotation (nested aggregates are not supported)",
        )
    })?;
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
    if is_many
        && matches!(
            spec.kind,
            LeafKind::TrustedAdvice | LeafKind::UntrustedAdvice
        )
    {
        return Err(syn::Error::new_spanned(
            &field.ty,
            "advice openings are scalar; a `Vec` advice field has no indexed id",
        ));
    }
    Ok(FieldPlan {
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
        LeafKind::Virtual { variant, payload } => {
            let polynomial = match (index, payload) {
                // Indexed family over an enum payload: the field is a `Vec` and the
                // annotation supplies the per-element payload *array*, so element `i`
                // maps to `Variant(ARRAY[i])` (e.g. `OpFlags(CIRCUIT_FLAGS[i])`).
                (Some(index), Some(payload)) => {
                    quote!(#jolt::JoltVirtualPolynomial::#variant(#payload[#index]))
                }
                // Indexed family over a `usize` payload: `Variant(i)`.
                (Some(index), None) => quote!(#jolt::JoltVirtualPolynomial::#variant(#index)),
                // Single payload-carrying variant: `Variant(PAYLOAD)`.
                (None, Some(payload)) => quote!(#jolt::JoltVirtualPolynomial::#variant(#payload)),
                // Single unit variant: `Variant`.
                (None, None) => quote!(#jolt::JoltVirtualPolynomial::#variant),
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
    let (value, impl_generics, ty_generics, where_clause) = cell_impl_pieces(&input.generics)?;
    let struct_relation = parse_struct_relation(&input.attrs)?;
    let fields = named_fields(&input.data, name.span())?;
    let plans = fields
        .iter()
        .map(|field| plan_field(field, struct_relation.as_ref()))
        .collect::<syn::Result<Vec<_>>>()?;

    let get = quote!(::jolt_claims::GetValue::value);
    let id_ty = quote!(::jolt_claims::protocols::jolt::JoltOpeningId);
    let mut value_chains = Vec::new();
    // `order_chains` mirrors `value_chains` one-for-one (same iteration, id instead
    // of value), so `canonical_order().len() == opening_values().len()` and
    // `canonical_order()[k]` is the id of `opening_values()[k]` by construction.
    let mut order_chains = Vec::new();
    let mut resolve_arms = Vec::new();

    for plan in &plans {
        let FieldPlan {
            ident,
            is_option,
            is_many,
            kind,
            relation,
        } = plan;
        if *is_many {
            let id = id_expr(kind, relation, Some(quote!(index)));
            value_chains.push(quote!(.chain(self.#ident.iter().map(|__cell| #get(__cell)))));
            order_chains.push(quote!(.chain(self.#ident.iter().enumerate().map(|(index, _)| #id))));
            resolve_arms.push(quote! {
                for (index, __cell) in self.#ident.iter().enumerate() {
                    if *id == #id {
                        return ::core::option::Option::Some(#get(__cell));
                    }
                }
            });
        } else if *is_option {
            let id = id_expr(kind, relation, None);
            value_chains.push(quote!(.chain(self.#ident.as_ref().map(|__cell| #get(__cell)))));
            order_chains.push(quote!(.chain(self.#ident.as_ref().map(|_| #id))));
            resolve_arms.push(quote! {
                if let ::core::option::Option::Some(__cell) = &self.#ident {
                    if *id == #id {
                        return ::core::option::Option::Some(#get(__cell));
                    }
                }
            });
        } else {
            let id = id_expr(kind, relation, None);
            value_chains.push(quote!(.chain(::core::iter::once(#get(&self.#ident)))));
            order_chains.push(quote!(.chain(::core::iter::once(#id))));
            resolve_arms.push(quote! {
                if *id == #id {
                    return ::core::option::Option::Some(#get(&self.#ident));
                }
            });
        }
    }

    // Field-wise zip of the value-only (`F`) and point-only (`Vec<F>`) cell forms
    // into the clear `OpeningClaim<F>` form: one `OpeningClaim` per leaf,
    // element-wise for `Vec` families, value-driven for `Option` leaves.
    let opening = quote!(::jolt_claims::OpeningClaim);
    let zip_field = Ident::new("__JoltZipField", proc_macro2::Span::call_site());
    let mut zip_inits = Vec::new();
    for plan in &plans {
        let ident = &plan.ident;
        if plan.is_many {
            zip_inits.push(quote! {
                #ident: values.#ident.iter().zip(points.#ident.iter())
                    .map(|(__value, __point)| #opening {
                        point: ::std::clone::Clone::clone(__point),
                        value: *__value,
                    })
                    .collect(),
            });
        } else if plan.is_option {
            zip_inits.push(quote! {
                #ident: values.#ident.as_ref().map(|__value| #opening {
                    point: points.#ident.clone().unwrap_or_default(),
                    value: *__value,
                }),
            });
        } else {
            zip_inits.push(quote! {
                #ident: #opening {
                    point: ::std::clone::Clone::clone(&points.#ident),
                    value: values.#ident,
                },
            });
        }
    }

    Ok(quote! {
        impl #impl_generics ::jolt_claims::OutputClaims<#value>
            for #name #ty_generics #where_clause
        {
            fn opening_values(&self) -> ::std::vec::Vec<#value> {
                ::core::iter::empty::<#value>()
                    #(#value_chains)*
                    .collect()
            }

            fn canonical_order(&self) -> ::std::vec::Vec<#id_ty> {
                ::core::iter::empty::<#id_ty>()
                    #(#order_chains)*
                    .collect()
            }

            fn resolve_output(
                &self,
                id: &::jolt_claims::protocols::jolt::JoltOpeningId,
            ) -> ::core::option::Option<#value> {
                #(#resolve_arms)*
                ::core::option::Option::None
            }
        }

        impl<#zip_field: ::jolt_field::Field> ::jolt_claims::ZipOpenings<#zip_field>
            for #name<#opening<#zip_field>>
        {
            type Values = #name<#zip_field>;
            type Points = #name<::std::vec::Vec<#zip_field>>;
            fn zip_openings(values: &Self::Values, points: &Self::Points) -> Self {
                #name {
                    #(#zip_inits)*
                }
            }
        }
    })
}

fn expand_input(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let (value, impl_generics, ty_generics, where_clause) = cell_impl_pieces(&input.generics)?;
    let fields = named_fields(&input.data, name.span())?;
    let plans = fields
        .iter()
        .map(|field| plan_field(field, None))
        .collect::<syn::Result<Vec<_>>>()?;

    let get = quote!(::jolt_claims::GetValue::value);
    let id_ty = quote!(::jolt_claims::protocols::jolt::JoltOpeningId);
    let mut resolve_arms = Vec::new();
    // Mirrors the resolve iteration (id per leaf, per `Vec` element, per `Some`
    // `Option`), so `canonical_order()` lists exactly the ids `resolve_input`
    // would hit, in field-declaration order.
    let mut order_chains = Vec::new();
    for plan in &plans {
        let FieldPlan {
            ident,
            is_option,
            is_many,
            kind,
            relation,
        } = plan;
        if *is_many {
            let id = id_expr(kind, relation, Some(quote!(index)));
            order_chains.push(quote!(.chain(self.#ident.iter().enumerate().map(|(index, _)| #id))));
            resolve_arms.push(quote! {
                for (index, __cell) in self.#ident.iter().enumerate() {
                    if *id == #id {
                        return ::core::option::Option::Some(#get(__cell));
                    }
                }
            });
        } else {
            let id = id_expr(kind, relation, None);
            if *is_option {
                order_chains.push(quote!(.chain(self.#ident.as_ref().map(|_| #id))));
            } else {
                order_chains.push(quote!(.chain(::core::iter::once(#id))));
            }
            let hit = if *is_option {
                // The field is `Option<C>`; surface the value if present.
                quote!(return self.#ident.as_ref().map(|__cell| #get(__cell));)
            } else {
                quote!(return ::core::option::Option::Some(#get(&self.#ident));)
            };
            resolve_arms.push(quote! {
                if *id == #id {
                    #hit
                }
            });
        }
    }

    Ok(quote! {
        impl #impl_generics ::jolt_claims::InputClaims<#value>
            for #name #ty_generics #where_clause
        {
            fn canonical_order(&self) -> ::std::vec::Vec<#id_ty> {
                ::core::iter::empty::<#id_ty>()
                    #(#order_chains)*
                    .collect()
            }

            fn resolve_input(
                &self,
                id: &::jolt_claims::protocols::jolt::JoltOpeningId,
            ) -> ::core::option::Option<#value> {
                #(#resolve_arms)*
                ::core::option::Option::None
            }
        }
    })
}

/// One challenge field: its identifier and the `SubEnum::Variant` path it names.
/// Challenge fields are always a scalar `F` (one drawn Fiat-Shamir scalar).
struct ChallengeFieldPlan {
    ident: Ident,
    path: syn::Path,
}

fn challenge_attr(field: &syn::Field) -> Option<&Attribute> {
    field
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("challenge"))
}

fn parse_challenge(attr: &Attribute) -> syn::Result<syn::Path> {
    attr.parse_args::<syn::Path>()
}

fn plan_challenge_field(field: &syn::Field) -> syn::Result<ChallengeFieldPlan> {
    let ident = field
        .ident
        .clone()
        .ok_or_else(|| syn::Error::new_spanned(field, "fields must be named"))?;
    let attr = challenge_attr(field).ok_or_else(|| {
        syn::Error::new_spanned(
            field,
            "every field needs a #[challenge(SubEnum::Variant)] annotation",
        )
    })?;
    let path = parse_challenge(attr)?;
    if is_vec_type(&field.ty) {
        return Err(syn::Error::new_spanned(
            &field.ty,
            "challenges are scalar; a `Vec` challenge field has no indexed id \
             (every challenge sub-enum variant is a unit variant)",
        ));
    }
    if is_option_type(&field.ty) {
        return Err(syn::Error::new_spanned(
            &field.ty,
            "challenge fields are an unconditional scalar `F`; a conditional \
             `Option<F>` challenge is not supported (no relation draws one, and the \
             `draw_challenges` default treats every field as one `challenge_scalar`)",
        ));
    }
    Ok(ChallengeFieldPlan { ident, path })
}

/// The single field type generic parameter (the field type, conventionally `F`).
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
                "expected a field-type generic parameter (e.g. `<F>`)",
            )
        })
}

fn expand_challenges(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let field = field_type_param(&input.generics)?;
    let fields = named_fields(&input.data, name.span())?;
    let plans = fields
        .iter()
        .map(plan_challenge_field)
        .collect::<syn::Result<Vec<_>>>()?;

    let mut resolve_arms = Vec::new();
    let mut build_stmts = Vec::new();
    let mut field_idents = Vec::new();
    // Every challenge field is a scalar, so the struct requires one drawn value per
    // field; `required` is the field count.
    let required = plans.len();
    for (index, plan) in plans.iter().enumerate() {
        let ChallengeFieldPlan { ident, path } = plan;
        field_idents.push(ident.clone());
        let id = quote!(::jolt_claims::protocols::jolt::JoltChallengeId::from(#path));
        resolve_arms.push(quote! {
            if *id == #id {
                return ::core::option::Option::Some(self.#ident);
            }
        });
        // Each scalar field consumes one drawn value; a dry stream is an error. The
        // already-populated count (`index`) is baked per field so the error reports
        // progress without a runtime counter.
        build_stmts.push(quote! {
            let #ident = __values.next().ok_or(
                ::jolt_claims::ChallengeDrawError {
                    required: #required,
                    populated: #index,
                },
            )?;
        });
    }

    Ok(quote! {
        impl<#field: ::jolt_field::Field> ::jolt_claims::SumcheckChallenges<#field> for #name<#field> {
            fn from_transcript_values<__I: ::core::iter::Iterator<Item = #field>>(
                values: __I,
            ) -> ::core::result::Result<Self, ::jolt_claims::ChallengeDrawError> {
                let mut __values = values;
                #(#build_stmts)*
                ::core::result::Result::Ok(Self {
                    #(#field_idents),*
                })
            }

            fn resolve_challenge(
                &self,
                id: &::jolt_claims::protocols::jolt::JoltChallengeId,
            ) -> ::core::option::Option<#field> {
                #(#resolve_arms)*
                ::core::option::Option::None
            }
        }
    })
}
