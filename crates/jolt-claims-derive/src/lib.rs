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
//! The claim struct is generic over an opening *cell*, instantiated at `F` (the
//! serialized wire value) or `Vec<F>` (the verifier-derived opening point). Each
//! derive emits the value resolver (`OutputClaims` / `InputClaims`) on the `F`
//! form and per-field point accessors on the `Vec<F>` form, so one struct
//! definition serves both forms with no `GetValue` / `GetPoint` cell trait
//! indirection.
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
//! A struct whose fields are all plain scalar cells additionally gets
//! `from_shared_point` on the point cell: the constructor for a relation whose
//! produced openings all open at one derived point.
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
//! `Vec<C>` field is an indexed family: element `i` maps to `Variant(i)` for a
//! `usize`-indexed variant, e.g. `#[opening(LookupTableFlag)]` →
//! `LookupTableFlag(i)`. A payload-carrying variant is always scalar; a family
//! keyed by an enum is declared as one field per element (see
//! `OuterRemainderOutputClaims` / `BytecodeReadRafAddressPhaseInputClaims`).
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
//!
//! ## `#[protocol(..)]` (all three derives)
//!
//! Selects the protocol id namespace the emitted impls resolve against. The
//! claim traits are generic over their id type (`OutputClaims<F, O>`,
//! `InputClaims<F, O>`, `SumcheckChallenges<F, C>`, defaulting to the jolt ids),
//! so each namespace is one more instantiation, not a new trait. Absent, the
//! namespace is `jolt` (`JoltOpeningId` / `JoltChallengeId` / ..); with
//! `#[protocol(field_inline)]` the impls target the field-inline id family
//! (`FieldInlineOpeningId` / `FieldInlineChallengeId` / ..). Advice openings are
//! a jolt-protocol concept and are rejected under any other namespace.

// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![forbid(unsafe_code)]
#![deny(
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::indexing_slicing
)]
// wildcard_enum_match_arm is omitted: this crate matches foreign syn AST enums,
// where wildcard fallbacks to Err/None are the correct, version-stable idiom.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{
    parse_macro_input, Attribute, Data, DeriveInput, Error, Field, Fields, GenericParam, Generics,
    Ident, Path, Result, Type,
};

/// Owning relation comes from the struct-level `#[relation(..)]`.
#[proc_macro_derive(OutputClaims, attributes(relation, opening, protocol))]
pub fn derive_output_claims(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_output(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Owning relation comes from each leaf field's `from = ..`.
#[proc_macro_derive(InputClaims, attributes(opening, protocol))]
pub fn derive_input_claims(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_input(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Each field names its challenge via `#[challenge(SubEnum::Variant)]`; the id is
/// `ChallengeId::from(SubEnum::Variant)` in the selected protocol namespace.
#[proc_macro_derive(SumcheckChallenges, attributes(challenge, protocol))]
pub fn derive_sumcheck_challenges(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_challenges(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// The protocol id namespace the emitted impls resolve against. Each namespace
/// names the id-family types of one `jolt_claims::protocols::*` module; the
/// derives stay a single implementation instantiated per namespace.
struct Namespace {
    opening_id: TokenStream2,
    relation_id: TokenStream2,
    virtual_polynomial: TokenStream2,
    committed_polynomial: TokenStream2,
    challenge_id: TokenStream2,
    /// Advice openings are jolt-protocol ids; other namespaces reject them.
    allows_advice: bool,
}

impl Namespace {
    fn jolt() -> Self {
        let module = quote!(::jolt_claims::protocols::jolt);
        Self {
            opening_id: quote!(#module::JoltOpeningId),
            relation_id: quote!(#module::JoltRelationId),
            virtual_polynomial: quote!(#module::JoltVirtualPolynomial),
            committed_polynomial: quote!(#module::JoltCommittedPolynomial),
            challenge_id: quote!(#module::JoltChallengeId),
            allows_advice: true,
        }
    }

    fn field_inline() -> Self {
        let module = quote!(::jolt_claims::protocols::field_inline);
        Self {
            opening_id: quote!(#module::FieldInlineOpeningId),
            relation_id: quote!(#module::FieldInlineRelationId),
            virtual_polynomial: quote!(#module::FieldInlineVirtualPolynomial),
            committed_polynomial: quote!(#module::FieldInlineCommittedPolynomial),
            challenge_id: quote!(#module::FieldInlineChallengeId),
            allows_advice: false,
        }
    }
}

/// Reads the optional struct-level `#[protocol(..)]` namespace selector;
/// defaults to the jolt namespace.
fn parse_namespace(attrs: &[Attribute]) -> Result<Namespace> {
    let mut selected: Option<(Ident, Namespace)> = None;
    for attr in attrs {
        if attr.path().is_ident("protocol") {
            if selected.is_some() {
                return Err(Error::new_spanned(
                    attr,
                    "duplicate #[protocol(..)] attribute",
                ));
            }
            let ident = attr.parse_args::<Ident>()?;
            let namespace = match ident.to_string().as_str() {
                "jolt" => Namespace::jolt(),
                "field_inline" => Namespace::field_inline(),
                other => {
                    return Err(Error::new_spanned(
                        &ident,
                        format!("unknown protocol namespace `{other}` (expected `jolt` or `field_inline`)"),
                    ));
                }
            };
            selected = Some((ident, namespace));
        }
    }
    Ok(selected.map_or_else(Namespace::jolt, |(_, namespace)| namespace))
}

enum LeafKind {
    /// A virtual-polynomial variant: its variant path plus an optional payload
    /// (`OpFlags(CircuitFlags::VirtualInstruction)` carries the `CircuitFlags::..`
    /// path as `payload`). A payload variant is always scalar — never indexed.
    Virtual {
        variant: Path,
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

fn named_fields(data: &Data, span: Span) -> Result<Vec<Field>> {
    match data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => Ok(fields.named.iter().cloned().collect()),
            _ => Err(Error::new(
                span,
                "OutputClaims/InputClaims require a struct with named fields",
            )),
        },
        _ => Err(Error::new(
            span,
            "OutputClaims/InputClaims can only be derived for structs",
        )),
    }
}

/// A claim struct must have exactly one generic type parameter (the opening
/// *cell*, conventionally `C`) and no lifetimes, consts, or where-clause: the
/// derive instantiates it at `F` (value form) and `Vec<F>` (point form), so any
/// other shape would make those instantiations ill-formed. Errors clearly rather
/// than emitting a wrongly instantiated impl.
fn ensure_single_cell_generic(generics: &Generics) -> Result<()> {
    let type_params = generics
        .params
        .iter()
        .filter(|param| matches!(param, GenericParam::Type(_)))
        .count();
    if generics.where_clause.is_some() || generics.params.len() != 1 || type_params != 1 {
        return Err(Error::new_spanned(
            generics,
            "OutputClaims/InputClaims require exactly one generic type parameter (the opening cell, e.g. `<C>`)",
        ));
    }
    Ok(())
}

fn parse_struct_relation(attrs: &[Attribute]) -> Result<Option<Ident>> {
    let mut relation = None;
    for attr in attrs {
        if attr.path().is_ident("relation") {
            if relation.is_some() {
                return Err(Error::new_spanned(
                    attr,
                    "duplicate #[relation(..)] attribute",
                ));
            }
            relation = Some(attr.parse_args::<Ident>()?);
        }
    }
    Ok(relation)
}

fn opening_attr(field: &Field) -> Option<&Attribute> {
    field
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("opening"))
}

fn parse_opening(attr: &Attribute) -> Result<OpeningSpec> {
    let mut virtual_variant: Option<(Path, Option<TokenStream2>)> = None;
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
        .ok_or_else(|| Error::new_spanned(attr, "#[opening(..)] must name one opening"))?;
    if selected.next().is_some() {
        return Err(Error::new_spanned(
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

fn plan_field(
    field: &Field,
    struct_relation: Option<&Ident>,
    namespace: &Namespace,
) -> Result<FieldPlan> {
    let ident = field
        .ident
        .clone()
        .ok_or_else(|| Error::new_spanned(field, "fields must be named"))?;
    let attr = opening_attr(field).ok_or_else(|| {
        Error::new_spanned(
            field,
            "every field needs an #[opening(..)] annotation (nested aggregates are not supported)",
        )
    })?;
    let spec = parse_opening(attr)?;
    if !namespace.allows_advice
        && matches!(
            spec.kind,
            LeafKind::TrustedAdvice | LeafKind::UntrustedAdvice
        )
    {
        return Err(Error::new_spanned(
            attr,
            "advice openings are jolt-protocol ids; this protocol namespace has no advice variant",
        ));
    }
    let relation = match (struct_relation, spec.from) {
        // OutputClaims: relation is struct-level; `from` is not allowed.
        (Some(relation), None) => relation.clone(),
        (Some(_), Some(from)) => {
            return Err(Error::new_spanned(
                from,
                "`from = ..` is only used by InputClaims; OutputClaims uses #[relation(..)]",
            ));
        }
        // InputClaims: relation is the per-field `from`.
        (None, Some(from)) => from,
        (None, None) => {
            return Err(Error::new_spanned(
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
        return Err(Error::new_spanned(
            &field.ty,
            "advice openings are scalar; a `Vec` advice field has no indexed id",
        ));
    }
    if is_many
        && matches!(
            &spec.kind,
            LeafKind::Virtual {
                payload: Some(_),
                ..
            }
        )
    {
        return Err(Error::new_spanned(
            &field.ty,
            "per-element payload arrays (e.g. `OpFlags(CIRCUIT_FLAGS)` on a `Vec`) are not \
             supported; declare one field per element instead",
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

/// Opening-id constructor for a leaf in the selected namespace, with an optional
/// index expression for indexed (`many`) families.
fn id_expr(
    ns: &Namespace,
    kind: &LeafKind,
    relation: &Ident,
    index: Option<TokenStream2>,
) -> TokenStream2 {
    let opening_id = &ns.opening_id;
    let relation_id = &ns.relation_id;
    let virtual_polynomial = &ns.virtual_polynomial;
    let committed_polynomial = &ns.committed_polynomial;
    let rel = quote!(#relation_id::#relation);
    match kind {
        LeafKind::Virtual { variant, payload } => {
            let polynomial = match (index, payload) {
                // A payload-carrying variant is always scalar; the `Vec`+payload
                // combination is rejected in `plan_field`.
                (Some(_), Some(_)) => {
                    unreachable!("Vec fields with payload annotations are rejected in plan_field")
                }
                // Indexed family over a `usize` payload: `Variant(i)`.
                (Some(index), None) => quote!(#virtual_polynomial::#variant(#index)),
                // Single payload-carrying variant: `Variant(PAYLOAD)`.
                (None, Some(payload)) => quote!(#virtual_polynomial::#variant(#payload)),
                // Single unit variant: `Variant`.
                (None, None) => quote!(#virtual_polynomial::#variant),
            };
            quote!(#opening_id::virtual_polynomial(#polynomial, #rel))
        }
        LeafKind::Committed(variant) => {
            let polynomial = if let Some(index) = index {
                quote!(#committed_polynomial::#variant(#index))
            } else {
                quote!(#committed_polynomial::#variant)
            };
            quote!(#opening_id::committed(#polynomial, #rel))
        }
        LeafKind::TrustedAdvice => quote!(#opening_id::trusted_advice(#rel)),
        LeafKind::UntrustedAdvice => quote!(#opening_id::untrusted_advice(#rel)),
    }
}

fn expand_output(input: DeriveInput) -> Result<TokenStream2> {
    let name = &input.ident;
    ensure_single_cell_generic(&input.generics)?;
    let namespace = parse_namespace(&input.attrs)?;
    let struct_relation = parse_struct_relation(&input.attrs)?;
    let fields = named_fields(&input.data, name.span())?;
    let plans = fields
        .iter()
        .map(|field| plan_field(field, struct_relation.as_ref(), &namespace))
        .collect::<Result<Vec<_>>>()?;

    let id_ty = namespace.opening_id.clone();
    // `order_chains` lists each leaf's id (per `Vec` element, per `Some` `Option`) in
    // field-declaration order, so it lists exactly the ids `resolve_output` hits.
    // `OutputClaims::opening_values` reconstructs the values from this order via
    // `resolve_output`, so the canonical order is single-sourced here.
    let mut order_chains = Vec::new();
    let mut resolve_arms = Vec::new();
    let mut construct_fields = Vec::new();

    for plan in &plans {
        let FieldPlan {
            ident,
            is_option,
            is_many,
            kind,
            relation,
        } = plan;
        if *is_many {
            let id = id_expr(&namespace, kind, relation, Some(quote!(index)));
            order_chains.push(quote!(.chain(self.#ident.iter().enumerate().map(|(index, _)| #id))));
            resolve_arms.push(quote! {
                for (index, __value) in self.#ident.iter().enumerate() {
                    if *id == #id {
                        return ::core::option::Option::Some(*__value);
                    }
                }
            });
            // An indexed family consumes indices `0, 1, ..` for as long as the
            // source answers — the family's length is instance data the source
            // defines.
            construct_fields.push(quote! {
                #ident: {
                    let mut __values = ::std::vec::Vec::new();
                    let mut index = 0usize;
                    while let ::core::option::Option::Some(__value) = resolve(&#id) {
                        __values.push(__value);
                        index += 1;
                    }
                    __values
                },
            });
        } else if *is_option {
            let id = id_expr(&namespace, kind, relation, None);
            order_chains.push(quote!(.chain(self.#ident.as_ref().map(|_| #id))));
            resolve_arms.push(quote! {
                if let ::core::option::Option::Some(__value) = &self.#ident {
                    if *id == #id {
                        return ::core::option::Option::Some(*__value);
                    }
                }
            });
            // An `Option` field is present iff the source answers its id.
            construct_fields.push(quote!(#ident: resolve(&#id),));
        } else {
            let id = id_expr(&namespace, kind, relation, None);
            order_chains.push(quote!(.chain(::core::iter::once(#id))));
            resolve_arms.push(quote! {
                if *id == #id {
                    return ::core::option::Option::Some(self.#ident);
                }
            });
            // A plain field's id must resolve; a miss is the caller's error.
            construct_fields.push(quote! {
                #ident: {
                    let __id = #id;
                    match resolve(&__id) {
                        ::core::option::Option::Some(__value) => __value,
                        ::core::option::Option::None => {
                            return ::core::result::Result::Err(
                                ::jolt_claims::MissingOpeningValue { id: __id },
                            );
                        }
                    }
                },
            });
        }
    }

    let point_accessors = plans.iter().map(point_accessor);
    // The shared-point constructor exists only for the all-scalar shape: a `Vec`
    // family or `Option` leaf has no single "every opening at one point" form.
    let from_shared_point = (!plans.is_empty()
        && plans.iter().all(|plan| !plan.is_many && !plan.is_option))
    .then(|| {
        let fields = plans.iter().enumerate().map(|(index, plan)| {
            let ident = &plan.ident;
            // the last field takes ownership of the point; the rest clone it
            if index + 1 == plans.len() {
                quote!(#ident: point,)
            } else {
                quote!(#ident: point.clone(),)
            }
        });
        quote! {
            /// Every produced opening at the one shared opening `point` — the
            /// point form of a relation whose produced openings all open at a
            /// single derived point. Its `derive_opening_points` builds the
            /// struct here, so the field enumeration stays owned by this derive
            /// instead of a hand-written struct literal.
            pub fn from_shared_point(point: ::std::vec::Vec<F>) -> Self {
                Self { #(#fields)* }
            }
        }
    });

    Ok(quote! {
        // The value resolver lives on the value cell (`C = F`): each field is read
        // as `F` (or `Vec<F>` / `Option<F>`) directly.
        impl<F: ::jolt_field::JoltField> ::jolt_claims::OutputClaims<F, #id_ty> for #name<F> {
            fn canonical_order(&self) -> ::std::vec::Vec<#id_ty> {
                ::core::iter::empty::<#id_ty>()
                    #(#order_chains)*
                    .collect()
            }

            fn resolve_output(
                &self,
                id: &#id_ty,
            ) -> ::core::option::Option<F> {
                #(#resolve_arms)*
                ::core::option::Option::None
            }

            fn from_opening_values(
                mut resolve: impl ::core::ops::FnMut(&#id_ty) -> ::core::option::Option<F>,
            ) -> ::core::result::Result<Self, ::jolt_claims::MissingOpeningValue<#id_ty>> {
                ::core::result::Result::Ok(Self {
                    #(#construct_fields)*
                })
            }
        }

        // The per-field opening-point accessors live on the point cell
        // (`C = Vec<F>`): each field is a `Vec<F>` point (or `Vec<Vec<F>>` /
        // `Option<Vec<F>>`). A field and its accessor share a name; `x` reads the
        // field, `x()` calls the accessor.
        impl<F: ::jolt_field::JoltField> #name<::std::vec::Vec<F>> {
            #from_shared_point
            #(#point_accessors)*
        }
    })
}

/// A per-field opening-point accessor on the point cell (`C = Vec<F>`): scalar
/// `fn f(&self) -> &[F]`, `Vec` `fn f(&self) -> &[Vec<F>]`, `Option`
/// `fn f(&self) -> Option<&[F]>`.
fn point_accessor(plan: &FieldPlan) -> TokenStream2 {
    let ident = &plan.ident;
    if plan.is_many {
        quote! {
            pub fn #ident(&self) -> &[::std::vec::Vec<F>] {
                &self.#ident
            }
        }
    } else if plan.is_option {
        quote! {
            pub fn #ident(&self) -> ::core::option::Option<&[F]> {
                self.#ident.as_deref()
            }
        }
    } else {
        quote! {
            pub fn #ident(&self) -> &[F] {
                &self.#ident
            }
        }
    }
}

fn expand_input(input: DeriveInput) -> Result<TokenStream2> {
    let name = &input.ident;
    ensure_single_cell_generic(&input.generics)?;
    let namespace = parse_namespace(&input.attrs)?;
    let fields = named_fields(&input.data, name.span())?;
    let plans = fields
        .iter()
        .map(|field| plan_field(field, None, &namespace))
        .collect::<Result<Vec<_>>>()?;

    let id_ty = namespace.opening_id.clone();
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
            let id = id_expr(&namespace, kind, relation, Some(quote!(index)));
            order_chains.push(quote!(.chain(self.#ident.iter().enumerate().map(|(index, _)| #id))));
            resolve_arms.push(quote! {
                for (index, __value) in self.#ident.iter().enumerate() {
                    if *id == #id {
                        return ::core::option::Option::Some(*__value);
                    }
                }
            });
        } else {
            let id = id_expr(&namespace, kind, relation, None);
            if *is_option {
                order_chains.push(quote!(.chain(self.#ident.as_ref().map(|_| #id))));
            } else {
                order_chains.push(quote!(.chain(::core::iter::once(#id))));
            }
            let hit = if *is_option {
                // The field is `Option<F>`; surface the value if present.
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

    let point_accessors = plans.iter().map(point_accessor);

    Ok(quote! {
        impl<F: ::jolt_field::JoltField> ::jolt_claims::InputClaims<F, #id_ty> for #name<F> {
            fn canonical_order(&self) -> ::std::vec::Vec<#id_ty> {
                ::core::iter::empty::<#id_ty>()
                    #(#order_chains)*
                    .collect()
            }

            fn resolve_input(
                &self,
                id: &#id_ty,
            ) -> ::core::option::Option<F> {
                #(#resolve_arms)*
                ::core::option::Option::None
            }
        }

        impl<F: ::jolt_field::JoltField> #name<::std::vec::Vec<F>> {
            #(#point_accessors)*
        }
    })
}

/// One challenge field: its identifier and the `SubEnum::Variant` path it names.
/// Challenge fields are always a scalar `F` (one drawn Fiat-Shamir scalar).
struct ChallengeFieldPlan {
    ident: Ident,
    path: Path,
}

fn challenge_attr(field: &Field) -> Option<&Attribute> {
    field
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("challenge"))
}

fn parse_challenge(attr: &Attribute) -> Result<Path> {
    attr.parse_args::<Path>()
}

fn plan_challenge_field(field: &Field) -> Result<ChallengeFieldPlan> {
    let ident = field
        .ident
        .clone()
        .ok_or_else(|| Error::new_spanned(field, "fields must be named"))?;
    let attr = challenge_attr(field).ok_or_else(|| {
        Error::new_spanned(
            field,
            "every field needs a #[challenge(SubEnum::Variant)] annotation",
        )
    })?;
    let path = parse_challenge(attr)?;
    if is_vec_type(&field.ty) {
        return Err(Error::new_spanned(
            &field.ty,
            "challenges are scalar; a `Vec` challenge field has no indexed id \
             (every challenge sub-enum variant is a unit variant)",
        ));
    }
    if is_option_type(&field.ty) {
        return Err(Error::new_spanned(
            &field.ty,
            "challenge fields are an unconditional scalar `F`; a conditional \
             `Option<F>` challenge is not supported (no relation draws one, and the \
             `draw_challenges` default treats every field as one `challenge_scalar`)",
        ));
    }
    Ok(ChallengeFieldPlan { ident, path })
}

/// The single field type generic parameter (the field type, conventionally `F`).
fn field_type_param(generics: &Generics) -> Result<Ident> {
    generics
        .params
        .iter()
        .find_map(|param| match param {
            GenericParam::Type(param) => Some(param.ident.clone()),
            _ => None,
        })
        .ok_or_else(|| {
            Error::new_spanned(
                generics,
                "expected a field-type generic parameter (e.g. `<F>`)",
            )
        })
}

fn expand_challenges(input: DeriveInput) -> Result<TokenStream2> {
    let name = &input.ident;
    let namespace = parse_namespace(&input.attrs)?;
    let challenge_id_ty = namespace.challenge_id.clone();
    let field = field_type_param(&input.generics)?;
    let fields = named_fields(&input.data, name.span())?;
    let plans = fields
        .iter()
        .map(plan_challenge_field)
        .collect::<Result<Vec<_>>>()?;

    let mut resolve_arms = Vec::new();
    let mut build_stmts = Vec::new();
    let mut field_idents = Vec::new();
    // Every challenge field is a scalar, so the struct requires one drawn value per
    // field; `required` is the field count.
    let required = plans.len();
    for (index, plan) in plans.iter().enumerate() {
        let ChallengeFieldPlan { ident, path } = plan;
        field_idents.push(ident.clone());
        let id = quote!(#challenge_id_ty::from(#path));
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
                ::jolt_claims::ChallengeDrawError::StreamExhausted {
                    required: #required,
                    populated: #index,
                },
            )?;
        });
    }

    Ok(quote! {
        impl<#field: ::jolt_field::JoltField> ::jolt_claims::SumcheckChallenges<#field, #challenge_id_ty> for #name<#field> {
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
                id: &#challenge_id_ty,
            ) -> ::core::option::Option<#field> {
                #(#resolve_arms)*
                ::core::option::Option::None
            }
        }
    })
}
