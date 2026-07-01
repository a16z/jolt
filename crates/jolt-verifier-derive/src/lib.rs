//! Derive macros for jolt-verifier's per-stage sumcheck-batch aggregates.
//!
//! [`macro@SumcheckBatch`] turns a per-stage source-of-truth struct whose fields
//! are the stage's `ConcreteSumcheck` instances into the stage's aggregate claim
//! types:
//!
//! ```ignore
//! #[derive(SumcheckBatch)]
//! struct Stage5Sumchecks<F: Field> {
//!     instruction_read_raf:     InstructionReadRaf<F>,
//!     ram_ra_claim_reduction:   RamRaClaimReduction<F>,
//!     registers_val_evaluation: RegistersValEvaluation<F>,
//! }
//! ```
//!
//! generates `Stage5InputClaims<F>`, `Stage5InputPoints<F>`,
//! `Stage5OutputClaims<F>`, `Stage5OutputPoints<F>`, and `Stage5Challenges<F>` (the
//! `Sumchecks` suffix is replaced by `InputClaims` / `InputPoints` / `OutputClaims`
//! / `OutputPoints` / `Challenges`), each with one field per instance projected
//! through the `SumcheckInputClaims` / `SumcheckInputPoints` / `SumcheckOutputClaims`
//! / `SumcheckOutputPoints` / `ConcreteSumcheckChallenges` aliases in
//! `jolt-verifier`'s `stages::relations` (the `*Claims` aggregates hold the wire
//! *values*, the `*Points` aggregates the derived opening points). A source field
//! `Option<Instance>` becomes an `Option<projection>` (a conditional instance),
//! chained only when `Some`.
//!
//! The projections are emitted *without* `Instance: ConcreteSumcheck<F>`
//! where-bounds on purpose: such a bound would make the compiler treat each
//! projection as an opaque associated type (from the bound) rather than
//! normalizing it to the concrete per-relation claim struct, which would break
//! the `Clone` / `Debug` / `serde` derives. Because each instance field is a
//! concrete type with a concrete `ConcreteSumcheck` impl, the projection
//! normalizes to the concrete per-relation claim struct, so the plain derives
//! apply.
//!
//! The set of generated *delegated impls* is intentionally minimal and grows as
//! the migration requires. Currently emitted for every stage:
//!
//! - the Fiat-Shamir opening plumbing (`opening_values` / `append_to_transcript`
//!   on the `OutputClaims` (values) aggregate, delegating to each member in
//!   declaration order); and
//! - the per-instance driver method on the source `StageNSumchecks` struct —
//!   `draw_challenges` (draw each member's challenges into `StageNChallenges`).
//!
//! See `specs/sumcheck-batch-derive.md`.
//!
//! The struct-level helper attribute `#[sumcheck_batch(custom_opening_values)]`
//! suppresses *only* that generated `opening_values` / `append_to_transcript`
//! inherent impl, leaving the five aggregate structs (and their derives /
//! serde) untouched. An alias-curated stage (one whose canonical opening order
//! skips cross-relation aliased openings) uses it to supply its own consistent
//! `opening_values` / `append_to_transcript` (and `validate`) as an inherent impl.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, Attribute, Data, DeriveInput, Fields, GenericArgument, GenericParam, Ident,
    Meta, PathArguments, Token, Type,
};

/// Generate a stage's aggregate claim types (`StageN{Input,Output}{Claims,Points}`
/// / `StageNChallenges`) from a struct of `ConcreteSumcheck` instances. See the
/// crate-level docs.
///
/// The struct-level helper attribute `#[sumcheck_batch(custom_opening_values)]`
/// opts a stage out of the generated `opening_values` / `append_to_transcript`
/// inherent impl on the `OutputClaims` (values) aggregate, so an alias-curated
/// stage can supply its own (e.g. one that skips cross-relation aliased openings).
/// The five aggregate structs and their derives are emitted unchanged.
#[proc_macro_derive(SumcheckBatch, attributes(sumcheck_batch))]
pub fn derive_sumcheck_batch(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// One instance field of the source struct: its name and `ConcreteSumcheck`
/// instance type. `is_option` records a conditional instance (`Option<Instance>`),
/// whose projections become `Option<..>` and chain only when present.
struct InstanceField {
    ident: Ident,
    instance: Type,
    is_option: bool,
}

fn expand(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let vis = &input.vis;

    let base = name
        .to_string()
        .strip_suffix("Sumchecks")
        .map(str::to_string)
        .filter(|base| !base.is_empty())
        .ok_or_else(|| {
            syn::Error::new_spanned(
                name,
                "a #[derive(SumcheckBatch)] struct must be named `<Stage>Sumchecks`",
            )
        })?;
    let input_claims_name = format_ident!("{base}InputClaims");
    let input_points_name = format_ident!("{base}InputPoints");
    let output_claims_name = format_ident!("{base}OutputClaims");
    let output_points_name = format_ident!("{base}OutputPoints");
    let challenges_name = format_ident!("{base}Challenges");

    let options = StageOptions::parse(&input.attrs)?;

    validate_generics(&input.generics)?;
    let f = field_type_param(&input.generics)?;
    let fields = named_fields(&input.data, name.span())?;
    let plans = fields
        .iter()
        .map(plan_field)
        .collect::<syn::Result<Vec<_>>>()?;
    if plans.is_empty() {
        return Err(syn::Error::new(
            name.span(),
            "SumcheckBatch requires at least one instance field",
        ));
    }

    let relations = quote!(crate::stages::relations);

    let project = |alias: &TokenStream2, plan: &InstanceField| {
        let instance = &plan.instance;
        let projected = quote!(#relations::#alias<#f, #instance>);
        if plan.is_option {
            quote!(::core::option::Option<#projected>)
        } else {
            projected
        }
    };

    let input_claims_alias = quote!(SumcheckInputClaims);
    let input_points_alias = quote!(SumcheckInputPoints);
    let output_claims_alias = quote!(SumcheckOutputClaims);
    let output_points_alias = quote!(SumcheckOutputPoints);
    let challenges_alias = quote!(ConcreteSumcheckChallenges);

    let field_decls = |alias: &TokenStream2| {
        plans
            .iter()
            .map(|plan| {
                let id = &plan.ident;
                let ty = project(alias, plan);
                quote!(pub #id: #ty)
            })
            .collect::<Vec<_>>()
    };
    let input_claims_fields = field_decls(&input_claims_alias);
    let input_points_fields = field_decls(&input_points_alias);
    let output_claims_fields = field_decls(&output_claims_alias);
    let output_points_fields = field_decls(&output_points_alias);
    let challenge_fields = field_decls(&challenges_alias);

    // `opening_values` over the wire cell (`C = F`): chain each member's
    // `OutputClaims::opening_values` in declaration order; `Option` members
    // contribute only when present.
    let opening_chain = plans.iter().map(|plan| {
        let id = &plan.ident;
        if plan.is_option {
            quote!(.chain(self.#id.as_ref().map(|member| member.opening_values()).unwrap_or_default()))
        } else {
            quote!(.chain(self.#id.opening_values()))
        }
    });

    // Per-instance driver plumbing on the source `StageNSumchecks` struct itself:
    // draw each member's challenges into the stage's challenge aggregate, delegating
    // to each member's `ConcreteSumcheck::draw_challenges` in declaration order;
    // `Option` members draw only when present. Always emitted — it compiles for
    // every stage because every member is a `ConcreteSumcheck` and the challenge
    // aggregate has exactly the member fields.
    let draw_fields = plans.iter().map(|plan| {
        let id = &plan.ident;
        if plan.is_option {
            quote! {
                #id: match self.#id.as_ref() {
                    ::core::option::Option::Some(member) => {
                        ::core::option::Option::Some(member.draw_challenges(transcript)?)
                    }
                    ::core::option::Option::None => ::core::option::Option::None,
                }
            }
        } else {
            quote!(#id: self.#id.draw_challenges(transcript)?)
        }
    });
    let driver_impl = quote! {
        impl<#f: ::jolt_field::Field> #name<#f> {
            /// Draw each instance's Fiat-Shamir challenges in declaration order,
            /// assembling the stage's challenge aggregate. Members with no
            /// challenges draw nothing; `Option` members draw only when present.
            /// This single-sources the stage's inline per-instance draw, so its
            /// Fiat-Shamir order follows member declaration order.
            pub fn draw_challenges<__T: ::jolt_transcript::Transcript<Challenge = #f>>(
                &self,
                transcript: &mut __T,
            ) -> ::core::result::Result<#challenges_name<#f>, crate::VerifierError> {
                use #relations::ConcreteSumcheck as _;
                ::core::result::Result::Ok(#challenges_name {
                    #(#draw_fields,)*
                })
            }
        }
    };

    // The generated `OutputClaims` opening plumbing. Gated out when the stage opts
    // in to `#[sumcheck_batch(custom_opening_values)]`, in which case the stage
    // supplies its own alias-curated `opening_values` / `append_to_transcript` (and
    // any `validate`) as an inherent impl on the generated struct.
    let opening_impl = if options.custom_opening_values {
        quote!()
    } else {
        quote! {
            impl<#f: ::jolt_field::Field> #output_claims_name<#f> {
                /// Produced opening scalars in canonical (field-declaration) order,
                /// delegating to each instance's `OutputClaims` in order.
                pub fn opening_values(&self) -> ::std::vec::Vec<#f> {
                    use ::jolt_claims::OutputClaims as _;
                    ::core::iter::empty::<#f>()
                        #(#opening_chain)*
                        .collect()
                }

                /// Append every produced opening to the transcript in canonical order,
                /// each under the `b"opening_claim"` label, matching the prover's
                /// commitment order.
                pub fn append_to_transcript<__T: ::jolt_transcript::Transcript<Challenge = #f>>(
                    &self,
                    transcript: &mut __T,
                ) {
                    for value in self.opening_values() {
                        transcript.append_labeled(b"opening_claim", &value);
                    }
                }
            }
        }
    };

    // Each opening cell instantiation is its own concrete aggregate: `*Claims`
    // holds the wire *values* (`Inputs<F>` / `Outputs<F>`); `*Points` holds the
    // derived opening points (`Inputs<Vec<F>>` / `Outputs<Vec<F>>`). Only the
    // `OutputClaims` (values) aggregate is serialized (the wire form), so it alone
    // derives serde; the empty serde bound suffices because `F: Field` already
    // implies `Serialize + DeserializeOwned` through the member structs.
    Ok(quote! {
        #[derive(Clone, Debug, PartialEq, Eq)]
        #vis struct #input_claims_name<#f: ::jolt_field::Field> {
            #(#input_claims_fields,)*
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        #vis struct #input_points_name<#f: ::jolt_field::Field> {
            #(#input_points_fields,)*
        }

        #[derive(Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize)]
        #[serde(bound(serialize = "", deserialize = ""))]
        #vis struct #output_claims_name<#f: ::jolt_field::Field> {
            #(#output_claims_fields,)*
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        #vis struct #output_points_name<#f: ::jolt_field::Field> {
            #(#output_points_fields,)*
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        #vis struct #challenges_name<#f: ::jolt_field::Field> {
            #(#challenge_fields,)*
        }

        #driver_impl

        #opening_impl
    })
}

/// Struct-level `#[sumcheck_batch(...)]` configuration. Parsed from the source
/// struct's attributes; recognizes only the flags below and errors clearly on
/// anything else.
#[derive(Default)]
struct StageOptions {
    /// `#[sumcheck_batch(custom_opening_values)]`: skip emitting the generated
    /// `OutputClaims` `opening_values` / `append_to_transcript` inherent impl so the
    /// stage can curate its own (e.g. skipping cross-relation aliased openings).
    custom_opening_values: bool,
}

impl StageOptions {
    fn parse(attrs: &[Attribute]) -> syn::Result<Self> {
        let mut options = StageOptions::default();
        for attr in attrs {
            if !attr.path().is_ident("sumcheck_batch") {
                continue;
            }
            // `#[sumcheck_batch(flag, flag, ...)]` — a comma-separated list of
            // bare-word flags (`Meta::Path`). Reject any non-flag form or unknown
            // flag with a span-pointed error.
            let flags = attr.parse_args_with(
                syn::punctuated::Punctuated::<Meta, Token![,]>::parse_terminated,
            )?;
            for flag in flags {
                let Meta::Path(path) = &flag else {
                    return Err(syn::Error::new_spanned(
                        &flag,
                        "expected a bare `sumcheck_batch` flag (e.g. `custom_opening_values`)",
                    ));
                };
                if path.is_ident("custom_opening_values") {
                    options.custom_opening_values = true;
                } else {
                    return Err(syn::Error::new_spanned(
                        path,
                        "unknown `sumcheck_batch` flag (supported: `custom_opening_values`)",
                    ));
                }
            }
        }
        Ok(options)
    }
}

/// The macro supports exactly one generic type parameter (the field `F`): no
/// extra generics, no lifetimes/consts, and no where-clause (any of which the
/// generated aggregates would silently drop). Reject anything else with a clear
/// error rather than mis-binding `F` or emitting wrong projections.
fn validate_generics(generics: &syn::Generics) -> syn::Result<()> {
    if let Some(where_clause) = &generics.where_clause {
        return Err(syn::Error::new_spanned(
            where_clause,
            "SumcheckBatch does not support a where-clause on the source struct",
        ));
    }
    let type_params = generics
        .params
        .iter()
        .filter(|param| matches!(param, GenericParam::Type(_)))
        .count();
    if generics.params.len() != 1 || type_params != 1 {
        return Err(syn::Error::new_spanned(
            generics,
            "SumcheckBatch requires exactly one generic type parameter (the field `F`)",
        ));
    }
    Ok(())
}

/// The first type generic parameter (the field, conventionally `F`).
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

fn named_fields(data: &Data, span: proc_macro2::Span) -> syn::Result<Vec<syn::Field>> {
    match data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => Ok(fields.named.iter().cloned().collect()),
            _ => Err(syn::Error::new(
                span,
                "SumcheckBatch requires a struct with named fields",
            )),
        },
        _ => Err(syn::Error::new(
            span,
            "SumcheckBatch can only be derived for structs",
        )),
    }
}

fn plan_field(field: &syn::Field) -> syn::Result<InstanceField> {
    let ident = field
        .ident
        .clone()
        .ok_or_else(|| syn::Error::new_spanned(field, "fields must be named"))?;
    let (is_option, instance) = match option_inner(&field.ty) {
        Some(inner) => (true, inner.clone()),
        None => (false, field.ty.clone()),
    };
    Ok(InstanceField {
        ident,
        instance,
        is_option,
    })
}

/// If `ty` is syntactically `Option<Inner>`, return `Inner`.
fn option_inner(ty: &Type) -> Option<&Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    })
}
