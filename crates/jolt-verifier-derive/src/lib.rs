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
//! generates `Stage5InputClaims<F, C>`, `Stage5OutputClaims<F, C>`, and
//! `Stage5Challenges<F>` (the `Sumchecks` suffix is replaced by `InputClaims` /
//! `OutputClaims` / `Challenges`), each with one field per instance projected
//! through the `ConcreteSumcheckInputs` / `ConcreteSumcheckOutputs` /
//! `ConcreteSumcheckChallenges` aliases in `jolt-verifier`'s `stages::relations`.
//! A source field `Option<Instance>` becomes an `Option<projection>` (a
//! conditional instance), chained only when `Some`.
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
//! The set of generated *delegated impls* is intentionally minimal — currently
//! the Fiat-Shamir opening plumbing consumed today (`OutputClaims`
//! `opening_values` / `append_to_transcript`, delegating to each member in
//! declaration order) — and grows as the migration requires. See
//! `specs/sumcheck-batch-derive.md`.
//!
//! The struct-level helper attribute `#[sumcheck_batch(custom_opening_values)]`
//! suppresses *only* that generated `opening_values` / `append_to_transcript`
//! inherent impl, leaving the three aggregate structs (and their derives /
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

/// Generate a stage's aggregate claim types (`StageNInputClaims` /
/// `StageNOutputClaims` / `StageNChallenges`) from a struct of `ConcreteSumcheck`
/// instances. See the crate-level docs.
///
/// The struct-level helper attribute `#[sumcheck_batch(custom_opening_values)]`
/// opts a stage out of the generated `OutputClaims` `opening_values` /
/// `append_to_transcript` inherent impl, so an alias-curated stage can supply its
/// own (e.g. one that skips cross-relation aliased openings). The three aggregate
/// structs and their derives are emitted unchanged.
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
    let input_name = format_ident!("{base}InputClaims");
    let output_name = format_ident!("{base}OutputClaims");
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

    let project = |alias: &TokenStream2, plan: &InstanceField, cell: Option<&TokenStream2>| {
        let instance = &plan.instance;
        let projected = if let Some(cell) = cell {
            quote!(#relations::#alias<#f, #instance, #cell>)
        } else {
            quote!(#relations::#alias<#f, #instance>)
        };
        if plan.is_option {
            quote!(::core::option::Option<#projected>)
        } else {
            projected
        }
    };

    let cell = quote!(__C);
    let inputs_alias = quote!(ConcreteSumcheckInputs);
    let outputs_alias = quote!(ConcreteSumcheckOutputs);
    let challenges_alias = quote!(ConcreteSumcheckChallenges);

    let input_fields = plans.iter().map(|plan| {
        let id = &plan.ident;
        let ty = project(&inputs_alias, plan, Some(&cell));
        quote!(pub #id: #ty)
    });
    let output_fields = plans.iter().map(|plan| {
        let id = &plan.ident;
        let ty = project(&outputs_alias, plan, Some(&cell));
        quote!(pub #id: #ty)
    });
    let challenge_fields = plans.iter().map(|plan| {
        let id = &plan.ident;
        let ty = project(&challenges_alias, plan, None);
        quote!(pub #id: #ty)
    });

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

    // The generated `OutputClaims` opening plumbing. Gated out when the stage opts
    // in to `#[sumcheck_batch(custom_opening_values)]`, in which case the stage
    // supplies its own alias-curated `opening_values` / `append_to_transcript` (and
    // any `validate`) as an inherent impl on the generated struct.
    let opening_impl = if options.custom_opening_values {
        quote!()
    } else {
        quote! {
            impl<#f: ::jolt_field::Field> #output_name<#f, #f> {
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

    // Derive sets start minimal: only what every per-relation member supports
    // today. The `Output` members derive the full standard set (incl. serde, as
    // the serialized wire form), so the `Output` aggregate does too. The `Input`
    // and `Challenges` members derive only `Clone`/`Debug`, so their aggregates
    // match; `PartialEq`/`Eq`/serde are added to those members (and here) when a
    // migration first needs them.
    Ok(quote! {
        #[derive(Clone, Debug, PartialEq, Eq)]
        #vis struct #input_name<#f: ::jolt_field::Field, #cell> {
            #(#input_fields,)*
        }

        #[derive(Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize)]
        #[serde(bound(
            serialize = "__C: ::serde::Serialize",
            deserialize = "__C: ::serde::Deserialize<'de>"
        ))]
        #vis struct #output_name<#f: ::jolt_field::Field, #cell> {
            #(#output_fields,)*
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        #vis struct #challenges_name<#f: ::jolt_field::Field> {
            #(#challenge_fields,)*
        }

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
