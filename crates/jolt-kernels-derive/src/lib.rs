//! Derive macros for `jolt-kernels` backend registries.
//!
//! [`macro@KernelSlots`] turns a plainly declared kernel registry — a struct
//! whose sumcheck slots are `Box<dyn PrepareKernel<F, R>>` fields — into its
//! type-indexed resolution: one `jolt_kernels::PrepareKernel<F, R>` impl per
//! slot, delegating `prepare` to `self.<field>`. The field's own type IS the
//! relation→slot mapping — declared once, restated nowhere. Fields whose type
//! never names `PrepareKernel` (bespoke slots such as commitment streaming or
//! the joint opening) are skipped. The test is syntactic: a field whose type
//! names `PrepareKernel` in any other shape (an extra object bound, a wrapper,
//! a wrong argument count) is a derive error at that field, while a slot
//! spelled through a type alias or a renamed import never names it, is
//! skipped, and surfaces as a missing `PrepareKernel` bound at the consumer
//! that needs it. The emitted impls reference `PrepareKernel` and its
//! signature types through the crate path from the serde-style
//! `#[kernel_slots(crate = "...")]` override — absolute `::jolt_kernels` by
//! default (external registries need nothing); `jolt-kernels` itself passes
//! `crate = "crate"` — and reuse the struct's own generics and where-clause
//! verbatim.
//!
//! See `specs/prover-stage-drivers.md`.

#[cfg(test)]
mod tests;

use proc_macro::TokenStream;
use proc_macro2::{TokenStream as TokenStream2, TokenTree};
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, Data, DeriveInput, Error, Fields, GenericArgument, Meta, PathArguments,
    Type, TypeParamBound,
};

/// Emit one `jolt_kernels::PrepareKernel<F, R>` impl per `Box<dyn
/// PrepareKernel<F, R>>` field of the registry struct, delegating `prepare`
/// to that field. Fields whose type does not name `PrepareKernel` are
/// skipped; any other field naming it is rejected.
/// `#[kernel_slots(crate = "...")]` overrides the `::jolt_kernels` path the
/// impls name the trait crate by (the defining crate passes `"crate"`). See
/// the crate-level docs.
#[proc_macro_derive(KernelSlots, attributes(kernel_slots))]
pub fn derive_kernel_slots(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn expand(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let krate = crate_path(&input.attrs)?.unwrap_or_else(|| syn::parse_quote!(::jolt_kernels));
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return Err(Error::new_spanned(
                    name,
                    "KernelSlots requires a struct with named fields",
                ))
            }
        },
        _ => {
            return Err(Error::new_spanned(
                name,
                "KernelSlots can only be derived for structs",
            ))
        }
    };

    let mut impls = Vec::new();
    let mut errors: Option<Error> = None;
    for field in fields {
        let ident = &field.ident;
        if !names_prepare_kernel(field.ty.to_token_stream()) {
            continue;
        }
        let Some((f, r)) = prepare_kernel_args(&field.ty) else {
            let error = Error::new_spanned(
                &field.ty,
                "a KernelSlots slot must be exactly `Box<dyn PrepareKernel<F, R>>`",
            );
            match &mut errors {
                Some(errors) => errors.combine(error),
                None => errors = Some(error),
            }
            continue;
        };
        impls.push(quote! {
            impl #impl_generics #krate::PrepareKernel<#f, #r> for #name #ty_generics
            #where_clause
            {
                fn prepare(
                    &self,
                    session: &mut #krate::ProofSession,
                    witness: &dyn ::jolt_witness::JoltWitnessPlane<#f>,
                    inputs: #krate::ProverInputs<'_, #f, #r>,
                ) -> ::core::result::Result<
                    ::std::boxed::Box<dyn #krate::SumcheckKernel<#f, Relation = #r>>,
                    #krate::KernelError<#f>,
                > {
                    self.#ident.prepare(session, witness, inputs)
                }
            }
        });
    }

    match errors {
        Some(errors) => Err(errors),
        None => Ok(quote!(#(#impls)*)),
    }
}

fn names_prepare_kernel(tokens: TokenStream2) -> bool {
    tokens.into_iter().any(|tree| match tree {
        TokenTree::Ident(ident) => ident == "PrepareKernel",
        TokenTree::Group(group) => names_prepare_kernel(group.stream()),
        TokenTree::Punct(_) | TokenTree::Literal(_) => false,
    })
}

fn crate_path(attrs: &[syn::Attribute]) -> syn::Result<Option<syn::Path>> {
    let mut krate = None;
    for attr in attrs {
        if !attr.path().is_ident("kernel_slots") {
            continue;
        }
        let entries = attr.parse_args_with(
            syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
        )?;
        for entry in entries {
            let Meta::NameValue(name_value) = &entry else {
                return Err(Error::new_spanned(
                    &entry,
                    "expected `crate = \"...\"` (the only `kernel_slots` entry)",
                ));
            };
            if !name_value.path.is_ident("crate") {
                return Err(Error::new_spanned(
                    &name_value.path,
                    "unknown `kernel_slots` entry (supported: `crate = \"...\"`)",
                ));
            }
            let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(lit),
                ..
            }) = &name_value.value
            else {
                return Err(Error::new_spanned(
                    &name_value.value,
                    "expected a string literal path, e.g. `crate = \"crate\"`",
                ));
            };
            krate = Some(lit.parse()?);
        }
    }
    Ok(krate)
}

fn prepare_kernel_args(ty: &Type) -> Option<(&Type, &Type)> {
    let Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != "Box" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let [GenericArgument::Type(Type::TraitObject(object))] =
        args.args.iter().collect::<Vec<_>>()[..]
    else {
        return None;
    };
    let [TypeParamBound::Trait(bound)] = object.bounds.iter().collect::<Vec<_>>()[..] else {
        return None;
    };
    let segment = bound.path.segments.last()?;
    if segment.ident != "PrepareKernel" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    match args.args.iter().collect::<Vec<_>>()[..] {
        [GenericArgument::Type(f), GenericArgument::Type(r)] => Some((f, r)),
        _ => None,
    }
}
