//! Derive macros for `jolt-kernels` backend registries.
//!
//! [`macro@KernelSlots`] turns a plainly declared kernel registry — a struct
//! whose sumcheck slots are `Box<dyn PrepareKernel<F, R>>` fields — into its
//! type-indexed resolution: one `jolt_kernels::HasKernel<F, R>` impl per slot,
//! returning `&*self.<field>`. The field's own type IS the relation→slot
//! mapping — declared once, restated nowhere. Every other field (bespoke
//! slots such as commitment streaming or the joint opening) is skipped
//! silently, so a mis-declared slot surfaces as a missing-`HasKernel` bound
//! error at the consuming stage impl, never as a derive error. The emitted
//! impls reference `HasKernel`/`PrepareKernel` through the crate path from
//! the serde-style `#[kernel_slots(crate = "...")]` override — absolute
//! `::jolt_kernels` by default (external registries need nothing);
//! `jolt-kernels` itself passes `crate = "crate"` — and reuse the struct's
//! own generics and where-clause verbatim.
//!
//! See `specs/prover-stage-drivers.md`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse_macro_input, Data, DeriveInput, Fields, GenericArgument, Meta, PathArguments, Type,
    TypeParamBound,
};

/// Emit one `jolt_kernels::HasKernel<F, R>` impl per `Box<dyn PrepareKernel<F,
/// R>>` field of the registry struct, resolving to that field. Fields of any
/// other type are skipped silently. `#[kernel_slots(crate = "...")]` overrides
/// the `::jolt_kernels` path the impls name the trait crate by (the defining
/// crate passes `"crate"`). See the crate-level docs.
#[proc_macro_derive(KernelSlots, attributes(kernel_slots))]
pub fn derive_kernel_slots(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
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
                return Err(syn::Error::new_spanned(
                    name,
                    "KernelSlots requires a struct with named fields",
                ))
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "KernelSlots can only be derived for structs",
            ))
        }
    };

    let impls = fields.iter().filter_map(|field| {
        let ident = field.ident.as_ref()?;
        let (f, r) = prepare_kernel_args(&field.ty)?;
        Some(quote! {
            impl #impl_generics #krate::HasKernel<#f, #r> for #name #ty_generics
            #where_clause
            {
                fn kernel(&self) -> &dyn #krate::PrepareKernel<#f, #r> {
                    &*self.#ident
                }
            }
        })
    });

    Ok(quote!(#(#impls)*))
}

/// Parse the serde-style `#[kernel_slots(crate = "...")]` override: a string
/// literal holding the path the emitted impls name `jolt-kernels` by
/// (`"crate"` in the defining crate). `None` means the absolute
/// `::jolt_kernels` default.
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
                return Err(syn::Error::new_spanned(
                    &entry,
                    "expected `crate = \"...\"` (the only `kernel_slots` entry)",
                ));
            };
            if !name_value.path.is_ident("crate") {
                return Err(syn::Error::new_spanned(
                    &name_value.path,
                    "unknown `kernel_slots` entry (supported: `crate = \"...\"`)",
                ));
            }
            let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(lit),
                ..
            }) = &name_value.value
            else {
                return Err(syn::Error::new_spanned(
                    &name_value.value,
                    "expected a string literal path, e.g. `crate = \"crate\"`",
                ));
            };
            krate = Some(lit.parse()?);
        }
    }
    Ok(krate)
}

/// If `ty` is syntactically `Box<dyn PrepareKernel<F, R>>` — a `Box` path with
/// one type argument that is a trait object with the single bound
/// `PrepareKernel<F, R>` (matched by its final path segment, with exactly two
/// type arguments) — return `(F, R)`.
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
