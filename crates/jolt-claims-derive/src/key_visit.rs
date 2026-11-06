use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Error, Fields, Result};

pub(crate) fn expand(input: DeriveInput) -> Result<TokenStream> {
    let Data::Enum(data) = &input.data else {
        return Err(Error::new_spanned(&input, "KeyVisit requires an enum"));
    };
    let name = &input.ident;
    let module = format_ident!("__recipe_key_{}", name.to_string().to_lowercase());
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let mut arms = Vec::new();
    for (index, variant) in data.variants.iter().enumerate() {
        if variant.discriminant.is_some() {
            return Err(Error::new_spanned(
                variant,
                "recipe tags use declaration order",
            ));
        }
        let tag = Literal::usize_unsuffixed(index);
        let variant_name = &variant.ident;
        let attrs = variant
            .attrs
            .iter()
            .filter(|attr| attr.path().is_ident("cfg"));
        let values: Vec<_> = variant
            .fields
            .iter()
            .enumerate()
            .map(|(index, field)| {
                field
                    .ident
                    .clone()
                    .unwrap_or_else(|| format_ident!("value{index}"))
                }
    })
            .collect();
        let pattern = match &variant.fields {
            Fields::Unit => quote!(),
            Fields::Unnamed(_) => quote!((#(#values),*)),
            Fields::Named(_) => quote!({#(#values),*}),
        };
        arms.push(quote! {
            #(#attrs)* Self::#variant_name #pattern => sink.word(#tag)
                #(&& ::jolt_claims::immutable_recipes::KeyVisit::visit(#values, sink))*,
            }
    });
    }
    Ok(quote! {
        mod #module {
            use super::#name;
        impl #impl_generics ::jolt_claims::immutable_recipes::KeyVisit for #name #ty_generics #where_clause {
            fn visit(&self, sink: &mut dyn ::jolt_claims::immutable_recipes::KeySink) -> bool {
                match self { #(#arms)* }
            }
        }
        }
    })
}
