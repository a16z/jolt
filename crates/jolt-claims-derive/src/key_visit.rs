use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Error, Fields, Path, Result};

pub(crate) fn expand(input: DeriveInput) -> Result<TokenStream> {
    let Data::Enum(data) = &input.data else {
        return Err(Error::new_spanned(&input, "KeyVisit requires an enum"));
    };
    let name = &input.ident;
    let namespace = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("recipe_namespace"))
        .ok_or_else(|| Error::new_spanned(&input, "expected #[recipe_namespace(path)]"))?
        .parse_args::<Path>()?;
    let namespace = quote!(#namespace).to_string().replace(' ', "");
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let mut arms = Vec::new();
    let mut render_arms = Vec::new();
    for (index, variant) in data.variants.iter().enumerate() {
        if variant.discriminant.is_some() {
            return Err(Error::new_spanned(
                variant,
                "recipe tags use declaration order",
            ));
        }
        let tag = Literal::usize_unsuffixed(index);
        let variant_name = &variant.ident;
        let attrs: Vec<_> = variant
            .attrs
            .iter()
            .filter(|attr| attr.path().is_ident("cfg"))
            .collect();
        let values: Vec<_> = variant
            .fields
            .iter()
            .enumerate()
            .map(|(index, field)| {
                field
                    .ident
                    .clone()
                    .unwrap_or_else(|| format_ident!("value{index}"))
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
        });
        let payload = match &variant.fields {
            Fields::Unit => quote!(),
            Fields::Unnamed(_) => quote! {
                writer.write_char('(')?;
                #(::jolt_claims::immutable_recipes::record::RenderId::render(#values, writer)?;
                  writer.write_char(',')?;)*
                writer.write_char(')')?;
            },
            Fields::Named(_) => quote! {
                writer.write_char('{')?;
                #(writer.write_str(concat!(stringify!(#values), ":"))?;
                  ::jolt_claims::immutable_recipes::record::RenderId::render(#values, writer)?;
                  writer.write_char(',')?;)*
                writer.write_char('}')?;
            },
        };
        render_arms.push(quote! {
            #(#attrs)* Self::#variant_name #pattern => {
                writer.write_str(concat!(#namespace, "::", stringify!(#name), "::", stringify!(#variant_name)))?;
                #payload
                Ok(())
            },
        });
    }
    Ok(quote! {
        impl #impl_generics ::jolt_claims::immutable_recipes::KeyVisit for #name #ty_generics #where_clause {
            fn visit(&self, sink: &mut dyn ::jolt_claims::immutable_recipes::KeySink) -> bool {
                match self { #(#arms)* }
            }
        }
        #[cfg(not(target_arch = "riscv64"))]
        impl #impl_generics ::jolt_claims::immutable_recipes::record::RenderId for #name #ty_generics #where_clause {
            fn render(&self, writer: &mut dyn ::std::fmt::Write) -> ::std::fmt::Result {
                match self { #(#render_arms)* }
            }
        }
    })
}
