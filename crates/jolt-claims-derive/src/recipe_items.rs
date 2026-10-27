use proc_macro2::TokenStream;
use quote::quote;
use syn::{parse2, parse_quote, Error, Fields, Item, ItemImpl, Type};

/// This dispatcher owns no relation formulas. Its input is the two adjacent
/// canonical item pairs; unknown/missing/duplicate items reject compilation.
pub fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let mut file = parse2::<syn::File>(input)?;
    let mut structures = [false; 2];
    let mut implementations = [false; 2];
    for item in &mut file.items {
        match item {
            Item::Struct(item) => {
                let index = named(&item.ident)?;
                if structures[index] {
                    return Err(Error::new_spanned(item, "duplicate recipe structure"));
                }
                let Fields::Named(fields) = &item.fields else {
                    return Err(Error::new_spanned(
                        item,
                        "recipe delegate must have one named base field",
                    ));
                };
                if fields.named.len() != 1
                    || !fields.named.first().is_some_and(|field| {
                        field.ident.as_ref().is_some_and(|name| name == "base")
                    })
                {
                    return Err(Error::new_spanned(item, "recipe delegate fields changed"));
                }
                structures[index] = true;
            }
            Item::Impl(item) => {
                let index = implementation(item)?;
                if implementations[index] {
                    return Err(Error::new_spanned(item, "duplicate recipe implementation"));
                }
                implementations[index] = true;
                item.items.push(match index {
                    0 => parse_quote!(crate::immutable_base_recipe!(3, COMPOSED, Composed);),
                    1 => parse_quote!(crate::immutable_base_recipe!(18, COMPOSED, Composed);),
                    _ => return Err(Error::new_spanned(item, "unknown recipe delegate")),
                });
            }
            item => {
                return Err(Error::new_spanned(
                    item,
                    "unexpected item in closed recipe dispatch",
                ))
            }
        }
    }
    if structures != [true; 2] || implementations != [true; 2] {
        return Err(Error::new(
            proc_macro2::Span::call_site(),
            "both exact recipe delegate pairs are required",
        ));
    }
    Ok(quote!(#file))
}

fn named(name: &syn::Ident) -> syn::Result<usize> {
    match name.to_string().as_str() {
        "ProductRemainder" => Ok(0),
        "ReadRafAddressPhase" => Ok(1),
        _ => Err(Error::new_spanned(name, "unknown recipe delegate name")),
    }
}

fn implementation(item: &ItemImpl) -> syn::Result<usize> {
    let Some((relation, _)) = &item.trait_ else {
        return Err(Error::new_spanned(
            item,
            "recipe dispatch requires the SymbolicSumcheck trait",
        ));
    };
    if item.modifiers.polarity.is_some() || !relation.is_ident("SymbolicSumcheck") {
        return Err(Error::new_spanned(item, "unexpected recipe trait"));
    }
    let Type::Path(path) = &*item.self_ty else {
        return Err(Error::new_spanned(item, "unexpected recipe self type"));
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return Err(Error::new_spanned(item, "unexpected qualified recipe type"));
    }
    let segment = path
        .path
        .segments
        .first()
        .ok_or_else(|| Error::new_spanned(item, "missing recipe type"))?;
    named(&segment.ident)
}

#[cfg(test)]
mod tests {
    use super::expand;
    use quote::quote;

    #[test]
    fn dispatch_rejects_missing_duplicate_and_unknown_items() {
        assert!(expand(quote!()).is_err());
        assert!(expand(quote!(
            struct Other;
        ))
        .is_err());
        assert!(expand(quote!(
            struct ProductRemainder {
                base: Base,
            }
            struct ProductRemainder {
                base: Base,
            }
        ))
        .is_err());
        assert!(expand(quote!(
            struct ProductRemainder {
                base: Base,
                extra: bool,
            }
        ))
        .is_err());
    }
}
