mod claims;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn sumcheck_claims(attr: TokenStream, item: TokenStream) -> TokenStream {
    claims::sumcheck_claims_impl(attr.into(), item.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
