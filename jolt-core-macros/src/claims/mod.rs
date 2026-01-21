mod codegen;
mod parse;

use proc_macro2::TokenStream;
use syn::{parse2, ItemImpl};

use parse::SumcheckClaimsAttr;

pub fn sumcheck_claims_impl(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let attr: SumcheckClaimsAttr = parse2(attr)?;
    let impl_block: ItemImpl = parse2(item)?;
    codegen::generate(attr, impl_block)
}
