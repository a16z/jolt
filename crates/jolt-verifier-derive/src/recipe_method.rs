use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use std::collections::BTreeMap;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::visit_mut::{self, VisitMut};
use syn::{parse_quote, Expr, ExprLit, FnArg, ImplItemFn, Lit, MetaNameValue, ReturnType, Token};

pub(crate) fn expand(arguments: TokenStream, original: TokenStream) -> syn::Result<TokenStream> {
    let mut expected = BTreeMap::new();
    let mut returns_set = false;
    for option in Punctuated::<MetaNameValue, Token![,]>::parse_terminated.parse2(arguments)? {
        let name = option
            .path
            .get_ident()
            .ok_or_else(|| syn::Error::new_spanned(&option, "expected seam name"))?
            .to_string();
        let Expr::Lit(ExprLit {
            lit: Lit::Int(value),
            ..
        }) = &option.value
        else {
            return Err(syn::Error::new_spanned(
                &option,
                "seam count must be a literal",
            ));
        };
        let count = value.base10_parse::<usize>()?;
        if name == "returns_set" {
            if count != 1 || returns_set {
                return Err(syn::Error::new_spanned(
                    &option,
                    "invalid set result option",
                ));
            }
            returns_set = true;
        } else if !matches!(
            name.as_str(),
            "input"
                | "output"
                | "support"
                | "call_input_claim"
                | "call_expected_output"
                | "call_evaluate_output_expression"
                | "call_expected_output_from_publics"
                | "call_wire_output_openings"
                | "call_begin_batch"
                | "call_expected_final_claim"
                | "call_validate_member_output_shape"
        ) {
            return Err(syn::Error::new_spanned(&option, "unknown recipe seam"));
        } else if count == 0 || expected.insert(name, count).is_some() {
            return Err(syn::Error::new_spanned(
                &option,
                "duplicate or zero seam count",
            ));
        }
    }
    let mut selected: ImplItemFn = syn::parse2(original.clone())?;
    let mut transform = Transform {
        counts: BTreeMap::new(),
        declared: expected.clone(),
        errors: None,
    };
    transform.visit_block_mut(&mut selected.block);
    if let Some(error) = transform.errors {
        return Err(error);
    }
    if transform.counts != expected {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "recipe seam inventory mismatch: expected {expected:?}, observed {:?}",
                transform.counts
            ),
        ));
    }
    selected.sig.ident = format_ident!("{}_with_recipes", selected.sig.ident);
    let offset = usize::from(matches!(
        selected.sig.inputs.first(),
        Some(FnArg::Receiver(_))
    ));
    selected.sig.inputs.insert(
        offset,
        parse_quote!(__recipe_context: &crate::immutable_recipes::RecipeContext<'_>),
    );
    if returns_set {
        let ReturnType::Type(_, ty) = &selected.sig.output else {
            return Err(syn::Error::new_spanned(
                &selected.sig,
                "set method has no return type",
            ));
        };
        selected.sig.output = parse_quote!(-> Result<#ty, crate::VerifierError>);
        let body = selected.block;
        selected.block = parse_quote!({ let __recipe_set = #body; Ok(__recipe_set) });
    }
    Ok(quote!(#original #selected))
}

struct Transform {
    counts: BTreeMap<String, usize>,
    declared: BTreeMap<String, usize>,
    errors: Option<syn::Error>,
}

impl Transform {
    fn count(&mut self, name: &str) {
        *self.counts.entry(name.to_owned()).or_default() += 1;
    }
    fn reject(&mut self, expression: &Expr, reason: &str) {
        let error = syn::Error::new_spanned(expression, reason);
        if let Some(previous) = &mut self.errors {
            previous.combine(error);
        } else {
            self.errors = Some(error);
        }
    }
}

impl VisitMut for Transform {
    fn visit_expr_mut(&mut self, expression: &mut Expr) {
        if let Expr::MethodCall(call) = expression {
            if call.method == "try_evaluate" {
                if let Expr::MethodCall(construction) = call.receiver.as_ref() {
                    let phase = match construction.method.to_string().as_str() {
                        "input_expression" => Some(("input", quote!(Input))),
                        "output_expression" => Some(("output", quote!(Output))),
                        _ => None,
                    };
                    if let Some((name, phase)) = phase {
                        if !construction.args.is_empty() || call.args.len() != 3 {
                            self.reject(expression, "unexpected expression evaluation arguments");
                            return;
                        }
                        let Some(types) = &construction.turbofish else {
                            self.reject(
                                expression,
                                "recipe evaluation requires explicit field type",
                            );
                            return;
                        };
                        if types.args.len() != 1 {
                            self.reject(expression, "recipe evaluation requires one field type");
                            return;
                        }
                        let field = types.args.first();
                        let relation = &construction.receiver;
                        let method = &construction.method;
                        let arguments = call.args.iter();
                        *expression = parse_quote!({
                            use crate::immutable_recipes::RecipeEvaluation as _;
                            let __recipe_symbolic = &(#relation);
                            __recipe_symbolic.evaluate_recipe::<#field>(
                                ::jolt_claims::immutable_recipes::Phase::#phase,
                                __recipe_context, || __recipe_symbolic.#method #types (), #(#arguments),*,
                            )
                        });
                        self.count(name);
                        return;
                    }
                }
            }
            if call.method == "expected_output_openings" {
                let Some(types) = &call.turbofish else {
                    self.reject(expression, "recipe support requires explicit field type");
                    return;
                };
                if types.args.len() != 1 || !call.args.is_empty() {
                    self.reject(expression, "unexpected support arguments");
                    return;
                }
                let field = types.args.first();
                let receiver = &call.receiver;
                *expression = parse_quote!({
                    use crate::immutable_recipes::RecipeEvaluation as _;
                    (#receiver).recipe_support::<#field>(__recipe_context)?
                });
                self.count("support");
                return;
            }
            let name = format!("call_{}", call.method);
            if self.declared.contains_key(&name) {
                call.method = format_ident!("{}_with_recipes", call.method);
                call.args.insert(0, parse_quote!(__recipe_context));
                self.count(&name);
                visit_mut::visit_expr_method_call_mut(self, call);
                if name == "call_wire_output_openings" {
                    let renamed = expression.clone();
                    *expression = parse_quote!((#renamed)?);
                }
                return;
            }
            if matches!(
                call.method.to_string().as_str(),
                "input_expression" | "output_expression"
            ) {
                self.reject(
                    expression,
                    "unhandled canonical construction in selected method",
                );
                return;
            }
        }
        if let Expr::Call(call) = expression {
            if let Expr::Path(path) = call.func.as_mut() {
                if let Some(segment) = path.path.segments.last_mut() {
                    let name = format!("call_{}", segment.ident);
                    if self.declared.contains_key(&name) {
                        segment.ident = format_ident!("{}_with_recipes", segment.ident);
                        call.args.insert(0, parse_quote!(__recipe_context));
                        self.count(&name);
                    }
                }
            }
        }
        visit_mut::visit_expr_mut(self, expression);
    }
}

/// The permitted transformations follow the already declared batch members.
pub(crate) fn batch_companions(
    driver: TokenStream,
    members: usize,
    krate: &syn::Path,
) -> syn::Result<TokenStream> {
    // Only the verifier's internal derives own this feature and context API.
    // External/default consumers retain their original expansion verbatim.
    if !krate.is_ident("crate") {
        return Ok(driver);
    }
    let item: syn::ItemImpl = syn::parse2(driver.clone())?;
    let mut companions = Vec::new();
    for method in &item.items {
        let syn::ImplItem::Fn(method) = method else {
            continue;
        };
        let arguments = match method.sig.ident.to_string().as_str() {
            "begin_batch" => quote!(call_input_claim = #members),
            "verify_clear" => quote!(call_begin_batch = 1, call_expected_final_claim = 1),
            "expected_final_claim" => quote!(call_expected_output = #members),
            "validate_output_claims" => quote!(call_validate_member_output_shape = #members),
            _ => continue,
        };
        let pair = expand(arguments, quote!(#method))?;
        let parsed: syn::ItemImpl = syn::parse2(quote!(impl Example { #pair }))?;
        let Some(syn::ImplItem::Fn(mut selected)) = parsed.items.last().cloned() else {
            return Err(syn::Error::new_spanned(
                method,
                "missing selected batch companion",
            ));
        };
        // This crate path is derived from SumcheckBatch's existing crate option.
        let offset = usize::from(matches!(
            selected.sig.inputs.first(),
            Some(FnArg::Receiver(_))
        ));
        let Some(argument) = selected.sig.inputs.iter_mut().nth(offset) else {
            return Err(syn::Error::new_spanned(
                method,
                "missing recipe context argument",
            ));
        };
        *argument = parse_quote!(__recipe_context: &#krate::immutable_recipes::RecipeContext<'_>);
        selected
            .attrs
            .push(parse_quote!(#[cfg(feature = "immutable-recipes")]));
        companions.push(selected);
    }
    let mut selected_impl = item;
    selected_impl.items = companions.into_iter().map(syn::ImplItem::Fn).collect();
    selected_impl
        .attrs
        .push(parse_quote!(#[cfg(feature = "immutable-recipes")]));
    Ok(quote!(#driver #selected_impl))
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "literal syntax fixtures must parse")]
mod tests {
    use super::expand;
    use quote::quote;

    #[test]
    fn trailing_callback_comma_produces_two_parseable_methods() {
        let original = quote! {
            fn expected_output(&self) -> Result<F, Error> {
                self.symbolic().output_expression::<F>().try_evaluate(
                    |id| opening(id), |id| challenge(id), |id| derived(id),
                )
            }
        };
        let expanded = expand(quote!(output = 1), original).expect("literal parser fixture");
        let implementation: syn::ItemImpl =
            syn::parse2(quote!(impl Fixture { #expanded })).expect("generated methods parse");
        assert_eq!(implementation.items.len(), 2);
        let Some(syn::ImplItem::Fn(selected)) = implementation.items.last() else {
            panic!("selected method missing");
        };
        assert_eq!(selected.sig.ident, "expected_output_with_recipes");
        assert_eq!(selected.sig.inputs.len(), 2);
    }

    #[test]
    fn undeclared_or_duplicate_seams_reject_generation() {
        let original = quote! {
            fn input_claim(&self) -> Result<F, Error> {
                self.symbolic().input_expression::<F>().try_evaluate(a, b, c)
            }
        };
        assert!(expand(quote!(), original.clone()).is_err());
        assert!(expand(quote!(input = 2), original.clone()).is_err());
        assert!(expand(quote!(input = 1, input = 1), original.clone()).is_err());
        assert!(expand(quote!(unknown = 1), original).is_err());
    }
}
