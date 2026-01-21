use proc_macro2::TokenStream;
use quote::quote;
use syn::{Expr, Ident, ItemImpl, Member};

use super::parse::{ClaimSpec, ClaimSpecContent, PolynomialRef, SumcheckClaimsAttr};

pub fn generate(attr: SumcheckClaimsAttr, impl_block: ItemImpl) -> syn::Result<TokenStream> {
    let self_ty = &impl_block.self_ty;
    let (impl_generics, _, where_clause) = impl_block.generics.split_for_impl();

    // Extract the field type parameter F from the impl block
    let field_ty = extract_field_type(&impl_block)?;

    let input_methods = generate_input_methods(&attr.input_claim, &field_ty)?;
    let output_methods = generate_output_methods(&attr.output_claim, &field_ty)?;

    // Keep the original impl block items
    let original_items = impl_block.items.iter();

    Ok(quote! {
        impl #impl_generics SumcheckInstanceParams<#field_ty> for #self_ty #where_clause {
            #input_methods
            #(#original_items)*
        }

        impl #impl_generics SumcheckClaims<#field_ty> for #self_ty #where_clause {
            #output_methods
        }
    })
}

fn extract_field_type(impl_block: &ItemImpl) -> syn::Result<syn::Type> {
    // Try to extract F from Self type like `RegistersClaimReductionSumcheckParams<F>`
    if let syn::Type::Path(type_path) = impl_block.self_ty.as_ref() {
        if let Some(segment) = type_path.path.segments.last() {
            if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                if let Some(syn::GenericArgument::Type(ty)) = args.args.first() {
                    return Ok(ty.clone());
                }
            }
        }
    }

    Err(syn::Error::new_spanned(
        &impl_block.self_ty,
        "expected type with generic parameter like MyType<F>",
    ))
}

fn generate_input_methods(spec: &ClaimSpec, field_ty: &syn::Type) -> syn::Result<TokenStream> {
    match spec {
        ClaimSpec::None => Ok(quote! {
            fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<#field_ty>) -> #field_ty {
                #field_ty::zero()
            }

            fn input_claim_constraint(&self) -> Option<InputClaimConstraint> {
                None
            }

            fn input_constraint_challenge_values(
                &self,
                _accumulator: &dyn OpeningAccumulator<#field_ty>,
            ) -> Vec<#field_ty> {
                Vec::new()
            }
        }),
        ClaimSpec::Some(content) => {
            let claim_fn = generate_input_claim_fn(content.as_ref(), field_ty)?;
            let constraint_fn = generate_input_constraint_fn(content.as_ref())?;
            let challenge_values_fn =
                generate_input_challenge_values_fn(content.as_ref(), field_ty)?;

            Ok(quote! {
                #claim_fn
                #constraint_fn
                #challenge_values_fn
            })
        }
    }
}

fn generate_output_methods(spec: &ClaimSpec, field_ty: &syn::Type) -> syn::Result<TokenStream> {
    match spec {
        ClaimSpec::None => Ok(quote! {
            fn expected_output_claim(
                &self,
                _accumulator: &dyn OpeningAccumulator<#field_ty>,
                _sumcheck_challenges: &[<#field_ty as JoltField>::Challenge],
            ) -> #field_ty {
                #field_ty::zero()
            }

            fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
                None
            }

            fn output_constraint_challenge_values(
                &self,
                _sumcheck_challenges: &[<#field_ty as JoltField>::Challenge],
            ) -> Vec<#field_ty> {
                Vec::new()
            }
        }),
        ClaimSpec::Some(content) => {
            let claim_fn = generate_output_claim_fn(content.as_ref(), field_ty)?;
            let constraint_fn = generate_output_constraint_fn(content.as_ref())?;
            let challenge_values_fn =
                generate_output_challenge_values_fn(content.as_ref(), field_ty)?;

            Ok(quote! {
                #claim_fn
                #constraint_fn
                #challenge_values_fn
            })
        }
    }
}

fn generate_input_claim_fn(
    content: &ClaimSpecContent,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    let opening_fetches = generate_opening_fetches(&content.openings);
    let expr = &content.expr;

    Ok(quote! {
        fn input_claim(&self, accumulator: &dyn OpeningAccumulator<#field_ty>) -> #field_ty {
            #opening_fetches
            #expr
        }
    })
}

fn generate_output_claim_fn(
    content: &ClaimSpecContent,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    let point_fetches = generate_point_fetches(&content.points);
    let opening_fetches = generate_opening_fetches(&content.openings);
    let derived_bindings = generate_derived_bindings(&content.derived);
    let expr = &content.expr;

    Ok(quote! {
        fn expected_output_claim(
            &self,
            accumulator: &dyn OpeningAccumulator<#field_ty>,
            sumcheck_challenges: &[<#field_ty as JoltField>::Challenge],
        ) -> #field_ty {
            let opening_point = self.normalize_opening_point(sumcheck_challenges);
            #point_fetches
            #opening_fetches
            #derived_bindings
            #expr
        }
    })
}

fn generate_opening_fetches(openings: &[(Ident, PolynomialRef)]) -> TokenStream {
    let fetches: Vec<_> = openings
        .iter()
        .map(|(name, poly_ref)| {
            let poly = &poly_ref.poly;
            let stage = &poly_ref.stage;
            quote! {
                let (_, #name) = accumulator.get_virtual_polynomial_opening(#poly, #stage);
            }
        })
        .collect();

    quote! { #(#fetches)* }
}

fn generate_point_fetches(points: &[(Ident, PolynomialRef)]) -> TokenStream {
    let fetches: Vec<_> = points
        .iter()
        .map(|(name, poly_ref)| {
            let poly = &poly_ref.poly;
            let stage = &poly_ref.stage;
            quote! {
                let (#name, _) = accumulator.get_virtual_polynomial_opening(#poly, #stage);
            }
        })
        .collect();

    quote! { #(#fetches)* }
}

fn generate_derived_bindings(derived: &[(Ident, Expr)]) -> TokenStream {
    let bindings: Vec<_> = derived
        .iter()
        .map(|(name, expr)| {
            quote! {
                let #name = #expr;
            }
        })
        .collect();

    quote! { #(#bindings)* }
}

fn generate_input_constraint_fn(content: &ClaimSpecContent) -> syn::Result<TokenStream> {
    let analysis = analyze_expr(&content.expr, &content.openings)?;
    let terms = generate_input_constraint_terms(&analysis, &content.openings)?;

    Ok(quote! {
        fn input_claim_constraint(&self) -> Option<InputClaimConstraint> {
            let terms = vec![#terms];
            Some(InputClaimConstraint::sum_of_products(terms))
        }
    })
}

fn generate_output_constraint_fn(content: &ClaimSpecContent) -> syn::Result<TokenStream> {
    let analysis = analyze_expr(&content.expr, &content.openings)?;
    let has_multiplier = !content.derived.is_empty();
    let terms = generate_output_constraint_terms(&analysis, &content.openings, has_multiplier)?;

    Ok(quote! {
        fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
            let terms = vec![#terms];
            Some(OutputClaimConstraint::sum_of_products(terms))
        }
    })
}

fn generate_input_challenge_values_fn(
    content: &ClaimSpecContent,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    let analysis = analyze_expr(&content.expr, &content.openings)?;
    let challenge_exprs: Vec<_> = analysis
        .challenge_refs
        .iter()
        .map(|field| quote! { self.#field })
        .collect();

    Ok(quote! {
        fn input_constraint_challenge_values(
            &self,
            _accumulator: &dyn OpeningAccumulator<#field_ty>,
        ) -> Vec<#field_ty> {
            vec![#(#challenge_exprs),*]
        }
    })
}

fn generate_output_challenge_values_fn(
    content: &ClaimSpecContent,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    // For output, we include derived values first, then self.field challenges
    // Points are bound from self instead of accumulator (no accumulator in this fn)
    let point_bindings = generate_self_point_bindings(&content.points);
    let derived_bindings = generate_derived_bindings(&content.derived);
    let derived_names: Vec<_> = content.derived.iter().map(|(name, _)| name).collect();

    let analysis = analyze_expr(&content.expr, &content.openings)?;
    let challenge_exprs: Vec<_> = analysis
        .challenge_refs
        .iter()
        .map(|field| quote! { self.#field })
        .collect();

    Ok(quote! {
        fn output_constraint_challenge_values(
            &self,
            sumcheck_challenges: &[<#field_ty as JoltField>::Challenge],
        ) -> Vec<#field_ty> {
            let opening_point = self.normalize_opening_point(sumcheck_challenges);
            #point_bindings
            #derived_bindings
            vec![#(#derived_names,)* #(#challenge_exprs),*]
        }
    })
}

/// Generate bindings from self fields instead of accumulator (for challenge_values fn)
fn generate_self_point_bindings(points: &[(Ident, PolynomialRef)]) -> TokenStream {
    let bindings: Vec<_> = points
        .iter()
        .map(|(name, _poly_ref)| {
            // Bind from self.field_name instead of accumulator
            quote! {
                let #name = &self.#name;
            }
        })
        .collect();

    quote! { #(#bindings)* }
}

#[derive(Debug, Default)]
struct ExprAnalysis {
    opening_refs: Vec<Ident>,
    challenge_refs: Vec<Ident>,
}

fn analyze_expr(expr: &Expr, openings: &[(Ident, PolynomialRef)]) -> syn::Result<ExprAnalysis> {
    let mut analysis = ExprAnalysis::default();
    let opening_names: Vec<_> = openings.iter().map(|(name, _)| name.to_string()).collect();

    collect_expr_info(expr, &opening_names, &mut analysis);

    Ok(analysis)
}

fn collect_expr_info(expr: &Expr, opening_names: &[String], analysis: &mut ExprAnalysis) {
    match expr {
        Expr::Binary(binary) => {
            collect_expr_info(&binary.left, opening_names, analysis);
            collect_expr_info(&binary.right, opening_names, analysis);
        }
        Expr::Paren(paren) => {
            collect_expr_info(&paren.expr, opening_names, analysis);
        }
        Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                let name = ident.to_string();
                if opening_names.contains(&name)
                    && !analysis.opening_refs.iter().any(|i| i == ident)
                {
                    analysis.opening_refs.push(ident.clone());
                }
            }
        }
        Expr::Field(field) => {
            // self.field references become challenges
            if let Expr::Path(path) = field.base.as_ref() {
                if path.path.is_ident("self") {
                    if let Member::Named(field_name) = &field.member {
                        if !analysis.challenge_refs.iter().any(|i| i == field_name) {
                            analysis.challenge_refs.push(field_name.clone());
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// Generate constraint terms for input claims (simple linear: a + γ*b + γ²*c)
fn generate_input_constraint_terms(
    analysis: &ExprAnalysis,
    openings: &[(Ident, PolynomialRef)],
) -> syn::Result<TokenStream> {
    let mut terms = Vec::new();
    let opening_map: std::collections::HashMap<_, _> = openings.iter().cloned().collect();

    // First opening has coefficient 1
    if let Some(first_opening) = analysis.opening_refs.first() {
        if let Some(poly_ref) = opening_map.get(first_opening) {
            let poly = &poly_ref.poly;
            let stage = &poly_ref.stage;
            terms.push(quote! {
                ProductTerm::single(ValueSource::Opening(
                    OpeningId::Virtual(#poly, #stage)
                ))
            });
        }
    }

    // Subsequent openings are scaled by challenges
    for (i, opening) in analysis.opening_refs.iter().skip(1).enumerate() {
        if let Some(poly_ref) = opening_map.get(opening) {
            let poly = &poly_ref.poly;
            let stage = &poly_ref.stage;
            let challenge_idx = i;
            terms.push(quote! {
                ProductTerm::scaled(
                    ValueSource::Challenge(#challenge_idx),
                    vec![ValueSource::Opening(OpeningId::Virtual(#poly, #stage))]
                )
            });
        }
    }

    Ok(quote! { #(#terms),* })
}

/// Generate constraint terms for output claims with multiplier (eq_eval * (a + γ*b + γ²*c))
fn generate_output_constraint_terms(
    analysis: &ExprAnalysis,
    openings: &[(Ident, PolynomialRef)],
    has_multiplier: bool,
) -> syn::Result<TokenStream> {
    let mut terms = Vec::new();
    let opening_map: std::collections::HashMap<_, _> = openings.iter().cloned().collect();

    if has_multiplier {
        // Pattern: eq_eval * (a + γ*b + γ²*c)
        // Challenges: [eq_eval, γ, γ²]
        // Terms: product([eq_eval, a]), product([eq_eval, γ, b]), product([eq_eval, γ², c])

        // First opening: eq_eval * opening
        if let Some(first_opening) = analysis.opening_refs.first() {
            if let Some(poly_ref) = opening_map.get(first_opening) {
                let poly = &poly_ref.poly;
                let stage = &poly_ref.stage;
                terms.push(quote! {
                    ProductTerm::product(vec![
                        ValueSource::Challenge(0),
                        ValueSource::Opening(OpeningId::Virtual(#poly, #stage))
                    ])
                });
            }
        }

        // Subsequent openings: eq_eval * challenge * opening
        for (i, opening) in analysis.opening_refs.iter().skip(1).enumerate() {
            if let Some(poly_ref) = opening_map.get(opening) {
                let poly = &poly_ref.poly;
                let stage = &poly_ref.stage;
                // Challenge indices: 0 = eq_eval, 1 = γ, 2 = γ², ...
                let challenge_idx = i + 1;
                terms.push(quote! {
                    ProductTerm::product(vec![
                        ValueSource::Challenge(0),
                        ValueSource::Challenge(#challenge_idx),
                        ValueSource::Opening(OpeningId::Virtual(#poly, #stage))
                    ])
                });
            }
        }
    } else {
        // No multiplier - same as input constraints
        return generate_input_constraint_terms(analysis, openings);
    }

    Ok(quote! { #(#terms),* })
}
