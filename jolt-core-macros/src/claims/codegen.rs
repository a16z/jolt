use proc_macro2::TokenStream;
use quote::quote;
use syn::{Expr, Ident, ItemImpl, Member};

use super::parse::{ClaimSpecContent, PolynomialRef, SumcheckClaimsAttr};

pub fn generate(attr: SumcheckClaimsAttr, impl_block: ItemImpl) -> syn::Result<TokenStream> {
    let self_ty = &impl_block.self_ty;
    let (impl_generics, _, where_clause) = impl_block.generics.split_for_impl();

    let field_ty = extract_field_type(&impl_block)?;
    let original_items: Vec<_> = impl_block.items.iter().collect();

    let input_methods = generate_input_methods(&attr.input_claim, &field_ty)?;
    let output_methods = generate_output_methods(&attr.output_claim, &field_ty)?;

    Ok(quote! {
        impl #impl_generics SumcheckInstanceParams<#field_ty> for #self_ty #where_clause {
            #input_methods
            #output_methods
            #(#original_items)*
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

fn generate_input_methods(
    content: &ClaimSpecContent,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    let claim_fn = generate_input_claim_fn(content, field_ty)?;
    let constraint_fn = generate_input_constraint_fn(content)?;
    let challenge_values_fn = generate_input_challenge_values_fn(content, field_ty)?;

    Ok(quote! {
        #claim_fn
        #constraint_fn
        #challenge_values_fn
    })
}

fn generate_output_methods(
    content: &ClaimSpecContent,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    let claim_fn = generate_output_claim_fn(content, field_ty)?;
    let constraint_fn = generate_output_constraint_fn(content)?;
    let challenge_values_fn = generate_output_challenge_values_fn(content, field_ty)?;

    Ok(quote! {
        #claim_fn
        #constraint_fn
        #challenge_values_fn
    })
}

fn generate_input_claim_fn(
    content: &ClaimSpecContent,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    let opening_fetches = generate_opening_fetches(&content.openings);
    let expr = &content.expr;

    // If no openings, we don't need accumulator
    let accumulator_param = if content.openings.is_empty() {
        quote! { _accumulator }
    } else {
        quote! { accumulator }
    };

    Ok(quote! {
        fn input_claim(&self, #accumulator_param: &dyn OpeningAccumulator<#field_ty>) -> #field_ty {
            #opening_fetches
            #expr
        }
    })
}

fn generate_output_claim_fn(
    content: &ClaimSpecContent,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    // Handle for_each with product_of (sum of products pattern)
    if let (Some(for_each), Some(product_of)) = (&content.for_each, &content.product_of) {
        return generate_sum_of_products_output_claim_fn(content, for_each, product_of, field_ty);
    }

    // Handle for_each alone (simple iteration)
    if let Some(for_each) = &content.for_each {
        return generate_for_each_output_claim_fn(content, for_each, field_ty);
    }

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

fn generate_for_each_output_claim_fn(
    content: &ClaimSpecContent,
    for_each: &super::parse::ForEachSpec,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    let outer_var = &for_each.var;
    let outer_len = &for_each.len_expr;

    let (opening_name, poly_ref) = content
        .openings
        .first()
        .ok_or_else(|| syn::Error::new_spanned(&content.expr, "for_each requires an opening"))?;

    let poly = &poly_ref.poly;
    let stage = &poly_ref.stage;
    let method = get_opening_method(poly_ref);

    let (coeff_name, coeff_expr) = content.derived.first().ok_or_else(|| {
        syn::Error::new_spanned(&content.expr, "for_each requires a derived expression")
    })?;

    let expr = &content.expr;

    Ok(quote! {
        fn expected_output_claim(
            &self,
            accumulator: &dyn OpeningAccumulator<#field_ty>,
            sumcheck_challenges: &[<#field_ty as JoltField>::Challenge],
        ) -> #field_ty {
            let opening_point = self.normalize_opening_point(sumcheck_challenges);
            let mut result = #field_ty::zero();
            for #outer_var in 0..#outer_len {
                let (_, #opening_name) = accumulator.#method(#poly, #stage);
                let #coeff_name = #coeff_expr;
                result += #expr;
            }
            result
        }
    })
}

/// Generate output claim for sum of products pattern:
/// `Σ_i coeff_i * ∏_j opening_{f(i,j)}`
fn generate_sum_of_products_output_claim_fn(
    content: &ClaimSpecContent,
    for_each: &super::parse::ForEachSpec,
    product_of: &super::parse::ForEachSpec,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    let outer_var = &for_each.var;
    let outer_len = &for_each.len_expr;
    let inner_var = &product_of.var;
    let inner_len = &product_of.len_expr;

    let (opening_name, poly_ref) = content
        .openings
        .first()
        .ok_or_else(|| syn::Error::new_spanned(&content.expr, "product_of requires an opening"))?;

    let poly = &poly_ref.poly;
    let stage = &poly_ref.stage;
    let method = get_opening_method(poly_ref);

    let derived_bindings = generate_derived_bindings(&content.derived);
    let expr = &content.expr;

    Ok(quote! {
        fn expected_output_claim(
            &self,
            accumulator: &dyn OpeningAccumulator<#field_ty>,
            sumcheck_challenges: &[<#field_ty as JoltField>::Challenge],
        ) -> #field_ty {
            let opening_point = self.normalize_opening_point(sumcheck_challenges);
            #derived_bindings
            let mut result = #field_ty::zero();
            for #outer_var in 0..#outer_len {
                let mut product = #field_ty::one();
                for #inner_var in 0..#inner_len {
                    let (_, #opening_name) = accumulator.#method(#poly, #stage);
                    product *= #opening_name;
                }
                result += #expr;
            }
            result
        }
    })
}

fn generate_opening_fetches(openings: &[(Ident, PolynomialRef)]) -> TokenStream {
    let fetches: Vec<_> = openings
        .iter()
        .map(|(name, poly_ref)| {
            let poly = &poly_ref.poly;
            let stage = &poly_ref.stage;
            let method = get_opening_method(poly_ref);
            quote! {
                let (_, #name) = accumulator.#method(#poly, #stage);
            }
        })
        .collect();

    quote! { #(#fetches)* }
}

fn get_opening_method(poly_ref: &PolynomialRef) -> syn::Ident {
    let poly_type = poly_ref
        .poly
        .segments
        .first()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();

    let method_name = match poly_type.as_str() {
        "VirtualPolynomial" => "get_virtual_polynomial_opening",
        "CommittedPolynomial" => "get_committed_polynomial_opening",
        _ => "get_virtual_polynomial_opening",
    };

    syn::Ident::new(method_name, proc_macro2::Span::call_site())
}

fn get_opening_id_variant(poly_ref: &PolynomialRef) -> TokenStream {
    let poly_type = poly_ref
        .poly
        .segments
        .first()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();

    let poly = &poly_ref.poly;
    let stage = &poly_ref.stage;

    match poly_type.as_str() {
        "CommittedPolynomial" => quote! { OpeningId::Committed(#poly, #stage) },
        _ => quote! { OpeningId::Virtual(#poly, #stage) },
    }
}

fn generate_point_fetches(points: &[(Ident, PolynomialRef)]) -> TokenStream {
    let fetches: Vec<_> = points
        .iter()
        .map(|(name, poly_ref)| {
            let poly = &poly_ref.poly;
            let stage = &poly_ref.stage;
            let method = get_opening_method(poly_ref);
            quote! {
                let (#name, _) = accumulator.#method(#poly, #stage);
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
    // If no openings, there's no constraint (pure self.field expression)
    if content.openings.is_empty() {
        return Ok(quote! {
            fn input_claim_constraint(&self) -> Option<InputClaimConstraint> {
                None
            }
        });
    }

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
    // Handle for_each with product_of (sum of products pattern)
    if let (Some(for_each), Some(product_of)) = (&content.for_each, &content.product_of) {
        return generate_sum_of_products_constraint_fn(content, for_each, product_of);
    }

    // Handle for_each alone
    if let Some(for_each) = &content.for_each {
        return generate_for_each_constraint_fn(content, for_each);
    }

    let analysis = analyze_expr(&content.expr, &content.openings)?;
    let has_derived = !content.derived.is_empty();
    let single_opening = analysis.opening_refs.len() == 1;
    let has_quadratic = has_quadratic_opening(&content.expr, &content.openings);

    let terms = if has_derived && single_opening && has_quadratic {
        // Pattern: derived * (o*o - o) → Challenge(0)*o² + Challenge(1)*o
        generate_quadratic_minus_linear_terms(&analysis, &content.openings)?
    } else if has_derived && single_opening {
        // Pattern: derived * single_opening → scaled(Challenge(0), Opening)
        generate_scaled_single_opening_terms(&analysis, &content.openings)?
    } else if has_derived {
        // Pattern: derived * (a + γ*b + ...) → multiplier pattern
        generate_output_constraint_terms(&analysis, &content.openings, true)?
    } else {
        // Pattern: no derived → each opening scaled by its own challenge
        generate_all_scaled_terms(&analysis, &content.openings)?
    };

    Ok(quote! {
        fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
            let terms = vec![#terms];
            Some(OutputClaimConstraint::sum_of_products(terms))
        }
    })
}

fn generate_for_each_constraint_fn(
    content: &ClaimSpecContent,
    for_each: &super::parse::ForEachSpec,
) -> syn::Result<TokenStream> {
    let outer_var = &for_each.var;
    let outer_len = &for_each.len_expr;

    let (opening_name, poly_ref) = content
        .openings
        .first()
        .ok_or_else(|| syn::Error::new_spanned(&content.expr, "for_each requires an opening"))?;

    let poly = &poly_ref.poly;
    let stage = &poly_ref.stage;

    let poly_type = poly_ref
        .poly
        .segments
        .first()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();

    let opening_id_expr = match poly_type.as_str() {
        "CommittedPolynomial" => quote! { OpeningId::Committed(#poly, #stage) },
        _ => quote! { OpeningId::Virtual(#poly, #stage) },
    };

    Ok(quote! {
        fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
            let terms: Vec<ProductTerm> = (0..#outer_len)
                .map(|#outer_var| {
                    let #opening_name = #opening_id_expr;
                    ProductTerm::scaled(
                        ValueSource::Challenge(#outer_var),
                        vec![ValueSource::Opening(#opening_name)]
                    )
                })
                .collect();
            Some(OutputClaimConstraint::sum_of_products(terms))
        }
    })
}

/// Generate constraint for sum of products pattern:
/// `Σ_i Challenge(i) * ∏_j Opening(f(i,j))`
fn generate_sum_of_products_constraint_fn(
    content: &ClaimSpecContent,
    for_each: &super::parse::ForEachSpec,
    product_of: &super::parse::ForEachSpec,
) -> syn::Result<TokenStream> {
    let outer_var = &for_each.var;
    let outer_len = &for_each.len_expr;
    let inner_var = &product_of.var;
    let inner_len = &product_of.len_expr;

    let (opening_name, poly_ref) = content
        .openings
        .first()
        .ok_or_else(|| syn::Error::new_spanned(&content.expr, "product_of requires an opening"))?;

    let poly = &poly_ref.poly;
    let stage = &poly_ref.stage;

    let poly_type = poly_ref
        .poly
        .segments
        .first()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();

    let opening_id_expr = match poly_type.as_str() {
        "CommittedPolynomial" => quote! { OpeningId::Committed(#poly, #stage) },
        _ => quote! { OpeningId::Virtual(#poly, #stage) },
    };

    Ok(quote! {
        fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
            let terms: Vec<ProductTerm> = (0..#outer_len)
                .map(|#outer_var| {
                    let factors: Vec<ValueSource> = (0..#inner_len)
                        .map(|#inner_var| {
                            let #opening_name = #opening_id_expr;
                            ValueSource::Opening(#opening_name)
                        })
                        .collect();
                    ProductTerm::scaled(ValueSource::Challenge(#outer_var), factors)
                })
                .collect();
            Some(OutputClaimConstraint::sum_of_products(terms))
        }
    })
}

/// Check if expression contains o * o pattern for any opening
fn has_quadratic_opening(expr: &Expr, openings: &[(Ident, PolynomialRef)]) -> bool {
    let opening_names: Vec<_> = openings.iter().map(|(name, _)| name.to_string()).collect();
    check_quadratic_recursive(expr, &opening_names)
}

fn check_quadratic_recursive(expr: &Expr, opening_names: &[String]) -> bool {
    match expr {
        Expr::Binary(binary) => {
            // Check for o * o pattern
            if matches!(binary.op, syn::BinOp::Mul(_)) {
                if let (Expr::Path(left), Expr::Path(right)) =
                    (binary.left.as_ref(), binary.right.as_ref())
                {
                    if let (Some(left_ident), Some(right_ident)) =
                        (left.path.get_ident(), right.path.get_ident())
                    {
                        let left_name = left_ident.to_string();
                        let right_name = right_ident.to_string();
                        if left_name == right_name && opening_names.contains(&left_name) {
                            return true;
                        }
                    }
                }
            }
            check_quadratic_recursive(&binary.left, opening_names)
                || check_quadratic_recursive(&binary.right, opening_names)
        }
        Expr::Paren(paren) => check_quadratic_recursive(&paren.expr, opening_names),
        _ => false,
    }
}

/// Generate terms for o*o - o pattern: Challenge(0)*o² + Challenge(1)*o
fn generate_quadratic_minus_linear_terms(
    analysis: &ExprAnalysis,
    openings: &[(Ident, PolynomialRef)],
) -> syn::Result<TokenStream> {
    let opening_map: std::collections::HashMap<_, _> = openings.iter().cloned().collect();

    if let Some(first_opening) = analysis.opening_refs.first() {
        if let Some(poly_ref) = opening_map.get(first_opening) {
            let opening_id = get_opening_id_variant(poly_ref);
            return Ok(quote! {
                ProductTerm::scaled(
                    ValueSource::Challenge(0),
                    vec![ValueSource::Opening(#opening_id), ValueSource::Opening(#opening_id)]
                ),
                ProductTerm::scaled(
                    ValueSource::Challenge(1),
                    vec![ValueSource::Opening(#opening_id)]
                )
            });
        }
    }

    Ok(quote! {})
}

/// Generate terms for single opening scaled by derived value: Challenge(0) * Opening
fn generate_scaled_single_opening_terms(
    analysis: &ExprAnalysis,
    openings: &[(Ident, PolynomialRef)],
) -> syn::Result<TokenStream> {
    let opening_map: std::collections::HashMap<_, _> = openings.iter().cloned().collect();

    if let Some(first_opening) = analysis.opening_refs.first() {
        if let Some(poly_ref) = opening_map.get(first_opening) {
            let opening_id = get_opening_id_variant(poly_ref);
            return Ok(quote! {
                ProductTerm::scaled(
                    ValueSource::Challenge(0),
                    vec![ValueSource::Opening(#opening_id)]
                )
            });
        }
    }

    Ok(quote! {})
}

/// Generate terms where each opening has its own challenge: Σ Challenge(i) * Opening(i)
fn generate_all_scaled_terms(
    analysis: &ExprAnalysis,
    openings: &[(Ident, PolynomialRef)],
) -> syn::Result<TokenStream> {
    let mut terms = Vec::new();
    let opening_map: std::collections::HashMap<_, _> = openings.iter().cloned().collect();

    for (i, opening) in analysis.opening_refs.iter().enumerate() {
        if let Some(poly_ref) = opening_map.get(opening) {
            let opening_id = get_opening_id_variant(poly_ref);
            terms.push(quote! {
                ProductTerm::scaled(
                    ValueSource::Challenge(#i),
                    vec![ValueSource::Opening(#opening_id)]
                )
            });
        }
    }

    Ok(quote! { #(#terms),* })
}

fn generate_input_challenge_values_fn(
    content: &ClaimSpecContent,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    // If no openings, return empty vec (no constraint)
    if content.openings.is_empty() {
        return Ok(quote! {
            fn input_constraint_challenge_values(
                &self,
                _accumulator: &dyn OpeningAccumulator<#field_ty>,
            ) -> Vec<#field_ty> {
                Vec::new()
            }
        });
    }

    let analysis = analyze_expr(&content.expr, &content.openings)?;

    // Combine simple field refs and indexed refs
    let simple_exprs: Vec<_> = analysis
        .challenge_refs
        .iter()
        .map(|field| quote! { self.#field })
        .collect();

    let indexed_exprs: Vec<_> = analysis
        .indexed_challenge_refs
        .iter()
        .map(|(array, idx)| quote! { self.#array[#idx] })
        .collect();

    Ok(quote! {
        fn input_constraint_challenge_values(
            &self,
            _accumulator: &dyn OpeningAccumulator<#field_ty>,
        ) -> Vec<#field_ty> {
            vec![#(#simple_exprs,)* #(#indexed_exprs),*]
        }
    })
}

fn generate_output_challenge_values_fn(
    content: &ClaimSpecContent,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    // Handle for_each with product_of (sum of products pattern)
    if let (Some(for_each), Some(_product_of)) = (&content.for_each, &content.product_of) {
        return generate_sum_of_products_challenge_values_fn(content, for_each, field_ty);
    }

    // Handle for_each alone
    if let Some(for_each) = &content.for_each {
        return generate_for_each_challenge_values_fn(content, for_each, field_ty);
    }

    let analysis = analyze_expr(&content.expr, &content.openings)?;
    let has_derived = !content.derived.is_empty();
    let single_opening = analysis.opening_refs.len() == 1;
    let has_quadratic = has_quadratic_opening(&content.expr, &content.openings);
    let num_derived = content.derived.len();
    let num_openings = analysis.opening_refs.len();

    let point_bindings = generate_self_point_bindings(&content.points);
    let derived_bindings = generate_derived_bindings(&content.derived);
    let derived_names: Vec<_> = content.derived.iter().map(|(name, _)| name).collect();

    // Combine simple field refs and indexed refs
    let simple_exprs: Vec<_> = analysis
        .challenge_refs
        .iter()
        .map(|field| quote! { self.#field })
        .collect();

    let indexed_exprs: Vec<_> = analysis
        .indexed_challenge_refs
        .iter()
        .map(|(array, idx)| quote! { self.#array[#idx] })
        .collect();

    let values_expr = if has_derived && single_opening && has_quadratic {
        // Quadratic pattern: derived * (o*o - o) → [derived, -derived]
        let first_derived = &derived_names[0];
        quote! { vec![#first_derived, -#first_derived] }
    } else if has_derived && single_opening {
        // Single derived value as the only challenge
        quote! { vec![#(#derived_names),*] }
    } else if has_derived && num_derived == num_openings {
        // One derived per opening - each derived IS the challenge
        quote! { vec![#(#derived_names),*] }
    } else if has_derived {
        // Multiplier pattern: [derived, self.gamma, self.gamma_sqr, ...]
        quote! { vec![#(#derived_names,)* #(#simple_exprs,)* #(#indexed_exprs),*] }
    } else {
        // All-scaled pattern: challenges from self.field or self.array[i]
        quote! { vec![#(#simple_exprs,)* #(#indexed_exprs),*] }
    };

    Ok(quote! {
        fn output_constraint_challenge_values(
            &self,
            sumcheck_challenges: &[<#field_ty as JoltField>::Challenge],
        ) -> Vec<#field_ty> {
            let opening_point = self.normalize_opening_point(sumcheck_challenges);
            #point_bindings
            #derived_bindings
            #values_expr
        }
    })
}

fn generate_for_each_challenge_values_fn(
    content: &ClaimSpecContent,
    for_each: &super::parse::ForEachSpec,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    let outer_var = &for_each.var;
    let outer_len = &for_each.len_expr;

    let (_, challenge_expr) = content.derived.first().ok_or_else(|| {
        syn::Error::new_spanned(
            &content.expr,
            "for_each requires a derived expression for challenge computation",
        )
    })?;

    Ok(quote! {
        fn output_constraint_challenge_values(
            &self,
            sumcheck_challenges: &[<#field_ty as JoltField>::Challenge],
        ) -> Vec<#field_ty> {
            let opening_point = self.normalize_opening_point(sumcheck_challenges);
            (0..#outer_len)
                .map(|#outer_var| #challenge_expr)
                .collect()
        }
    })
}

/// Generate challenge values for sum of products pattern.
/// Uses the derived expression to compute Challenge(i) for each outer iteration.
fn generate_sum_of_products_challenge_values_fn(
    content: &ClaimSpecContent,
    for_each: &super::parse::ForEachSpec,
    field_ty: &syn::Type,
) -> syn::Result<TokenStream> {
    let outer_var = &for_each.var;
    let outer_len = &for_each.len_expr;

    let (_, challenge_expr) = content.derived.first().ok_or_else(|| {
        syn::Error::new_spanned(
            &content.expr,
            "product_of requires a derived expression for challenge computation",
        )
    })?;

    Ok(quote! {
        fn output_constraint_challenge_values(
            &self,
            sumcheck_challenges: &[<#field_ty as JoltField>::Challenge],
        ) -> Vec<#field_ty> {
            let opening_point = self.normalize_opening_point(sumcheck_challenges);
            (0..#outer_len)
                .map(|#outer_var| #challenge_expr)
                .collect()
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
    /// Simple field references: self.gamma, self.gamma_sqr
    challenge_refs: Vec<Ident>,
    /// Array index references: self.gamma_powers[0], self.gamma_powers[1]
    /// Stored as (array_name, index_expr)
    indexed_challenge_refs: Vec<(Ident, Expr)>,
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
        Expr::Index(index) => {
            // self.array[i] references - handle array indexing
            if let Expr::Field(field) = index.expr.as_ref() {
                if let Expr::Path(path) = field.base.as_ref() {
                    if path.path.is_ident("self") {
                        if let Member::Named(array_name) = &field.member {
                            let idx_expr = (*index.index).clone();
                            // Check if we already have this exact reference
                            let exists =
                                analysis.indexed_challenge_refs.iter().any(|(name, idx)| {
                                    name == array_name
                                        && quote!(#idx).to_string() == quote!(#idx_expr).to_string()
                                });
                            if !exists {
                                analysis
                                    .indexed_challenge_refs
                                    .push((array_name.clone(), idx_expr));
                            }
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
            let opening_id = get_opening_id_variant(poly_ref);
            terms.push(quote! {
                ProductTerm::single(ValueSource::Opening(#opening_id))
            });
        }
    }

    // Subsequent openings are scaled by challenges
    for (i, opening) in analysis.opening_refs.iter().skip(1).enumerate() {
        if let Some(poly_ref) = opening_map.get(opening) {
            let opening_id = get_opening_id_variant(poly_ref);
            let challenge_idx = i;
            terms.push(quote! {
                ProductTerm::scaled(
                    ValueSource::Challenge(#challenge_idx),
                    vec![ValueSource::Opening(#opening_id)]
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
                let opening_id = get_opening_id_variant(poly_ref);
                terms.push(quote! {
                    ProductTerm::product(vec![
                        ValueSource::Challenge(0),
                        ValueSource::Opening(#opening_id)
                    ])
                });
            }
        }

        // Subsequent openings: eq_eval * challenge * opening
        for (i, opening) in analysis.opening_refs.iter().skip(1).enumerate() {
            if let Some(poly_ref) = opening_map.get(opening) {
                let opening_id = get_opening_id_variant(poly_ref);
                // Challenge indices: 0 = eq_eval, 1 = γ, 2 = γ², ...
                let challenge_idx = i + 1;
                terms.push(quote! {
                    ProductTerm::product(vec![
                        ValueSource::Challenge(0),
                        ValueSource::Challenge(#challenge_idx),
                        ValueSource::Opening(#opening_id)
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
