//! Derive macros for jolt-verifier's per-stage sumcheck-batch aggregates.
//!
//! [`macro@SumcheckBatch`] turns a per-stage source-of-truth struct whose fields
//! are the stage's `ConcreteSumcheck` instances into the stage's aggregate claim
//! types:
//!
//! ```ignore
//! #[derive(SumcheckBatch)]
//! struct Stage5Sumchecks<F: Field> {
//!     instruction_read_raf:     InstructionReadRaf<F>,
//!     ram_ra_claim_reduction:   RamRaClaimReduction<F>,
//!     registers_val_evaluation: RegistersValEvaluation<F>,
//! }
//! ```
//!
//! generates `Stage5InputClaims<F>`, `Stage5InputPoints<F>`,
//! `Stage5OutputClaims<F>`, `Stage5OutputPoints<F>`, `Stage5Challenges<F>`, and
//! `Stage5BatchingCoefficients<F>` (the `Sumchecks` suffix is replaced by
//! `InputClaims` / `InputPoints` / `OutputClaims` / `OutputPoints` / `Challenges` /
//! `BatchingCoefficients`). The claim/point/challenge aggregates project each field
//! through the `SumcheckInputClaims` / `SumcheckInputPoints` / `SumcheckOutputClaims`
//! / `SumcheckOutputPoints` / `ConcreteSumcheckChallenges` aliases in
//! `jolt-verifier`'s `stages::relations` (the `*Claims` aggregates hold the wire
//! *values*, the `*Points` aggregates the derived opening points); the
//! `BatchingCoefficients` aggregate holds one `F` per member (the scalar the batched
//! verifier draws for that instance). A source field `Option<Instance>` becomes an
//! `Option<projection>` (a conditional instance), chained only when `Some`. The
//! generated drivers treat a *present* instance with an absent input / challenge /
//! claim / point / coefficient cell as a wiring error (attributed to the member's
//! relation id), never as an absent member: silently skipping such a member would
//! desynchronize the Fiat-Shamir transcript (a missed absorb or coefficient draw)
//! or drop a term from the final-claim fold.
//!
//! The projections are emitted *without* `Instance: ConcreteSumcheck<F>`
//! where-bounds on purpose: such a bound would make the compiler treat each
//! projection as an opaque associated type (from the bound) rather than
//! normalizing it to the concrete per-relation claim struct, which would break
//! the `Clone` / `Debug` / `serde` derives. Because each instance field is a
//! concrete type with a concrete `ConcreteSumcheck` impl, the projection
//! normalizes to the concrete per-relation claim struct, so the plain derives
//! apply.
//!
//! The set of generated *delegated impls* is intentionally minimal and grows as
//! the migration requires. Currently emitted for every stage:
//!
//! - the Fiat-Shamir opening plumbing (`opening_values` / `append_to_transcript`
//!   on the `OutputClaims` (values) aggregate, delegating to each member in
//!   declaration order); and
//! - the per-instance driver method on the source `StageNSumchecks` struct —
//!   `draw_challenges` (draw each member's challenges into `StageNChallenges`).
//!
//! Opt-in per method via `#[sumcheck_batch(...)]` flags, emitted on the source
//! `StageNSumchecks` struct only for stages that request them:
//!
//! - `verify_clear` — the clear-path batched-verify driver: fold the members into
//!   one combined claim (max `(num_vars, degree)`, absorb each `input_claim`, draw
//!   the batching coefficients, random-linear-combine the padded sums) and reduce it
//!   through the single-instance `SumcheckProof::verify_compressed_boolean`. The
//!   batching lives here rather than in `jolt-sumcheck`'s `BatchedSumcheckVerifier`.
//! - `verify_zk` — the ZK-path driver: fold the members' dimensions, draw the
//!   batching coefficients, and check committed consistency through
//!   `SumcheckProof::verify_committed_consistency_dims`.
//! - `derive_opening_points` — slice each member's opening point from the batch
//!   challenge vector (its length-`rounds` suffix, per the front-loaded batching
//!   layout) and map it through the member's `ConcreteSumcheck::derive_opening_points`
//!   into the stage's `OutputPoints` aggregate. Takes the challenge vector as `&[F]`,
//!   so the clear and ZK paths each pass their own.
//! - `expected_final_claim` — fold the members' `ConcreteSumcheck::expected_output`
//!   with the batch coefficients (`StageNBatchingCoefficients`) into the final claim
//!   the reduction is checked against. `verify_clear` returns the coefficients inside
//!   `StageNClearBatch { reduction, coefficients }`.
//! - `output_shape` — `output_claim_count` (total produced openings, e.g. the ZK
//!   commitment count) and `validate_output_claims` (assert the proof's output claims
//!   match the dims-derived shape), both via each member's
//!   `SymbolicSumcheck::expected_output_openings` (derived from its output `Expr`).
//!
//! The `verify_*` drivers never name `SumcheckClaim` / `SumcheckStatement`; those
//! stay internal to `jolt-sumcheck`.
//!
//! See `specs/sumcheck-batch-derive.md`.
//!
//! The struct-level helper attribute `#[sumcheck_batch(custom_opening_values)]`
//! suppresses *only* that generated `opening_values` / `append_to_transcript`
//! inherent impl, leaving the aggregate structs (and their derives / serde)
//! untouched. An alias-curated stage (one whose canonical opening order skips
//! cross-relation aliased openings) uses it to supply its own consistent
//! `opening_values` / `append_to_transcript` (and `validate`) as an inherent impl.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, Attribute, Data, DeriveInput, Fields, GenericArgument, GenericParam, Ident,
    Meta, PathArguments, Token, Type,
};

/// Generate a stage's aggregate claim types (`StageN{Input,Output}{Claims,Points}`
/// / `StageNChallenges`) from a struct of `ConcreteSumcheck` instances. See the
/// crate-level docs.
///
/// The struct-level helper attribute `#[sumcheck_batch(custom_opening_values)]`
/// opts a stage out of the generated `opening_values` / `append_to_transcript`
/// inherent impl on the `OutputClaims` (values) aggregate, so an alias-curated
/// stage can supply its own (e.g. one that skips cross-relation aliased openings).
/// The aggregate structs and their derives are emitted unchanged.
#[proc_macro_derive(SumcheckBatch, attributes(sumcheck_batch))]
pub fn derive_sumcheck_batch(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// One instance field of the source struct: its name and `ConcreteSumcheck`
/// instance type. `is_option` records a conditional instance (`Option<Instance>`),
/// whose projections become `Option<..>` and chain only when present.
struct InstanceField {
    ident: Ident,
    instance: Type,
    is_option: bool,
}

fn expand(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let vis = &input.vis;

    let base = name
        .to_string()
        .strip_suffix("Sumchecks")
        .map(str::to_string)
        .filter(|base| !base.is_empty())
        .ok_or_else(|| {
            syn::Error::new_spanned(
                name,
                "a #[derive(SumcheckBatch)] struct must be named `<Stage>Sumchecks`",
            )
        })?;
    let input_claims_name = format_ident!("{base}InputClaims");
    let input_points_name = format_ident!("{base}InputPoints");
    let output_claims_name = format_ident!("{base}OutputClaims");
    let output_points_name = format_ident!("{base}OutputPoints");
    let challenges_name = format_ident!("{base}Challenges");
    let batching_coefficients_name = format_ident!("{base}BatchingCoefficients");
    let clear_batch_name = format_ident!("{base}ClearBatch");

    let options = StageOptions::parse(&input.attrs)?;

    validate_generics(&input.generics)?;
    let f = field_type_param(&input.generics)?;
    let fields = named_fields(&input.data, name.span())?;
    let plans = fields
        .iter()
        .map(plan_field)
        .collect::<syn::Result<Vec<_>>>()?;
    if plans.is_empty() {
        return Err(syn::Error::new(
            name.span(),
            "SumcheckBatch requires at least one instance field",
        ));
    }

    let relations = quote!(crate::stages::relations);

    let project = |alias: &TokenStream2, plan: &InstanceField| {
        let instance = &plan.instance;
        let projected = quote!(#relations::#alias<#f, #instance>);
        if plan.is_option {
            quote!(::core::option::Option<#projected>)
        } else {
            projected
        }
    };

    let input_claims_alias = quote!(SumcheckInputClaims);
    let input_points_alias = quote!(SumcheckInputPoints);
    let output_claims_alias = quote!(SumcheckOutputClaims);
    let output_points_alias = quote!(SumcheckOutputPoints);
    let challenges_alias = quote!(ConcreteSumcheckChallenges);

    let field_decls = |alias: &TokenStream2| {
        plans
            .iter()
            .map(|plan| {
                let id = &plan.ident;
                let ty = project(alias, plan);
                quote!(pub #id: #ty)
            })
            .collect::<Vec<_>>()
    };
    let input_claims_fields = field_decls(&input_claims_alias);
    let input_points_fields = field_decls(&input_points_alias);
    let output_claims_fields = field_decls(&output_claims_alias);
    let output_points_fields = field_decls(&output_points_alias);
    let challenge_fields = field_decls(&challenges_alias);

    // The per-instance batching coefficients: one `F` per member (the scalar the
    // batched verifier draws for that instance), `Option<F>` for a conditional
    // member (present iff the instance ran). A named, typed view of what was a
    // positional coefficient `Vec`, in member declaration order.
    let batching_coefficient_fields = plans
        .iter()
        .map(|plan| {
            let id = &plan.ident;
            if plan.is_option {
                quote!(pub #id: ::core::option::Option<#f>)
            } else {
                quote!(pub #id: #f)
            }
        })
        .collect::<Vec<_>>();

    // `opening_values` over the wire cell (`C = F`): chain each member's
    // `OutputClaims::opening_values` in declaration order; `Option` members
    // contribute only when present.
    let opening_chain = plans.iter().map(|plan| {
        let id = &plan.ident;
        if plan.is_option {
            quote!(.chain(self.#id.as_ref().map(|member| member.opening_values()).unwrap_or_default()))
        } else {
            quote!(.chain(self.#id.opening_values()))
        }
    });

    // Per-instance driver plumbing on the source `StageNSumchecks` struct itself:
    // draw each member's challenges into the stage's challenge aggregate, delegating
    // to each member's `ConcreteSumcheck::draw_challenges` in declaration order;
    // `Option` members draw only when present. Always emitted — it compiles for
    // every stage because every member is a `ConcreteSumcheck` and the challenge
    // aggregate has exactly the member fields.
    let draw_fields = plans.iter().map(|plan| {
        let id = &plan.ident;
        if plan.is_option {
            quote! {
                #id: match self.#id.as_ref() {
                    ::core::option::Option::Some(member) => {
                        ::core::option::Option::Some(member.draw_challenges(transcript)?)
                    }
                    ::core::option::Option::None => ::core::option::Option::None,
                }
            }
        } else {
            quote!(#id: self.#id.draw_challenges(transcript)?)
        }
    });
    // The representative relation id used in a batch-level sumcheck error: the first
    // non-`Option` member (the batch's leading instance), matching the hand-written
    // stages' choice. Falls back to the first member if every member is optional.
    let stage_id_ident = plans
        .iter()
        .find(|plan| !plan.is_option)
        .map_or(&plans[0].ident, |plan| &plan.ident);

    // Fold each member's `(rounds, degree)` into the batch's `(max_num_vars,
    // max_degree)` — the front-loaded batching layout's combined dimensions. Reused
    // by both the clear and ZK drivers, so it is a closure re-invoked per block (a
    // `quote!` interpolation consumes its iterator).
    let max_fold = || {
        plans.iter().map(|plan| {
            let id = &plan.ident;
            if plan.is_option {
                quote! {
                    if let ::core::option::Option::Some(__member) = self.#id.as_ref() {
                        __max_num_vars = ::core::cmp::max(__max_num_vars, __member.rounds());
                        __max_degree = ::core::cmp::max(__max_degree, __member.degree());
                    }
                }
            } else {
                quote! {
                    __max_num_vars = ::core::cmp::max(__max_num_vars, self.#id.rounds());
                    __max_degree = ::core::cmp::max(__max_degree, self.#id.degree());
                }
            }
        })
    };

    // The clear-path batched-verify driver: compute the combined `(max_num_vars,
    // max_degree, claimed_sum)` from the members (absorb sums, draw coefficients,
    // random-linear-combine), then reduce through the single-instance
    // `SumcheckProof::verify_compressed_boolean`. Opt-in via
    // `#[sumcheck_batch(verify_clear)]`.
    let verify_clear_method = if options.verify_clear {
        let max_fold = max_fold();
        let sum_idents = plans
            .iter()
            .map(|plan| format_ident!("__sum_{}", plan.ident))
            .collect::<Vec<_>>();
        let coeff_idents = plans
            .iter()
            .map(|plan| format_ident!("__coeff_{}", plan.ident))
            .collect::<Vec<_>>();

        // Each member's claimed sum (its `input_claim`), bound to `__sum_<member>`.
        // `Option` members bind an `Option<F>`, present iff the instance is. A
        // present instance with a missing input or challenge cell is a WIRING BUG,
        // not an absent member: silently skipping it would drop this member's sum
        // absorb from the transcript (a Fiat-Shamir divergence that surfaces as an
        // unattributable batch failure downstream), so it errors here with the
        // member's relation id instead.
        let sum_bindings = plans.iter().zip(&sum_idents).map(|(plan, sum)| {
            let id = &plan.ident;
            if plan.is_option {
                quote! {
                    let #sum = match (self.#id.as_ref(), inputs.#id.as_ref(), challenges.#id.as_ref()) {
                        (
                            ::core::option::Option::Some(__member),
                            ::core::option::Option::Some(__inputs),
                            ::core::option::Option::Some(__challenges),
                        ) => ::core::option::Option::Some(__member.input_claim(__inputs, __challenges)?),
                        (::core::option::Option::None, _, _) => ::core::option::Option::None,
                        (::core::option::Option::Some(__member), __inputs, _) => {
                            return ::core::result::Result::Err(
                                crate::VerifierError::StageClaimSumcheckFailed {
                                    stage: __member.id(),
                                    reason: if __inputs.is_none() {
                                        "present instance is missing its input values"
                                    } else {
                                        "present instance is missing its challenges"
                                    }
                                    .to_string(),
                                },
                            );
                        }
                    };
                }
            } else {
                quote!(let #sum = self.#id.input_claim(&inputs.#id, &challenges.#id)?;)
            }
        });

        // Absorb each present member's claimed sum into the transcript, in
        // declaration order — the Fiat-Shamir binding that must precede the
        // batching-coefficient draw.
        let sum_absorbs = plans.iter().zip(&sum_idents).map(|(plan, sum)| {
            if plan.is_option {
                quote! {
                    if let ::core::option::Option::Some(__sum) = #sum.as_ref() {
                        ::jolt_sumcheck::append_sumcheck_claim(transcript, __sum);
                    }
                }
            } else {
                quote!(::jolt_sumcheck::append_sumcheck_claim(transcript, &#sum);)
            }
        });

        // Draw one batching coefficient per present member (declaration order),
        // binding it to `__coeff_<member>`. `Option` members key the draw on the
        // member's bound sum — the same resolution the absorb used — so an absorb
        // and its coefficient draw can never disagree about presence.
        let coeff_draws =
            plans
                .iter()
                .zip(sum_idents.iter().zip(&coeff_idents))
                .map(|(plan, (sum, coeff))| {
                    if plan.is_option {
                        quote! {
                            let #coeff = if #sum.is_some() {
                                ::core::option::Option::Some(transcript.challenge_scalar())
                            } else {
                                ::core::option::Option::None
                            };
                        }
                    } else {
                        quote!(let #coeff = transcript.challenge_scalar();)
                    }
                });

        // Pack the drawn coefficients into the named aggregate, in member order.
        let coeff_fields = plans.iter().zip(&coeff_idents).map(|(plan, coeff)| {
            let id = &plan.ident;
            quote!(#id: #coeff)
        });

        // Each member's contribution to the combined claim: `coeff * sum * 2^(max -
        // rounds)` (front-loaded padding scale), pushed into `__terms` and summed.
        let term_pushes = plans
            .iter()
            .zip(sum_idents.iter().zip(&coeff_idents))
            .map(|(plan, (sum, coeff))| {
                let id = &plan.ident;
                if plan.is_option {
                    quote! {
                        if let (
                            ::core::option::Option::Some(__coeff),
                            ::core::option::Option::Some(__sum),
                            ::core::option::Option::Some(__member),
                        ) = (#coeff, #sum, self.#id.as_ref())
                        {
                            __terms.push(__coeff * __sum.mul_pow_2(__max_num_vars - __member.rounds()));
                        }
                    }
                } else {
                    quote! {
                        __terms.push(#coeff * #sum.mul_pow_2(__max_num_vars - self.#id.rounds()));
                    }
                }
            });

        quote! {
            pub fn verify_clear<__C, __T>(
                &self,
                inputs: &#input_claims_name<#f>,
                challenges: &#challenges_name<#f>,
                proof: &::jolt_sumcheck::SumcheckProof<#f, __C>,
                transcript: &mut __T,
            ) -> ::core::result::Result<#clear_batch_name<#f>, crate::VerifierError>
            where
                __C: ::core::clone::Clone + ::jolt_transcript::AppendToTranscript,
                __T: ::jolt_transcript::Transcript<Challenge = #f>,
            {
                use #relations::ConcreteSumcheck as _;
                use ::jolt_field::MulPow2 as _;

                #(#sum_bindings)*

                let mut __max_num_vars = 0usize;
                let mut __max_degree = 0usize;
                #(#max_fold)*

                #(#sum_absorbs)*

                #(#coeff_draws)*
                let __coefficients = #batching_coefficients_name {
                    #(#coeff_fields,)*
                };

                let mut __terms = ::std::vec::Vec::new();
                #(#term_pushes)*
                let __claimed_sum: #f = __terms.into_iter().sum();

                let __reduction = proof
                    .verify_compressed_boolean(__max_num_vars, __max_degree, __claimed_sum, transcript)
                    .map_err(|error| crate::VerifierError::StageClaimSumcheckFailed {
                        stage: self.#stage_id_ident.id(),
                        reason: error.to_string(),
                    })?;

                ::core::result::Result::Ok(#clear_batch_name {
                    reduction: __reduction,
                    coefficients: __coefficients,
                })
            }
        }
    } else {
        quote!()
    };

    // The ZK-path batched-verify driver: compute the combined `(max_num_vars,
    // max_degree)`, draw the batching coefficients, then check committed consistency
    // through `SumcheckProof::verify_committed_consistency_dims`. Committed proofs
    // never reveal claim scalars, so no claimed sums are absorbed. Opt-in via
    // `#[sumcheck_batch(verify_zk)]`.
    let verify_zk_method = if options.verify_zk {
        let max_fold = max_fold();

        // Draw one batching coefficient per present member (ZK path): no claimed sums
        // are absorbed, so only the coefficients are recorded, in declaration order.
        let coeff_draws_zk = plans.iter().map(|plan| {
            let id = &plan.ident;
            if plan.is_option {
                quote! {
                    if self.#id.is_some() {
                        __batching_coefficients.push(transcript.challenge_scalar());
                    }
                }
            } else {
                quote!(__batching_coefficients.push(transcript.challenge_scalar());)
            }
        });

        quote! {
            pub fn verify_zk<__C, __T>(
                &self,
                proof: &::jolt_sumcheck::SumcheckProof<#f, __C>,
                transcript: &mut __T,
            ) -> ::core::result::Result<
                ::jolt_sumcheck::BatchedCommittedSumcheckConsistency<#f, __C>,
                crate::VerifierError,
            >
            where
                __C: ::core::clone::Clone + ::jolt_transcript::AppendToTranscript,
                __T: ::jolt_transcript::Transcript<Challenge = #f>,
            {
                use #relations::ConcreteSumcheck as _;

                let mut __max_num_vars = 0usize;
                let mut __max_degree = 0usize;
                #(#max_fold)*

                let mut __batching_coefficients = ::std::vec::Vec::new();
                #(#coeff_draws_zk)*

                let __consistency = proof
                    .verify_committed_consistency_dims(__max_num_vars, __max_degree, transcript)
                    .map_err(|error| crate::VerifierError::StageClaimSumcheckFailed {
                        stage: self.#stage_id_ident.id(),
                        reason: error.to_string(),
                    })?;

                ::core::result::Result::Ok(::jolt_sumcheck::BatchedCommittedSumcheckConsistency {
                    consistency: __consistency,
                    batching_coefficients: __batching_coefficients,
                    max_num_vars: __max_num_vars,
                    max_degree: __max_degree,
                })
            }
        }
    } else {
        quote!()
    };

    // Map each member's opening point through its
    // `ConcreteSumcheck::derive_opening_points` into the stage's `OutputPoints`
    // aggregate. Takes the batch challenge vector directly: under the front-loaded
    // batching layout an instance's point is the length-`rounds` suffix of that
    // vector, so no batch-result abstraction is needed and one method serves both the
    // clear and ZK paths (each supplies its own challenge vector). Opt-in via
    // `#[sumcheck_batch(derive_opening_points)]`.
    let derive_points_method = if options.derive_opening_points {
        let field_bindings = plans.iter().map(|plan| {
            let id = &plan.ident;
            let field = format_ident!("__points_{}", id);
            let binding = if plan.is_option {
                quote! {
                    let #field = match (self.#id.as_ref(), input_points.#id.as_ref()) {
                        (
                            ::core::option::Option::Some(__member),
                            ::core::option::Option::Some(__input_points),
                        ) => {
                            let __point = __instance_point(
                                batch_point,
                                __member.instance_point_offset(batch_point.len())?,
                                __member.rounds(),
                                __member.id(),
                            )?;
                            ::core::option::Option::Some(
                                __member.derive_opening_points(__point, __input_points)?,
                            )
                        }
                        (::core::option::Option::None, _) => ::core::option::Option::None,
                        (::core::option::Option::Some(__member), ::core::option::Option::None) => {
                            return ::core::result::Result::Err(
                                crate::VerifierError::StageClaimSumcheckFailed {
                                    stage: __member.id(),
                                    reason: "present instance is missing its input opening points"
                                        .to_string(),
                                },
                            );
                        }
                    };
                }
            } else {
                quote! {
                    let #field = self.#id.derive_opening_points(
                        __instance_point(
                            batch_point,
                            self.#id.instance_point_offset(batch_point.len())?,
                            self.#id.rounds(),
                            self.#id.id(),
                        )?,
                        &input_points.#id,
                    )?;
                }
            };
            (binding, id, field)
        }).collect::<Vec<_>>();
        let point_bindings = field_bindings.iter().map(|(binding, _, _)| binding);
        let point_fields = field_bindings
            .iter()
            .map(|(_, id, field)| quote!(#id: #field));

        quote! {
            pub fn derive_opening_points(
                &self,
                batch_point: &[#f],
                input_points: &#input_points_name<#f>,
            ) -> ::core::result::Result<#output_points_name<#f>, crate::VerifierError> {
                use #relations::ConcreteSumcheck as _;

                // An instance with `rounds` variables is bound on
                // `batch_point[offset .. offset + rounds]`, where `offset` is the
                // member's `instance_point_offset` (the front-loaded suffix by
                // default; the two-phase address relations use the offset-0 prefix,
                // the stage-2 RAM relations their phase-1 offset).
                fn __instance_point<__F: ::jolt_field::Field>(
                    batch_point: &[__F],
                    offset: usize,
                    rounds: usize,
                    stage: ::jolt_claims::protocols::jolt::JoltRelationId,
                ) -> ::core::result::Result<&[__F], crate::VerifierError> {
                    offset
                        .checked_add(rounds)
                        .and_then(|__end| batch_point.get(offset..__end))
                        .ok_or(crate::VerifierError::StageClaimSumcheckFailed {
                            stage,
                            reason: ::std::format!(
                                "instance point [{offset}, {offset} + {rounds}) exceeds the batch \
                                 challenge vector ({} entries)",
                                batch_point.len(),
                            ),
                        })
                }

                #(#point_bindings)*
                ::core::result::Result::Ok(#output_points_name {
                    #(#point_fields,)*
                })
            }
        }
    } else {
        quote!()
    };

    // Fold the members' expected output claims with the batch coefficients into the
    // final claim the reduction is checked against: `Σ coeff_m * expected_output_m`.
    // A present `Option` member with any absent cell errors rather than silently
    // dropping its term (which would surface as an opaque final-claim mismatch).
    // Opt-in via `#[sumcheck_batch(expected_final_claim)]`.
    let expected_final_claim_method = if options.expected_final_claim {
        let output_terms = plans.iter().map(|plan| {
            let id = &plan.ident;
            if plan.is_option {
                quote! {
                    if let ::core::option::Option::Some(__member) = self.#id.as_ref() {
                        let (
                            ::core::option::Option::Some(__coeff),
                            ::core::option::Option::Some(__input_points),
                            ::core::option::Option::Some(__output_values),
                            ::core::option::Option::Some(__output_points),
                            ::core::option::Option::Some(__challenges),
                        ) = (
                            coefficients.#id,
                            input_points.#id.as_ref(),
                            output_values.#id.as_ref(),
                            output_points.#id.as_ref(),
                            challenges.#id.as_ref(),
                        ) else {
                            return ::core::result::Result::Err(
                                crate::VerifierError::StageClaimSumcheckFailed {
                                    stage: __member.id(),
                                    reason: "present instance is missing a coefficient, claim, \
                                             point, or challenge cell for the final-claim fold"
                                        .to_string(),
                                },
                            );
                        };
                        __terms.push(__coeff * __member.expected_output(
                            __input_points,
                            __output_values,
                            __output_points,
                            __challenges,
                        )?);
                    }
                }
            } else {
                quote! {
                    __terms.push(coefficients.#id * self.#id.expected_output(
                        &input_points.#id,
                        &output_values.#id,
                        &output_points.#id,
                        &challenges.#id,
                    )?);
                }
            }
        });
        quote! {
            pub fn expected_final_claim(
                &self,
                coefficients: &#batching_coefficients_name<#f>,
                input_points: &#input_points_name<#f>,
                output_values: &#output_claims_name<#f>,
                output_points: &#output_points_name<#f>,
                challenges: &#challenges_name<#f>,
            ) -> ::core::result::Result<#f, crate::VerifierError> {
                use #relations::ConcreteSumcheck as _;
                let mut __terms = ::std::vec::Vec::new();
                #(#output_terms)*
                let __expected: #f = __terms.into_iter().sum();
                ::core::result::Result::Ok(__expected)
            }
        }
    } else {
        quote!()
    };

    // The output-claim shape helpers: the total produced-opening count (for the ZK
    // commitment count) and a validator that the proof-supplied output claims match
    // the dims-derived expected shape (per member, comparing `canonical_order` id-sets
    // against `expected_output_openings`). Opt-in via `#[sumcheck_batch(output_shape)]`.
    let output_shape_methods = if options.output_shape {
        let count_terms = plans.iter().map(|plan| {
            let id = &plan.ident;
            if plan.is_option {
                quote! {
                    if let ::core::option::Option::Some(__member) = self.#id.as_ref() {
                        __count += __member.symbolic().expected_output_openings::<#f>().len();
                    }
                }
            } else {
                quote!(__count += self.#id.symbolic().expected_output_openings::<#f>().len();)
            }
        });
        let validate_checks = plans.iter().map(|plan| {
            let id = &plan.ident;
            let check = quote! {
                let __expected = __member.symbolic().expected_output_openings::<#f>();
                let __provided: ::std::collections::BTreeSet<_> =
                    __claims.canonical_order().into_iter().collect();
                if __provided != __expected {
                    return ::core::result::Result::Err(
                        crate::VerifierError::StageClaimPublicInputFailed {
                            stage: __member.id(),
                            reason: ::std::format!(
                                "output claim shape mismatch: expected {} openings, got {}",
                                __expected.len(),
                                __provided.len(),
                            ),
                        },
                    );
                }
            };
            if plan.is_option {
                quote! {
                    match (self.#id.as_ref(), claims.#id.as_ref()) {
                        (
                            ::core::option::Option::Some(__member),
                            ::core::option::Option::Some(__claims),
                        ) => { #check }
                        (::core::option::Option::None, ::core::option::Option::None) => {}
                        (::core::option::Option::Some(__member), ::core::option::Option::None) => {
                            return ::core::result::Result::Err(
                                crate::VerifierError::StageClaimPublicInputFailed {
                                    stage: __member.id(),
                                    reason: "present instance is missing its output claims"
                                        .to_string(),
                                },
                            );
                        }
                        (::core::option::Option::None, ::core::option::Option::Some(_)) => {}
                    }
                }
            } else {
                quote! {
                    {
                        let __member = &self.#id;
                        let __claims = &claims.#id;
                        #check
                    }
                }
            }
        });
        quote! {
            /// The total number of produced opening claims across the batch (the
            /// dims-derived expected shape), e.g. the committed-output-claim count.
            pub fn output_claim_count(&self) -> usize {
                use #relations::ConcreteSumcheck as _;
                use ::jolt_claims::SymbolicSumcheck as _;
                let mut __count = 0usize;
                #(#count_terms)*
                __count
            }

            /// Assert the proof-supplied output claims match the expected shape: per
            /// member, the provided `canonical_order` id-set equals the relation's
            /// dims-derived `expected_output_openings`.
            pub fn validate_output_claims(
                &self,
                claims: &#output_claims_name<#f>,
            ) -> ::core::result::Result<(), crate::VerifierError> {
                use #relations::ConcreteSumcheck as _;
                use ::jolt_claims::OutputClaims as _;
                use ::jolt_claims::SymbolicSumcheck as _;
                #(#validate_checks)*
                ::core::result::Result::Ok(())
            }
        }
    } else {
        quote!()
    };

    let driver_impl = quote! {
        impl<#f: ::jolt_field::Field> #name<#f> {
            /// Draw each instance's Fiat-Shamir challenges in declaration order,
            /// assembling the stage's challenge aggregate. Members with no
            /// challenges draw nothing; `Option` members draw only when present.
            /// This single-sources the stage's inline per-instance draw, so its
            /// Fiat-Shamir order follows member declaration order.
            pub fn draw_challenges<__T: ::jolt_transcript::Transcript<Challenge = #f>>(
                &self,
                transcript: &mut __T,
            ) -> ::core::result::Result<#challenges_name<#f>, crate::VerifierError> {
                use #relations::ConcreteSumcheck as _;
                ::core::result::Result::Ok(#challenges_name {
                    #(#draw_fields,)*
                })
            }

            #verify_clear_method
            #verify_zk_method
            #derive_points_method
            #expected_final_claim_method
            #output_shape_methods
        }
    };

    // The generated `OutputClaims` opening plumbing. Gated out when the stage opts
    // in to `#[sumcheck_batch(custom_opening_values)]`, in which case the stage
    // supplies its own alias-curated `opening_values` / `append_to_transcript` (and
    // any `validate`) as an inherent impl on the generated struct.
    let opening_impl = if options.custom_opening_values {
        quote!()
    } else {
        quote! {
            impl<#f: ::jolt_field::Field> #output_claims_name<#f> {
                /// Produced opening scalars in canonical (field-declaration) order,
                /// delegating to each instance's `OutputClaims` in order.
                pub fn opening_values(&self) -> ::std::vec::Vec<#f> {
                    use ::jolt_claims::OutputClaims as _;
                    ::core::iter::empty::<#f>()
                        #(#opening_chain)*
                        .collect()
                }

                /// Append every produced opening to the transcript in canonical order,
                /// each under the `b"opening_claim"` label, matching the prover's
                /// commitment order.
                pub fn append_to_transcript<__T: ::jolt_transcript::Transcript<Challenge = #f>>(
                    &self,
                    transcript: &mut __T,
                ) {
                    for value in self.opening_values() {
                        transcript.append_labeled(b"opening_claim", &value);
                    }
                }
            }
        }
    };

    // Each opening cell instantiation is its own concrete aggregate: `*Claims`
    // holds the wire *values* (`Inputs<F>` / `Outputs<F>`); `*Points` holds the
    // derived opening points (`Inputs<Vec<F>>` / `Outputs<Vec<F>>`). Only the
    // `OutputClaims` (values) aggregate is serialized (the wire form), so it alone
    // derives serde; the empty serde bound suffices because `F: Field` already
    // implies `Serialize + DeserializeOwned` through the member structs.
    // `verify_clear`'s result: the single-instance reduction plus the named batching
    // coefficients. Emitted only when `verify_clear` is generated (its return type).
    let clear_batch_struct = if options.verify_clear {
        quote! {
            #[derive(Clone, Debug, PartialEq, Eq)]
            #vis struct #clear_batch_name<#f: ::jolt_field::Field> {
                pub reduction: ::jolt_sumcheck::EvaluationClaim<#f>,
                pub coefficients: #batching_coefficients_name<#f>,
            }
        }
    } else {
        quote!()
    };

    Ok(quote! {
        #[derive(Clone, Debug, PartialEq, Eq)]
        #vis struct #input_claims_name<#f: ::jolt_field::Field> {
            #(#input_claims_fields,)*
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        #vis struct #input_points_name<#f: ::jolt_field::Field> {
            #(#input_points_fields,)*
        }

        #[derive(Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize)]
        #[serde(bound(serialize = "", deserialize = ""))]
        #vis struct #output_claims_name<#f: ::jolt_field::Field> {
            #(#output_claims_fields,)*
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        #vis struct #output_points_name<#f: ::jolt_field::Field> {
            #(#output_points_fields,)*
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        #vis struct #challenges_name<#f: ::jolt_field::Field> {
            #(#challenge_fields,)*
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        #vis struct #batching_coefficients_name<#f: ::jolt_field::Field> {
            #(#batching_coefficient_fields,)*
        }

        #clear_batch_struct

        #driver_impl

        #opening_impl
    })
}

/// Struct-level `#[sumcheck_batch(...)]` configuration. Parsed from the source
/// struct's attributes; recognizes only the flags below and errors clearly on
/// anything else.
#[derive(Default)]
struct StageOptions {
    /// `#[sumcheck_batch(custom_opening_values)]`: skip emitting the generated
    /// `OutputClaims` `opening_values` / `append_to_transcript` inherent impl so the
    /// stage can curate its own (e.g. skipping cross-relation aliased openings).
    custom_opening_values: bool,
    /// `#[sumcheck_batch(verify_clear)]`: emit the clear-path batched-verify driver
    /// (`verify_clear`) that folds the members into one combined claim (absorb sums,
    /// draw coefficients, RLC) and reduces it through the single-instance verifier.
    verify_clear: bool,
    /// `#[sumcheck_batch(verify_zk)]`: emit the ZK-path batched-verify driver
    /// (`verify_zk`) that folds the members' dimensions and checks committed
    /// consistency through the single-instance verifier.
    verify_zk: bool,
    /// `#[sumcheck_batch(derive_opening_points)]`: emit `derive_opening_points`,
    /// which slices each member's opening point from the batch challenge vector (its
    /// length-`rounds` suffix) and maps it through the member's
    /// `ConcreteSumcheck::derive_opening_points` into the stage's `OutputPoints`
    /// aggregate.
    derive_opening_points: bool,
    /// `#[sumcheck_batch(expected_final_claim)]`: emit `expected_final_claim`, which
    /// folds the members' `expected_output` with the batch coefficients into the
    /// final claim the reduction is checked against.
    expected_final_claim: bool,
    /// `#[sumcheck_batch(output_shape)]`: emit `output_claim_count` and
    /// `validate_output_claims`, which derive the expected output-claim shape from each
    /// member's `expected_output_openings` (its output `Expr`).
    output_shape: bool,
}

impl StageOptions {
    fn parse(attrs: &[Attribute]) -> syn::Result<Self> {
        let mut options = StageOptions::default();
        for attr in attrs {
            if !attr.path().is_ident("sumcheck_batch") {
                continue;
            }
            // `#[sumcheck_batch(flag, flag, ...)]` — a comma-separated list of
            // bare-word flags (`Meta::Path`). Reject any non-flag form or unknown
            // flag with a span-pointed error.
            let flags = attr.parse_args_with(
                syn::punctuated::Punctuated::<Meta, Token![,]>::parse_terminated,
            )?;
            for flag in flags {
                let Meta::Path(path) = &flag else {
                    return Err(syn::Error::new_spanned(
                        &flag,
                        "expected a bare `sumcheck_batch` flag (e.g. `custom_opening_values`)",
                    ));
                };
                if path.is_ident("custom_opening_values") {
                    options.custom_opening_values = true;
                } else if path.is_ident("verify_clear") {
                    options.verify_clear = true;
                } else if path.is_ident("verify_zk") {
                    options.verify_zk = true;
                } else if path.is_ident("derive_opening_points") {
                    options.derive_opening_points = true;
                } else if path.is_ident("expected_final_claim") {
                    options.expected_final_claim = true;
                } else if path.is_ident("output_shape") {
                    options.output_shape = true;
                } else {
                    return Err(syn::Error::new_spanned(
                        path,
                        "unknown `sumcheck_batch` flag (supported: `custom_opening_values`, \
                         `verify_clear`, `verify_zk`, `derive_opening_points`, \
                         `expected_final_claim`, `output_shape`)",
                    ));
                }
            }
        }
        Ok(options)
    }
}

/// The macro supports exactly one generic type parameter (the field `F`): no
/// extra generics, no lifetimes/consts, and no where-clause (any of which the
/// generated aggregates would silently drop). Reject anything else with a clear
/// error rather than mis-binding `F` or emitting wrong projections.
fn validate_generics(generics: &syn::Generics) -> syn::Result<()> {
    if let Some(where_clause) = &generics.where_clause {
        return Err(syn::Error::new_spanned(
            where_clause,
            "SumcheckBatch does not support a where-clause on the source struct",
        ));
    }
    let type_params = generics
        .params
        .iter()
        .filter(|param| matches!(param, GenericParam::Type(_)))
        .count();
    if generics.params.len() != 1 || type_params != 1 {
        return Err(syn::Error::new_spanned(
            generics,
            "SumcheckBatch requires exactly one generic type parameter (the field `F`)",
        ));
    }
    Ok(())
}

/// The first type generic parameter (the field, conventionally `F`).
fn field_type_param(generics: &syn::Generics) -> syn::Result<Ident> {
    generics
        .params
        .iter()
        .find_map(|param| match param {
            GenericParam::Type(param) => Some(param.ident.clone()),
            _ => None,
        })
        .ok_or_else(|| {
            syn::Error::new_spanned(
                generics,
                "expected a field-type generic parameter (e.g. `<F>`)",
            )
        })
}

fn named_fields(data: &Data, span: proc_macro2::Span) -> syn::Result<Vec<syn::Field>> {
    match data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => Ok(fields.named.iter().cloned().collect()),
            _ => Err(syn::Error::new(
                span,
                "SumcheckBatch requires a struct with named fields",
            )),
        },
        _ => Err(syn::Error::new(
            span,
            "SumcheckBatch can only be derived for structs",
        )),
    }
}

fn plan_field(field: &syn::Field) -> syn::Result<InstanceField> {
    let ident = field
        .ident
        .clone()
        .ok_or_else(|| syn::Error::new_spanned(field, "fields must be named"))?;
    let (is_option, instance) = match option_inner(&field.ty) {
        Some(inner) => (true, inner.clone()),
        None => (false, field.ty.clone()),
    };
    Ok(InstanceField {
        ident,
        instance,
        is_option,
    })
}

/// If `ty` is syntactically `Option<Inner>`, return `Inner`.
fn option_inner(ty: &Type) -> Option<&Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    })
}
