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
//! `Stage5BatchingCoefficients<F>` (one cell per member, in declaration order),
//! projecting each field through the `Sumcheck{Input,Output}{Claims,Points}` /
//! `ConcreteSumcheckChallenges` aliases in `jolt-verifier`'s `stages::relations`
//! (the `*Claims` aggregates hold the wire *values*, the `*Points` aggregates the
//! derived opening points; `BatchingCoefficients` holds the batched verifier's
//! per-instance scalar). A source field `Option<Instance>` becomes an
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
//! Emitted for every stage, as inherent methods on the source `StageNSumchecks`
//! struct (each generated method's own doc has the details; the shared
//! per-member logic lives as generic functions in `stages::relations`, so the
//! macro emits only the typed per-member dispatch):
//!
//! - `begin_batch` — the batched-sumcheck *head*, shared by `verify_clear` and
//!   the prove-side stage recipes: per-member `input_claim` (declaration order)
//!   → `recorder.absorb_input_claims` → one batching-coefficient draw per
//!   present member → the `2^(max − rounds)`-padded random linear combination.
//!   Generic over `jolt_sumcheck::SumcheckRecorder`, the clear/ZK seam: whether
//!   the claim absorb writes transcript bytes is decided by the recorder TYPE
//!   (clear appends, committed no-ops), never by a runtime flag. Returns the
//!   engine-form `jolt_sumcheck::BatchPrelude` paired with the stage's named
//!   `BatchingCoefficients`.
//! - `<snake_case_struct>_members` — an inert, `#[macro_export]`ed callback
//!   macro carrying the batch declaration as a structured token list (member
//!   names, generics-stripped relation paths, presence, aggregate type names,
//!   output-shape flag), forwarded to a caller-chosen macro. The derive's
//!   ONLY prover-facing emission — the single-sourcing handoff from which
//!   `jolt-prover`'s `impl_stage_prover` expands the prove-side stage-driver
//!   impls (`StageProver`/`KernelSource`), so no stage's member list, order,
//!   or presence is ever restated. See `specs/prover-stage-drivers.md`.
//! - `verify_clear` — the composed clear-path driver: `begin_batch` with a clear
//!   recorder, reduce the combined claim through the single-instance
//!   `SumcheckProof::verify_compressed_boolean`, `derive_opening_points` at the
//!   reduced point, then the `expected_final_claim` equality check. The batching
//!   lives in the generated head; `jolt-sumcheck` provides only the single-instance
//!   verifier. Returns the stage's produced `OutputPoints`.
//! - `verify_zk` — the ZK-path driver: fold the members' dimensions, draw the
//!   batching coefficients, and check committed consistency through
//!   `SumcheckProof::verify_committed_consistency_dims`. Committed proofs reveal no
//!   claim scalars, so the caller must have absorbed claim COMMITMENTS beforehand
//!   (the coefficients are squeezed before the consistency rounds).
//! - `derive_opening_points` — slice each member's point from the batch challenge
//!   vector (`ConcreteSumcheck::instance_point`, at the member's overridable
//!   `instance_point_offset`) and map it through the member's
//!   `ConcreteSumcheck::derive_opening_points`.
//!   Takes the challenge vector as `&[F]`, so the clear and ZK paths each pass
//!   their own.
//! - `expected_final_claim` — run `validate_aliases`, then fold the members'
//!   `ConcreteSumcheck::expected_output` with the batch coefficients.
//! - `validate_aliases` — each member's declared `(aliased, source)` opening pairs,
//!   via `relations::validate_member_aliases` against the batch-wide resolver. Run
//!   unskippably by `expected_final_claim`, so declaring a pair on a relation
//!   enforces it everywhere.
//! - `output_claim_count` / `validate_output_claims` — the wire-shape helpers
//!   (via `relations::validate_member_{presence, output_shape}`), deriving each
//!   member's expected openings from `ConcreteSumcheck::wire_output_openings`.
//! - `draw_challenges`, `empty_input_points`, and the absorb plumbing
//!   (`opening_values` / `append_output_claims`, via
//!   `relations::absorbed_opening_values`).
//!
//! Every `#[sumcheck_batch(...)]` flag is an opt-OUT (`StageOptions` below is
//! the canonical reference): `no_opening_values`, `no_output_shape`, and
//! `no_draw_challenges` each suppress generated methods that would be WRONG to
//! call on their stage (a member-interleaved or runtime-deduped absorb order; a
//! runtime-deduped wire shape; stage-level challenge provenance) — suppressed
//! rather than overridden so they cannot be miscalled. A flagless stage gets
//! the full method suite.
//!
//! The one non-flag entry is the serde-style crate-path override
//! `#[sumcheck_batch(crate = "...")]`: the path the generated code names
//! `jolt-verifier` by. Defaults to the absolute `::jolt_verifier`, so external
//! users need nothing; `jolt-verifier` itself passes `crate = "crate"`. The
//! emitted member-list callback macro is NOT affected — its tokens resolve at
//! the consumer's invocation site (cross-crate, in `jolt-prover`), never
//! against this override.
//!
//! The `verify_*` drivers never name `SumcheckClaim` / `SumcheckStatement`; those
//! stay internal to `jolt-sumcheck`.
//!
//! See `specs/sumcheck-batch-derive.md`.

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
/// The struct-level `#[sumcheck_batch(...)]` flags are opt-outs
/// (`no_opening_values`, `no_output_shape`, `no_draw_challenges`), each
/// suppressing generated methods that would be wrong to call on the flagged
/// stage, which supplies its own replacement where one is needed. The
/// aggregate structs and their derives are emitted unchanged.
/// `#[sumcheck_batch(crate = "...")]` overrides the `::jolt_verifier` path
/// the generated code names this crate by (the defining crate passes
/// `"crate"`).
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

/// Wrap `body` in the member-presence scaffold shared by the per-member driver
/// blocks: bind each `(name, expr)` cell by reference for a plain member, or via
/// `if let Some(..)` over `expr.as_ref()` for an `Option` member (skipping
/// `body` when absent). Only for infallible-skip sites; blocks that must ERROR
/// on a present/absent cell mismatch keep their bespoke matches.
fn per_member(
    is_option: bool,
    cells: &[(&Ident, TokenStream2)],
    body: TokenStream2,
) -> TokenStream2 {
    if is_option {
        let patterns = cells
            .iter()
            .map(|(name, _)| quote!(::core::option::Option::Some(#name)));
        let scrutinees = cells.iter().map(|(_, expr)| quote!(#expr.as_ref()));
        quote! {
            if let (#(#patterns,)*) = (#(#scrutinees,)*) {
                #body
            }
        }
    } else {
        let bindings = cells.iter().map(|(name, expr)| quote!(let #name = &#expr;));
        quote! {
            {
                #(#bindings)*
                #body
            }
        }
    }
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
    // The batch's string label for stage-level sumcheck errors.
    let base_lit = syn::LitStr::new(&base, name.span());
    let input_claims_name = format_ident!("{base}InputClaims");
    let input_points_name = format_ident!("{base}InputPoints");
    let output_claims_name = format_ident!("{base}OutputClaims");
    let output_points_name = format_ident!("{base}OutputPoints");
    let challenges_name = format_ident!("{base}Challenges");
    let batching_coefficients_name = format_ident!("{base}BatchingCoefficients");

    let options = StageOptions::parse(&input.attrs)?;
    // The path the generated code names `jolt-verifier` by: the absolute
    // default serves external deriving crates; `jolt-verifier` itself passes
    // `crate = "crate"` (no `extern crate self` alias).
    let krate = options
        .krate
        .clone()
        .unwrap_or_else(|| syn::parse_quote!(::jolt_verifier));

    let f = validated_field_param(&input.generics)?;
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

    let relations = quote!(#krate::stages::relations);

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

    // The instance-based absorb: each member contributes its
    // `absorbed_opening_values` (its claims' `canonical_order`-aligned values
    // minus its aliased opening ids) in declaration order. `Option` members
    // follow the claims cell (validators police presence mismatches; the absorb
    // itself is infallible).
    let claims_ident = format_ident!("__claims");
    let member_ident = format_ident!("__member");
    let opening_extends = plans.iter().map(|plan| {
        let id = &plan.ident;
        let instance = &plan.instance;
        per_member(
            plan.is_option,
            &[(&claims_ident, quote!(claims.#id))],
            quote! {
                __values.extend(#relations::absorbed_opening_values::<#f, #instance>(__claims));
            },
        )
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
                #id: self
                    .#id
                    .as_ref()
                    .map(|__member| __member.draw_challenges(transcript))
                    .transpose()?
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
    let stage_relation_id_method = quote! {
        /// The batch's representative relation id for batch-level error
        /// attribution: the first non-`Option` member's relation id. The single
        /// source read by the generated verify drivers and by `jolt-prover`'s
        /// `impl_stage_prover!` expansion.
        pub fn stage_relation_id(&self) -> ::jolt_claims::protocols::jolt::JoltRelationId {
            #relations::ConcreteSumcheck::id(&self.#stage_id_ident)
        }
    };

    // Fold each member's `(rounds, degree)` into the batch's `(max_num_vars,
    // max_degree)` — the front-loaded batching layout's combined dimensions. Reused
    // by both the clear and ZK drivers, so it is a closure re-invoked per block (a
    // `quote!` interpolation consumes its iterator).
    let max_fold = || {
        plans.iter().map(|plan| {
            let id = &plan.ident;
            per_member(
                plan.is_option,
                &[(&member_ident, quote!(self.#id))],
                quote! {
                    __max_num_vars = ::core::cmp::max(__max_num_vars, __member.rounds());
                    __max_degree = ::core::cmp::max(__max_degree, __member.degree());
                },
            )
        })
    };

    // The batched-sumcheck head, shared by the clear verify driver and the
    // prove-side stage recipes: compute each member's input claim, absorb the
    // claims through the recorder (the clear/ZK seam — a clear recorder appends
    // them, a committed recorder no-ops), draw the batching coefficients, and
    // random-linear-combine the padded sums. Every Fiat-Shamir-ordered step of
    // the head lives here, so the two sides cannot drift on it.
    let begin_batch_method = {
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
                                #krate::VerifierError::StageClaimSumcheckFailed {
                                    stage: #base_lit.to_string(),
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

        // Collect each present member's claimed sum, in declaration order, for
        // the recorder absorb — the Fiat-Shamir binding that must precede the
        // batching-coefficient draw (a clear recorder appends each under
        // `b"sumcheck_claim"`).
        let sum_ident = format_ident!("__sum");
        let sum_collects = plans.iter().zip(&sum_idents).map(|(plan, sum)| {
            per_member(
                plan.is_option,
                &[(&sum_ident, quote!(#sum))],
                quote!(__input_claims.push(*__sum);),
            )
        });

        // Draw one batching coefficient per present member (declaration order),
        // binding it to `__coeff_<member>`. `Option` members key the draw on the
        // member's bound sum — the same resolution the absorb used — so an absorb
        // and its coefficient draw can never disagree about presence (the squeeze
        // side effect fires inside the `map` exactly when the sum is present).
        let coeff_draws =
            plans
                .iter()
                .zip(sum_idents.iter().zip(&coeff_idents))
                .map(|(plan, (sum, coeff))| {
                    if plan.is_option {
                        quote! {
                            let #coeff = #sum.as_ref().map(|_| transcript.challenge_scalar());
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

        // Each present member's engine-form entry `{ input_claim, coefficient,
        // rounds }`, in declaration order. `BatchPrelude::new` folds these into
        // the combined claim `Σ coeff · sum · 2^(max − rounds)` (the front-loaded
        // padding scale).
        let member_pushes =
            plans
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
                                __members.push(::jolt_sumcheck::BatchMember {
                                    input_claim: __sum,
                                    coefficient: __coeff,
                                    rounds: __member.rounds(),
                                    offset: __member.instance_point_offset(__max_num_vars)?,
                                });
                            }
                        }
                    } else {
                        quote! {
                            __members.push(::jolt_sumcheck::BatchMember {
                                input_claim: #sum,
                                coefficient: #coeff,
                                rounds: self.#id.rounds(),
                                offset: self.#id.instance_point_offset(__max_num_vars)?,
                            });
                        }
                    }
                });

        quote! {
            /// The batched-sumcheck head, shared by `verify_clear` and the
            /// prove-side stage recipes: per-member `input_claim` (declaration
            /// order) → `recorder.absorb_input_claims` → one batching-coefficient
            /// draw per present member → the `2^(max − rounds)`-padded random
            /// linear combination. Whether the claim absorb writes transcript
            /// bytes is decided by the recorder type (clear appends, committed
            /// no-ops), never by a runtime flag. Returns the engine-form
            /// `jolt_sumcheck::BatchPrelude` paired with the stage's named
            /// batching coefficients.
            pub fn begin_batch<__R, __T>(
                &self,
                inputs: &#input_claims_name<#f>,
                challenges: &#challenges_name<#f>,
                recorder: &mut __R,
                transcript: &mut __T,
            ) -> ::core::result::Result<
                (::jolt_sumcheck::BatchPrelude<#f>, #batching_coefficients_name<#f>),
                #krate::VerifierError,
            >
            where
                __R: ::jolt_sumcheck::SumcheckRecorder<#f>,
                __T: ::jolt_transcript::Transcript<Challenge = #f>,
            {
                use #relations::ConcreteSumcheck as _;

                #(#sum_bindings)*

                let mut __max_num_vars = 0usize;
                let mut __max_degree = 0usize;
                #(#max_fold)*

                let mut __input_claims = ::std::vec::Vec::new();
                #(#sum_collects)*
                recorder.absorb_input_claims(&__input_claims, transcript);

                #(#coeff_draws)*
                let __coefficients = #batching_coefficients_name {
                    #(#coeff_fields,)*
                };

                let mut __members = ::std::vec::Vec::new();
                #(#member_pushes)*

                ::core::result::Result::Ok((
                    ::jolt_sumcheck::BatchPrelude::new(__members, __max_num_vars, __max_degree),
                    __coefficients,
                ))
            }
        }
    };

    // The composed clear-path driver: begin the batch (clear recorder, so the
    // claim absorbs are appended), verify the single-instance compressed-boolean
    // sumcheck, derive the produced opening points at the reduced point, and check
    // the reduced claim against the expected final-claim fold — the tail every
    // stage repeats verbatim. Validation (`validate_output_claims`,
    // member-presence guards) and the canonical opening absorb stay with the
    // caller: their position and form are stage-specific.
    let verify_clear_method = quote! {
        /// Run the clear-path batched verification in one call: `begin_batch`
        /// (with a clear recorder, so the claim absorbs are appended), the
        /// single-instance `SumcheckProof::verify_compressed_boolean`,
        /// [`Self::derive_opening_points`] at the reduced point, then the
        /// [`Self::expected_final_claim`] equality check (attributed to `stage`
        /// on mismatch). Returns the produced opening points.
        #[expect(
            clippy::too_many_arguments,
            reason = "the composed tail threads every per-stage aggregate once"
        )]
        pub fn verify_clear<__C, __T>(
            &self,
            inputs: &#input_claims_name<#f>,
            input_points: &#input_points_name<#f>,
            challenges: &#challenges_name<#f>,
            claims: &#output_claims_name<#f>,
            proof: &::jolt_sumcheck::SumcheckProof<#f, __C>,
            transcript: &mut __T,
            stage: usize,
        ) -> ::core::result::Result<#output_points_name<#f>, #krate::VerifierError>
        where
            __C: ::core::clone::Clone + ::jolt_transcript::AppendToTranscript,
            __T: ::jolt_transcript::Transcript<Challenge = #f>,
        {
            use #relations::ConcreteSumcheck as _;

            let mut __recorder = ::jolt_sumcheck::ClearSumcheckRecorder::<#f, __C>::new();
            let (__batch, __coefficients) =
                self.begin_batch(inputs, challenges, &mut __recorder, transcript)?;

            let __reduction = proof
                .verify_compressed_boolean(
                    __batch.max_num_vars,
                    __batch.max_degree,
                    __batch.claimed_sum,
                    transcript,
                )
                .map_err(|error| #krate::VerifierError::StageClaimSumcheckFailed {
                    stage: #base_lit.to_string(),
                    reason: error.to_string(),
                })?;

            let __output_points =
                self.derive_opening_points(__reduction.point.as_slice(), input_points)?;
            let __expected_final_claim = self.expected_final_claim(
                &__coefficients,
                input_points,
                claims,
                &__output_points,
                challenges,
            )?;
            if __reduction.value != __expected_final_claim {
                return ::core::result::Result::Err(
                    #krate::VerifierError::StageClaimOutputMismatch { stage },
                );
            }
            ::core::result::Result::Ok(__output_points)
        }
    };

    // The ZK-path batched-verify driver: compute the combined `(max_num_vars,
    // max_degree)`, draw the batching coefficients, then check committed consistency
    // through `SumcheckProof::verify_committed_consistency_dims`. Committed proofs
    // never reveal claim scalars, so no claimed sums are absorbed.
    //
    // Soundness ordering: because the batching coefficients are squeezed here
    // (before the consistency rounds) and no claim scalars are absorbed, the
    // caller MUST have already absorbed commitments to those claim scalars into
    // the transcript before invoking this driver. Without that prior binding a
    // malicious prover could choose its claim openings adaptively after seeing
    // the batching challenge.
    let verify_zk_method = {
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
                #krate::VerifierError,
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
                    .map_err(|error| #krate::VerifierError::StageClaimSumcheckFailed {
                        stage: #base_lit.to_string(),
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
    };

    // Map each member's opening point through its
    // `ConcreteSumcheck::derive_opening_points` into the stage's `OutputPoints`
    // aggregate. Takes the batch challenge vector directly: an instance's point
    // is the length-`rounds` slice of that vector starting at its
    // `instance_point_offset`, so no batch-result abstraction is needed and one
    // method serves both the clear and ZK paths (each supplies its own
    // challenge vector).
    let derive_points_method = {
        let field_bindings = plans.iter().map(|plan| {
            let id = &plan.ident;
            let field = format_ident!("__points_{}", id);
            let binding = if plan.is_option {
                quote! {
                    let #field = match (self.#id.as_ref(), input_points.#id.as_ref()) {
                        (
                            ::core::option::Option::Some(__member),
                            ::core::option::Option::Some(__input_points),
                        ) => ::core::option::Option::Some(__member.derive_opening_points(
                            __member.instance_point(batch_point)?,
                            __input_points,
                        )?),
                        (::core::option::Option::None, _) => ::core::option::Option::None,
                        (::core::option::Option::Some(__member), ::core::option::Option::None) => {
                            return ::core::result::Result::Err(
                                #krate::VerifierError::StageClaimSumcheckFailed {
                                    stage: #base_lit.to_string(),
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
                        self.#id.instance_point(batch_point)?,
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
            ) -> ::core::result::Result<#output_points_name<#f>, #krate::VerifierError> {
                use #relations::ConcreteSumcheck as _;

                #(#point_bindings)*
                ::core::result::Result::Ok(#output_points_name {
                    #(#point_fields,)*
                })
            }
        }
    };

    // Enforce each member's declared cross-relation opening aliases: the wire's
    // aliased cell must equal its canonical source's cell (resolved by id across
    // the batch, so exactly one member answers). Value-only: opening points are
    // derived, not wire data, and an alias is declarable only when both relations
    // bind the same batch-point slice identically — a structural invariant.
    // Always generated and run by `expected_final_claim` (the fold is where the
    // aliased values get consumed), so declaring a pair on a relation enforces it
    // everywhere: the check cannot be skipped by a stage.
    let validate_aliases_method = {
        let resolve_arms = plans.iter().map(|plan| {
            let id = &plan.ident;
            if plan.is_option {
                quote! {
                    .or_else(|| {
                        output_values
                            .#id
                            .as_ref()
                            .and_then(|__claims| __claims.resolve_output(__id))
                    })
                }
            } else {
                quote!(.or_else(|| output_values.#id.resolve_output(__id)))
            }
        });
        let claims_cell_ident = format_ident!("__claims_cell");
        let alias_checks = plans.iter().map(|plan| {
            let id = &plan.ident;
            let instance = &plan.instance;
            per_member(
                plan.is_option,
                &[
                    (&member_ident, quote!(self.#id)),
                    (&claims_cell_ident, quote!(output_values.#id)),
                ],
                quote! {
                    #relations::validate_member_aliases::<#f, #instance>(
                        __member,
                        __claims_cell,
                        &__resolve,
                    )?;
                },
            )
        });
        quote! {
            /// Enforce every member's declared cross-relation opening aliases
            /// (`ConcreteSumcheck::aliased_output_openings`): each aliased wire
            /// cell must equal its canonical source opening, resolved by id across
            /// the batch. Load-bearing — see `relations::validate_member_aliases`.
            /// Run by `expected_final_claim`; callable directly by tests.
            pub fn validate_aliases(
                &self,
                output_values: &#output_claims_name<#f>,
            ) -> ::core::result::Result<(), #krate::VerifierError> {
                use ::jolt_claims::OutputClaims as _;
                let __resolve = |__id: &::jolt_claims::protocols::jolt::JoltOpeningId| {
                    ::core::option::Option::<#f>::None
                        #(#resolve_arms)*
                };
                #(#alias_checks)*
                ::core::result::Result::Ok(())
            }
        }
    };

    // Fold the members' expected output claims with the batch coefficients into the
    // final claim the reduction is checked against: `Σ coeff_m * expected_output_m`.
    // A present `Option` member with any absent cell errors rather than silently
    // dropping its term (which would surface as an opaque final-claim mismatch).
    let expected_final_claim_method = {
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
                                #krate::VerifierError::StageClaimSumcheckFailed {
                                    stage: #base_lit.to_string(),
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
            ) -> ::core::result::Result<#f, #krate::VerifierError> {
                use #relations::ConcreteSumcheck as _;
                // The fold consumes the aliased wire copies, so their equality
                // with the canonical sources is enforced here, unskippably.
                self.validate_aliases(output_values)?;
                let mut __terms = ::std::vec::Vec::new();
                #(#output_terms)*
                let __expected: #f = __terms.into_iter().sum();
                ::core::result::Result::Ok(__expected)
            }
        }
    };

    // An all-default `InputPoints` constructor for relations that read no input
    // opening points: each non-`Option` cell is `Default::default()`, each
    // `Option` cell tracks its member's presence (exactly the invariant the
    // generated drivers require).
    let empty_input_points_method = {
        let cell_inits = plans.iter().map(|plan| {
            let id = &plan.ident;
            if plan.is_option {
                quote!(#id: self.#id.as_ref().map(|_| ::core::default::Default::default()))
            } else {
                quote!(#id: ::core::default::Default::default())
            }
        });
        quote! {
            /// An all-default consumed-`InputPoints` aggregate for a stage whose
            /// relations read no input opening points: each non-`Option` cell is
            /// `Default::default()`, each `Option` cell tracks its member's presence.
            pub fn empty_input_points(&self) -> #input_points_name<#f> {
                #input_points_name {
                    #(#cell_inits,)*
                }
            }
        }
    };

    // The output-claim shape helpers: the total produced-opening count (for the ZK
    // commitment count) and a validator that the proof-supplied output claims match
    // the dims-derived expected shape. Both delegate per member to the generic
    // `relations` helpers; an `Option` member's presence guards run first (a stage
    // that curates its own shape checks calls `validate_member_presence` by hand —
    // stage 6b). Suppressed by `#[sumcheck_batch(no_output_shape)]` for a stage
    // whose wire shape is runtime-deduped (the count/validator would be wrong).
    let output_shape_methods = if options.no_output_shape {
        quote!()
    } else {
        let count_terms = plans.iter().map(|plan| {
            let id = &plan.ident;
            per_member(
                plan.is_option,
                &[(&member_ident, quote!(self.#id))],
                quote!(__count += __member.wire_output_openings().len();),
            )
        });
        let validate_checks = plans.iter().map(|plan| {
            let id = &plan.ident;
            let instance = &plan.instance;
            let shape = per_member(
                plan.is_option,
                &[
                    (&member_ident, quote!(self.#id)),
                    (&claims_ident, quote!(claims.#id)),
                ],
                quote! {
                    #relations::validate_member_output_shape::<#f, #instance>(
                        __member, __claims,
                    )?;
                },
            );
            if plan.is_option {
                quote! {
                    #relations::validate_member_presence::<#f, #instance>(
                        self.#id.as_ref(),
                        claims.#id.as_ref(),
                    )?;
                    #shape
                }
            } else {
                shape
            }
        });
        quote! {
            /// The total number of absorbed/committed opening claims across the
            /// batch (each member's `ConcreteSumcheck::wire_output_openings`),
            /// e.g. the committed-output-claim count.
            pub fn output_claim_count(&self) -> usize {
                use #relations::ConcreteSumcheck as _;
                let mut __count = 0usize;
                #(#count_terms)*
                __count
            }

            /// Assert the proof-supplied output claims match the expected shape: per
            /// member, the provided `canonical_order` id-set (minus the member's
            /// aliased openings) equals the relation's
            /// `ConcreteSumcheck::wire_output_openings`; an `Option` member's claim
            /// presence must agree with the instance's.
            pub fn validate_output_claims(
                &self,
                claims: &#output_claims_name<#f>,
            ) -> ::core::result::Result<(), #krate::VerifierError> {
                #(#validate_checks)*
                ::core::result::Result::Ok(())
            }
        }
    };

    // Suppressed by `#[sumcheck_batch(no_draw_challenges)]`: a stage whose member
    // challenges have stage-level provenance (shared squeezes, pre-batch draws,
    // value re-rolls) hand-assembles its challenge aggregate, and a generated
    // per-member draw would squeeze at the wrong transcript position if ever
    // called — so it must not exist to be miscalled.
    let draw_challenges_method = if options.no_draw_challenges {
        quote!()
    } else {
        quote! {
            /// Draw each instance's Fiat-Shamir challenges in declaration order,
            /// assembling the stage's challenge aggregate. Members with no
            /// challenges draw nothing; `Option` members draw only when present.
            /// This single-sources the stage's inline per-instance draw, so its
            /// Fiat-Shamir order follows member declaration order.
            pub fn draw_challenges<__T: ::jolt_transcript::Transcript<Challenge = #f>>(
                &self,
                transcript: &mut __T,
            ) -> ::core::result::Result<#challenges_name<#f>, #krate::VerifierError> {
                use #relations::ConcreteSumcheck as _;
                ::core::result::Result::Ok(#challenges_name {
                    #(#draw_fields,)*
                })
            }
        }
    };

    // The generated absorb plumbing, on the source `StageNSumchecks` struct (it
    // consults each member's `aliased_output_openings` skip-set, an instance
    // method). Gated out when the stage opts in to
    // `#[sumcheck_batch(no_opening_values)]`, in which case the stage supplies
    // its own absorb (e.g. one whose order interleaves members, or whose dedup is
    // runtime point-driven).
    let absorb_methods = if options.no_opening_values {
        quote!()
    } else {
        quote! {
            /// Produced opening scalars in canonical order — member declaration
            /// order, each member's claims in its `canonical_order` — skipping
            /// each member's aliased openings (absorbed once via their canonical
            /// source relation). This is the Fiat-Shamir order and MUST match the
            /// prover's commitment order.
            pub fn opening_values(&self, claims: &#output_claims_name<#f>) -> ::std::vec::Vec<#f> {
                let mut __values = ::std::vec::Vec::new();
                #(#opening_extends)*
                __values
            }

            /// Append every absorbed opening to the transcript in canonical order,
            /// each under the `b"opening_claim"` label, matching the prover's
            /// commitment order.
            pub fn append_output_claims<__T: ::jolt_transcript::Transcript<Challenge = #f>>(
                &self,
                transcript: &mut __T,
                claims: &#output_claims_name<#f>,
            ) {
                for value in self.opening_values(claims) {
                    transcript.append_labeled(b"opening_claim", &value);
                }
            }
        }
    };

    // The prover-facing single-sourcing handoff: an inert, exported callback
    // macro carrying this batch's declaration — member names, relation paths
    // (generics stripped; the consumer re-applies its own field parameter),
    // presence, the aggregate type names, and the output-shape flag — as a
    // structured token list forwarded to a caller-chosen macro. `jolt-prover`'s
    // `impl_stage_prover` expands its `StageProver`/`KernelSource` impls from
    // it, so no stage's member list, order, or presence is ever restated.
    // Tokens resolve at the consumer's invocation site (which imports the
    // batch's relation and aggregate names); extra invocation tokens (e.g. a
    // curation override) are forwarded ahead of the list.
    let members_macro = {
        let macro_name = format_ident!("{}_members", snake_case(&name.to_string()));
        let macro_doc = format!(
            "The member-list callback macro for [`{name}`], emitted by \
             `#[derive(SumcheckBatch)]`: forwards the batch's declaration (member names, \
             relation paths, presence, aggregate names, output-shape flag) to a caller-chosen \
             macro. See `specs/prover-stage-drivers.md`."
        );
        let shape = if options.no_output_shape {
            format_ident!("unchecked")
        } else {
            format_ident!("checked")
        };
        let member_entries = plans
            .iter()
            .map(|plan| {
                let id = &plan.ident;
                let relation = relation_path(&plan.instance)?;
                let presence = if plan.is_option {
                    format_ident!("optional")
                } else {
                    format_ident!("required")
                };
                Ok(quote! {
                    { name: #id, relation: #relation, presence: #presence },
                })
            })
            .collect::<syn::Result<Vec<_>>>()?;
        quote! {
            #[doc = #macro_doc]
            #[macro_export]
            macro_rules! #macro_name {
                ($cb:ident $($extra:tt)*) => {
                    $cb! {
                        $($extra)*
                        batch = #name,
                        aggregates = {
                            input_claims = #input_claims_name,
                            input_points = #input_points_name,
                            output_claims = #output_claims_name,
                            output_points = #output_points_name,
                            challenges = #challenges_name,
                        },
                        shape = #shape,
                        members = [
                            #(#member_entries)*
                        ]
                    }
                };
            }
        }
    };

    let driver_impl = quote! {
        impl<#f: ::jolt_field::Field> #name<#f> {
            #stage_relation_id_method
            #draw_challenges_method

            #begin_batch_method
            #verify_clear_method
            #verify_zk_method
            #derive_points_method
            #validate_aliases_method
            #expected_final_claim_method
            #output_shape_methods
            #empty_input_points_method
            #absorb_methods
        }
    };

    // Each opening cell instantiation is its own concrete aggregate: `*Claims`
    // holds the wire *values* (`Inputs<F>` / `Outputs<F>`); `*Points` holds the
    // derived opening points (`Inputs<Vec<F>>` / `Outputs<Vec<F>>`). Only the
    // `OutputClaims` (values) aggregate is serialized (the wire form), so it alone
    // derives serde. `F: Field` does not imply the serde traits, so the bounds are
    // spelled explicitly (the workspace convention for claim structs), fully
    // qualified so call sites need no serde imports.
    let serialize_bound = format!("{f}: ::serde::Serialize");
    let deserialize_bound = format!("{f}: for<'a> ::serde::Deserialize<'a>");

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
        #[serde(bound(serialize = #serialize_bound, deserialize = #deserialize_bound))]
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

        #members_macro

        #driver_impl
    })
}

/// Struct-level `#[sumcheck_batch(...)]` configuration. Parsed from the source
/// struct's attributes; recognizes only the flags below and errors clearly on
/// anything else.
#[derive(Default)]
struct StageOptions {
    /// `#[sumcheck_batch(no_opening_values)]`: skip emitting the generated
    /// `opening_values` / `append_output_claims` absorb methods so the stage can
    /// supply its own (a member-interleaved order or a runtime point-driven dedup).
    no_opening_values: bool,
    /// `#[sumcheck_batch(no_output_shape)]`: skip emitting `output_claim_count` and
    /// `validate_output_claims` (which derive the expected output-claim shape from
    /// each member's `wire_output_openings`) for a stage whose wire shape is not
    /// statically derivable (a runtime point-driven dedup) — the count/validator
    /// would be wrong there, so they must not exist to be miscalled.
    no_output_shape: bool,
    /// `#[sumcheck_batch(no_draw_challenges)]`: skip emitting the generated
    /// `draw_challenges` for a stage whose member challenges have stage-level
    /// provenance (shared squeezes, pre-batch draws, value re-rolls) — calling a
    /// per-member draw there would squeeze at the wrong transcript position, so
    /// the method must not exist to be miscalled.
    no_draw_challenges: bool,
    /// `#[sumcheck_batch(crate = "...")]`: the path the generated code names
    /// `jolt-verifier` by (serde's `crate` attribute shape). `None` means the
    /// absolute `::jolt_verifier` default; the defining crate passes
    /// `"crate"`. Never applied to the emitted member-list callback macro,
    /// whose tokens resolve at the (cross-crate) invocation site.
    krate: Option<syn::Path>,
}

impl StageOptions {
    fn parse(attrs: &[Attribute]) -> syn::Result<Self> {
        let mut options = StageOptions::default();
        for attr in attrs {
            if !attr.path().is_ident("sumcheck_batch") {
                continue;
            }
            // `#[sumcheck_batch(flag, ..., crate = "path")]` — a
            // comma-separated list of bare-word flags (`Meta::Path`) plus the
            // optional crate-path override. Reject any other form or unknown
            // flag with a span-pointed error.
            let flags = attr.parse_args_with(
                syn::punctuated::Punctuated::<Meta, Token![,]>::parse_terminated,
            )?;
            for flag in flags {
                if let Meta::NameValue(name_value) = &flag {
                    if name_value.path.is_ident("crate") {
                        options.krate = Some(parse_crate_path(name_value)?);
                        continue;
                    }
                }
                let Meta::Path(path) = &flag else {
                    return Err(syn::Error::new_spanned(
                        &flag,
                        "expected a bare `sumcheck_batch` flag (e.g. `no_opening_values`) or \
                         `crate = \"...\"`",
                    ));
                };
                if path.is_ident("no_opening_values") {
                    options.no_opening_values = true;
                } else if path.is_ident("no_output_shape") {
                    options.no_output_shape = true;
                } else if path.is_ident("no_draw_challenges") {
                    options.no_draw_challenges = true;
                } else {
                    return Err(syn::Error::new_spanned(
                        path,
                        "unknown `sumcheck_batch` flag (supported: `no_opening_values`, \
                         `no_output_shape`, `no_draw_challenges`, `crate = \"...\"`)",
                    ));
                }
            }
        }
        Ok(options)
    }
}

/// Parse the serde-style `crate = "..."` value: a string literal holding the
/// path the generated code names the defining crate by (`"crate"` in
/// `jolt-verifier` itself, a re-export path in a wrapping crate).
fn parse_crate_path(name_value: &syn::MetaNameValue) -> syn::Result<syn::Path> {
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
    lit.parse()
}

/// The macro supports exactly one generic type parameter (the field `F`,
/// returned): no extra generics, no lifetimes/consts, and no where-clause (any
/// of which the generated aggregates would silently drop). Reject anything else
/// with a clear error rather than binding the wrong `F` or emitting wrong
/// projections.
fn validated_field_param(generics: &syn::Generics) -> syn::Result<Ident> {
    if let Some(where_clause) = &generics.where_clause {
        return Err(syn::Error::new_spanned(
            where_clause,
            "SumcheckBatch does not support a where-clause on the source struct",
        ));
    }
    match generics.params.iter().collect::<Vec<_>>().as_slice() {
        [GenericParam::Type(param)] => Ok(param.ident.clone()),
        _ => Err(syn::Error::new_spanned(
            generics,
            "SumcheckBatch requires exactly one generic type parameter (the field `F`)",
        )),
    }
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

/// `CamelCase` → `snake_case` for the emitted member-list macro's name
/// (`Stage1BatchSumchecks` → `stage1_batch_sumchecks`).
fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (index, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if index != 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// The relation path of a member field with its generic arguments stripped
/// (`SpartanShift<F>` → `SpartanShift`): the consumer macro re-applies its own
/// field-type parameter, so the emitted token needs no hygiene agreement on
/// the parameter name.
fn relation_path(instance: &Type) -> syn::Result<syn::Path> {
    let Type::Path(path) = instance else {
        return Err(syn::Error::new_spanned(
            instance,
            "SumcheckBatch member types must be paths",
        ));
    };
    let mut path = path.path.clone();
    if let Some(segment) = path.segments.last_mut() {
        segment.arguments = PathArguments::None;
    }
    Ok(path)
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
