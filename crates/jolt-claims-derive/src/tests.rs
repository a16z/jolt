//! Expansion snapshots and error-path tests for the claim derives.
//!
//! The macro's output is invisible to mutation testing and to reviewers of
//! consumer crates, so the expansions of representative structs (covering the
//! full `#[opening(..)]` grammar) are pinned as pretty-printed snapshots in
//! `testdata/`. A drifted snapshot means the generated wiring changed for
//! every claim struct in the workspace; regenerate deliberately with
//! `JOLT_CLAIMS_DERIVE_REGENERATE_SNAPSHOTS=1` and review the diff.
#![expect(
    clippy::expect_used,
    reason = "snapshot harness should fail loudly on I/O or parse errors"
)]

use std::{env, fs, path::PathBuf};

use proc_macro2::TokenStream as TokenStream2;
use syn::parse_quote;

use crate::{expand_challenges, expand_input, expand_output};

const REGENERATE_ENV: &str = "JOLT_CLAIMS_DERIVE_REGENERATE_SNAPSHOTS";

fn pretty(tokens: TokenStream2) -> String {
    let file = syn::parse2::<syn::File>(tokens).expect("expansion must parse as a list of items");
    prettyplease::unparse(&file)
}

fn assert_snapshot(name: &str, actual: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(format!("{name}.expanded.rs"));
    if env::var_os(REGENERATE_ENV).is_some() {
        fs::write(&path, actual).expect("failed to write snapshot");
        return;
    }
    let expected = fs::read_to_string(&path).expect(
        "missing expansion snapshot; generate it with JOLT_CLAIMS_DERIVE_REGENERATE_SNAPSHOTS=1",
    );
    assert_eq!(
        actual, expected,
        "macro expansion drifted from testdata/{name}.expanded.rs; this changes the generated \
         claim wiring for every derive site. If intentional, regenerate with \
         {REGENERATE_ENV}=1 and review the diff."
    );
}

/// Covers the full `OutputClaims` grammar: scalar virtual, payload-carrying
/// virtual, indexed (`Vec`) family, conditional (`Option`) committed, and
/// scalar advice openings.
fn representative_output_struct() -> syn::DeriveInput {
    parse_quote! {
        #[relation(SpartanOuter)]
        struct DemoOutputClaims<C> {
            #[opening(PC)]
            pc: C,
            #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
            virtual_flag: C,
            #[opening(LookupTableFlag)]
            table_flags: Vec<C>,
            #[opening(committed = RamInc)]
            ram_inc: Option<C>,
            #[opening(trusted_advice)]
            advice: C,
        }
    }
}

#[test]
fn output_claims_expansion_matches_snapshot() {
    let tokens = expand_output(representative_output_struct()).expect("expansion should succeed");
    assert_snapshot("output_claims", &pretty(tokens));
}

#[test]
fn input_claims_expansion_matches_snapshot() {
    let input: syn::DeriveInput = parse_quote! {
        struct DemoInputClaims<C> {
            #[opening(PC, from = SpartanOuter)]
            pc: C,
            #[opening(committed = RamInc, from = RamReadWriteChecking)]
            ram_inc: Option<C>,
            #[opening(LookupTableFlag, from = InstructionReadRaf)]
            table_flags: Vec<C>,
            #[opening(untrusted_advice, from = AdviceClaimReduction)]
            advice: C,
        }
    };
    let tokens = expand_input(input).expect("expansion should succeed");
    assert_snapshot("input_claims", &pretty(tokens));
}

#[test]
fn sumcheck_challenges_expansion_matches_snapshot() {
    let input: syn::DeriveInput = parse_quote! {
        struct DemoChallenges<F> {
            #[challenge(SpartanChallenges::Tau)]
            tau: F,
            #[challenge(SpartanChallenges::RInner)]
            r_inner: F,
        }
    };
    let tokens = expand_challenges(input).expect("expansion should succeed");
    assert_snapshot("sumcheck_challenges", &pretty(tokens));
}

/// The macro's core contract: the canonical opening order is single-sourced
/// from field declaration order. The ids must appear in the expansion in the
/// same order the fields are declared, for both derives.
#[test]
fn canonical_order_follows_field_declaration_order() {
    let tokens = expand_output(representative_output_struct()).expect("expansion should succeed");
    let expansion = pretty(tokens);
    let canonical_order = expansion
        .split("fn canonical_order")
        .nth(1)
        .and_then(|rest| rest.split("fn resolve_output").next())
        .expect("expansion contains canonical_order before resolve_output");
    let id_positions: Vec<usize> = [
        "JoltVirtualPolynomial::PC",
        "CircuitFlags::VirtualInstruction",
        "JoltVirtualPolynomial::LookupTableFlag",
        "JoltCommittedPolynomial::RamInc",
        "trusted_advice",
    ]
    .iter()
    .map(|needle| {
        canonical_order
            .find(needle)
            .expect("canonical_order emits an id for every declared field")
    })
    .collect();
    assert!(
        id_positions.windows(2).all(|pair| pair[0] < pair[1]),
        "canonical_order ids out of declaration order: positions {id_positions:?}"
    );
}

#[track_caller]
fn expect_output_error(input: syn::DeriveInput, expected: &str) {
    let error = expand_output(input)
        .expect_err("expansion should be rejected")
        .to_string();
    assert_eq!(error, expected);
}

#[test]
fn output_claims_rejects_non_structs() {
    expect_output_error(
        parse_quote! { enum Demo<C> { A(C) } },
        "OutputClaims/InputClaims can only be derived for structs",
    );
}

#[test]
fn output_claims_rejects_tuple_structs() {
    expect_output_error(
        parse_quote! { #[relation(SpartanOuter)] struct Demo<C>(C); },
        "OutputClaims/InputClaims require a struct with named fields",
    );
}

#[test]
fn output_claims_rejects_extra_generics() {
    expect_output_error(
        parse_quote! {
            #[relation(SpartanOuter)]
            struct Demo<C, D> {
                #[opening(PC)]
                pc: C,
                other: D,
            }
        },
        "OutputClaims/InputClaims require exactly one generic type parameter (the opening cell, e.g. `<C>`)",
    );
}

#[test]
fn output_claims_rejects_where_clauses() {
    expect_output_error(
        parse_quote! {
            #[relation(SpartanOuter)]
            struct Demo<C>
            where
                C: Clone,
            {
                #[opening(PC)]
                pc: C,
            }
        },
        "OutputClaims/InputClaims require exactly one generic type parameter (the opening cell, e.g. `<C>`)",
    );
}

#[test]
fn output_claims_rejects_lifetime_parameters() {
    expect_output_error(
        parse_quote! {
            #[relation(SpartanOuter)]
            struct Demo<'a, C> {
                #[opening(PC)]
                pc: &'a C,
            }
        },
        "OutputClaims/InputClaims require exactly one generic type parameter (the opening cell, e.g. `<C>`)",
    );
}

#[test]
fn output_claims_rejects_duplicate_relation_attrs() {
    expect_output_error(
        parse_quote! {
            #[relation(SpartanOuter)]
            #[relation(SpartanInner)]
            struct Demo<C> {
                #[opening(PC)]
                pc: C,
            }
        },
        "duplicate #[relation(..)] attribute",
    );
}

#[test]
fn output_claims_rejects_from_on_output_openings() {
    expect_output_error(
        parse_quote! {
            #[relation(SpartanOuter)]
            struct Demo<C> {
                #[opening(PC, from = SpartanOuter)]
                pc: C,
            }
        },
        "`from = ..` is only used by InputClaims; OutputClaims uses #[relation(..)]",
    );
}

#[test]
fn output_claims_rejects_unannotated_fields() {
    expect_output_error(
        parse_quote! {
            #[relation(SpartanOuter)]
            struct Demo<C> {
                pc: C,
            }
        },
        "every field needs an #[opening(..)] annotation (nested aggregates are not supported)",
    );
}

#[test]
fn output_claims_rejects_multiple_kinds_in_one_opening() {
    expect_output_error(
        parse_quote! {
            #[relation(SpartanOuter)]
            struct Demo<C> {
                #[opening(PC, committed = RamInc)]
                pc: C,
            }
        },
        "#[opening(..)] must name exactly one opening",
    );
}

#[test]
fn output_claims_rejects_vec_advice_fields() {
    expect_output_error(
        parse_quote! {
            #[relation(SpartanOuter)]
            struct Demo<C> {
                #[opening(trusted_advice)]
                advice: Vec<C>,
            }
        },
        "advice openings are scalar; a `Vec` advice field has no indexed id",
    );
}

#[test]
fn output_claims_rejects_vec_payload_fields() {
    expect_output_error(
        parse_quote! {
            #[relation(SpartanOuter)]
            struct Demo<C> {
                #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
                flags: Vec<C>,
            }
        },
        "per-element payload arrays (e.g. `OpFlags(CIRCUIT_FLAGS)` on a `Vec`) are not supported; declare one field per element instead",
    );
}

#[test]
fn input_claims_rejects_leaves_without_from() {
    let error = expand_input(parse_quote! {
        struct Demo<C> {
            #[opening(PC)]
            pc: C,
        }
    })
    .expect_err("expansion should be rejected")
    .to_string();
    assert_eq!(
        error,
        "missing `from = ProducingRelation` on an input opening (or missing struct-level #[relation(..)] for an output opening)"
    );
}

#[track_caller]
fn expect_challenges_error(input: syn::DeriveInput, expected: &str) {
    let error = expand_challenges(input)
        .expect_err("expansion should be rejected")
        .to_string();
    assert_eq!(error, expected);
}

#[test]
fn challenges_reject_vec_fields() {
    expect_challenges_error(
        parse_quote! {
            struct Demo<F> {
                #[challenge(SpartanChallenges::Tau)]
                tau: Vec<F>,
            }
        },
        "challenges are scalar; a `Vec` challenge field has no indexed id (every challenge sub-enum variant is a unit variant)",
    );
}

#[test]
fn challenges_reject_option_fields() {
    expect_challenges_error(
        parse_quote! {
            struct Demo<F> {
                #[challenge(SpartanChallenges::Tau)]
                tau: Option<F>,
            }
        },
        "challenge fields are an unconditional scalar `F`; a conditional `Option<F>` challenge is not supported (no relation draws one, and the `draw_challenges` default treats every field as one `challenge_scalar`)",
    );
}

#[test]
fn challenges_reject_missing_generic_parameter() {
    expect_challenges_error(
        parse_quote! {
            struct Demo {
                #[challenge(SpartanChallenges::Tau)]
                tau: u64,
            }
        },
        "expected a field-type generic parameter (e.g. `<F>`)",
    );
}

#[test]
fn challenges_reject_unannotated_fields() {
    expect_challenges_error(
        parse_quote! {
            struct Demo<F> {
                tau: F,
            }
        },
        "every field needs a #[challenge(SubEnum::Variant)] annotation",
    );
}
