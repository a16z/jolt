#![expect(
    clippy::expect_used,
    reason = "catalog tests should fail loudly when a table or grid is malformed"
)]

//! The Jolt-owned schedule catalogs: coverage and drift guards.

use akita_config::{honest_fold_policy_of, CommitmentConfig};
use akita_types::{
    commit_only_setup_field_elements, setup_matrix_capacity_for_schedule, AkitaScheduleLookupKey,
};
use jolt_akita::configs::{JoltDenseBounded, JoltOneHotK16, JoltOneHotK256};
use jolt_akita::schedule_registry::{FIXTURE_K16_FINAL_NUM_VARS, FIXTURE_TRUSTED_ADVICE_GROUP};
use jolt_akita::schedules::emit::{
    family_specs, keys, K16_NUM_VARS, K256_NUM_VARS, ONE_HOT_TRACE_NUM_POLYS,
};
use jolt_akita::schedules::{jolt_fp128_onehot_k16_table, jolt_fp128_onehot_k256_table};

/// Every key of a family grid resolves from its checked-in table (binary
/// lookup over sorted entries) — no planner-DP fallback for reachable
/// `OneHotTrace` shapes. Identity validity is exercised by every akita e2e (an
/// identity mismatch hard-errors instead of falling back).
#[test]
fn catalogs_cover_every_reachable_one_hot_trace_shape() {
    for (table, num_polys, num_vars) in [
        (
            jolt_fp128_onehot_k16_table().expect("K16 catalog is checked in"),
            ONE_HOT_TRACE_NUM_POLYS,
            K16_NUM_VARS,
        ),
        (
            jolt_fp128_onehot_k256_table().expect("K256 catalog is checked in"),
            ONE_HOT_TRACE_NUM_POLYS,
            K256_NUM_VARS,
        ),
    ] {
        let grid = keys(num_polys, num_vars);
        assert!(!grid.is_empty());
        for key in grid {
            assert!(
                table.entries.iter().any(|entry| {
                    entry.root.final_group.layout == key
                        && entry.root.precommitted_groups.is_empty()
                }),
                "missing catalog entry for {key:?}"
            );
        }
        assert_eq!(
            table.identity.key_count,
            table.entries.len(),
            "identity key count must match the table"
        );
    }
}

/// Production trusted-advice precommit layout (`2^20` u64 words) and the
/// SHA2-chain packed-trace layout at `log_T = 26`, K=256. No grouped row is
/// checked in for either — this test is what asserts that.
const TRUSTED_ADVICE_GROUP: akita_types::PolynomialGroupLayout =
    akita_types::PolynomialGroupLayout::new(20, 1);
const TRUSTED_ADVICE_K256_FINAL_GROUP: akita_types::PolynomialGroupLayout =
    akita_types::PolynomialGroupLayout::new(39, 1);

fn trusted_advice_grouped_key() -> AkitaScheduleLookupKey {
    let trusted_profile =
        JoltDenseBounded::profile_without_precommitted_groups(TRUSTED_ADVICE_GROUP)
            .expect("trusted advice standalone row must resolve");
    AkitaScheduleLookupKey {
        final_group: TRUSTED_ADVICE_K256_FINAL_GROUP,
        precommitteds: vec![trusted_profile],
    }
}

/// No grouped advice row is checked in: a grouped row is keyed on the frozen
/// precommit profiles, which follow the program's advice capacity, so
/// preprocessing plans every one. Provisioning a single production-shaped key
/// must install it and make it resolve through the public hooks.
#[test]
fn grouped_advice_rows_are_planned_not_cataloged() {
    let key = trusted_advice_grouped_key();
    assert!(
        JoltOneHotK256::resolve_catalog_row_for_key(&key).is_err(),
        "a grouped advice row must never be checked in"
    );

    let rows = jolt_akita::schedule_registry::provision::<JoltOneHotK256>(
        std::slice::from_ref(&key.precommitteds),
        honest_fold_policy_of::<JoltDenseBounded>(),
        [key.final_group.num_vars()],
    )
    .expect("preprocessing must plan the production grouped row");
    assert_eq!(rows.rows().count(), 1);

    let resolved = JoltOneHotK256::resolve_catalog_row_for_key(&key)
        .expect("the provisioned row must resolve by key");
    assert_eq!(resolved.profiles().precommitteds, key.precommitteds);
    assert_eq!(resolved.profiles().final_group.group, key.final_group);
    assert_eq!(resolved.schedule().root.params.precommitted_groups.len(), 1);

    // The verifier only ever sees the public selection.
    let by_selection = JoltOneHotK256::resolve_schedule_selection(resolved.selection())
        .expect("the provisioned row must resolve by public selection");
    assert_eq!(by_selection.profiles(), resolved.profiles());
}

/// Two dense precommits ahead of the same final trace group — the
/// `[UntrustedAdvice, TrustedAdvice, OneHotTrace]` shape. Untrusted and trusted
/// share the production arity, so both are the same frozen profile.
#[test]
fn two_precommit_grouped_advice_row_is_planned() {
    let mut key = trusted_advice_grouped_key();
    key.precommitteds.push(key.precommitteds[0]);

    let rows = jolt_akita::schedule_registry::provision::<JoltOneHotK256>(
        std::slice::from_ref(&key.precommitteds),
        honest_fold_policy_of::<JoltDenseBounded>(),
        [key.final_group.num_vars()],
    )
    .expect("preprocessing must plan the two-precommit grouped row");
    assert_eq!(rows.rows().count(), 1);

    let resolved = JoltOneHotK256::resolve_catalog_row_for_key(&key)
        .expect("the provisioned row must resolve by key");
    assert_eq!(resolved.profiles().precommitteds, key.precommitteds);
    assert_eq!(resolved.schedule().root.params.precommitted_groups.len(), 2);

    assert!(key
        .fits_setup_capacity(39, 3)
        .expect("grouped capacity arithmetic must not overflow"));
    assert!(!key
        .fits_setup_capacity(39, 2)
        .expect("grouped capacity arithmetic must not overflow"));
}

#[test]
fn grouped_setup_capacity_covers_precommit_and_complete_schedule() {
    let key = trusted_advice_grouped_key();
    assert!(key
        .fits_setup_capacity(39, 2)
        .expect("grouped capacity arithmetic must not overflow"));
    assert!(!key
        .fits_setup_capacity(39, 1)
        .expect("grouped capacity arithmetic must not overflow"));

    // Grouped advice rows are planned, not cataloged, so install it first.
    let _rows = jolt_akita::schedule_registry::provision::<JoltOneHotK256>(
        std::slice::from_ref(&key.precommitteds),
        honest_fold_policy_of::<JoltDenseBounded>(),
        [key.final_group.num_vars()],
    )
    .expect("preprocessing must plan the production grouped row");
    let resolved = JoltOneHotK256::resolve_catalog_row_for_key(&key)
        .expect("trusted advice plus K256 final row must resolve");
    let full_capacity = setup_matrix_capacity_for_schedule(resolved.schedule())
        .expect("grouped schedule capacity must be valid");
    let prefix = key.precommitteds[0];
    let precommit_capacity = commit_only_setup_field_elements(
        &prefix.inner_commit_matrix,
        &prefix.outer_commit_matrix,
        prefix.outer_slice_count,
    )
    .expect("trusted precommit capacity must be valid");

    let setup_capacity = JoltOneHotK256::setup_matrix_capacity(39, 2)
        .expect("grouped production setup shape must be catalog-backed");
    assert!(setup_capacity.num_field_elements >= full_capacity.num_field_elements);
    assert!(setup_capacity.num_field_elements >= precommit_capacity);

    let scalar_only_capacity = JoltOneHotK256::setup_matrix_capacity(39, 1)
        .expect("scalar production setup shape must remain catalog-backed");
    assert!(scalar_only_capacity.num_field_elements >= precommit_capacity);
}

/// No grouped advice row is checked in for either K=16 config, and setup
/// capacity still covers the grouped shapes preprocessing will plan. The
/// production and grouped-capable configs share one catalog, so the assertion
/// covers both.
#[test]
fn no_family_catalogs_a_grouped_advice_row() {
    let trusted_profile =
        JoltDenseBounded::profile_without_precommitted_groups(FIXTURE_TRUSTED_ADVICE_GROUP)
            .expect("fixture trusted advice standalone row must resolve");
    let table = jolt_fp128_onehot_k16_table().expect("K16 catalog is checked in");
    assert!(
        table
            .entries
            .iter()
            .all(|entry| entry.root.precommitted_groups.is_empty()),
        "the K=16 catalog must carry no grouped rows"
    );

    // One and two dense precommits, covering the single-advice and both-advice
    // batch shapes. The fixture's untrusted and trusted layouts share an arity,
    // so one frozen profile stands for either kind.
    for precommitteds in [
        vec![trusted_profile],
        vec![trusted_profile, trusted_profile],
    ] {
        let batch_polys = precommitteds.len() + 1;
        for num_vars in FIXTURE_K16_FINAL_NUM_VARS.0..=FIXTURE_K16_FINAL_NUM_VARS.1 {
            let key = AkitaScheduleLookupKey {
                final_group: akita_types::PolynomialGroupLayout::new(num_vars, 1),
                precommitteds: precommitteds.clone(),
            };
            assert!(JoltOneHotK16::resolve_catalog_row_for_key(&key).is_err());
            assert!(JoltOneHotK16::setup_matrix_capacity(num_vars, batch_polys).is_ok());
        }
    }
}

/// Drops the module's leading import boilerplate — everything through the
/// closing `};` of the `use super::{…};` block. rustfmt sorts and wraps that
/// list, so it cannot token-match the emitter's fixed header; the schedule
/// data below it is what this oracle guards.
fn strip_import_header(source: &str) -> &str {
    source
        .find("use super::{")
        .and_then(|start| {
            let rest = &source[start..];
            rest.find("};").map(|end| &rest[end + 2..])
        })
        .unwrap_or(source)
}

/// Splits Rust source into a whitespace-insensitive token stream:
/// identifier/number runs stay whole, every other non-whitespace character is
/// its own token. The planner emits unformatted source while the checked-in
/// modules are rustfmt-formatted (outside the `#[rustfmt::skip]` tables), so a
/// byte-for-byte oracle reports pure formatting as drift; token equality
/// detects every semantic change while ignoring layout. The checked-in file's
/// formatting itself is enforced by the workspace `cargo fmt` lane.
fn source_tokens(source: &str) -> Vec<String> {
    let source = strip_import_header(source);
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in source.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            if !ch.is_whitespace() {
                tokens.push(ch.to_string());
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Regenerates both family modules through the planner DP and compares their
/// token streams against the checked-in tables. Slow (re-runs every DP
/// solve) — run explicitly:
/// `cargo nextest run -p jolt-akita catalogs_match_planner --run-ignored all`
#[test]
#[ignore = "regenerates every schedule through the planner DP (minutes)"]
fn catalogs_match_planner_regeneration() {
    for spec in
        family_specs(std::path::PathBuf::new()).expect("every family must declare a valid contract")
    {
        let regenerated =
            akita_planner::emit::emit_family_module(&spec).expect("regeneration must succeed");
        let checked_in = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/schedules")
                .join(format!("{}.rs", spec.module_name)),
        )
        .expect("checked-in table must exist");
        let regenerated = source_tokens(&regenerated);
        let checked_in = source_tokens(&checked_in);
        if let Some(index) = (0..regenerated.len().max(checked_in.len()))
            .find(|&index| regenerated.get(index) != checked_in.get(index))
        {
            let context = |tokens: &[String]| {
                tokens[index.saturating_sub(8)..(index + 8).min(tokens.len())].join(" ")
            };
            assert_eq!(
                regenerated.get(index),
                checked_in.get(index),
                "{} drifted from the planner DP — regenerate via gen_jolt_schedules\n  \
                 first mismatch at token {index}\n  planner:    …{}…\n  checked-in: …{}…",
                spec.module_name,
                context(&regenerated),
                context(&checked_in),
            );
        }
    }
}
