#![expect(
    clippy::expect_used,
    reason = "catalog tests should fail loudly when a table or grid is malformed"
)]

//! The Jolt-owned schedule catalogs: coverage and drift guards.

use jolt_akita::schedules::emit::family_specs;
use jolt_akita::schedules::{jolt_fp128_d64_onehot_k16_table, jolt_fp128_d64_onehot_k256_table};

/// Every key of a family grid resolves from its checked-in table (binary
/// lookup over sorted entries) — no planner-DP fallback for reachable
/// `OneHotTrace` shapes. Identity validity is exercised by every akita e2e (an
/// identity mismatch hard-errors instead of falling back).
///
/// The planner's feasibility floor excludes some wide-column low-variable
/// corners of the grid (e.g. maximal K=16 chunk counts at the minimum padded
/// trace length): those keys are legitimately absent iff the planner DP
/// itself rejects them, so absence is verified against a fresh DP run —
/// catalogued ⟺ plannable, never a silent gap.
#[test]
fn catalogs_cover_every_reachable_one_hot_trace_shape() {
    for (spec, table) in family_specs(std::path::PathBuf::new()).into_iter().zip([
        jolt_fp128_d64_onehot_k16_table().expect("K16 catalog is checked in"),
        jolt_fp128_d64_onehot_k256_table().expect("K256 catalog is checked in"),
    ]) {
        assert!(!spec.keys.is_empty());
        for key in &spec.keys {
            let catalogued = table.entries.iter().any(|entry| {
                entry.root.final_group.layout == *key && entry.root.precommitted_groups.is_empty()
            });
            if !catalogued {
                let planned = (spec.regen)(*key);
                assert!(
                    planned.is_err(),
                    "missing catalog entry for plannable key {key:?}"
                );
            }
        }
        assert_eq!(
            table.identity.key_count,
            table.entries.len(),
            "identity key count must match the table"
        );
    }
}

/// Splits Rust source into a whitespace-insensitive token stream:
/// identifier/number runs stay whole, every other non-whitespace character is
/// its own token. The planner emits unformatted source while the checked-in
/// modules are rustfmt-formatted (outside the `#[rustfmt::skip]` tables), so a
/// byte-for-byte oracle reports pure formatting as drift; token equality
/// detects every semantic change while ignoring layout. The checked-in file's
/// formatting itself is enforced by the workspace `cargo fmt` lane.
fn source_tokens(source: &str) -> Vec<String> {
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
    for spec in family_specs(std::path::PathBuf::new()) {
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
