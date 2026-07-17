#![expect(
    clippy::expect_used,
    reason = "catalog tests should fail loudly when a table or grid is malformed"
)]

//! The Jolt-owned schedule catalogs: coverage and drift guards.

use jolt_akita::schedules::emit::{
    family_specs, keys, K16_NUM_POLYS, K16_NUM_VARS, K256_NUM_POLYS, K256_NUM_VARS,
};
use jolt_akita::schedules::{jolt_fp128_d64_onehot_k16_table, jolt_fp128_d64_onehot_k256_table};

/// Every key of a family grid resolves from its checked-in table (binary
/// lookup over sorted entries) — no planner-DP fallback for reachable
/// `W_jolt` shapes. Identity validity is exercised by every akita e2e (an
/// identity mismatch hard-errors instead of falling back).
#[test]
fn catalogs_cover_every_reachable_wjolt_shape() {
    for (table, num_polys, num_vars) in [
        (
            jolt_fp128_d64_onehot_k16_table().expect("K16 catalog is checked in"),
            K16_NUM_POLYS,
            K16_NUM_VARS,
        ),
        (
            jolt_fp128_d64_onehot_k256_table().expect("K256 catalog is checked in"),
            K256_NUM_POLYS,
            K256_NUM_VARS,
        ),
    ] {
        let grid = keys(num_polys, num_vars);
        assert!(!grid.is_empty());
        for key in grid {
            assert!(
                table
                    .entries
                    .iter()
                    .any(|entry| entry.final_group == key && entry.precommitteds.is_empty()),
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

/// Regenerates both family modules through the planner DP and compares them
/// byte-for-byte against the checked-in tables. Slow (re-runs every DP
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
        assert_eq!(
            regenerated, checked_in,
            "{} drifted from the planner DP — regenerate via gen_jolt_schedules",
            spec.module_name
        );
    }
}
