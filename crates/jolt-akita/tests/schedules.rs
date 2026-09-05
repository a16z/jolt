#![expect(
    clippy::expect_used,
    reason = "catalog tests should fail loudly when an artifact or grid is malformed"
)]

//! Coverage, setup-sizing, and regeneration guards for Jolt's external catalogs.

use std::path::PathBuf;

use akita_config::trusted_setup_matrix_capacity;
use akita_planner::emit::MaterializationDiagnostics;
use akita_schedules::{ResolvedScheduleRow, TrustedScheduleCatalog};
use akita_types::{
    commit_only_setup_field_elements, setup_matrix_capacity_for_schedule, AkitaScheduleLookupKey,
    FoldSchedule, PolynomialGroupLayout,
};
use jolt_akita::configs::{JoltDenseBounded, JoltOneHotK16, JoltOneHotK256};
use jolt_akita::schedule_registry::{
    dense_precommit_profile, FIXTURE_K16_FINAL_NUM_VARS, FIXTURE_TRUSTED_ADVICE_GROUP,
};
use jolt_akita::schedules::emit::{
    family_specs, keys, K16_NUM_VARS, K16_PACKING_VARIABLES, K256_NUM_VARS, K256_PACKING_VARIABLES,
    ONE_HOT_TRACE_NUM_POLYS, RECURSIVE_TRACE_LOG_T_CUTOVER,
};
use jolt_akita::{AkitaScheduleArtifacts, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};

fn artifacts() -> AkitaScheduleArtifacts {
    AkitaScheduleArtifacts::from_directory(AkitaScheduleArtifacts::packaged_directory())
        .expect("checked-in Jolt schedule artifacts")
}

fn dense_catalog() -> TrustedScheduleCatalog {
    artifacts().dense_catalog().expect("dense catalog")
}

fn one_hot_catalog(one_hot_k: usize) -> TrustedScheduleCatalog {
    artifacts()
        .one_hot_catalog(one_hot_k)
        .expect("one-hot catalog")
}

#[test]
fn catalogs_cover_every_reachable_one_hot_trace_shape() {
    for (catalog, num_vars) in [
        (one_hot_catalog(AKITA_ONE_HOT_K16), K16_NUM_VARS),
        (one_hot_catalog(AKITA_ONE_HOT_K256), K256_NUM_VARS),
    ] {
        let grid = keys(ONE_HOT_TRACE_NUM_POLYS, num_vars);
        assert!(!grid.is_empty());
        for key in &grid {
            let resolved = catalog
                .resolve_key(&AkitaScheduleLookupKey::single(*key))
                .expect("reachable scalar shape must resolve");
            assert!(resolved.profiles().precommitteds.is_empty());
        }
        assert_eq!(catalog.len(), grid.len());
    }
}

fn scalar_schedule(catalog: &TrustedScheduleCatalog, num_vars: usize) -> FoldSchedule {
    catalog
        .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
            num_vars, 1,
        )))
        .expect("cutover row must resolve")
        .schedule()
        .clone()
}

fn uses_setup_offloading(schedule: &FoldSchedule) -> bool {
    schedule
        .recursive_folds
        .iter()
        .any(|fold| fold.params.setup_prefix().is_some())
}

#[test]
fn one_hot_catalogs_switch_to_setup_offloading_at_the_trace_cutover() {
    for (catalog, packing_variables) in [
        (one_hot_catalog(AKITA_ONE_HOT_K16), K16_PACKING_VARIABLES),
        (one_hot_catalog(AKITA_ONE_HOT_K256), K256_PACKING_VARIABLES),
    ] {
        let cutover_num_vars = RECURSIVE_TRACE_LOG_T_CUTOVER + packing_variables;
        assert!(!uses_setup_offloading(&scalar_schedule(
            &catalog,
            cutover_num_vars - 1
        )));
        assert!(uses_setup_offloading(&scalar_schedule(
            &catalog,
            cutover_num_vars
        )));
    }
}

const TRUSTED_ADVICE_GROUP: PolynomialGroupLayout = PolynomialGroupLayout::new(20, 1);
const TRUSTED_ADVICE_K256_FINAL_GROUP: PolynomialGroupLayout = PolynomialGroupLayout::new(39, 1);

fn trusted_advice_grouped_key(dense: &TrustedScheduleCatalog) -> AkitaScheduleLookupKey {
    let trusted_profile = dense_precommit_profile(dense, TRUSTED_ADVICE_GROUP)
        .expect("trusted advice standalone row must resolve");
    AkitaScheduleLookupKey {
        final_group: TRUSTED_ADVICE_K256_FINAL_GROUP,
        precommitteds: vec![trusted_profile],
    }
}

fn assert_adaptation_preserves_main_skeleton(
    base: &TrustedScheduleCatalog,
    resolved: &ResolvedScheduleRow,
    final_group: PolynomialGroupLayout,
) {
    let main = base
        .resolve_key(&AkitaScheduleLookupKey::single(final_group))
        .expect("main scalar row");
    assert_eq!(
        resolved.schedule().root.params.own_group(),
        main.schedule().root.params.own_group(),
        "adaptation must preserve the central trace root geometry",
    );
    assert_eq!(
        resolved
            .schedule()
            .recursive_folds
            .iter()
            .map(|fold| fold.params.setup_prefix().is_some())
            .collect::<Vec<_>>(),
        main.schedule()
            .recursive_folds
            .iter()
            .map(|fold| fold.params.setup_prefix().is_some())
            .collect::<Vec<_>>(),
        "adaptation must preserve the direct/setup-offloaded topology",
    );
}

#[test]
fn grouped_advice_rows_are_setup_owned_not_in_the_base_artifact() {
    let dense = dense_catalog();
    let base = one_hot_catalog(AKITA_ONE_HOT_K256);
    let key = trusted_advice_grouped_key(&dense);
    assert!(base.resolve_key(&key).is_err());

    let rows = jolt_akita::schedule_registry::provision::<JoltOneHotK256, JoltDenseBounded>(
        &base,
        std::slice::from_ref(&key.precommitteds),
        [key.final_group.num_vars()],
    )
    .expect("preprocessing must adapt the production grouped row");
    assert_eq!(rows.rows().len(), 1);

    let setup_catalog =
        jolt_akita::schedule_registry::extend_catalog::<JoltOneHotK256>(&base, &rows)
            .expect("freeze setup-owned catalog");
    let resolved = setup_catalog
        .resolve_key(&key)
        .expect("setup-owned row must resolve by key");
    assert_eq!(resolved.profiles().precommitteds, key.precommitteds);
    assert_adaptation_preserves_main_skeleton(&base, &resolved, key.final_group);
    assert_eq!(
        setup_catalog
            .resolve_selection(resolved.selection())
            .expect("row must resolve by proof selection")
            .profiles(),
        resolved.profiles()
    );
}

#[test]
fn grouped_adaptation_preserves_direct_and_recursive_k16_trace_skeletons() {
    let dense = dense_catalog();
    let base = one_hot_catalog(AKITA_ONE_HOT_K16);
    let precommit = dense_precommit_profile(&dense, FIXTURE_TRUSTED_ADVICE_GROUP)
        .expect("trusted advice profile");
    for final_num_vars in [
        RECURSIVE_TRACE_LOG_T_CUTOVER + K16_PACKING_VARIABLES - 1,
        RECURSIVE_TRACE_LOG_T_CUTOVER + K16_PACKING_VARIABLES,
    ] {
        let rows = jolt_akita::schedule_registry::provision::<JoltOneHotK16, JoltDenseBounded>(
            &base,
            &[vec![precommit]],
            [final_num_vars],
        )
        .expect("adapt the grouped K=16 row");
        let setup_catalog =
            jolt_akita::schedule_registry::extend_catalog::<JoltOneHotK16>(&base, &rows)
                .expect("freeze adapted K=16 catalog");
        let final_group = PolynomialGroupLayout::new(final_num_vars, 1);
        let resolved = setup_catalog
            .resolve_key(&AkitaScheduleLookupKey {
                final_group,
                precommitteds: vec![precommit],
            })
            .expect("adapted K=16 row");
        assert_adaptation_preserves_main_skeleton(&base, &resolved, final_group);
    }
}

#[test]
fn grouped_setup_capacity_covers_precommit_and_complete_schedule() {
    let dense = dense_catalog();
    let base = one_hot_catalog(AKITA_ONE_HOT_K256);
    let key = trusted_advice_grouped_key(&dense);
    let rows = jolt_akita::schedule_registry::provision::<JoltOneHotK256, JoltDenseBounded>(
        &base,
        std::slice::from_ref(&key.precommitteds),
        [key.final_group.num_vars()],
    )
    .expect("adapt grouped row");
    let setup_catalog =
        jolt_akita::schedule_registry::extend_catalog::<JoltOneHotK256>(&base, &rows)
            .expect("freeze setup catalog");
    let resolved = setup_catalog.resolve_key(&key).expect("grouped row");
    let full_capacity =
        setup_matrix_capacity_for_schedule(resolved.schedule()).expect("grouped schedule capacity");
    let prefix = key.precommitteds[0];
    let precommit_capacity = commit_only_setup_field_elements(
        &prefix.inner.matrix,
        &prefix.outer.matrix,
        prefix.outer_slice_count,
    )
    .expect("precommit capacity");
    let setup_capacity = trusted_setup_matrix_capacity::<JoltOneHotK256>(&setup_catalog, 39, 2)
        .expect("catalog-backed setup capacity");
    assert!(setup_capacity.num_field_elements >= full_capacity.num_field_elements);
    assert!(setup_capacity.num_field_elements >= precommit_capacity);
}

#[test]
fn base_catalogs_contain_no_grouped_advice_rows() {
    let dense = dense_catalog();
    let base = one_hot_catalog(AKITA_ONE_HOT_K16);
    assert!(base
        .rows()
        .all(|row| row.profiles().precommitteds.is_empty()));

    let trusted_profile = dense_precommit_profile(&dense, FIXTURE_TRUSTED_ADVICE_GROUP)
        .expect("fixture dense profile");
    for precommitteds in [
        vec![trusted_profile],
        vec![trusted_profile, trusted_profile],
    ] {
        for num_vars in FIXTURE_K16_FINAL_NUM_VARS.0..=FIXTURE_K16_FINAL_NUM_VARS.1 {
            let key = AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(num_vars, 1),
                precommitteds: precommitteds.clone(),
            };
            assert!(base.resolve_key(&key).is_err());
        }
    }
}

#[test]
fn grouped_provisioning_rejects_out_of_family_final_arity() {
    let dense = dense_catalog();
    let base = one_hot_catalog(AKITA_ONE_HOT_K16);
    let error = jolt_akita::schedule_registry::provision_precommitted_for_k(
        &dense,
        &base,
        None,
        Some(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars()),
        &[],
        AKITA_ONE_HOT_K16,
        K16_NUM_VARS.0 - 1,
    )
    .expect_err("a declared reachable arity outside the family must fail setup");
    assert!(error.to_string().contains("outside the supported range"));
}

/// The emit specs are the single source of truth for what the generator
/// writes; each checked-in one-hot catalog must be exactly its family's grid —
/// the forward inclusion is checked above, so a length match plus a
/// reverse-inclusion sweep rules out stale or duplicated entries.
#[test]
fn emit_specs_and_checked_in_catalogs_agree_exactly() {
    let [k16_spec, k256_spec, _dense_spec] = family_specs(PathBuf::new()).expect("emit specs");
    let cases = [
        (
            k16_spec,
            "jolt-fp128-onehot-k16",
            one_hot_catalog(AKITA_ONE_HOT_K16),
        ),
        (
            k256_spec,
            "jolt-fp128-onehot-k256",
            one_hot_catalog(AKITA_ONE_HOT_K256),
        ),
    ];
    for (spec, family_name, catalog) in cases {
        assert_eq!(spec.family_name, family_name, "spec order regressed");
        assert!(
            spec.grouped_requests.is_empty(),
            "Jolt one-hot families emit scalar single-group schedules only"
        );
        assert_eq!(
            spec.keys.len(),
            catalog.len(),
            "{family_name}: grid and catalog must have the same key count"
        );
        for row in catalog.rows() {
            assert!(
                row.profiles().precommitteds.is_empty(),
                "{family_name}: Jolt one-hot catalogs are scalar-only"
            );
            assert!(
                spec.keys.contains(&row.profiles().final_group.group),
                "{family_name}: stale catalog entry {:?} is not a reachable shape",
                row.profiles().final_group.group
            );
        }
        for (index, key) in spec.keys.iter().enumerate() {
            assert!(
                !spec.keys[..index].contains(key),
                "{family_name}: duplicate grid key {key:?}"
            );
        }
    }
}

/// Re-run every planner solve and byte-compare canonical artifacts.
#[test]
#[ignore = "regenerates every schedule through the planner DP (minutes)"]
fn catalogs_match_planner_regeneration() {
    let output =
        std::env::temp_dir().join(format!("jolt-akita-schedule-check-{}", std::process::id()));
    std::fs::create_dir_all(&output).expect("temporary artifact directory");
    let specs = family_specs(output.clone()).expect("valid family specs");
    let rendered = akita_planner::emit::render_schedule_artifact_outputs_with_validation(
        &specs,
        MaterializationDiagnostics::default(),
        |_, _| Ok(()),
    )
    .expect("regenerate artifacts");
    let generated = akita_planner::emit::publish_artifact_outputs(rendered)
        .expect("publish temporary artifacts");
    for generated in generated {
        let checked_in = AkitaScheduleArtifacts::packaged_directory()
            .join(generated.file_name().expect("generated artifact file name"));
        assert_eq!(
            std::fs::read(&generated).expect("generated artifact"),
            std::fs::read(&checked_in).expect("checked-in artifact"),
            "{} drifted from planner output",
            checked_in.display()
        );
    }
    std::fs::remove_dir_all(output).expect("remove temporary artifacts");
}
