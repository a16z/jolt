#![expect(
    clippy::expect_used,
    reason = "catalog tests should fail loudly when an artifact or grid is malformed"
)]

//! Coverage and setup-sizing guards for Jolt's external catalogs.

use std::path::PathBuf;

use akita_config::{SetupRequirements, TrustedScheduleCatalog};
use akita_schedules::{ResolvedScheduleRow, ValidatedScheduleCatalog};
use akita_types::{
    commit_only_setup_field_elements, setup_matrix_capacity_for_schedule, AkitaScheduleLookupKey,
    FoldSchedule, PolynomialGroupLayout,
};
use jolt_akita::configs::{JoltOneHotK16, JoltOneHotK256};
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

fn dense_catalog() -> ValidatedScheduleCatalog {
    artifacts().dense_catalog().expect("dense catalog")
}

fn full_dense_catalog() -> ValidatedScheduleCatalog {
    artifacts()
        .full_dense_catalog()
        .expect("full-width dense catalog")
}

fn one_hot_catalog(one_hot_k: usize) -> ValidatedScheduleCatalog {
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

fn scalar_schedule(catalog: &ValidatedScheduleCatalog, num_vars: usize) -> FoldSchedule {
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

fn trusted_advice_grouped_key(dense: &ValidatedScheduleCatalog) -> AkitaScheduleLookupKey {
    let trusted_profile = dense_precommit_profile(dense, TRUSTED_ADVICE_GROUP)
        .expect("trusted advice standalone row must resolve");
    AkitaScheduleLookupKey {
        final_group: TRUSTED_ADVICE_K256_FINAL_GROUP,
        precommitteds: vec![trusted_profile],
    }
}

fn assert_adaptation_preserves_main_skeleton(
    base: &ValidatedScheduleCatalog,
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

    let rows = jolt_akita::schedule_registry::provision_precommitted_for_k(
        &dense,
        &full_dense_catalog(),
        &base,
        None,
        Some(TRUSTED_ADVICE_GROUP.num_vars()),
        &[],
        AKITA_ONE_HOT_K256,
        key.final_group.num_vars(),
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
    assert_adaptation_preserves_main_skeleton(&base, resolved, key.final_group);
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
        let rows = jolt_akita::schedule_registry::provision_precommitted_for_k(
            &dense,
            &full_dense_catalog(),
            &base,
            None,
            Some(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars()),
            &[],
            AKITA_ONE_HOT_K16,
            final_num_vars,
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
        assert_adaptation_preserves_main_skeleton(&base, resolved, final_group);
    }
}

#[test]
fn grouped_setup_capacity_covers_precommit_and_complete_schedule() {
    let dense = dense_catalog();
    let base = one_hot_catalog(AKITA_ONE_HOT_K256);
    let key = trusted_advice_grouped_key(&dense);
    let rows = jolt_akita::schedule_registry::provision_precommitted_for_k(
        &dense,
        &full_dense_catalog(),
        &base,
        None,
        Some(TRUSTED_ADVICE_GROUP.num_vars()),
        &[],
        AKITA_ONE_HOT_K256,
        key.final_group.num_vars(),
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
    let trusted_catalog =
        TrustedScheduleCatalog::<JoltOneHotK256>::new(setup_catalog).expect("config-bound catalog");
    let setup_capacity = SetupRequirements::from_catalog(&trusted_catalog, 39, 2)
        .expect("catalog-backed setup capacity")
        .matrix_capacity;
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
        &full_dense_catalog(),
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
    let [k16_spec, k256_spec, dense_spec, full_dense_spec] =
        family_specs(PathBuf::new()).expect("emit specs");
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
        (dense_spec, "jolt-fp128-dense-bounded", dense_catalog()),
        (
            full_dense_spec,
            "jolt-fp128-dense-full",
            full_dense_catalog(),
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

/// Full-width field increments must compose with the trace and every
/// combination of bounded advice objects.
#[cfg(feature = "field-inline")]
mod field_inc {
    #![expect(
        clippy::panic,
        reason = "pin tests attribute a failing arity in the panic message"
    )]

    use akita_config::CommitmentConfig;
    use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};
    use jolt_akita::configs::{JoltOneHotK16, JoltOneHotK256};
    use jolt_akita::schedule_registry::{
        dense_precommit_profile, extend_catalog, provision_precommitted_for_k,
        FIXTURE_K16_FINAL_NUM_VARS, FIXTURE_TRUSTED_ADVICE_GROUP,
    };
    use jolt_akita::schedules::emit::{K16_NUM_VARS, K256_NUM_VARS};
    use jolt_akita::{DensePrecommitLayout, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};
    use jolt_claims::protocols::field_inline::lattice::FieldIncLayout;
    use jolt_claims::protocols::jolt::lattice::packing::one_hot_trace_column_capacity;

    use super::{dense_catalog, full_dense_catalog, one_hot_catalog};

    fn trace_arity_overhead(one_hot_k: usize) -> usize {
        let log_k_chunk = one_hot_k.ilog2() as usize;
        log_k_chunk
            + one_hot_trace_column_capacity(log_k_chunk)
                .expect("one-hot column capacity")
                .ilog2() as usize
    }

    /// The prover pads Akita traces to at least 2^12 cycles.
    const PROVER_MIN_LOG_T: usize = 12;

    fn field_inline_rows_plan_and_resolve_at_every_arity<Cfg: CommitmentConfig>(
        one_hot_k: usize,
        (declared_min, ceiling): (usize, usize),
    ) {
        let dense = dense_catalog();
        let full_dense = full_dense_catalog();
        let base = one_hot_catalog(one_hot_k);
        let overhead = trace_arity_overhead(one_hot_k);
        let reachable_min = (overhead + PROVER_MIN_LOG_T).max(declared_min);
        for final_num_vars in reachable_min..=ceiling {
            let layout = FieldIncLayout::new(final_num_vars - overhead);
            let rows = provision_precommitted_for_k(
                &dense,
                &full_dense,
                &base,
                None,
                None,
                &[DensePrecommitLayout::FullWidth { num_vars: layout.num_vars() }],
                one_hot_k,
                final_num_vars,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "K={one_hot_k} final arity {final_num_vars}: field-inline provisioning failed: {error}"
                )
            });
            assert_eq!(
                rows.rows().len(),
                1,
                "K={one_hot_k} final arity {final_num_vars} must plan its field-inline row"
            );
            let key = AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(final_num_vars, 1),
                precommitteds: vec![dense_precommit_profile(
                    &full_dense,
                    PolynomialGroupLayout::new(layout.num_vars(), 1),
                )
                .expect("field increment profile")],
            };
            let setup_catalog =
                extend_catalog::<Cfg>(&base, &rows).expect("freeze the field-inline setup catalog");
            let resolved = setup_catalog.resolve_key(&key).unwrap_or_else(|error| {
                panic!(
                    "K={one_hot_k} final arity {final_num_vars} must resolve its field-inline row: {error}"
                )
            });
            assert_eq!(resolved.profiles().precommitteds, key.precommitteds);
        }
    }

    #[test]
    fn field_inline_rows_plan_and_resolve_at_every_k16_arity() {
        field_inline_rows_plan_and_resolve_at_every_arity::<JoltOneHotK16>(
            AKITA_ONE_HOT_K16,
            K16_NUM_VARS,
        );
    }

    #[test]
    fn field_inline_rows_plan_and_resolve_at_every_k256_arity() {
        field_inline_rows_plan_and_resolve_at_every_arity::<JoltOneHotK256>(
            AKITA_ONE_HOT_K256,
            K256_NUM_VARS,
        );
    }

    #[test]
    fn full_width_replanning_is_limited_to_one_inc_and_two_advice_groups() {
        let full = DensePrecommitLayout::FullWidth { num_vars: 30 };
        let dense = dense_catalog();
        let full_dense = full_dense_catalog();
        let base = one_hot_catalog(AKITA_ONE_HOT_K256);
        for layouts in [
            vec![
                DensePrecommitLayout::Bounded { num_vars: 14 },
                DensePrecommitLayout::Bounded { num_vars: 15 },
                DensePrecommitLayout::Bounded { num_vars: 16 },
                full,
            ],
            vec![full, full],
        ] {
            assert!(
                provision_precommitted_for_k(
                    &dense,
                    &full_dense,
                    &base,
                    None,
                    None,
                    &layouts,
                    AKITA_ONE_HOT_K256,
                    43,
                )
                .is_err(),
                "unsupported batch shapes must retain the guided-planning rejection"
            );
        }
    }

    #[test]
    fn field_inline_rows_append_the_inc_group_to_every_advice_combination() {
        let dense = dense_catalog();
        let full_dense = full_dense_catalog();
        let base = one_hot_catalog(AKITA_ONE_HOT_K16);
        let final_num_vars = FIXTURE_K16_FINAL_NUM_VARS.1;
        let layout = FieldIncLayout::new(final_num_vars - trace_arity_overhead(AKITA_ONE_HOT_K16));
        let trusted = FIXTURE_TRUSTED_ADVICE_GROUP.num_vars();
        let rows = provision_precommitted_for_k(
            &dense,
            &full_dense,
            &base,
            Some(trusted + 1),
            Some(trusted),
            &[DensePrecommitLayout::FullWidth {
                num_vars: layout.num_vars(),
            }],
            AKITA_ONE_HOT_K16,
            final_num_vars,
        )
        .expect("provisioning with field-inline must plan every combination");
        assert_eq!(rows.rows().len(), 4);
        let inc = dense_precommit_profile(
            &full_dense,
            PolynomialGroupLayout::new(layout.num_vars(), 1),
        )
        .expect("field increment profile");
        for row in rows.rows() {
            assert_eq!(row.profiles().precommitteds.last(), Some(&inc));
        }
        let catalog =
            extend_catalog::<JoltOneHotK16>(&base, &rows).expect("freeze grouped catalog");
        for num_vars in [trusted, trusted + 1] {
            let widened_advice =
                dense_precommit_profile(&full_dense, PolynomialGroupLayout::new(num_vars, 1))
                    .expect("full-width advice-shaped profile");
            assert!(catalog
                .resolve_key(&AkitaScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(final_num_vars, 1),
                    precommitteds: vec![widened_advice, inc],
                })
                .is_err());
        }
    }
}
