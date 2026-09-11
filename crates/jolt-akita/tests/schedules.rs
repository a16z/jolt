#![expect(
    clippy::expect_used,
    reason = "catalog tests should fail loudly when an artifact or grid is malformed"
)]

//! Coverage, setup-sizing, and regeneration guards for Jolt's external catalogs.

use akita_config::{SetupRequirements, TrustedScheduleCatalog};
use akita_planner::emit::MaterializationDiagnostics;
use akita_schedules::{ResolvedScheduleRow, ValidatedScheduleCatalog};
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

fn dense_catalog() -> ValidatedScheduleCatalog {
    artifacts().dense_catalog().expect("dense catalog")
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
        assert_adaptation_preserves_main_skeleton(&base, resolved, final_group);
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
        &base,
        None,
        Some(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars()),
        &[],
        #[cfg(feature = "field-inline")]
        None,
        AKITA_ONE_HOT_K16,
        K16_NUM_VARS.0 - 1,
    )
    .expect_err("a declared reachable arity outside the family must fail setup");
    assert!(error.to_string().contains("outside the supported range"));
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

/// The FR limb group's provisioning pins: the carried arity line equals the
/// jolt-claims packing law, every reachable final arity plans and resolves
/// its FR row, and the limb group closes every advice combination.
#[cfg(feature = "field-inline")]
mod field_inc_limbs {
    #![expect(
        clippy::panic,
        reason = "pin tests attribute a failing arity in the panic message"
    )]

    use akita_config::CommitmentConfig;
    use akita_schedules::TrustedScheduleCatalog;
    use akita_types::{AkitaScheduleLookupKey, GroupCommitPhaseParams, PolynomialGroupLayout};
    use jolt_akita::configs::{JoltOneHotK16, JoltOneHotK256};
    use jolt_akita::schedule_registry::{
        dense_precommit_profile, extend_catalog, provision_precommitted_for_k,
        FIXTURE_K16_FINAL_NUM_VARS, FIXTURE_TRUSTED_ADVICE_GROUP,
    };
    use jolt_akita::schedules::emit::{K16_NUM_VARS, K256_NUM_VARS};
    use jolt_akita::{
        AkitaField, FieldIncLimbScheduleParams, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
    };
    use jolt_claims::lattice::MIN_DENSE_OBJECT_NUM_VARS;
    use jolt_claims::protocols::field_inline::lattice::{
        field_inc_limb_count, FieldIncLimbPackingPlan, FieldIncLimbShape,
    };
    use jolt_claims::protocols::jolt::lattice::packing::one_hot_trace_column_capacity;

    use super::{dense_catalog, one_hot_catalog};

    /// The packed trace's arity overhead over its own `log_T`: the chunk plus
    /// selector variables, constant per K.
    fn trace_arity_overhead(one_hot_k: usize) -> usize {
        let log_k_chunk = one_hot_k.ilog2() as usize;
        log_k_chunk
            + one_hot_trace_column_capacity(log_k_chunk)
                .expect("one-hot column capacity")
                .ilog2() as usize
    }

    /// The production caller's derivation of the FR arity line, from the
    /// jolt-claims laws: the packed trace's arity overhead over `log_T` and
    /// the limb plan's floor/selector geometry.
    fn law_derived_params(one_hot_k: usize) -> FieldIncLimbScheduleParams {
        let limbs = field_inc_limb_count::<AkitaField>();
        FieldIncLimbScheduleParams::new(
            trace_arity_overhead(one_hot_k),
            MIN_DENSE_OBJECT_NUM_VARS,
            limbs.next_power_of_two().ilog2() as usize,
        )
    }

    fn limb_profile(
        dense: &TrustedScheduleCatalog,
        params: FieldIncLimbScheduleParams,
        final_num_vars: usize,
    ) -> GroupCommitPhaseParams {
        dense_precommit_profile(
            dense,
            PolynomialGroupLayout::new(
                params
                    .physical_num_vars(final_num_vars)
                    .expect("reachable arity"),
                1,
            ),
        )
        .expect("limb profile resolves in the dense catalog")
    }

    /// The carried arity line must equal the jolt-claims packing law at every
    /// final arity, in both K regimes.
    #[test]
    fn carried_arity_line_matches_the_packing_law() {
        let limbs = field_inc_limb_count::<AkitaField>();
        assert_eq!(limbs, 2, "fp128 decomposes into two u64 limbs");
        for (one_hot_k, (min, max)) in [
            (AKITA_ONE_HOT_K16, K16_NUM_VARS),
            (AKITA_ONE_HOT_K256, K256_NUM_VARS),
        ] {
            let params = law_derived_params(one_hot_k);
            for final_num_vars in min..=max {
                let carried = params.physical_num_vars(final_num_vars);
                let expected = final_num_vars
                    .checked_sub(trace_arity_overhead(one_hot_k))
                    .map(|log_t| {
                        FieldIncLimbPackingPlan::new(&FieldIncLimbShape { limbs, log_t })
                            .expect("limb packing plan")
                            .packing()
                            .packed_num_vars()
                    });
                assert_eq!(
                    carried, expected,
                    "K={one_hot_k} final arity {final_num_vars}: carried arity diverges from \
                     the packing law"
                );
            }
        }
    }

    /// The prover pads packed traces to `MIN_PADDED_TRACE_LENGTH`
    /// (jolt-prover, `1 << 12` on akita builds), so the smallest reachable
    /// FR final arity is `overhead + 12`.
    const PROVER_MIN_LOG_T: usize = 12;

    /// Every reachable final arity of the K catalog provisions its own FR row
    /// (production provisions the setup's single final arity) that resolves
    /// through the frozen setup catalog. Doubles as the norm-budget check:
    /// the rows plan under the same u64-bounded dense fold policy advice
    /// uses, so a planned row means the limb words fit that budget. Arities
    /// below the prover's trace floor are unreachable and not swept (the
    /// dense catalog need not carry their limb layouts).
    fn fr_rows_plan_and_resolve_at_every_arity<Cfg: CommitmentConfig>(
        one_hot_k: usize,
        (declared_min, ceiling): (usize, usize),
    ) {
        let dense = dense_catalog();
        let base = one_hot_catalog(one_hot_k);
        let params = law_derived_params(one_hot_k);
        let reachable_min = (trace_arity_overhead(one_hot_k) + PROVER_MIN_LOG_T).max(declared_min);
        for final_num_vars in reachable_min..=ceiling {
            let rows = provision_precommitted_for_k(
                &dense,
                &base,
                None,
                None,
                &[],
                Some(params),
                one_hot_k,
                final_num_vars,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "K={one_hot_k} final arity {final_num_vars}: FR provisioning failed: {error}"
                )
            });
            assert_eq!(
                rows.rows().len(),
                1,
                "K={one_hot_k} final arity {final_num_vars} must plan its FR row"
            );
            let key = AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(final_num_vars, 1),
                precommitteds: vec![limb_profile(&dense, params, final_num_vars)],
            };
            let setup_catalog =
                extend_catalog::<Cfg>(&base, &rows).expect("freeze the FR setup catalog");
            let resolved = setup_catalog.resolve_key(&key).unwrap_or_else(|error| {
                panic!(
                    "K={one_hot_k} final arity {final_num_vars} must resolve its FR row: {error}"
                )
            });
            assert_eq!(resolved.profiles().precommitteds, key.precommitteds);
        }
    }

    #[test]
    fn fr_rows_plan_and_resolve_at_every_k16_arity() {
        fr_rows_plan_and_resolve_at_every_arity::<JoltOneHotK16>(AKITA_ONE_HOT_K16, K16_NUM_VARS);
    }

    #[test]
    fn fr_rows_plan_and_resolve_at_every_k256_arity() {
        fr_rows_plan_and_resolve_at_every_arity::<JoltOneHotK256>(
            AKITA_ONE_HOT_K256,
            K256_NUM_VARS,
        );
    }

    /// With both advice kinds declared, every advice presence combination is
    /// provisioned with the FR profile as its last group and no FR-absent row
    /// exists: an FR-on prover commits the group on every proof, so none is
    /// constructible.
    #[test]
    fn fr_rows_append_the_limb_group_to_every_advice_combination() {
        let dense = dense_catalog();
        let base = one_hot_catalog(AKITA_ONE_HOT_K16);
        let params = law_derived_params(AKITA_ONE_HOT_K16);
        let final_num_vars = FIXTURE_K16_FINAL_NUM_VARS.1;
        let trusted = FIXTURE_TRUSTED_ADVICE_GROUP.num_vars();
        let rows = provision_precommitted_for_k(
            &dense,
            &base,
            Some(trusted + 1),
            Some(trusted),
            &[],
            Some(params),
            AKITA_ONE_HOT_K16,
            final_num_vars,
        )
        .expect("FR-composed provisioning must plan every combination");
        assert_eq!(rows.rows().len(), 4);
        let limb = limb_profile(&dense, params, final_num_vars);
        for row in rows.rows() {
            assert_eq!(row.profiles().precommitteds.last(), Some(&limb));
        }
    }
}
