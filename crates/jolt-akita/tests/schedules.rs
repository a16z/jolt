#![expect(
    clippy::expect_used,
    reason = "catalog tests should fail loudly when an artifact or grid is malformed"
)]

use jolt_akita::schedule_registry::GroupedScheduleParams;
use std::collections::BTreeSet;
use std::path::PathBuf;

use akita_config::{CommitmentConfig, SetupRequirements, TrustedScheduleCatalog};
use akita_params::{
    commit_only_setup_field_elements, setup_matrix_capacity_for_schedule, ChunkedWitnessCfg,
    FoldSchedule, PolynomialGroupLayout, ScheduleLookupKey,
};
use akita_schedules::{ResolvedScheduleRow, ValidatedScheduleCatalog};
use jolt_akita::configs::{JoltDenseBounded, JoltDenseFull, JoltOneHotK16, JoltOneHotK256};
use jolt_akita::schedule_registry::{
    dense_group_profile, FIXTURE_K16_FINAL_NUM_VARS, FIXTURE_TRUSTED_ADVICE_GROUP,
};
use jolt_akita::schedules::emit::{
    family_specs, K16_COLUMN_VARIABLES, K16_NUM_VARS, K256_COLUMN_VARIABLES,
    RECURSIVE_TRACE_LOG_T_CUTOVER,
};
use jolt_akita::{
    AkitaChunkProfile, AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams, AKITA_ONE_HOT_K16,
    AKITA_ONE_HOT_K256,
};
use jolt_claims::protocols::jolt::lattice::strategy::MAX_ONE_HOT_TRACE_COLUMNS;
use jolt_claims::protocols::jolt::lattice::{
    one_hot_trace_columns, OneHotTraceShape, ONE_HOT_TRACE_LAYOUT,
};
use jolt_claims::protocols::jolt::{JoltFormulaDimensions, JoltOneHotDimensions};
use jolt_openings::{CommitmentScheme, OpeningsError};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn artifacts() -> AkitaScheduleArtifacts {
    AkitaScheduleArtifacts::from_directory(
        AkitaScheduleArtifacts::packaged_directory(),
        AkitaChunkProfile::new(8, 2),
    )
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

fn one_hot_catalog(one_hot_k: usize, profile: AkitaChunkProfile) -> ValidatedScheduleCatalog {
    artifacts()
        .one_hot_catalog_for_profile(one_hot_k, profile)
        .expect("one-hot catalog")
}

#[test]
fn four_file_directory_supports_single_profile() {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after Unix epoch")
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "jolt-akita-four-artifacts-{}-{suffix}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).expect("temporary artifact directory");
    for family in [
        JoltDenseBounded::schedule_family_name(),
        JoltDenseFull::schedule_family_name(),
        JoltOneHotK16::schedule_family_name(),
        JoltOneHotK256::schedule_family_name(),
    ] {
        let name = format!("{family}.aks");
        let _ = std::fs::copy(
            AkitaScheduleArtifacts::packaged_directory().join(&name),
            directory.join(&name),
        )
        .expect("copy original schedule artifact");
    }

    let loaded = AkitaScheduleArtifacts::from_directory(&directory, AkitaChunkProfile::new(8, 2))
        .expect("base four-file directory must load");
    let _ = loaded
        .one_hot_catalog(AKITA_ONE_HOT_K16)
        .expect("Single catalog must remain available");
    let error = AkitaScheme::setup(
        AkitaSetupParams::one_hot_only(16, 1, [3; 32], AKITA_ONE_HOT_K16, Arc::new(loaded))
            .with_akita_chunk_profile(AkitaChunkProfile::new(8, 2)),
    )
    .expect_err("selecting a missing companion catalog must fail setup");
    assert!(matches!(error, OpeningsError::InvalidSetup(_)));
    std::fs::remove_dir_all(directory).expect("remove temporary artifact directory");
}

#[test]
fn k16_catalogs_cover_production_trace_geometry() {
    let mut expected = BTreeSet::new();
    for bytecode_bits in 1..=32 {
        for ram_bits in 1..=61 {
            let dimensions = JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                log_t: 0,
                instruction_address_bits: 128,
                bytecode_k: 1usize << bytecode_bits,
                ram_k: 1usize << ram_bits,
                committed_chunk_bits: K16_COLUMN_VARIABLES,
                lookup_virtual_chunk_bits: 32,
            })
            .expect("RV64 address dimensions");
            let mut shape = OneHotTraceShape {
                ra_layout: dimensions.ra_layout,
                log_t: 0,
                log_k_chunk: K16_COLUMN_VARIABLES,
            };
            if one_hot_trace_columns(&shape).expect("trace columns").len()
                > MAX_ONE_HOT_TRACE_COLUMNS
            {
                continue;
            }
            for log_t in 12..=30 {
                shape.log_t = log_t;
                let plan = ONE_HOT_TRACE_LAYOUT.plan(&shape).expect("trace layout");
                let _ = expected.insert((plan.num_vars(), plan.ids().len()));
            }
        }
    }
    let catalog = one_hot_catalog(AKITA_ONE_HOT_K16, AkitaChunkProfile::new(1, 0));
    for &(num_vars, num_polys) in &expected {
        let key = PolynomialGroupLayout::new(num_vars, num_polys);
        let _ = catalog
            .resolve_key(&ScheduleLookupKey::single(key))
            .expect("default production trace geometry must resolve");
    }
}

#[test]
fn k256_catalogs_reject_unprovisioned_trace_shapes() {
    let unused_trace = ScheduleLookupKey::single(PolynomialGroupLayout::new(33, 28));
    for profile in [AkitaChunkProfile::new(1, 0), AkitaChunkProfile::new(8, 2)] {
        let catalog = one_hot_catalog(AKITA_ONE_HOT_K256, profile);
        assert!(catalog.resolve_key(&unused_trace).is_err());
        let params = GroupedScheduleParams::new(
            None,
            Some(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars()),
            Vec::new(),
            unused_trace.final_group,
        );
        let error = AkitaScheme::setup(
            AkitaSetupParams::one_hot_only_grouped(
                unused_trace.final_group.num_vars(),
                unused_trace.final_group.num_polynomials(),
                unused_trace.final_group.num_polynomials() + 1,
                [3; 32],
                AKITA_ONE_HOT_K256,
                Some(params),
                Arc::new(artifacts()),
            )
            .with_akita_chunk_profile(profile),
        )
        .expect_err("grouped setup must explain an unsupported K=256 trace shape");
        assert!(matches!(error, OpeningsError::InvalidSetup(_)));
        let message = error.to_string();
        assert!(message.contains("K=256 final shape (num_vars=33, num_polys=28)"));
        assert!(message.contains("fixture and benchmark shapes only"));
        assert!(message.contains("use K=16 for production traces"));
        assert!(message.contains("supply a catalog containing the exact K=256 shape"));
    }
}

fn trace_schedule(
    catalog: &ValidatedScheduleCatalog,
    num_vars: usize,
    num_polys: usize,
) -> FoldSchedule {
    catalog
        .resolve_key(&ScheduleLookupKey::single(PolynomialGroupLayout::new(
            num_vars, num_polys,
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
    for (catalog, column_variables, num_polys) in [
        (
            one_hot_catalog(AKITA_ONE_HOT_K16, AkitaChunkProfile::new(1, 0)),
            K16_COLUMN_VARIABLES,
            51,
        ),
        (
            one_hot_catalog(AKITA_ONE_HOT_K256, AkitaChunkProfile::new(1, 0)),
            K256_COLUMN_VARIABLES,
            27,
        ),
    ] {
        let cutover_num_vars = RECURSIVE_TRACE_LOG_T_CUTOVER + column_variables;
        assert!(!uses_setup_offloading(&trace_schedule(
            &catalog,
            cutover_num_vars - 1,
            num_polys,
        )));
        assert!(uses_setup_offloading(&trace_schedule(
            &catalog,
            cutover_num_vars,
            num_polys,
        )));
    }
}

const TRUSTED_ADVICE_GROUP: PolynomialGroupLayout = PolynomialGroupLayout::new(20, 1);
const TRUSTED_ADVICE_K256_FINAL_GROUP: PolynomialGroupLayout = PolynomialGroupLayout::new(34, 27);

fn trusted_advice_grouped_key(dense: &ValidatedScheduleCatalog) -> ScheduleLookupKey {
    let trusted_profile = dense_group_profile(dense, TRUSTED_ADVICE_GROUP)
        .expect("trusted advice standalone row must resolve");
    ScheduleLookupKey {
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
        .resolve_key(&ScheduleLookupKey::single(final_group))
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
    let base = one_hot_catalog(AKITA_ONE_HOT_K256, AkitaChunkProfile::new(1, 0));
    let key = trusted_advice_grouped_key(&dense);
    assert!(base.resolve_key(&key).is_err());

    let rows = jolt_akita::schedule_registry::provision_groups_for_k(
        &dense,
        &full_dense_catalog(),
        &base,
        &GroupedScheduleParams::new(
            None,
            Some(TRUSTED_ADVICE_GROUP.num_vars()),
            Vec::new(),
            key.final_group,
        ),
        AKITA_ONE_HOT_K256,
    )
    .expect("preprocessing must adapt the production grouped row");
    assert_eq!(rows.rows().len(), 1);

    let setup_catalog = jolt_akita::schedule_registry::extend_catalog::<JoltOneHotK256>(
        &TrustedScheduleCatalog::<JoltOneHotK256>::new(
            base.clone(),
            AkitaChunkProfile::default().witness_cfg(),
        )
        .expect("single-chunk fixture must bind"),
        &rows,
    )
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
    let base = one_hot_catalog(AKITA_ONE_HOT_K16, AkitaChunkProfile::new(1, 0));
    let precommit =
        dense_group_profile(&dense, FIXTURE_TRUSTED_ADVICE_GROUP).expect("trusted advice profile");
    for final_num_vars in [
        RECURSIVE_TRACE_LOG_T_CUTOVER + K16_COLUMN_VARIABLES - 1,
        RECURSIVE_TRACE_LOG_T_CUTOVER + K16_COLUMN_VARIABLES,
    ] {
        let rows = jolt_akita::schedule_registry::provision_groups_for_k(
            &dense,
            &full_dense_catalog(),
            &base,
            &GroupedScheduleParams::new(
                None,
                Some(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars()),
                Vec::new(),
                PolynomialGroupLayout::new(final_num_vars, 51),
            ),
            AKITA_ONE_HOT_K16,
        )
        .expect("adapt the grouped K=16 row");
        let setup_catalog = jolt_akita::schedule_registry::extend_catalog::<JoltOneHotK16>(
            &TrustedScheduleCatalog::<JoltOneHotK16>::new(
                base.clone(),
                AkitaChunkProfile::default().witness_cfg(),
            )
            .expect("single-chunk fixture must bind"),
            &rows,
        )
        .expect("freeze adapted K=16 catalog");
        let final_group = PolynomialGroupLayout::new(final_num_vars, 51);
        let resolved = setup_catalog
            .resolve_key(&ScheduleLookupKey {
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
    let base = one_hot_catalog(AKITA_ONE_HOT_K256, AkitaChunkProfile::new(1, 0));
    let key = trusted_advice_grouped_key(&dense);
    let rows = jolt_akita::schedule_registry::provision_groups_for_k(
        &dense,
        &full_dense_catalog(),
        &base,
        &GroupedScheduleParams::new(
            None,
            Some(TRUSTED_ADVICE_GROUP.num_vars()),
            Vec::new(),
            key.final_group,
        ),
        AKITA_ONE_HOT_K256,
    )
    .expect("adapt grouped row");
    let setup_catalog = jolt_akita::schedule_registry::extend_catalog::<JoltOneHotK256>(
        &TrustedScheduleCatalog::<JoltOneHotK256>::new(
            base.clone(),
            AkitaChunkProfile::default().witness_cfg(),
        )
        .expect("single-chunk fixture must bind"),
        &rows,
    )
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
    let trusted_catalog = TrustedScheduleCatalog::<JoltOneHotK256>::new(
        setup_catalog,
        ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect("config-bound catalog");
    let setup_capacity = SetupRequirements::from_catalog(&trusted_catalog, 38, 28)
        .expect("catalog-backed setup capacity")
        .matrix_capacity();
    assert!(setup_capacity.num_field_elements >= full_capacity.num_field_elements);
    assert!(setup_capacity.num_field_elements >= precommit_capacity);
}

#[test]
fn base_catalogs_contain_no_grouped_advice_rows() {
    let dense = dense_catalog();
    let base = one_hot_catalog(AKITA_ONE_HOT_K16, AkitaChunkProfile::new(1, 0));
    assert!(base
        .rows()
        .all(|row| row.profiles().precommitteds.is_empty()));

    let trusted_profile =
        dense_group_profile(&dense, FIXTURE_TRUSTED_ADVICE_GROUP).expect("fixture dense profile");
    for precommitteds in [
        vec![trusted_profile],
        vec![trusted_profile, trusted_profile],
    ] {
        for num_vars in FIXTURE_K16_FINAL_NUM_VARS.0..=FIXTURE_K16_FINAL_NUM_VARS.1 {
            let key = ScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(num_vars, 51),
                precommitteds: precommitteds.clone(),
            };
            assert!(base.resolve_key(&key).is_err());
        }
    }
}

#[test]
fn grouped_provisioning_rejects_unadmitted_final_shape() {
    let final_num_vars = K16_NUM_VARS.0 - 1;
    let request = GroupedScheduleParams::new(
        None,
        Some(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars()),
        Vec::new(),
        PolynomialGroupLayout::new(final_num_vars, 51),
    );
    let error = AkitaScheme::setup(AkitaSetupParams::one_hot_only_grouped(
        final_num_vars,
        51,
        52,
        [3; 32],
        AKITA_ONE_HOT_K16,
        Some(request),
        Arc::new(artifacts()),
    ))
    .expect_err("a declared reachable arity outside the family must fail setup");
    assert!(error.to_string().contains("outside the admitted catalog"));
}

#[test]
fn grouped_provisioning_rejects_out_of_family_final_arity() {
    for one_hot_k in [AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256] {
        for (profile, final_num_vars) in [
            (AkitaChunkProfile::new(1, 0), 11),
            (AkitaChunkProfile::new(8, 2), 13),
        ] {
            for grouped in [false, true] {
                let request = grouped.then(|| {
                    GroupedScheduleParams::new(
                        None,
                        Some(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars()),
                        Vec::new(),
                        PolynomialGroupLayout::new(final_num_vars, 1),
                    )
                });
                let error = AkitaScheme::setup(
                    AkitaSetupParams::one_hot_only_grouped(
                        final_num_vars,
                        1,
                        2,
                        [3; 32],
                        one_hot_k,
                        request,
                        Arc::new(artifacts()),
                    )
                    .with_akita_chunk_profile(profile),
                )
                .expect_err("an arity below the profile floor must fail setup");
                let message = error.to_string();
                assert!(message.contains("outside the supported range"));
                assert!(message.contains(&format!("profile {profile:?}")));
                assert!(message.contains(&format!("K={one_hot_k}")));
            }
        }
    }
}

/// Checks key membership only. `gen_jolt_schedules --check` replans and
/// compares complete artifact contents against the pinned backend.
#[test]
fn catalogs_have_exact_emitted_keys() {
    let specs = family_specs(PathBuf::new()).expect("emit specs");
    let cases = [
        (
            "jolt-fp128-onehot-k16",
            one_hot_catalog(AKITA_ONE_HOT_K16, AkitaChunkProfile::new(1, 0)),
        ),
        (
            "jolt-fp128-onehot-k256",
            one_hot_catalog(AKITA_ONE_HOT_K256, AkitaChunkProfile::new(1, 0)),
        ),
        (
            "jolt-fp128-onehot-k16",
            one_hot_catalog(AKITA_ONE_HOT_K16, AkitaChunkProfile::new(8, 2)),
        ),
        (
            "jolt-fp128-onehot-k256",
            one_hot_catalog(AKITA_ONE_HOT_K256, AkitaChunkProfile::new(8, 2)),
        ),
        ("jolt-fp128-dense-bounded", dense_catalog()),
        ("jolt-fp128-dense-full", full_dense_catalog()),
    ];
    assert_eq!(specs.len(), cases.len());
    assert_eq!(
        specs
            .get(specs.len() - 2)
            .expect("dense emit spec")
            .spec
            .family_name,
        JoltDenseBounded::schedule_family_name()
    );
    for (fixture, (family_name, catalog)) in specs.iter().zip(cases) {
        let spec = &fixture.spec;
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

#[cfg(feature = "field-inline")]
mod field_inc {
    #![expect(
        clippy::panic,
        reason = "pin tests attribute a failing arity in the panic message"
    )]

    use akita_config::TrustedScheduleCatalog;
    use akita_params::{PolynomialGroupLayout, ScheduleLookupKey};
    use jolt_akita::configs::AkitaChunkProfile;
    use jolt_akita::configs::JoltOneHotK16;
    use jolt_akita::schedule_registry::GroupedScheduleParams;
    use jolt_akita::schedule_registry::{
        dense_group_profile, extend_catalog, provision_groups_for_k, FIXTURE_K16_FINAL_NUM_VARS,
        FIXTURE_TRUSTED_ADVICE_GROUP,
    };
    use jolt_akita::schedules::emit::one_hot_keys;
    use jolt_akita::{DenseGroupLayout, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};
    use jolt_claims::protocols::field_inline::lattice::FieldIncLayout;

    use super::{dense_catalog, full_dense_catalog, one_hot_catalog};

    fn trace_arity_overhead(one_hot_k: usize) -> usize {
        one_hot_k.ilog2() as usize
    }

    fn field_inline_rows_plan_and_resolve(
        one_hot_k: usize,
        profile: AkitaChunkProfile,
        final_groups: impl IntoIterator<Item = PolynomialGroupLayout>,
    ) {
        let dense = dense_catalog();
        let full_dense = full_dense_catalog();
        let base = one_hot_catalog(one_hot_k, profile);
        let overhead = trace_arity_overhead(one_hot_k);
        for final_group in final_groups {
            let final_num_vars = final_group.num_vars();
            let layout = FieldIncLayout::new(final_num_vars - overhead);
            let params = GroupedScheduleParams::new(
                None,
                None,
                vec![DenseGroupLayout::FullWidth {
                    num_vars: layout.num_vars(),
                }],
                final_group,
            );
            let setup_catalog = params
                .extend_catalog(&dense, &full_dense, &base, one_hot_k, profile)
                .unwrap_or_else(|error| {
                    panic!("K={one_hot_k} {profile:?} {final_group:?}: field-inline provisioning failed: {error}")
                });
            let key = ScheduleLookupKey {
                final_group,
                precommitteds: vec![dense_group_profile(
                    &full_dense,
                    PolynomialGroupLayout::new(layout.num_vars(), 1),
                )
                .expect("field increment profile")],
            };
            let resolved = setup_catalog.resolve_key(&key).unwrap_or_else(|error| {
                panic!(
                    "K={one_hot_k} final arity {final_num_vars} must resolve its field-inline row: {error}"
                )
            });
            assert_eq!(resolved.profiles().precommitteds, key.precommitteds);
            assert_eq!(resolved.profiles().final_group.group, final_group);
            assert_eq!(
                resolved.schedule().root.params.witness_chunk.num_chunks,
                profile.num_chunks()
            );
        }
    }

    fn assert_k16_production_coverage(profile: AkitaChunkProfile) {
        let groups = one_hot_keys(AKITA_ONE_HOT_K16, profile)
            .expect("canonical production grid")
            .into_iter()
            .filter(|group| group.num_polynomials() > 2)
            .collect::<Vec<_>>();
        assert_eq!(groups.len(), 266);
        field_inline_rows_plan_and_resolve(AKITA_ONE_HOT_K16, profile, groups);
    }

    #[test]
    fn field_inline_rows_cover_k16_single_production_grid() {
        assert_k16_production_coverage(AkitaChunkProfile::new(1, 0));
    }

    fn assert_k16_chunked_boundary_coverage(profile: AkitaChunkProfile) {
        let groups = [51, 64].into_iter().flat_map(|width| {
            [16, 25, 34]
                .into_iter()
                .map(move |arity| PolynomialGroupLayout::new(arity, width))
        });
        field_inline_rows_plan_and_resolve(AKITA_ONE_HOT_K16, profile, groups);
    }

    #[test]
    fn field_inline_rows_cover_k16_w8r2_boundaries() {
        assert_k16_chunked_boundary_coverage(AkitaChunkProfile::new(8, 2));
    }

    #[test]
    fn field_inline_rows_plan_and_resolve_for_k256_trace_fixtures() {
        field_inline_rows_plan_and_resolve(
            AKITA_ONE_HOT_K256,
            AkitaChunkProfile::new(1, 0),
            [(20, 29), (28, 27), (29, 27), (34, 27)]
                .map(|(num_vars, num_polys)| PolynomialGroupLayout::new(num_vars, num_polys)),
        );
    }

    #[test]
    fn full_width_replanning_is_limited_to_one_inc_and_two_advice_groups() {
        let full = DenseGroupLayout::FullWidth { num_vars: 30 };
        let dense = dense_catalog();
        let full_dense = full_dense_catalog();
        let base = one_hot_catalog(AKITA_ONE_HOT_K256, AkitaChunkProfile::new(1, 0));
        for layouts in [
            vec![
                DenseGroupLayout::Bounded { num_vars: 14 },
                DenseGroupLayout::Bounded { num_vars: 15 },
                DenseGroupLayout::Bounded { num_vars: 16 },
                full,
            ],
            vec![full, full],
        ] {
            let error = provision_groups_for_k(
                &dense,
                &full_dense,
                &base,
                &GroupedScheduleParams::new(
                    None,
                    None,
                    layouts,
                    PolynomialGroupLayout::new(34, 27),
                ),
                AKITA_ONE_HOT_K256,
            )
            .expect_err(
                "unsupported full-width batch shapes must be rejected before planner search",
            );
            assert!(error.to_string().contains(
                "full-width batches support one field increment and at most two advice groups"
            ));
        }
    }

    #[test]
    fn field_inline_rows_append_the_inc_group_to_every_advice_combination() {
        let dense = dense_catalog();
        let full_dense = full_dense_catalog();
        let base = one_hot_catalog(AKITA_ONE_HOT_K16, AkitaChunkProfile::new(1, 0));
        let final_num_vars = FIXTURE_K16_FINAL_NUM_VARS.1;
        let layout = FieldIncLayout::new(final_num_vars - trace_arity_overhead(AKITA_ONE_HOT_K16));
        let trusted = FIXTURE_TRUSTED_ADVICE_GROUP.num_vars();
        let rows = provision_groups_for_k(
            &dense,
            &full_dense,
            &base,
            &GroupedScheduleParams::new(
                Some(trusted + 1),
                Some(trusted),
                vec![DenseGroupLayout::FullWidth {
                    num_vars: layout.num_vars(),
                }],
                PolynomialGroupLayout::new(final_num_vars, 51),
            ),
            AKITA_ONE_HOT_K16,
        )
        .expect("provisioning with field-inline must plan every combination");
        assert_eq!(rows.rows().len(), 4);
        let inc = dense_group_profile(
            &full_dense,
            PolynomialGroupLayout::new(layout.num_vars(), 1),
        )
        .expect("field increment profile");
        for row in rows.rows() {
            assert_eq!(row.profiles().precommitteds.last(), Some(&inc));
        }
        let catalog = extend_catalog::<JoltOneHotK16>(
            &TrustedScheduleCatalog::<JoltOneHotK16>::new(
                base.clone(),
                AkitaChunkProfile::default().witness_cfg(),
            )
            .expect("single-chunk fixture must bind"),
            &rows,
        )
        .expect("freeze grouped catalog");
        for num_vars in [trusted, trusted + 1] {
            let widened_advice =
                dense_group_profile(&full_dense, PolynomialGroupLayout::new(num_vars, 1))
                    .expect("full-width advice-shaped profile");
            assert!(catalog
                .resolve_key(&ScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(final_num_vars, 51),
                    precommitteds: vec![widened_advice, inc],
                })
                .is_err());
        }
    }
}
