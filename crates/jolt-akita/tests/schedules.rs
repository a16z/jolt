#![expect(
    clippy::expect_used,
    reason = "catalog tests should fail loudly when an artifact or grid is malformed"
)]

//! Coverage, setup-sizing, and regeneration guards for Jolt's external catalogs.

use akita_config::{SetupRequirements, TrustedScheduleCatalog};
#[cfg(all(feature = "metal", target_os = "macos"))]
use akita_metal::MetalBackend;
use akita_planner::emit::{
    GroupedGenerationRequest, MaterializationDiagnostics, PrecommittedProducer,
};
use akita_schedules::{ResolvedScheduleRow, ValidatedScheduleCatalog};
use akita_types::{
    commit_only_setup_field_elements, setup_matrix_capacity_for_schedule, AkitaScheduleLookupKey,
    FoldSchedule, GroupCommitPhaseParams, PolynomialGroupLayout,
};
use jolt_akita::configs::{
    JoltDenseBounded, JoltFieldDigits, JoltOneHotK16, JoltOneHotK256, JoltSignedBytes,
};
use jolt_akita::schedule_registry::{
    dense_precommit_profile, extend_catalog, precommitted_producer, provision,
    PrecommittedScheduleParams, TraceFamily, FIXTURE_K16_FINAL_NUM_VARS,
    FIXTURE_TRUSTED_ADVICE_GROUP,
};
use jolt_akita::schedules::emit::{
    family_specs, keys, FIELD_DIGIT_GROUPS, K16_NUM_VARS, K16_PACKING_VARIABLES, K256_NUM_VARS,
    K256_PACKING_VARIABLES, ONE_HOT_TRACE_NUM_POLYS, RECURSIVE_TRACE_LOG_T_CUTOVER,
    SIGNED_BYTE_NUM_VARS, SIGNED_BYTE_PACKING_VARIABLES, SIGNED_BYTE_PINNED_ROOTS,
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

fn dense_producers(profiles: &[GroupCommitPhaseParams]) -> Vec<PrecommittedProducer> {
    profiles
        .iter()
        .map(|profile| precommitted_producer::<JoltDenseBounded>(*profile).expect("dense producer"))
        .collect()
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

#[test]
fn k256_t28_catalog_preserves_prover_optimized_root() {
    let schedule = scalar_schedule(&one_hot_catalog(AKITA_ONE_HOT_K256), 41);
    let inner = &schedule.root.params.final_group().profile.inner.matrix;
    assert_eq!(inner.ring_dimension(), 128);
    assert_eq!(inner.output_rank(), 3);
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

    for key in keys(ONE_HOT_TRACE_NUM_POLYS, K256_NUM_VARS) {
        one_hot_catalog(AKITA_ONE_HOT_K256)
            .resolve_key(&AkitaScheduleLookupKey::single(key))
            .expect("K256 catalog row must resolve")
            .validate_opening_layout(
                &akita_types::OpeningClaimsLayout::from_groups(vec![key])
                    .expect("K256 Metal catalog key must form an opening layout"),
            )
            .expect("K256 Metal catalog row must validate and resolve");
    }
}

/// The shared CPU/Metal catalog preserves every row digest across transport.
#[test]
fn shared_k256_catalog_preserves_row_digests_across_transport() {
    let catalog = one_hot_catalog(AKITA_ONE_HOT_K256);
    let transported = TrustedScheduleCatalog::<JoltOneHotK256>::from_artifact_bytes(
        &catalog.to_artifact_bytes().expect("encode shared catalog"),
    )
    .expect("decode shared catalog");
    for key in keys(ONE_HOT_TRACE_NUM_POLYS, K256_NUM_VARS) {
        let lookup = AkitaScheduleLookupKey::single(key);
        let original = catalog
            .resolve_key(&lookup)
            .expect("original row must resolve");
        let decoded = transported
            .resolve_key(&lookup)
            .expect("transported row must resolve");
        assert_eq!(
            original.selection().row_digest,
            decoded.selection().row_digest,
            "shared K256 row digest changed during transport at {key:?}"
        );
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

    let rows = provision::<JoltOneHotK256>(
        &base,
        &[dense_producers(&key.precommitteds)],
        [key.final_group.num_vars()],
    )
    .expect("preprocessing must adapt the production grouped row");
    assert_eq!(rows.rows().len(), 1);

    let setup_catalog =
        extend_catalog::<JoltOneHotK256>(&base, &rows).expect("freeze setup-owned catalog");
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
        let rows =
            provision::<JoltOneHotK16>(&base, &[dense_producers(&[precommit])], [final_num_vars])
                .expect("adapt the grouped K=16 row");
        let setup_catalog =
            extend_catalog::<JoltOneHotK16>(&base, &rows).expect("freeze adapted K=16 catalog");
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
    let rows = provision::<JoltOneHotK256>(
        &base,
        &[dense_producers(&key.precommitteds)],
        [key.final_group.num_vars()],
    )
    .expect("adapt grouped row");
    let setup_catalog =
        extend_catalog::<JoltOneHotK256>(&base, &rows).expect("freeze setup catalog");
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
    let error = PrecommittedScheduleParams::new(
        None,
        Some(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars()),
        K16_NUM_VARS.0 - 1,
    )
    .extend_catalog(&dense, &[], &base, TraceFamily::OneHotK16)
    .expect_err("a declared reachable arity outside the family must fail setup");
    assert!(error.to_string().contains("outside the supported range"));
}

#[derive(Debug, PartialEq, Eq)]
struct RootGeometry {
    positions_per_block: usize,
    ring_dimension: usize,
    log_basis: u32,
    digits: usize,
    rank: usize,
}

fn root_geometry(profile: &GroupCommitPhaseParams) -> RootGeometry {
    RootGeometry {
        positions_per_block: profile.blocks.positions_per_block,
        ring_dimension: profile.inner.matrix.ring_dimension(),
        log_basis: profile.inner.digits.log_basis,
        digits: profile.inner.digits.num_digits,
        rank: profile.inner.matrix.output_rank(),
    }
}

fn scalar_profile(
    catalog: &ValidatedScheduleCatalog,
    group: PolynomialGroupLayout,
) -> GroupCommitPhaseParams {
    catalog
        .resolve_key(&AkitaScheduleLookupKey::single(group))
        .expect("scalar row must resolve")
        .profiles()
        .final_group
}

#[test]
fn signed_byte_rows_commit_one_d128_byte_plane_and_pin_their_roots() {
    let catalog = artifacts()
        .signed_byte_catalog()
        .expect("signed-byte catalog");
    let grid = keys(&[1], SIGNED_BYTE_NUM_VARS);
    assert_eq!(catalog.len(), grid.len());
    for key in grid {
        let profile = scalar_profile(&catalog, key);
        let geometry = root_geometry(&profile);
        assert_eq!(
            (geometry.log_basis, geometry.digits, geometry.ring_dimension),
            (8, 1, 128),
            "{key:?}"
        );
        #[cfg(all(feature = "metal", target_os = "macos"))]
        assert!(MetalBackend::commits_signed_bytes(&profile), "{key:?}");
        for (num_vars, root) in SIGNED_BYTE_PINNED_ROOTS {
            if key.num_vars() == num_vars {
                assert_eq!(geometry.positions_per_block, root.positions_per_block);
            }
        }
    }
}

#[test]
fn field_digit_rows_commit_sixteen_byte_planes() {
    let catalog = artifacts()
        .field_digit_catalog()
        .expect("field-digit catalog");
    assert_eq!(catalog.len(), FIELD_DIGIT_GROUPS.len());
    let [triples, ram] = FIELD_DIGIT_GROUPS;
    assert_eq!(
        root_geometry(&scalar_profile(&catalog, triples)),
        RootGeometry {
            positions_per_block: 1024,
            ring_dimension: 128,
            log_basis: 8,
            digits: 16,
            rank: 4,
        }
    );
    assert_eq!(
        root_geometry(&scalar_profile(&catalog, ram)),
        RootGeometry {
            positions_per_block: 32,
            ring_dimension: 128,
            log_basis: 8,
            digits: 16,
            rank: 3,
        }
    );
}

#[test]
fn signed_byte_grouped_rows_keep_each_trace_root_beside_advice_and_field_digits() {
    let artifacts = artifacts();
    let dense = dense_catalog();
    let base = artifacts
        .signed_byte_catalog()
        .expect("signed-byte catalog");
    let field_digits = artifacts
        .field_digit_catalog()
        .expect("field-digit catalog");
    let mut producers =
        dense_producers(&[
            dense_precommit_profile(&dense, FIXTURE_TRUSTED_ADVICE_GROUP)
                .expect("trusted advice profile"),
        ]);
    producers.extend(FIELD_DIGIT_GROUPS.map(|group| {
        precommitted_producer::<JoltFieldDigits>(scalar_profile(&field_digits, group))
            .expect("field-digit producer")
    }));
    for final_num_vars in [
        16 + SIGNED_BYTE_PACKING_VARIABLES,
        20 + SIGNED_BYTE_PACKING_VARIABLES,
        29 + SIGNED_BYTE_PACKING_VARIABLES,
    ] {
        let final_group = PolynomialGroupLayout::new(final_num_vars, 1);
        let rows = provision::<JoltSignedBytes>(&base, &[producers.clone()], [final_num_vars])
            .expect("adapt the grouped signed-byte row");
        let catalog =
            extend_catalog::<JoltSignedBytes>(&base, &rows).expect("freeze the setup catalog");
        let resolved = catalog
            .resolve_key(&GroupedGenerationRequest::new(final_group, producers.clone()).key())
            .expect("grouped signed-byte row");
        assert_adaptation_preserves_main_skeleton(&base, resolved, final_group);
        #[cfg(all(feature = "metal", target_os = "macos"))]
        assert!(MetalBackend::commits_signed_bytes(
            &resolved.profiles().final_group
        ));
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
