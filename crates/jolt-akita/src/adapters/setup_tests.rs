#![expect(
    clippy::expect_used,
    reason = "tests assert successful setup provisioning"
)]

use super::*;
use crate::schedule_registry::DenseGroupLayout;
use crate::{AkitaChunkProfile, AkitaScheme};
use akita_params::PolynomialGroupLayout;
use jolt_openings::CommitmentScheme;

fn setup() -> AkitaProverSetup {
    AkitaScheme::setup(AkitaSetupParams::new(
        16,
        2,
        [7; 32],
        AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
    ))
    .expect("both setup flavors should build")
    .0
}

#[test]
fn covered_setup_domain_reuses_prepared_state() {
    let original = setup();
    let covered = original
        .clone()
        .with_setup_domain(1)
        .expect("covered domain");
    assert!(std::ptr::eq(
        original.one_hot_backend().expect("original backend").0,
        covered.one_hot_backend().expect("covered backend").0,
    ));
    assert!(std::ptr::eq(
        original.one_hot_backend().expect("original backend").1,
        covered.one_hot_backend().expect("covered backend").1,
    ));
    assert!(Arc::ptr_eq(
        &original.verifier.backend_cache.one_hot,
        &covered.verifier.backend_cache.one_hot,
    ));
    assert!(Arc::ptr_eq(
        &original.verifier.backend_cache.dense,
        &covered.verifier.backend_cache.dense,
    ));
}

#[test]
fn growing_setup_domain_preserves_prefixes_and_other_flavors() {
    let original = setup();
    let original_key = original.one_hot_backend().expect("original backend").0;
    let old_len = original_key.expanded.descriptor().num_field_elements;
    let new_len = old_len
        .checked_next_power_of_two()
        .and_then(|n| n.checked_mul(2))
        .expect("small fixture domain");
    let grown = original
        .clone()
        .with_setup_domain(new_len)
        .expect("larger domain");
    let grown_key = grown.one_hot_backend().expect("grown backend").0;
    assert_eq!(grown_key.expanded.descriptor().num_field_elements, new_len);
    assert_eq!(
        grown_key
            .expanded
            .shared_matrix()
            .as_field_slice()
            .get(..old_len)
            .expect("original prefix"),
        original_key.expanded.shared_matrix().as_field_slice(),
    );
    assert_eq!(grown_key.prefix_slots, original_key.prefix_slots);
    assert!(std::ptr::eq(
        original.dense_backend().expect("original dense backend").1,
        grown.dense_backend().expect("preserved dense backend").1,
    ));
    assert!(Arc::ptr_eq(
        &original.verifier.backend_cache.dense,
        &grown.verifier.backend_cache.dense,
    ));
    assert!(Arc::ptr_eq(
        &original.verifier.backend_cache.one_hot_scheme,
        &grown.verifier.backend_cache.one_hot_scheme,
    ));
    assert!(!Arc::ptr_eq(
        &original.verifier.backend_cache.one_hot,
        &grown.verifier.backend_cache.one_hot,
    ));
    assert_eq!(
        original
            .one_hot_backend()
            .expect("original backend")
            .0
            .expanded
            .descriptor()
            .num_field_elements,
        old_len
    );
    assert!(original.verifier.backend_cache.one_hot.get().is_some());
    assert!(grown.verifier.backend_cache.one_hot.get().is_some());
}

#[test]
fn setup_domain_rejects_non_power_of_two_sizes() {
    let original = setup();
    for invalid in [0, 3, usize::MAX] {
        assert!(original.clone().with_setup_domain(invalid).is_err());
    }
}

/// A setup whose grouped schedule admits one full-width dense group at 14
/// variables.
fn full_width_setup(params: AkitaSetupParams) -> AkitaProverSetup {
    let mut params = params;
    params.grouped_schedule = Some(GroupedScheduleParams::new(
        None,
        None,
        vec![DenseGroupLayout::FullWidth { num_vars: 14 }],
        PolynomialGroupLayout::new(14, 1),
    ));
    AkitaScheme::setup(params)
        .expect("grouped setup should build")
        .0
}

#[test]
fn streamed_ring_switch_reaches_every_backend_and_survives_a_larger_domain() {
    let artifacts =
        || AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
    let limit = |backend: &AkitaBackend| backend.max_cached_ring_switch_elements();

    let cached = setup();
    assert_ne!(limit(cached.dense_backend().expect("dense backend").1), 0);
    assert_ne!(
        limit(cached.one_hot_backend().expect("one-hot backend").1),
        0
    );
    let streamed = cached
        .with_streamed_ring_switch()
        .expect("streamed ring switch");
    assert_eq!(limit(streamed.dense_backend().expect("dense backend").1), 0);
    let (one_hot, backend) = streamed.one_hot_backend().expect("one-hot backend");
    assert_eq!(limit(backend), 0);
    let grown = streamed
        .clone()
        .with_setup_domain(2 * one_hot.expanded.descriptor().num_field_elements)
        .expect("larger domain");
    assert_eq!(limit(grown.one_hot_backend().expect("grown backend").1), 0);

    // A full-width group that commits on the one-hot backend keeps doing so.
    let shared = full_width_setup(AkitaSetupParams::one_hot_only_grouped(
        14,
        1,
        2,
        [7; 32],
        AKITA_ONE_HOT_K16,
        None,
        artifacts(),
    ))
    .with_streamed_ring_switch()
    .expect("streamed ring switch");
    let full_width = shared.full_width.get(&14).expect("full-width group");
    assert_eq!(limit(&full_width.backend), 0);
    assert!(std::ptr::eq(
        full_width.backend.as_ref(),
        shared.one_hot_backend().expect("one-hot backend").1,
    ));

    // Without a one-hot backend the group owns its backend and its matrix.
    let owned = full_width_setup(AkitaSetupParams::dense_only(14, 1, [7; 32], artifacts()));
    let matrix = Arc::clone(
        &owned
            .full_width
            .get(&14)
            .expect("full-width group")
            .expanded,
    );
    let owned = owned
        .with_streamed_ring_switch()
        .expect("streamed ring switch");
    let full_width = owned.full_width.get(&14).expect("full-width group");
    assert_eq!(limit(&full_width.backend), 0);
    assert!(Arc::ptr_eq(&matrix, &full_width.expanded));
}
