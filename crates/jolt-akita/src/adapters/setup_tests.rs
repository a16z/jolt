#![expect(
    clippy::expect_used,
    reason = "tests assert successful setup provisioning"
)]

use super::*;
use crate::{AkitaChunkProfile, AkitaScheme};
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

#[test]
#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
fn prover_pool_is_local_to_prover_setup() {
    let original = setup();
    let pool = AkitaProverPool::new(
        NonZeroUsize::new(2).expect("nonzero worker count"),
        Some(Duration::from_micros(64)),
    )
    .expect("prover workers spawn");
    let configured = original.clone().with_prover_pool(pool.clone());
    let cloned = configured.clone();
    assert!(original.prover_pool.is_none());
    assert!(Arc::ptr_eq(
        &cloned.prover_pool.as_ref().expect("configured pool").0,
        &pool.0,
    ));
    assert_eq!(original.verifier, configured.verifier);
    assert_eq!(
        bincode::serde::encode_to_vec(&original.verifier, bincode::config::standard())
            .expect("original verifier setup serializes"),
        bincode::serde::encode_to_vec(&configured.verifier, bincode::config::standard())
            .expect("configured verifier setup serializes"),
    );
    configured.with_prover_backend(|| {
        assert!(pool.0.current_thread_index().is_some());
        cloned.with_prover_backend(|| {
            assert!(pool.0.current_thread_index().is_some());
        });
        with_backend_pool(|| {
            assert!(pool.0.current_thread_index().is_none());
            assert!(backend_pool()
                .expect("default workers spawn")
                .current_thread_index()
                .is_some());
        });
    });
}
