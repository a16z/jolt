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

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
#[test]
fn prover_pool_reuses_workers_without_changing_backend_routing() {
    let pool =
        AkitaProverPool::new(NonZeroUsize::new(2).expect("nonzero workers")).expect("prover pool");
    pool.install(|| {
        let _ = rayon::broadcast(|_| {
            with_prover_backend(|| {
                assert!(pool.0.current_thread_index().is_some());
            });
            with_backend_pool(|| {
                assert!(pool.0.current_thread_index().is_none());
            });
        });
    });
    with_prover_backend(|| {
        assert!(pool.0.current_thread_index().is_none());
        assert!(backend_pool()
            .expect("threaded host")
            .current_thread_index()
            .is_some());
    });
}

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
#[test]
fn prover_pool_active_wait_runs_parallel_work_and_broadcasts() {
    use rayon::prelude::*;
    for workers in [1, 4] {
        let sum = AkitaProverPool::with_active_wait(
            NonZeroUsize::new(workers).expect("active-wait test fixture"),
            || {
                let visits = rayon::broadcast(|_| {
                    assert!(IN_PROVER_POOL.with(Cell::get));
                    with_prover_backend(rayon::current_num_threads)
                });
                assert_eq!(visits, vec![workers; workers]);
                (0usize..1000).into_par_iter().sum::<usize>()
            },
        )
        .expect("active-wait test fixture");
        assert_eq!(sum, 499_500);
    }
}

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
#[test]
fn prover_pool_active_wait_returns_before_detached_jobs_need_the_caller() {
    use std::{sync::mpsc, time::Duration};
    let (release_tx, release_rx) = mpsc::channel();
    let (returned_tx, returned_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let _worker = std::thread::spawn(move || {
        AkitaProverPool::with_active_wait(
            NonZeroUsize::new(2).expect("active-wait test fixture"),
            || {
                let (started_tx, started_rx) = mpsc::channel();
                rayon::spawn(move || {
                    started_tx.send(()).expect("active-wait test fixture");
                    release_rx.recv().expect("active-wait test fixture");
                    finished_tx.send(()).expect("active-wait test fixture");
                });
                started_rx
                    .recv_timeout(Duration::from_secs(30))
                    .expect("active-wait test fixture");
            },
        )
        .expect("active-wait test fixture");
        returned_tx.send(()).expect("active-wait test fixture");
    });
    // Joining helper threads inside the API would deadlock before this send.
    returned_rx
        .recv_timeout(Duration::from_secs(30))
        .expect("active-wait test fixture");
    release_tx.send(()).expect("active-wait test fixture");
    finished_rx
        .recv_timeout(Duration::from_secs(30))
        .expect("active-wait test fixture");
}

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
#[test]
fn prover_pool_active_wait_preserves_cross_pool_cooperation() {
    use std::{sync::mpsc, time::Duration};
    let (tx, rx) = mpsc::channel();
    let _worker = std::thread::spawn(move || {
        let outer = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("active-wait test fixture");
        let answer = outer.install(|| {
            AkitaProverPool::with_active_wait(
                NonZeroUsize::new(1).expect("active-wait test fixture"),
                || outer.install(|| 42),
            )
            .expect("active-wait test fixture")
        });
        tx.send(answer).expect("active-wait test fixture");
    });
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(30))
            .expect("active-wait test fixture"),
        42
    );
}

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
#[test]
fn prover_pool_active_wait_supports_nested_and_concurrent_calls() {
    use std::{sync::mpsc, time::Duration};
    let (tx, rx) = mpsc::channel();
    for index in 0..2 {
        let tx = tx.clone();
        let _worker = std::thread::spawn(move || {
            let answer = AkitaProverPool::with_active_wait(
                NonZeroUsize::new(2).expect("active-wait test fixture"),
                || {
                    AkitaProverPool::with_active_wait(
                        NonZeroUsize::new(1).expect("active-wait test fixture"),
                        || index + 10,
                    )
                    .expect("active-wait test fixture")
                },
            )
            .expect("active-wait test fixture");
            tx.send(answer).expect("active-wait test fixture");
        });
    }
    let mut answers = (0..2)
        .map(|_| {
            rx.recv_timeout(Duration::from_secs(30))
                .expect("active-wait test fixture")
        })
        .collect::<Vec<_>>();
    answers.sort_unstable();
    assert_eq!(answers, vec![10, 11]);
}

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
#[test]
fn prover_pool_active_wait_stops_workers_after_normal_return_and_panic() {
    use std::{
        panic::AssertUnwindSafe, sync::mpsc, thread::Result as ThreadResult, time::Duration,
    };
    for panic in [false, true] {
        let (tx, rx) = mpsc::channel();
        let pool = ThreadPoolBuilder::new()
            .num_threads(2)
            .exit_handler(move |_| {
                let _send_result = tx.send(());
            })
            .build()
            .expect("active-wait test fixture");
        let result: ThreadResult<()> = std::panic::catch_unwind(AssertUnwindSafe(|| {
            with_fresh_active_workers(pool, || assert!(!panic, "intentional root panic"));
        }));
        assert_eq!(result.is_err(), panic);
        for _ in 0..2 {
            rx.recv_timeout(Duration::from_secs(30))
                .expect("active-wait test fixture");
        }
    }
}
