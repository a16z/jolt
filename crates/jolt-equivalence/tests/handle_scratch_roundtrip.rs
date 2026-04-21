//! Opaque handle lifecycle on `CpuBackend`: open -> bind -> query -> close.
//!
//! Exercises the infra landed in iter 73 P76-B. Runs `HandleShape::Scratch`
//! end-to-end under a Rayon parallel context to stress the interior mutex.
//! Does NOT exercise any hot path; that lands in later iterations when
//! `HandleShape::Eq` is wired.

use jolt_compute::{BindingOrder, ComputeBackend, HandleShape};
use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use rayon::prelude::*;

fn make_challenges(n: usize) -> Vec<Fr> {
    let mut rng = ark_std::test_rng();
    (0..n).map(|_| Fr::random(&mut rng)).collect()
}

#[test]
fn scratch_roundtrip_serial() {
    let backend = CpuBackend;
    let challenges = make_challenges(8);

    let id = backend.open_handle::<Fr>(HandleShape::Scratch { size: 8 });
    for (round, &r) in challenges.iter().enumerate() {
        backend.bind_handle::<Fr>(id, round, r);
    }
    for (round, &r) in challenges.iter().enumerate() {
        assert_eq!(backend.query_handle::<Fr>(id, round), r);
    }
    backend.close_handle(id);
}

/// Many handles, opened and driven concurrently by Rayon. Validates that the
/// internal store's `Mutex` + `AtomicU32` counter stay coherent.
#[test]
fn scratch_roundtrip_parallel() {
    let backend = CpuBackend;
    let n_handles = 64usize;
    let rounds = 16usize;

    let challenges: Vec<Vec<Fr>> = (0..n_handles).map(|_| make_challenges(rounds)).collect();

    let ids: Vec<_> = (0..n_handles)
        .into_par_iter()
        .map(|i| {
            let id = backend.open_handle::<Fr>(HandleShape::Scratch { size: rounds });
            for (round, &r) in challenges[i].iter().enumerate() {
                backend.bind_handle::<Fr>(id, round, r);
            }
            id
        })
        .collect();

    let mismatches: usize = ids
        .par_iter()
        .enumerate()
        .map(|(i, &id)| {
            let mut wrong = 0;
            for (round, &expected) in challenges[i].iter().enumerate() {
                if backend.query_handle::<Fr>(id, round) != expected {
                    wrong += 1;
                }
            }
            wrong
        })
        .sum();
    assert_eq!(mismatches, 0, "handle query mismatches under par context");

    for id in ids {
        backend.close_handle(id);
    }
}

/// Sanity: each `open_handle` returns a distinct `HandleId`, even under
/// concurrent contention.
#[test]
fn scratch_handle_ids_unique_under_par() {
    let backend = CpuBackend;
    let mut ids: Vec<_> = (0..1024)
        .into_par_iter()
        .map(|_| backend.open_handle::<Fr>(HandleShape::Scratch { size: 1 }))
        .collect();
    ids.sort_by_key(|h| h.0);
    ids.dedup();
    assert_eq!(ids.len(), 1024);
    for id in ids {
        backend.close_handle(id);
    }
}

/// `HandleShape::Eq` is declared on the trait but not yet implemented —
/// attempts to open one panic. This test pins that contract so that the iter
/// that wires Eq has a clear "remove this test" marker.
#[test]
#[should_panic(expected = "HandleShape::Eq")]
fn eq_shape_panics_until_wired() {
    let backend = CpuBackend;
    let challenges = make_challenges(4);
    let _ = backend.open_handle::<Fr>(HandleShape::Eq {
        challenges: &challenges,
        order: BindingOrder::LowToHigh,
    });
}
