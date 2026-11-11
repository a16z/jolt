use super::alloc::{Allocate, TableInit};
use super::failure_policy::FailAt;
use crate::{AllocationCause, Budget, EqPolynomial, EqTableError, TensorEqTable};
use jolt_field::{Fr, Prime128Offset275, Ring};
use jolt_utils::alloc::ZeroAllocError;
use std::error::Error;
use std::panic::{catch_unwind, set_hook, take_hook, Location, UnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

const UNLIMITED: Budget = Budget {
    max_bytes: usize::MAX,
};

#[test]
fn each_owned_allocation_can_fail_without_leaking() {
    let calls = AtomicUsize::new(0);
    for n in 0..=3 {
        let point = vec![Fr::from_u64(2); n];
        for kind in 0..6 {
            let allocations = match kind {
                3 | 4 => n + 2,
                5 => 2,
                _ => 1,
            };
            for target in 1..=allocations + 1 {
                calls.store(0, Ordering::SeqCst);
                let policy = FailAt {
                    calls: &calls,
                    target,
                    budget: UNLIMITED,
                };
                let result = match kind {
                    0 => EqPolynomial::<Fr>::evals_in(&point, None, policy).map(drop),
                    1 => EqPolynomial::<Fr>::evals_mapped_in(&point, None, |x| x, policy).map(drop),
                    2 => EqPolynomial::<Fr>::evals_prefix_in(&point, None, 1, policy).map(drop),
                    3 => EqPolynomial::<Fr>::evals_cached_msb_in(&point, None, policy).map(drop),
                    4 => EqPolynomial::<Fr>::evals_cached_lsb_in(&point, None, policy).map(drop),
                    5 => TensorEqTable::from_point_in(&point, Fr::from_u64(1), n / 2, policy)
                        .map(drop),
                    _ => unreachable!(),
                };
                if target <= allocations {
                    assert!(matches!(result, Err(EqTableError::AllocationFailed { .. })));
                } else {
                    assert!(result.is_ok());
                    assert_eq!(calls.load(Ordering::SeqCst), allocations);
                }
                FailAt::assert_released();
            }
        }
    }
    // Both tensor admissions must precede even the first factor allocation.
    let policy = FailAt {
        calls: &calls,
        target: 1,
        budget: Budget {
            max_bytes: size_of::<Fr>(),
        },
    };
    calls.store(0, Ordering::SeqCst);
    assert!(matches!(
        TensorEqTable::from_point_in(&[Fr::from_u64(2); 3], Fr::from_u64(1), 0, policy),
        Err(EqTableError::OverBudget { .. })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let policy = FailAt {
        budget: UNLIMITED,
        ..policy
    };
    for n in [usize::BITS as usize, usize::BITS as usize - 1] {
        let point = vec![Fr::from_u64(2); n];
        for result in [
            EqPolynomial::<Fr>::evals_in(&point, None, policy).map(drop),
            EqPolynomial::<Fr>::evals_mapped_in(&point, None, |x| x, policy).map(drop),
            EqPolynomial::<Fr>::evals_prefix_in(&point, None, 1usize << (usize::BITS - 1), policy)
                .map(drop),
            EqPolynomial::<Fr>::evals_cached_msb_in(&point, None, policy).map(drop),
            EqPolynomial::<Fr>::evals_cached_lsb_in(&point, None, policy).map(drop),
            TensorEqTable::from_point_in(&point, Fr::from_u64(1), 0, policy).map(drop),
        ] {
            assert!(matches!(
                result,
                Err(EqTableError::DimensionOverflow { .. }
                    | EqTableError::SizeOverflow { .. }
                    | EqTableError::LengthMismatch { .. })
            ));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
    // Arithmetic-only admission of both 1 GiB factors of the 52-variable tensor.
    assert_eq!(size_of::<Prime128Offset275>(), 16);
    let policy = FailAt {
        budget: Budget { max_bytes: 1 << 30 },
        target: 0,
        ..policy
    };
    assert!(matches!(
        TensorEqTable::from_point_in(
            &[Prime128Offset275::from_u64(2); 52],
            Prime128Offset275::from_u64(1),
            26,
            policy
        ),
        Err(EqTableError::AllocationFailed { .. })
    ));
    FailAt::assert_released();
}

#[test]
fn budget_caps_each_family_in_requested_bytes() {
    let point = [Fr::from_u64(2); 3];
    let unit = size_of::<Fr>();
    for (elements, kind) in [(8, 0), (8, 1), (5, 2), (15, 3), (15, 4), (4, 5)] {
        for extra in [0, 1] {
            let budget = Budget {
                max_bytes: elements * unit - extra,
            };
            let result = match kind {
                0 => EqPolynomial::<Fr>::evals_in(&point, None, budget).map(drop),
                1 => EqPolynomial::<Fr>::evals_mapped_in(&point, None, |x| x, budget).map(drop),
                2 => EqPolynomial::<Fr>::evals_prefix_in(&point, None, 5, budget).map(drop),
                3 => EqPolynomial::<Fr>::evals_cached_msb_in(&point, None, budget).map(drop),
                4 => EqPolynomial::<Fr>::evals_cached_lsb_in(&point, None, budget).map(drop),
                5 => TensorEqTable::from_point_in(&point, Fr::from_u64(1), 1, budget).map(drop),
                _ => unreachable!(),
            };
            if extra == 0 {
                assert!(result.is_ok());
            } else {
                assert!(
                    matches!(result, Err(EqTableError::OverBudget { bytes, max_bytes, .. }) if bytes == elements * unit && bytes == max_bytes + 1)
                );
            }
        }
    }
    assert!(matches!(
        UNLIMITED.admit::<Fr>(usize::MAX, Location::caller()),
        Err(EqTableError::SizeOverflow { .. })
    ));
    assert_eq!(
        EqPolynomial::<Fr>::evals_prefix_in(&point, None, 0, Budget { max_bytes: 0 }).unwrap(),
        []
    );
    assert!(matches!(
        EqPolynomial::<Fr>::evals_in(&[] as &[Fr], None, Budget { max_bytes: 0 }),
        Err(EqTableError::OverBudget { .. })
    ));
}

#[test]
fn budget_preserves_allocator_error_causes() {
    FailAt::fail_next_allocation();
    let result = UNLIMITED.table::<Fr>(4, TableInit::Zero);
    assert!(matches!(
        result,
        Err(EqTableError::AllocationFailed {
            elements: 4,
            cause: AllocationCause::Zero(ZeroAllocError::AllocFailed { .. })
        })
    ));
    FailAt::fail_next_allocation();
    let result = UNLIMITED.table(4, TableInit::Repeat(Fr::from_u64(1)));
    assert!(matches!(
        result,
        Err(EqTableError::AllocationFailed {
            elements: 4,
            cause: AllocationCause::Reserve(_)
        })
    ));
    FailAt::fail_next_allocation();
    let result = UNLIMITED.storage::<Vec<Fr>>(4);
    assert!(matches!(
        result,
        Err(EqTableError::AllocationFailed {
            elements: 4,
            cause: AllocationCause::Reserve(_)
        })
    ));

    assert!(matches!(
        UNLIMITED.table::<Fr>(usize::MAX, TableInit::Zero),
        Err(EqTableError::AllocationFailed {
            elements: usize::MAX,
            cause: AllocationCause::Zero(ZeroAllocError::CapacityOverflow)
        })
    ));
    for result in [
        UNLIMITED.storage::<Fr>(usize::MAX),
        UNLIMITED.table(usize::MAX, TableInit::Repeat(Fr::from_u64(1))),
    ] {
        match result {
            Err(EqTableError::AllocationFailed {
                elements: usize::MAX,
                cause: AllocationCause::Reserve(cause),
            }) => assert_ne!(cause.to_string(), ""),
            _ => panic!("reservation cause was lost"),
        }
    }
}

#[test]
fn argument_errors_are_typed() {
    let point = [Fr::from_u64(2)];
    assert!(matches!(
        EqPolynomial::<Fr>::evals_prefix_in(&point, None, 3, UNLIMITED),
        Err(EqTableError::LengthMismatch {
            expected: 2,
            got: 3
        })
    ));
    assert!(matches!(
        EqPolynomial::<Fr>::prefix_sum_in(&point, 3, UNLIMITED),
        Err(EqTableError::LengthMismatch {
            expected: 2,
            got: 3
        })
    ));
    assert!(matches!(
        TensorEqTable::from_point_in(&point, Fr::from_u64(1), 2, UNLIMITED),
        Err(EqTableError::InvalidSplit {
            outer_vars: 2,
            num_vars: 1
        })
    ));
    let table = TensorEqTable::from_point_in(&point, Fr::from_u64(1), 0, UNLIMITED).unwrap();
    assert!(matches!(
        table.evaluate_index_in(2, UNLIMITED),
        Err(EqTableError::IndexOutOfRange { index: 2, len: 2 })
    ));
}

#[test]
fn admission_reports_the_external_caller() {
    #[track_caller]
    fn check_locations() {
        let expected = Location::caller();
        let point = [Fr::from_u64(2)];
        let budget = Budget { max_bytes: 0 };
        for error in [
            EqPolynomial::<Fr>::evals_in(&point, None, budget).unwrap_err(),
            EqPolynomial::<Fr>::evals_mapped_in(&point, None, |x| x, budget).unwrap_err(),
            EqPolynomial::<Fr>::evals_prefix_in(&point, None, 1, budget).unwrap_err(),
            EqPolynomial::<Fr>::evals_cached_msb_in(&point, None, budget).unwrap_err(),
            EqPolynomial::<Fr>::evals_cached_lsb_in(&point, None, budget).unwrap_err(),
            TensorEqTable::from_point_in(&point, Fr::from_u64(1), 0, budget).unwrap_err(),
        ] {
            let EqTableError::OverBudget { at, .. } = error else {
                panic!("wrong error")
            };
            assert_eq!(at, expected);
        }
    }
    check_locations();
}

#[test]
fn allocation_errors_chain_the_allocator_error_once() {
    let reserve = UNLIMITED.storage::<Fr>(usize::MAX).unwrap_err();
    let zero = UNLIMITED
        .table::<Fr>(usize::MAX, TableInit::Zero)
        .unwrap_err();
    for (error, allocator_error) in [
        (
            &reserve as &dyn Error,
            Vec::<Fr>::new()
                .try_reserve_exact(usize::MAX)
                .unwrap_err()
                .to_string(),
        ),
        (&zero, ZeroAllocError::CapacityOverflow.to_string()),
    ] {
        let source = error.source().unwrap();
        assert_eq!(source.to_string(), allocator_error);
        assert!(source.source().is_none());
        assert!(!error.to_string().contains(&allocator_error));
    }
}

#[test]
fn over_budget_states_the_request_and_the_budget() {
    let error =
        EqPolynomial::<Fr>::evals_in(&[Fr::from_u64(2); 20], None, Budget { max_bytes: 1 << 20 })
            .unwrap_err();
    let EqTableError::OverBudget { at, .. } = error else {
        panic!("expected OverBudget, got {error:?}")
    };
    assert_eq!(
        error.to_string(),
        format!(
            "eq table of 1048576 entries needs 33554432 bytes, over the 1048576-byte budget (requested at {at})"
        )
    );
}

/// Runs `build`, which must panic, and returns the panic's message and file.
fn global_panic(build: impl FnOnce() + UnwindSafe) -> (String, String) {
    let file = Arc::new(Mutex::new(None));
    let record = Arc::clone(&file);
    let test_thread = thread::current().id();
    let hook = take_hook();
    set_hook(Box::new(move |info| {
        if thread::current().id() == test_thread {
            *record.lock().unwrap() = info.location().map(|at| at.file().to_owned());
        }
    }));
    let payload = catch_unwind(build);
    set_hook(hook);
    let message = *payload.unwrap_err().downcast::<String>().unwrap();
    let file = file.lock().unwrap().take().unwrap();
    (message, file)
}

#[test]
fn global_panics_with_the_typed_error() {
    let empty: &[Fr] = &[];
    let cases: [(Box<dyn FnOnce() + UnwindSafe>, EqTableError); 6] = [
        (
            Box::new(|| {
                let _ = EqPolynomial::<Fr>::prefix_sum(empty, 2);
            }),
            EqTableError::LengthMismatch {
                expected: 1,
                got: 2,
            },
        ),
        (
            Box::new(|| {
                let _ = EqPolynomial::<Fr>::evals_prefix(empty, None, 2);
            }),
            EqTableError::LengthMismatch {
                expected: 1,
                got: 2,
            },
        ),
        (
            Box::new(|| {
                let _ = TensorEqTable::<Fr>::new_with_scaling_and_split(empty, Fr::from_u64(1), 1);
            }),
            EqTableError::InvalidSplit {
                outer_vars: 1,
                num_vars: 0,
            },
        ),
        (
            Box::new(|| {
                let table =
                    TensorEqTable::<Fr>::new_with_scaling_and_split(empty, Fr::from_u64(1), 0);
                let _ = table.evaluate_index(1);
            }),
            EqTableError::IndexOutOfRange { index: 1, len: 1 },
        ),
        (
            Box::new(|| {
                let _ = EqPolynomial::<Fr>::evals(&[Fr::from_u64(2); usize::BITS as usize], None);
            }),
            EqTableError::DimensionOverflow {
                num_vars: usize::BITS as usize,
            },
        ),
        (
            Box::new(|| {
                let _ =
                    EqPolynomial::<Fr>::evals(&[Fr::from_u64(2); usize::BITS as usize - 5], None);
            }),
            EqTableError::SizeOverflow {
                elements: 1 << (usize::BITS - 5),
                element_size: size_of::<Fr>(),
            },
        ),
    ];
    for (build, expected) in cases {
        // The panic names this file, not jolt-poly's internals.
        assert_eq!(
            global_panic(build),
            (expected.to_string(), file!().to_owned())
        );
    }
}
