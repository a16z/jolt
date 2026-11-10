#![expect(
    clippy::panic,
    reason = "test failures and the OOM child must fail loudly"
)]
#![expect(
    clippy::unwrap_used,
    reason = "test setup and expected successful results"
)]

mod support;

use jolt_field::{Field, Fr, Ring};
use jolt_poly::{
    BindingOrder, Endianness, EqBudget, EqPolynomial, EqTableError, GruenSplitEqPolynomial,
    PointRef, TensorEqTable, HIGH_TO_LOW, LOW_TO_HIGH,
};
use jolt_utils::alloc::{try_zeroed_vec, ZeroAllocError};
#[cfg(feature = "parallel")]
use rayon::ThreadPoolBuilder;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
#[cfg(unix)]
use std::process::Command;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use support::{NonZeroRepr, Zst};

struct FailingAllocator;

#[global_allocator]
static ALLOCATOR: FailingAllocator = FailingAllocator;
thread_local! {
    // try_* fills are serial; unrelated runtime threads must not enter the sweep.
    static ARMED: Cell<bool> = const { Cell::new(false) };
}
static SESSION: AtomicUsize = AtomicUsize::new(0);
static ATTEMPTS: AtomicUsize = AtomicUsize::new(0);
static FAIL_AT: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: layout extension preserves payload alignment, and allocation/deallocation
// use the same System layout. The header distinguishes builder and harness storage.
unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc supplies a valid layout.
        unsafe { Self::allocate(layout, false) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc supplies a valid layout.
        unsafe { Self::allocate(layout, true) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let (full, offset) = Layout::new::<usize>().extend(layout).unwrap();
        // SAFETY: allocate returned base + offset with the session at base.
        unsafe {
            let base = ptr.sub(offset);
            let session = base.cast::<usize>().read_unaligned();
            if session != 0 && session == SESSION.load(Ordering::SeqCst) {
                let _previous = LIVE.fetch_sub(1, Ordering::SeqCst);
            }
            System.dealloc(base, full);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_layout = Layout::from_size_align(new_size, layout.align()).unwrap();
        // SAFETY: the two allocations have matching layouts and disjoint storage.
        unsafe {
            let new_ptr = self.alloc(new_layout);
            if !new_ptr.is_null() {
                ptr::copy_nonoverlapping(ptr, new_ptr, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
            new_ptr
        }
    }
}

impl FailingAllocator {
    unsafe fn allocate(layout: Layout, zeroed: bool) -> *mut u8 {
        let armed = ARMED.with(Cell::get);
        let session = if armed {
            SESSION.load(Ordering::SeqCst)
        } else {
            0
        };
        if armed && ATTEMPTS.fetch_add(1, Ordering::SeqCst) + 1 == FAIL_AT.load(Ordering::SeqCst) {
            return ptr::null_mut();
        }
        let Ok((full, offset)) = Layout::new::<usize>().extend(layout) else {
            return ptr::null_mut();
        };
        // SAFETY: full is a valid nonzero layout; the header and payload do not overlap.
        unsafe {
            let base = if zeroed {
                System.alloc_zeroed(full)
            } else {
                System.alloc(full)
            };
            if base.is_null() {
                return base;
            }
            base.cast::<usize>().write_unaligned(session);
            if armed {
                let _previous = LIVE.fetch_add(1, Ordering::SeqCst);
                let _previous = BYTES.fetch_add(layout.size(), Ordering::SeqCst);
            }
            base.add(offset)
        }
    }
}

fn measure<T>(
    fail_at: usize,
    build: impl FnOnce() -> Result<T, EqTableError>,
) -> (bool, usize, usize, usize) {
    let _previous = SESSION.fetch_add(1, Ordering::SeqCst);
    ATTEMPTS.store(0, Ordering::SeqCst);
    LIVE.store(0, Ordering::SeqCst);
    BYTES.store(0, Ordering::SeqCst);
    FAIL_AT.store(fail_at, Ordering::SeqCst);
    ARMED.with(|armed| armed.set(true));
    let result = build();
    let expected = if fail_at == 0 {
        result.is_ok()
    } else {
        matches!(result, Err(EqTableError::AllocationFailed { .. }))
    };
    drop(result);
    ARMED.with(|armed| armed.set(false));
    (
        expected,
        ATTEMPTS.load(Ordering::SeqCst),
        LIVE.load(Ordering::SeqCst),
        BYTES.load(Ordering::SeqCst),
    )
}

fn sweep<T>(allocations: usize, bytes: usize, build: impl Fn() -> Result<T, EqTableError>) {
    let (ok, attempts, live, allocated) = measure(0, &build);
    assert!(ok);
    assert_eq!(attempts, allocations);
    assert_eq!(allocated, bytes);
    assert_eq!(live, 0);
    for k in 1..=allocations {
        let (ok, attempts, live, _) = measure(k, &build);
        assert!(ok, "allocation {k} did not return AllocationFailed");
        assert_eq!(attempts, k);
        assert_eq!(live, 0, "allocation {k} leaked an earlier buffer");
    }
}

fn builders<F: Field>() {
    let coords = [F::from_u64(2); 5];
    let budget = EqBudget {
        max_bytes: usize::MAX,
    };
    let bytes = size_of::<F>();
    for n in [0, 5] {
        let point = PointRef::low_to_high(&coords[..n]);
        let len = 1 << n;
        sweep(1, len * bytes, || {
            EqPolynomial::<F>::try_evals(&point, None, budget)
        });
        sweep(1, len * bytes, || {
            EqPolynomial::<F>::try_evals_mapped(&point, None, |x| x * x, budget)
        });
        let prefix_len = len.min(7);
        sweep(1, prefix_len * bytes, || {
            EqPolynomial::<F>::try_evals_prefix(&point, None, prefix_len, budget)
        });
        sweep(0, 0, || {
            EqPolynomial::<F>::try_evals_prefix(&point, None, 0, budget)
        });
        for msb in [true, false] {
            sweep(
                n + 2,
                (n + 1) * size_of::<Vec<F>>() + (2 * len - 1) * bytes,
                || {
                    if msb {
                        EqPolynomial::<F>::try_evals_cached_msb(&point, None, budget)
                    } else {
                        EqPolynomial::<F>::try_evals_cached_lsb(&point, None, budget)
                    }
                },
            );
        }
        let outer_vars = n / 2;
        let factor_len = (1 << outer_vars) + (1 << (n - outer_vars));
        sweep(2, factor_len * bytes, || {
            TensorEqTable::<F>::try_from_point(&point, F::from_u64(3), outer_vars, budget)
        });
        let tensor =
            TensorEqTable::<F>::try_from_point(&point, F::from_u64(3), outer_vars, budget).unwrap();
        sweep(0, 0, || tensor.try_evaluate_index(0));
        sweep(0, 0, || {
            EqPolynomial::<F>::try_prefix_sum(&point, prefix_len)
        });
    }
}

fn gruen_builders<F: Field, const E: Endianness>() {
    let coords = [F::from_u64(2); 6];
    // Frozen buffer counts include the owned point, both cache headers and every leaf.
    for (n, allocations, elements, headers) in [
        (0, 4, 2, 2),
        (1, 5, 3, 2),
        (2, 6, 6, 3),
        (5, 9, 19, 6),
        (6, 10, 28, 7),
    ] {
        let point = PointRef::<E, _>::new(&coords[..n]);
        let bytes = elements * size_of::<F>();
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            sweep(allocations, bytes + headers * size_of::<Vec<F>>(), || {
                GruenSplitEqPolynomial::<F>::try_from_point(
                    &point,
                    order,
                    Some(F::from_u64(7)),
                    EqBudget { max_bytes: bytes },
                )
            });
            assert!(GruenSplitEqPolynomial::<F>::try_from_point(
                &point,
                order,
                None,
                EqBudget {
                    max_bytes: bytes + 1
                }
            )
            .is_ok());
            sweep(0, 0, || {
                let error = GruenSplitEqPolynomial::<F>::try_from_point(
                    &point,
                    order,
                    None,
                    EqBudget {
                        max_bytes: bytes - 1,
                    },
                )
                .unwrap_err();
                assert!(matches!(
                    error,
                    EqTableError::OverBudget { bytes: actual, max_bytes, at }
                        if actual == bytes && max_bytes == bytes - 1
                            && at.file().ends_with("eq_allocations.rs")
                ));
                Ok(())
            });
        }
    }
    let long = [F::from_u64(2); usize::BITS as usize];
    for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
        sweep(0, 0, || {
            assert!(matches!(
                GruenSplitEqPolynomial::<F>::try_from_point(
                    &long,
                    order,
                    None,
                    EqBudget {
                        max_bytes: usize::MAX
                    }
                ),
                Err(EqTableError::DimensionOverflow { .. })
            ));
            Ok(())
        });
    }
}

#[test]
fn every_builder_allocation_is_fallible_and_partial_failures_release_storage() {
    if let Some(mode) = std::env::var_os("JOLT_EQ_ALLOC_ABORT") {
        FAIL_AT.store(1, Ordering::SeqCst);
        ARMED.with(|armed| armed.set(true));
        if mode == "marker" {
            drop(Fr::zeroed_vec(32));
        } else {
            drop(NonZeroRepr::zeroed_vec(32));
        }
        panic!("infallible allocator returned after injected OOM");
    }
    let cold = [Fr::from_u64(2); 17];
    let budget = EqBudget {
        max_bytes: usize::MAX,
    };
    sweep(1, (1 << 17) * size_of::<Fr>(), || {
        EqPolynomial::<Fr>::try_evals(&cold, None, budget)
    });
    sweep(1, (1 << 17) * size_of::<Fr>(), || {
        EqPolynomial::<Fr>::try_evals_mapped(&cold, None, |x| x * x, budget)
    });
    builders::<Fr>();
    builders::<NonZeroRepr>();
    gruen_builders::<Fr, HIGH_TO_LOW>();
    gruen_builders::<Fr, LOW_TO_HIGH>();
    gruen_builders::<NonZeroRepr, HIGH_TO_LOW>();
    gruen_builders::<NonZeroRepr, LOW_TO_HIGH>();
    #[cfg(feature = "parallel")]
    {
        let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        // Complete worker startup before arming the builder-only sweep.
        drop(pool.broadcast(|_| ()));
        pool.install(|| {
            let warm = [Fr::from_u64(2); 17];
            for n in [14, 15, 16, 17] {
                let point = &warm[..n];
                let budget = EqBudget {
                    max_bytes: usize::MAX,
                };
                sweep(1, (1 << n) * size_of::<Fr>(), || {
                    EqPolynomial::<Fr>::try_evals(point, None, budget)
                });
                sweep(1, (1 << n) * size_of::<Fr>(), || {
                    EqPolynomial::<Fr>::try_evals_mapped(point, None, |x| x * x, budget)
                });
            }
        });
    }
    for len in [0, 8, usize::MAX] {
        sweep(0, 0, || {
            Zst::try_zeroed_vec(len).map_err(|_| EqTableError::AllocationFailed { elements: len })
        });
        sweep(0, 0, || {
            try_zeroed_vec::<Zst>(len).map_err(|_| EqTableError::AllocationFailed { elements: len })
        });
        assert_eq!(Zst::try_zeroed_vec(len).unwrap().len(), len);
        assert_eq!(try_zeroed_vec::<Zst>(len).unwrap().len(), len);
    }
    assert!(matches!(
        Fr::try_zeroed_vec(usize::MAX),
        Err(ZeroAllocError::CapacityOverflow)
    ));
    assert!(matches!(
        NonZeroRepr::try_zeroed_vec(usize::MAX),
        Err(ZeroAllocError::CapacityOverflow)
    ));
    assert!(std::panic::catch_unwind(|| Fr::zeroed_vec(usize::MAX)).is_err());
    assert!(std::panic::catch_unwind(|| NonZeroRepr::zeroed_vec(usize::MAX)).is_err());
    #[cfg(unix)]
    for mode in ["marker", "safe"] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "every_builder_allocation_is_fallible_and_partial_failures_release_storage",
            ])
            .env("JOLT_EQ_ALLOC_ABORT", mode)
            .output()
            .unwrap();
        assert_eq!(output.status.signal(), Some(6));
        assert!(String::from_utf8_lossy(&output.stderr).contains("memory allocation of"));
    }
}
