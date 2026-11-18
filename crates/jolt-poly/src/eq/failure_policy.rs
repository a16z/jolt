use super::alloc::{Allocate, Budget, EqAlloc, EqTableError, TableInit};
use jolt_field::Field;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::num::NonZeroUsize;
use std::panic::Location;
use std::sync::atomic::{AtomicUsize, Ordering};

// Only buffers returned by FailingPolicy are registered; Rayon and test-runner
// allocations are excluded. Registration and the call counter are atomic, but the
// tensor builder allocates its two factors, and the Gruen builder its two cached
// tables, inside `rayon::join`. The order of each pair is unspecified, so a
// `FailAt::Nth` aimed at one allocation of a pair may fail the other.
static LIVE: [AtomicUsize; 16] = [const { AtomicUsize::new(0) }; 16];
thread_local! {
    static FAIL_NEXT: Cell<bool> = const { Cell::new(false) };
}
struct TrackingAllocator;
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

// SAFETY: successful allocations and all deallocations use System; injected
// allocation failures return null, as permitted by GlobalAlloc.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if FAIL_NEXT
            .try_with(|flag| flag.replace(false))
            .unwrap_or(false)
        {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller supplies the GlobalAlloc layout contract.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if FAIL_NEXT
            .try_with(|flag| flag.replace(false))
            .unwrap_or(false)
        {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller supplies the GlobalAlloc layout contract.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        for slot in &LIVE {
            if slot
                .compare_exchange(ptr as usize, 0, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                break;
            }
        }
        // SAFETY: the pointer and layout are the original System allocation.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Which registered allocations fail.
#[derive(Clone, Copy)]
pub(super) enum FailAt {
    Every,
    /// The `n`-th allocation, counting from one.
    Nth(NonZeroUsize),
}

#[derive(Clone, Copy)]
pub(super) struct FailingPolicy<'a> {
    pub calls: &'a AtomicUsize,
    pub fail: FailAt,
    pub budget: Budget,
}
impl FailingPolicy<'_> {
    pub fn fail_next_allocation() {
        FAIL_NEXT.with(|flag| flag.set(true));
    }
    fn before(self) -> Result<(), EqTableError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let fails = match self.fail {
            FailAt::Every => true,
            FailAt::Nth(n) => call == n.get(),
        };
        if fails {
            // Capacity overflow deterministically produces the allocator's real error
            // without asking the OS for memory; it is the injected failure cause.
            let _ = self.budget.storage::<u8>(usize::MAX)?;
        }
        Ok(())
    }
    fn track<T>(values: Vec<T>) -> Vec<T> {
        if values.capacity() > 0 && size_of::<T>() > 0 {
            let address = values.as_ptr() as usize;
            assert!(LIVE.iter().any(|slot| slot
                .compare_exchange(0, address, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()));
        }
        values
    }
    pub fn assert_released() {
        assert!(LIVE.iter().all(|slot| slot.load(Ordering::SeqCst) == 0));
    }
}
impl EqAlloc for FailingPolicy<'_> {
    type Error = EqTableError;
}
impl Allocate for FailingPolicy<'_> {
    fn reject(self, error: EqTableError) -> <Self as EqAlloc>::Error {
        error
    }
    fn admit<F: Field>(
        self,
        elements: usize,
        at: &'static Location<'static>,
    ) -> Result<(), <Self as EqAlloc>::Error> {
        self.budget.admit::<F>(elements, at)
    }
    fn table<F: Field>(
        self,
        len: usize,
        init: TableInit<F>,
    ) -> Result<Vec<F>, <Self as EqAlloc>::Error> {
        self.before()?;
        self.budget.table(len, init).map(Self::track)
    }
    fn storage<T>(self, len: usize) -> Result<Vec<T>, <Self as EqAlloc>::Error> {
        self.before()?;
        self.budget.storage(len).map(Self::track)
    }
}
