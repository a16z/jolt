use super::alloc::{sealed::Sealed, Budget, EqAlloc, EqTableError, TableInit};
use jolt_field::Field;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::panic::Location;
use std::sync::atomic::{AtomicUsize, Ordering};

// Only buffers returned by FailAt are registered; Rayon and test-runner allocations
// are excluded. The failure test runs all registered allocations sequentially.
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

#[derive(Clone, Copy)]
pub(super) struct FailAt<'a> {
    pub calls: &'a AtomicUsize,
    pub target: usize,
    pub budget: Budget,
}
impl Sealed for FailAt<'_> {}
impl FailAt<'_> {
    pub fn fail_next_allocation() {
        FAIL_NEXT.with(|flag| flag.set(true));
    }
    fn before(self) -> Result<(), EqTableError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.target == 0 || call == self.target {
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
impl EqAlloc for FailAt<'_> {
    type Error = EqTableError;
    fn reject(self, error: EqTableError) -> EqTableError {
        error
    }
    fn admit<F: Field>(
        self,
        elements: usize,
        at: &'static Location<'static>,
    ) -> Result<(), EqTableError> {
        self.budget.admit::<F>(elements, at)
    }
    fn table<F: Field>(self, len: usize, init: TableInit<F>) -> Result<Vec<F>, EqTableError> {
        self.before()?;
        self.budget.table(len, init).map(Self::track)
    }
    fn storage<T>(self, len: usize) -> Result<Vec<T>, EqTableError> {
        self.before()?;
        self.budget.storage(len).map(Self::track)
    }
}
