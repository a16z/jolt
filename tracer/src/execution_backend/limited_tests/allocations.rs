use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;

#[derive(Clone)]
pub(super) struct Requests {
    pub live: i128,
    events: [(usize, usize); 512],
    len: usize,
    enabled: bool,
}

thread_local! {
    static REQUESTS: RefCell<Requests> = const { RefCell::new(Requests {
        live: 0, events: [(0, 0); 512], len: 0, enabled: false,
    }) };
}

impl Requests {
    pub fn start() {
        REQUESTS.with(|requests| {
            let mut requests = requests.borrow_mut();
            requests.live = 0;
            requests.len = 0;
            requests.enabled = true;
        });
    }

    pub fn stop() -> Self {
        REQUESTS.with(|requests| {
            let mut requests = requests.borrow_mut();
            requests.enabled = false;
            assert!(requests.len <= requests.events.len());
            requests.clone()
        })
    }

    pub fn bytes(&self, ptr: *const u8) -> usize {
        self.events[..self.len]
            .iter()
            .rev()
            .find(|(address, _)| *address == ptr as usize)
            .unwrap()
            .1
    }

    pub fn count_requests(&self, bytes: usize) -> usize {
        self.events[..self.len]
            .iter()
            .filter(|(_, size)| *size == bytes)
            .count()
    }
}

fn record(ptr: *mut u8, bytes: usize, freed: usize) {
    let _ = REQUESTS.try_with(|requests| {
        let Ok(mut requests) = requests.try_borrow_mut() else {
            return;
        };
        if requests.enabled {
            requests.live = requests
                .live
                .saturating_add(bytes as i128)
                .saturating_sub(freed as i128);
            if bytes != 0 {
                let index = requests.len;
                if index < requests.events.len() {
                    requests.events[index] = (ptr as usize, bytes);
                }
                requests.len += 1;
            }
        }
    });
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: every request is forwarded unchanged to System; recording only uses
// fixed-size thread-local storage and does not allocate.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies the valid allocation layout.
        let ptr = unsafe { System.alloc(layout) };
        record(ptr, layout.size(), 0);
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies the valid allocation layout.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        record(ptr, layout.size(), 0);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record(ptr, 0, layout.size());
        // SAFETY: the caller supplies System's live allocation and layout.
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller supplies System's live allocation and valid size.
        let next = unsafe { System.realloc(ptr, layout, size) };
        record(next, size, layout.size());
        next
    }
}
