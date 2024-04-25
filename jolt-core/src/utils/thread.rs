use std::thread::{self, JoinHandle};

use crate::poly::field::JoltField;

pub fn drop_in_background_thread<T>(data: T)
where
    T: Send + 'static,
{
    // h/t https://abrams.cc/rust-dropping-things-in-another-thread
    rayon::spawn(move || drop(data));
}

pub fn allocate_vec_in_background<T: Clone + Send + 'static>(
    value: T,
    size: usize,
) -> JoinHandle<Vec<T>> {
    thread::spawn(move || vec![value; size])
}

#[tracing::instrument(skip_all, name = "unsafe_allocate_zero_vec")]
pub fn unsafe_allocate_zero_vec<F: JoltField + Sized>(size: usize) -> Vec<F> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    // Check for safety of 0 allocation
    unsafe {
        let value = &F::zero();
        let ptr = value as *const F as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<F>;
    unsafe {
        let layout = std::alloc::Layout::array::<F>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut F;

        if ptr.is_null() {
            panic!("Zero vec allocaiton failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}

pub fn join_triple<A, B, C, RA, RB, RC>(oper_a: A, oper_b: B, oper_c: C) -> (RA, RB, RC)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    C: FnOnce() -> RC + Send,
    RA: Send,
    RB: Send,
    RC: Send,
{
    let (res_a, (res_b, res_c)) = rayon::join(oper_a, || rayon::join(oper_b, oper_c));
    (res_a, res_b, res_c)
}
