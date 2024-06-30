use rayon::prelude::*;
use std::thread::{self, JoinHandle};

use crate::field::JoltField;

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

#[tracing::instrument(skip_all)]
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

#[tracing::instrument(skip_all)]
pub fn unsafe_allocate_sparse_zero_vec<F: JoltField + Sized>(size: usize) -> Vec<(F, usize)> {
    // Check for safety of 0 allocation
    unsafe {
        let value = &F::zero();
        let ptr = value as *const F as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<(F, usize)>;
    unsafe {
        let layout = std::alloc::Layout::array::<(F, usize)>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut (F, usize);

        if ptr.is_null() {
            panic!("Zero vec allocation failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}

#[tracing::instrument(skip_all)]
pub fn par_flatten_triple<T: Send + Sync + Copy, F: Fn(usize) -> Vec<T>>(
    triple: Vec<(Vec<T>, Vec<T>, Vec<T>)>,
    allocate: F,
    excess_alloc: usize,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let az_len: usize = triple.iter().map(|item| item.0.len()).sum();
    let bz_len: usize = triple.iter().map(|item| item.1.len()).sum();
    let cz_len: usize = triple.iter().map(|item| item.2.len()).sum();

    let (mut a_sparse, mut b_sparse, mut c_sparse): (Vec<T>, Vec<T>, Vec<T>) =
        (allocate(az_len), allocate(bz_len), allocate(cz_len));

    let mut a_slices = Vec::with_capacity(triple.len() + excess_alloc);
    let mut b_slices = Vec::with_capacity(triple.len() + excess_alloc);
    let mut c_slices = Vec::with_capacity(triple.len() + excess_alloc);

    let mut a_rest: &mut [T] = a_sparse.as_mut_slice();
    let mut b_rest: &mut [T] = b_sparse.as_mut_slice();
    let mut c_rest: &mut [T] = c_sparse.as_mut_slice();

    for item in &triple {
        let (a_chunk, a_new_rest) = a_rest.split_at_mut(item.0.len());
        a_slices.push(a_chunk);
        a_rest = a_new_rest;

        let (b_chunk, b_new_rest) = b_rest.split_at_mut(item.1.len());
        b_slices.push(b_chunk);
        b_rest = b_new_rest;

        let (c_chunk, c_new_rest) = c_rest.split_at_mut(item.2.len());
        c_slices.push(c_chunk);
        c_rest = c_new_rest;
    }

    triple
        .into_par_iter()
        .zip(
            a_slices
                .par_iter_mut()
                .zip(b_slices.par_iter_mut().zip(c_slices.par_iter_mut())),
        )
        .for_each(|(chunk, (a, (b, c)))| {
            join_triple(
                || a.copy_from_slice(&chunk.0),
                || b.copy_from_slice(&chunk.1),
                || c.copy_from_slice(&chunk.2),
            );
        });

    (a_sparse, b_sparse, c_sparse)
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
