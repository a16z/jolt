#![allow(clippy::uninlined_format_args)]
use ark_bn254::Fr;
use dory::arithmetic::Field;
use rayon::prelude::*;

pub fn unsafe_allocate_zero_vec(size: usize) -> Vec<Fr> {
    #[cfg(test)]
    {
        unsafe {
            let value = &Fr::zero();
            let ptr = value as *const Fr as *const u8;
            let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<Fr>());
            assert!(bytes.iter().all(|&byte| byte == 0));
        }
    }

    unsafe {
        let layout = std::alloc::Layout::array::<Fr>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut Fr;

        if ptr.is_null() {
            panic!("Zero vec allocation failed");
        }

        Vec::from_raw_parts(ptr, size, size)
    }
}

/// Computes eq(z, x) for all x in {0,1}^n serially.
pub fn evals_serial(z: &[Fr], scaling_factor: Option<Fr>) -> Vec<Fr> {
    let final_size = 1 << z.len();
    let mut evals = vec![scaling_factor.unwrap_or(Fr::one()); final_size];
    let mut size = 1;
    for &zi in z {
        size *= 2;
        for i in (0..size).rev().step_by(2) {
            let scalar = evals[i / 2];
            evals[i] = scalar * zi;
            evals[i - 1] = scalar - evals[i];
        }
    }

    evals
}

pub fn evals_parallel(r: &[Fr], scaling_factor: Option<Fr>) -> Vec<Fr> {
    let final_size = 1 << r.len();
    let mut evals = unsafe_allocate_zero_vec(final_size);
    let mut size = 1;
    evals[0] = scaling_factor.unwrap_or(Fr::one());

    for r in r.iter().rev() {
        let (evals_left, evals_right) = evals.split_at_mut(size);
        let (evals_right, _) = evals_right.split_at_mut(size);

        evals_left
            .par_iter_mut()
            .zip(evals_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * *r;
                *x -= *y;
            });

        size *= 2;
    }

    evals
}

pub fn evals_parallel_dynamic(r: &[Fr], scaling_factor: Option<Fr>) -> Vec<Fr> {
    let final_size = 1 << r.len();
    let mut evals = vec![scaling_factor.unwrap_or(Fr::one()); final_size];
    let mut size = 1;
    evals[0] = scaling_factor.unwrap_or(Fr::one());

    for r in r.iter().rev() {
        let (evals_left, evals_right) = evals.split_at_mut(size);
        let (evals_right, _) = evals_right.split_at_mut(size);
        evals_left
            .par_iter_mut()
            .zip(evals_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * *r;
                *x -= *y;
            });
        size *= 2;
    }

    evals
}
