#![allow(clippy::uninlined_format_args)]
use ark_bn254::Fr;
use ark_std::test_rng;
use ark_std::UniformRand;
use dory::arithmetic::Field;
use rayon::prelude::*;

pub fn unsafe_allocate_zero_vec(size: usize) -> Vec<Fr> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    #[cfg(test)]
    {
        // Check for safety of 0 allocation
        unsafe {
            let value = &F::zero();
            let ptr = value as *const F as *const u8;
            let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
            assert!(bytes.iter().all(|&byte| byte == 0));
        }
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<Fr>;
    unsafe {
        let layout = std::alloc::Layout::array::<Fr>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut Fr;

        if ptr.is_null() {
            panic!("Zero vec allocation failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}

/// Computes eq(z, x) for all x in {0,1}^n serially.
/// More efficient for small `z.len()`.
fn evals_serial(z: &[Fr], scaling_factor: Option<Fr>) -> Vec<Fr> {
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
    let mut evals: Vec<Fr> = unsafe_allocate_zero_vec(final_size);
    //let mut evals = vec![Fr::zero(); final_size];
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
    let mut evals: Vec<Fr> = unsafe_allocate_zero_vec(final_size);
    //let mut evals = vec![Fr::zero(); final_size];
    let mut size = 1;
    evals[0] = scaling_factor.unwrap_or(Fr::one());

    for r in r.iter().rev() {
        let (evals_left, evals_right) = evals.split_at_mut(size);
        let (evals_right, _) = evals_right.split_at_mut(size);

        if size >= 1 << 16 {
            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * *r;
                    *x -= *y;
                });
        } else {
            for (x, y) in evals_left.iter_mut().zip(evals_right.iter_mut()) {
                *y = *x * *r;
                *x -= *y;
            }
        }

        size *= 2;
    }
    evals
}

fn main() {
    let mut rng = test_rng();
    let n = 15;
    let z: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    let now = std::time::Instant::now();
    let serial = evals_serial(&z, None);
    let serial_time = now.elapsed();
    println!("Serial time: {:.2?}", serial_time);

    let now = std::time::Instant::now();
    let parallel = evals_parallel(&z, None);
    let parallel_time = now.elapsed();
    println!("Old parallel time: {:.2?}", parallel_time);

    let now = std::time::Instant::now();
    let parallel4 = evals_parallel_dynamic(&z, None);
    let parallel4_time = now.elapsed();
    println!("Dynamic time: {:.2?}", parallel4_time);

    assert_eq!(serial, parallel);
    assert_eq!(parallel4, serial);
    println!("Success: Serial and parallel outputs match!");

    //println!("z = {:?}", z);
    //println!("Evaluations:");
    //for (i, val) in serial.iter().enumerate() {
    //    println!("x = {:0width$b} -> {}", i, val, width = n);
    //}
}
