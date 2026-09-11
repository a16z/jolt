#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;
use alloc::vec::Vec;

#[repr(align(8))]
struct Aligned([i32; 64]);

#[repr(C, align(8))]
struct WordAligned {
    pad: i32,
    data: [i32; 64],
}

#[jolt::provable(heap_size = 32768, stack_size = 32768, max_trace_length = 131072)]
fn ntt(input: Vec<i32>, psi: Vec<i32>, twiddles: Vec<i32>, p: i32, pinv: i32) -> u64 {
    let mut state = Aligned(input.as_slice().try_into().unwrap());
    let psi = WordAligned {
        pad: 0,
        data: psi.as_slice().try_into().unwrap(),
    };
    let twiddles = Aligned(twiddles.as_slice().try_into().unwrap());
    jolt_inlines_ntt::forward_ntt64(&mut state.0, &psi.data, &twiddles.0, p, pinv);
    state.0.iter().enumerate().fold(0, |sum, (i, a)| {
        sum + (i as u64 + 1) * i64::from(*a).rem_euclid(i64::from(p)) as u64
    })
}
