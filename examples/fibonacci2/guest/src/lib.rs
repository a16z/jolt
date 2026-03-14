#![cfg_attr(feature = "guest", no_std)]
use jolt::{end_cycle_tracking, start_cycle_tracking};

#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn fib(n: u32) -> u128 {
    let mut a: u128 = 0;
    let mut b: u128 = 1;
    let mut sum: u128;

    start_cycle_tracking("fib_loop");
    for _ in 1..n {
        sum = a + b;
        a = b;
        b = sum;
    }
    end_cycle_tracking("fib_loop");
    b
}

#[cfg(any(feature = "guest", feature = "zk"))]
#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn fib_with_private_input(n: u32, private_bump: jolt::PrivateInput<u32>) -> u128 {
    let adjusted_n = n + (*private_bump % 3);

    let mut a: u128 = 0;
    let mut b: u128 = 1;
    let mut sum: u128;

    start_cycle_tracking("fib_loop_private");
    for _ in 1..adjusted_n {
        sum = a + b;
        a = b;
        b = sum;
    }
    end_cycle_tracking("fib_loop_private");
    b
}

#[jolt::provable(
    heap_size = 32768,
    max_trace_length = 65536,
    max_untrusted_advice_size = 131072
)]
fn fib_with_large_advice_input(n: u32, advice: jolt::UntrustedAdvice<&[u8]>) -> u128 {
    let advice = *advice;
    jolt::check_advice!(advice.len() >= 2, "advice must contain at least 2 entries");

    let last_idx = advice.len() - 1;
    jolt::check_advice_eq!(
        advice[last_idx] as u64,
        7u64,
        "expected fixed marker in last advice byte"
    );

    let sampled_idx = (n as usize) % advice.len();
    jolt::check_advice_eq!(
        advice[sampled_idx] as u64,
        7u64,
        "expected fixed marker in sampled advice byte"
    );

    let mut a: u128 = 0;
    let mut b: u128 = 1;
    let mut sum: u128;

    start_cycle_tracking("fib_loop_large_advice");
    for _ in 1..n {
        sum = a + b;
        a = b;
        b = sum;
    }
    end_cycle_tracking("fib_loop_large_advice");

    b
}
