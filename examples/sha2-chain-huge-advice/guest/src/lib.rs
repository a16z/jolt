#![cfg_attr(feature = "guest", no_std)]
use jolt::{end_cycle_tracking, start_cycle_tracking};

#[jolt::provable(
    heap_size = 32768,
    max_trace_length = 4194304,
    // Keep advice capacity very large, but below the threshold that would place
    // the untrusted-advice base at address 0 in the guest memory layout.
    max_untrusted_advice_size = 16777216,
    backtrace = "off"
)]
fn fib_huge_advice(n: u32, huge_advice: jolt::UntrustedAdvice<&[u8]>) -> u128 {
    let advice = *huge_advice;
    jolt::check_advice!(advice.len() >= 2, "advice must contain at least 2 bytes");
    let last_idx = advice.len() - 1;
    let sampled_idx = (n as usize) % advice.len();

    jolt::check_advice_eq!(advice[0] as u64, 7u64, "unexpected first advice byte");
    jolt::check_advice_eq!(advice[last_idx] as u64, 7u64, "unexpected last advice byte");
    jolt::check_advice_eq!(
        advice[sampled_idx] as u64,
        7u64,
        "unexpected sampled advice byte"
    );

    let mut a: u128 = 0;
    let mut b: u128 = 1;
    let mut sum: u128;

    start_cycle_tracking("fib_loop_huge_advice");
    for _ in 1..n {
        sum = a + b;
        a = b;
        b = sum;
    }
    end_cycle_tracking("fib_loop_huge_advice");

    b
}
