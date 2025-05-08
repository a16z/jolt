#![cfg_attr(feature = "guest", no_std)]
use jolt::{start_cycle_tracking, end_cycle_tracking};

#[jolt::provable]
fn fib(n: u32) -> u128 {
    let mut a: u128 = 0;
    let mut b: u128 = 1;
    let mut sum: u128;

    // ----- begin measured span ----------------------------------------
    start_cycle_tracking("fib_loop");

    for _ in 1..n {
        sum = a + b;
        a = b;
        b = sum;
    }

    end_cycle_tracking("fib_loop");
    // ----- end measured span ------------------------------------------

    b
}
