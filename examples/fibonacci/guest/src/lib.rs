#![cfg_attr(feature = "guest", no_std)]
use jolt::{end_cycle_tracking, start_cycle_tracking};


// come back to this at the end
#[jolt::provable]
fn fib(n: u32) -> u128 {
    let mut a: u128 = 0;
    start_cycle_tracking("empty");
    let mut b: u128 = 1;
    end_cycle_tracking("empty");
    let mut sum: u128;

    start_cycle_tracking("fib_loop");

    for _ in 1..n {
        sum = a + b;
        // ----- end measured span ------------
        a = b;
        b = sum;
    }
    end_cycle_tracking("fib_loop");

    // end_cycle_tracking("fib_loop");
    // // ----- end measured span ------------------------------------------

    b
}
