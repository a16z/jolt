#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable]
fn collatz_convergence_range(start: u128, end: u128) -> u128 {
    let mut max_num_steps = 0;
    for n in start..end {
        let num_steps = collatz_convergence(n);
        if num_steps > max_num_steps {
            max_num_steps = num_steps;
        }
    }
    max_num_steps
}

#[jolt::provable]
fn collatz_convergence(n: u128) -> u128 {
    let mut n = n;
    let mut num_steps = 0;
    while n != 1 {
        if n % 2 == 0 {
            n /= 2;
        } else {
            n += (n << 1) + 1;
        }
        num_steps += 1;
    }
    return num_steps;
}
