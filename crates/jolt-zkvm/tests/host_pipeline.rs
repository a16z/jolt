//! End-to-end integration tests with Dory commitment scheme.
//!
//! Exercises the `prove` → `verify` pipeline with real RISC-V guest programs
//! compiled via jolt-host and the real Dory polynomial commitment scheme.
//!
//! These are the closest tests to production: real ELF → tracer → witness →
//! Dory commit → Spartan → sumcheck → Dory openings → verify.

mod common;

use common::*;

#[test]
fn muldiv_basic() {
    let inputs = postcard::to_stdvec(&(9u32, 5u32, 3u32)).unwrap();
    prove_and_verify_guest("muldiv-guest", &inputs);
}

#[test]
fn muldiv_identity() {
    let inputs = postcard::to_stdvec(&(1u32, 1u32, 1u32)).unwrap();
    prove_and_verify_guest("muldiv-guest", &inputs);
}

#[test]
fn muldiv_zero_numerator() {
    let inputs = postcard::to_stdvec(&(0u32, 5u32, 1u32)).unwrap();
    prove_and_verify_guest("muldiv-guest", &inputs);
}

#[test]
fn muldiv_zero_factor() {
    let inputs = postcard::to_stdvec(&(7u32, 0u32, 1u32)).unwrap();
    prove_and_verify_guest("muldiv-guest", &inputs);
}

#[test]
fn muldiv_large_product() {
    let inputs = postcard::to_stdvec(&(10000u32, 20000u32, 7u32)).unwrap();
    prove_and_verify_guest("muldiv-guest", &inputs);
}

#[test]
fn muldiv_divide_by_one() {
    let inputs = postcard::to_stdvec(&(42u32, 13u32, 1u32)).unwrap();
    prove_and_verify_guest("muldiv-guest", &inputs);
}

#[test]
fn muldiv_same_values() {
    let inputs = postcard::to_stdvec(&(5u32, 5u32, 5u32)).unwrap();
    prove_and_verify_guest("muldiv-guest", &inputs);
}

#[test]
fn muldiv_large_divisor() {
    let inputs = postcard::to_stdvec(&(100u32, 3u32, 200u32)).unwrap();
    prove_and_verify_guest("muldiv-guest", &inputs);
}

#[test]
fn muldiv_powers_of_two() {
    let inputs = postcard::to_stdvec(&(256u32, 128u32, 64u32)).unwrap();
    prove_and_verify_guest("muldiv-guest", &inputs);
}

#[test]
fn muldiv_primes() {
    let inputs = postcard::to_stdvec(&(97u32, 89u32, 83u32)).unwrap();
    prove_and_verify_guest("muldiv-guest", &inputs);
}

#[test]
fn fibonacci_small() {
    let inputs = postcard::to_stdvec(&5u32).unwrap();
    prove_and_verify_guest("fibonacci-guest", &inputs);
}

#[test]
fn fibonacci_ten() {
    let inputs = postcard::to_stdvec(&10u32).unwrap();
    prove_and_verify_guest("fibonacci-guest", &inputs);
}

#[test]
fn fibonacci_one() {
    let inputs = postcard::to_stdvec(&1u32).unwrap();
    prove_and_verify_guest("fibonacci-guest", &inputs);
}

#[test]
fn fibonacci_two() {
    let inputs = postcard::to_stdvec(&2u32).unwrap();
    prove_and_verify_guest("fibonacci-guest", &inputs);
}

#[test]
fn fibonacci_fifteen() {
    let inputs = postcard::to_stdvec(&15u32).unwrap();
    prove_and_verify_guest("fibonacci-guest", &inputs);
}

#[test]
fn fibonacci_twenty() {
    let inputs = postcard::to_stdvec(&20u32).unwrap();
    prove_and_verify_guest("fibonacci-guest", &inputs);
}

// Guest has two #[jolt::provable] functions; must use set_func to select.

#[test]
fn collatz_one() {
    let inputs = postcard::to_stdvec(&1u128).unwrap();
    prove_and_verify_guest_func("collatz-guest", "collatz_convergence", &inputs);
}

#[test]
fn collatz_two() {
    let inputs = postcard::to_stdvec(&2u128).unwrap();
    prove_and_verify_guest_func("collatz-guest", "collatz_convergence", &inputs);
}

#[test]
fn collatz_seven() {
    let inputs = postcard::to_stdvec(&7u128).unwrap();
    prove_and_verify_guest_func("collatz-guest", "collatz_convergence", &inputs);
}

#[test]
fn collatz_ten() {
    let inputs = postcard::to_stdvec(&10u128).unwrap();
    prove_and_verify_guest_func("collatz-guest", "collatz_convergence", &inputs);
}

#[test]
fn collatz_power_of_two() {
    // Powers of 2 converge in exactly log2(n) steps (all halvings).
    let inputs = postcard::to_stdvec(&16u128).unwrap();
    prove_and_verify_guest_func("collatz-guest", "collatz_convergence", &inputs);
}

#[test]
fn collatz_twenty_seven() {
    // 27 is famous for taking 111 steps. Large trace.
    let inputs = postcard::to_stdvec(&27u128).unwrap();
    prove_and_verify_guest_func("collatz-guest", "collatz_convergence", &inputs);
}

#[test]
fn alloc_first_element() {
    let inputs = postcard::to_stdvec(&0u32).unwrap();
    prove_and_verify_guest("alloc-guest", &inputs);
}

#[test]
fn alloc_middle_element() {
    let inputs = postcard::to_stdvec(&50u32).unwrap();
    prove_and_verify_guest("alloc-guest", &inputs);
}

#[test]
fn alloc_last_element() {
    let inputs = postcard::to_stdvec(&99u32).unwrap();
    prove_and_verify_guest("alloc-guest", &inputs);
}

#[test]
fn alloc_small_index() {
    let inputs = postcard::to_stdvec(&3u32).unwrap();
    prove_and_verify_guest("alloc-guest", &inputs);
}

#[test]
fn alloc_near_end() {
    let inputs = postcard::to_stdvec(&97u32).unwrap();
    prove_and_verify_guest("alloc-guest", &inputs);
}

#[test]
fn multi_function_add() {
    let inputs = postcard::to_stdvec(&(7u32, 3u32)).unwrap();
    prove_and_verify_guest_func("multi-function-guest", "add", &inputs);
}

#[test]
fn multi_function_mul() {
    let inputs = postcard::to_stdvec(&(7u32, 3u32)).unwrap();
    prove_and_verify_guest_func("multi-function-guest", "mul", &inputs);
}

#[test]
fn multi_function_add_zeros() {
    let inputs = postcard::to_stdvec(&(0u32, 0u32)).unwrap();
    prove_and_verify_guest_func("multi-function-guest", "add", &inputs);
}

#[test]
fn multi_function_mul_by_one() {
    let inputs = postcard::to_stdvec(&(42u32, 1u32)).unwrap();
    prove_and_verify_guest_func("multi-function-guest", "mul", &inputs);
}

#[test]
fn multi_function_mul_by_zero() {
    let inputs = postcard::to_stdvec(&(42u32, 0u32)).unwrap();
    prove_and_verify_guest_func("multi-function-guest", "mul", &inputs);
}

#[test]
fn multi_function_add_large() {
    let inputs = postcard::to_stdvec(&(1_000_000_u32, 2_000_000_u32)).unwrap();
    prove_and_verify_guest_func("multi-function-guest", "add", &inputs);
}

#[test]
fn multi_function_mul_large() {
    let inputs = postcard::to_stdvec(&(1234u32, 5678u32)).unwrap();
    prove_and_verify_guest_func("multi-function-guest", "mul", &inputs);
}

#[test]
fn memory_ops_pipeline() {
    prove_and_verify_guest("memory-ops-guest", &[]);
}

// These use the existing pipeline with random claim reduction data
// (not the host layer) — validates witness generation on real traces.

#[test]
fn muldiv_synthetic_pipeline() {
    let inputs = postcard::to_stdvec(&(9u32, 5u32, 3u32)).unwrap();
    run_real_program_synthetic("muldiv-guest", &inputs, b"jolt-muldiv");
}

#[test]
fn fibonacci_synthetic_pipeline() {
    let inputs = postcard::to_stdvec(&10u32).unwrap();
    run_real_program_synthetic("fibonacci-guest", &inputs, b"jolt-fib");
}

#[test]
fn collatz_synthetic_pipeline() {
    let inputs = postcard::to_stdvec(&7u128).unwrap();
    run_real_program_synthetic_func(
        "collatz-guest",
        "collatz_convergence",
        &inputs,
        b"jolt-collatz",
    );
}

#[test]
fn alloc_synthetic_pipeline() {
    let inputs = postcard::to_stdvec(&42u32).unwrap();
    run_real_program_synthetic("alloc-guest", &inputs, b"jolt-alloc");
}
