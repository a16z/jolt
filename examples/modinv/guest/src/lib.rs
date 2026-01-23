#![cfg_attr(feature = "guest", no_std)]

/// Computes the modular multiplicative inverse of `a` modulo `m` using the Extended Euclidean Algorithm.
/// Returns `Some(x)` where `a * x ≡ 1 (mod m)`, or `None` if the inverse doesn't exist.
///
/// This function is marked with `#[jolt::advice]`, which means:
/// - On the first emulation pass (with `compute_advice` feature), it executes the body and writes the result to the advice tape
/// - On the second pass (without `compute_advice`), it reads the precomputed result from the advice tape
///
/// The advice function wraps the result in `UntrustedAdvice<T>`, signaling that the value must be verified
/// before use.
#[jolt::advice]
fn compute_modinv(a: u64, m: u64) -> jolt::UntrustedAdvice<Option<u64>> {
    if m == 0 {
        return jolt::UntrustedAdvice::new(None);
    }

    // Extended Euclidean Algorithm
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }

    // old_r is the GCD
    if old_r != 1 {
        // No inverse exists
        return jolt::UntrustedAdvice::new(None);
    }

    // Ensure the result is positive
    let result = if old_s < 0 {
        (old_s + m as i128) as u64
    } else {
        old_s as u64
    };

    jolt::UntrustedAdvice::new(Some(result))
}

/// Simple modular inverse example demonstrating runtime advice.
///
/// This example shows how to use the advice system to provide non-deterministic
/// witness data (the modular inverse) which is then verified in the guest.
#[jolt::provable(memory_size = 32768, max_trace_length = 65536)]
fn modinv(a: u64, m: u64) -> u64 {
    use core::ops::Deref;

    // Get the modular inverse from the advice tape (precomputed on first pass)
    let inv_advice = compute_modinv(a, m);

    // Extract the value from the UntrustedAdvice wrapper using Deref
    let inv_option = inv_advice.deref();

    // For this example, we assume the inverse exists
    let inv = inv_option.unwrap_or(0);

    // CRITICAL: Verify that the advice is correct!
    // This checks that a * inv ≡ 1 (mod m)
    // Using u128 to avoid overflow during multiplication
    let product = ((a as u128) * (inv as u128)) % (m as u128);

    // Use check_advice! to ensure the multiplication produces 1 mod m
    jolt::check_advice!(product == 1);

    inv
}
