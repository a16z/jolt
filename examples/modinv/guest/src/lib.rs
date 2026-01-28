use jolt::{end_cycle_tracking, start_cycle_tracking};

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
fn modinv_advice(a: u64, m: u64) -> jolt::UntrustedAdvice<u64> {
    modinv_naive(a, m)
}

/// Simple modular inverse example demonstrating runtime advice.
///
/// This example shows how to use the advice system to provide non-deterministic
/// witness data (the modular inverse) which is then verified in the guest.
#[jolt::provable]
fn modinv(a: u64, m: u64) -> u64 {
    let inv_advice = {
        start_cycle_tracking("modinv advice");
        use core::ops::Deref;

        // Get the modular inverse from the advice tape (precomputed on first pass)
        let inv_advice = modinv_advice(a, m);

        // Extract the value from the UntrustedAdvice wrapper using Deref
        let inv = *inv_advice.deref();

        // CRITICAL: Verify that the advice is correct!
        // This checks that a * inv ≡ 1 (mod m)
        // Using u128 to avoid overflow during multiplication
        let product = ((a as u128) * (inv as u128)) % (m as u128);

        // Use check_advice! to ensure the multiplication produces 1 mod m
        jolt::check_advice!(product == 1);

        end_cycle_tracking("modinv advice");
        inv
    };

    let inv_naive = {
        start_cycle_tracking("modinv naive");
        let inv = modinv_naive(a, m);
        end_cycle_tracking("modinv naive");
        inv
    };

    assert_eq!(inv_advice, inv_naive);

    inv_advice
}

/// Naive modular inverse implementation that computes directly without runtime advice.
///
/// This version performs the Extended Euclidean Algorithm entirely within the guest,
/// without leveraging the advice system. This allows us to compare the cycle counts
/// to demonstrate the efficiency gains from using runtime advice.
#[jolt::provable]
fn modinv_naive(a: u64, m: u64) -> u64 {
    if m == 0 {
        return 0;
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
        return 0;
    }

    // Ensure the result is positive
    if old_s < 0 {
        (old_s + m as i128) as u64
    } else {
        old_s as u64
    }
}
