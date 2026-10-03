use jolt::{end_cycle_tracking, start_cycle_tracking};

#[jolt::advice]
fn modinv_advice(a: u64, m: u64) -> jolt::UntrustedAdvice<(u64, u64)> {
    let inv = modinv_naive(a, m);
    let quo = if m == 0 {
        0
    } else {
        (a as u128 * inv as u128 / m as u128) as u64
    };
    (inv, quo)
}

#[jolt::provable]
fn modinv(a: u64, m: u64) -> u64 {
    let inv_advice = {
        start_cycle_tracking("modinv advice");

        let adv = modinv_advice(a, m);

        let (inv, quo) = *adv;

        // CRITICAL: Verify that the advice is correct!
        // This uses check_advice! to ensure that a * inv ≡ 1 (mod m)
        // and that inv < m
        let product = (a as u128) * (inv as u128) - (quo as u128) * (m as u128);
        jolt::check_advice!(product == 1u128 && inv < m);

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
fn modinv_naive(a: u64, m: u64) -> u64 {
    if m == 0 {
        return 0;
    }

    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }

    if old_r != 1 {
        return 0;
    }

    if old_s < 0 {
        (old_s + m as i128) as u64
    } else {
        old_s as u64
    }
}
