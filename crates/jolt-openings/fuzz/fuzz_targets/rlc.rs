#![no_main]
use jolt_field::{Field, Fr};
use jolt_openings::{rlc_combine, rlc_combine_scalars};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Need at least 32 bytes for rho + 32 bytes for one evaluation
    if data.len() < 64 {
        return;
    }

    let rho = <Fr as Field>::from_bytes(&data[..32]);
    let remaining = &data[32..];

    // Build 1-4 polynomials of length 1-8 from remaining bytes
    let num_evals = remaining.len() / 32;
    if num_evals == 0 {
        return;
    }

    let evals: Vec<Fr> = (0..num_evals)
        .map(|i| {
            let start = i * 32;
            let end = (start + 32).min(remaining.len());
            if end - start < 32 {
                Fr::from_u64(remaining[start] as u64)
            } else {
                <Fr as Field>::from_bytes(&remaining[start..end])
            }
        })
        .collect();

    // rlc_combine_scalars must not panic
    let scalar_result = rlc_combine_scalars(&evals, rho);

    // rlc_combine with single polynomial must equal the polynomial itself
    let single_result = rlc_combine(&[&evals], rho);
    assert_eq!(single_result, evals, "single-poly RLC must be identity");

    // rlc_combine_scalars of [v] must equal v regardless of rho
    if evals.len() == 1 {
        assert_eq!(scalar_result, evals[0]);
    }

    // Split evals into two halves and verify consistency:
    // rlc_combine([a, b], rho) pointwise must equal a[i] + rho * b[i]
    if evals.len() >= 2 && evals.len() % 2 == 0 {
        let half = evals.len() / 2;
        let a = &evals[..half];
        let b = &evals[half..];

        let combined = rlc_combine(&[a, b], rho);
        for i in 0..half {
            assert_eq!(combined[i], a[i] + rho * b[i]);
        }

        // Scalar consistency: rlc_combine_scalars([va, vb], rho) == va + rho * vb
        let va = rlc_combine_scalars(&[a[0]], Fr::from_u64(0));
        let vb = rlc_combine_scalars(&[b[0]], Fr::from_u64(0));
        let combined_scalar = rlc_combine_scalars(&[va, vb], rho);
        assert_eq!(combined_scalar, va + rho * vb);
    }
});
