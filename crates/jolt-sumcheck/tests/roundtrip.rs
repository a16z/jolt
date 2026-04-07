//! Integration tests: full prover-verifier roundtrips with product compositions.
//!
//! These tests construct honest sumcheck proofs for polynomials of varying
//! degree and verify them, exercising the complete protocol flow including
//! transcript synchronization.

use jolt_field::{Field, Fr};
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::proof::SumcheckProof;
use jolt_sumcheck::round::ClearRoundVerifier;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckVerifier};
use jolt_transcript::{AppendToTranscript, LabelWithCount, MockTranscript, Transcript};

type F = Fr;

/// Prove a sumcheck for the product of `polys` multilinear polynomials.
///
/// Given d multilinear polynomials over n variables, proves the claim
/// `sum_{x in {0,1}^n} prod_j f_j(x) = C`. The round polynomial in
/// round i is degree d, requiring d+1 evaluation points.
///
/// Returns (proof, claimed_sum).
fn prove_product(
    polys: &[Vec<F>],
    num_vars: usize,
    transcript: &mut MockTranscript<F>,
) -> (SumcheckProof<F>, F) {
    let degree = polys.len();
    let n = 1 << num_vars;
    assert!(polys.iter().all(|p| p.len() == n));

    // Compute claimed sum
    let claimed_sum: F = (0..n)
        .map(|i| polys.iter().map(|p| p[i]).product::<F>())
        .sum();

    let mut bufs: Vec<Vec<F>> = polys.to_vec();
    let mut round_polys = Vec::with_capacity(num_vars);

    for _round in 0..num_vars {
        let half = bufs[0].len() / 2;

        // Evaluate the round polynomial at points 0, 1, ..., degree.
        // At point t, for each pair (lo, hi):
        //   f_j(t) = lo + t*(hi - lo)
        // The round poly value is: sum_i prod_j f_j(t)
        let evals: Vec<F> = (0..=degree)
            .map(|t| {
                let ft = F::from_u64(t as u64);
                let mut sum = F::from_u64(0);
                for i in 0..half {
                    let mut prod = F::from_u64(1);
                    for buf in &bufs {
                        let lo = buf[i];
                        let hi = buf[i + half];
                        prod *= lo + ft * (hi - lo);
                    }
                    sum += prod;
                }
                sum
            })
            .collect();

        // Interpolate to get the degree-d round polynomial in coefficient form
        let points: Vec<(F, F)> = evals
            .iter()
            .enumerate()
            .map(|(i, &v)| (F::from_u64(i as u64), v))
            .collect();
        let round_poly = UnivariatePoly::interpolate(&points);

        // Absorb coefficients (matching ClearRoundVerifier::new())
        for &coeff in round_poly.coefficients() {
            coeff.append_to_transcript(transcript);
        }

        let r: F = transcript.challenge();
        round_polys.push(round_poly);

        // Bind all polynomials (HighToLow)
        for buf in &mut bufs {
            for i in 0..half {
                buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
            }
            buf.truncate(half);
        }
    }

    (
        SumcheckProof {
            round_polynomials: round_polys,
        },
        claimed_sum,
    )
}

#[test]
fn degree2_product_roundtrip() {
    // f(x) * g(x) where f, g are multilinear over 4 variables
    let num_vars = 4;
    let n = 1 << num_vars;

    let f: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 + 1)).collect();
    let g: Vec<F> = (0..n).map(|i| F::from_u64((i * 3 + 7) as u64)).collect();

    let mut pt = MockTranscript::<F>::default();
    let (proof, claimed_sum) = prove_product(&[f, g], num_vars, &mut pt);

    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };

    let mut vt = MockTranscript::<F>::default();
    let clear = ClearRoundVerifier::new();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt, &clear);
    assert!(result.is_ok(), "degree-2 verify failed: {:?}", result.err());
}

#[test]
fn degree3_product_roundtrip() {
    // f(x) * g(x) * h(x), degree 3, 3 variables
    let num_vars = 3;
    let n = 1 << num_vars;

    let f: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 + 1)).collect();
    let g: Vec<F> = (0..n).map(|i| F::from_u64((i * 2 + 3) as u64)).collect();
    let h: Vec<F> = (0..n).map(|i| F::from_u64((i + 10) as u64)).collect();

    let mut pt = MockTranscript::<F>::default();
    let (proof, claimed_sum) = prove_product(&[f, g, h], num_vars, &mut pt);

    let claim = SumcheckClaim {
        num_vars,
        degree: 3,
        claimed_sum,
    };

    let mut vt = MockTranscript::<F>::default();
    let clear = ClearRoundVerifier::new();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt, &clear);
    assert!(result.is_ok(), "degree-3 verify failed: {:?}", result.err());
}

#[test]
fn degree3_final_eval_correct() {
    // Verify that the final eval matches the product of individual evals at the point
    let num_vars = 3;
    let n = 1 << num_vars;

    let f_evals: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 + 1)).collect();
    let g_evals: Vec<F> = (0..n).map(|i| F::from_u64((i * 5 + 2) as u64)).collect();
    let h_evals: Vec<F> = (0..n).map(|i| F::from_u64((i + 7) as u64)).collect();

    let mut pt = MockTranscript::<F>::default();
    let (proof, claimed_sum) = prove_product(
        &[f_evals.clone(), g_evals.clone(), h_evals.clone()],
        num_vars,
        &mut pt,
    );

    let claim = SumcheckClaim {
        num_vars,
        degree: 3,
        claimed_sum,
    };

    let mut vt = MockTranscript::<F>::default();
    let clear = ClearRoundVerifier::new();
    let (final_eval, challenges) =
        SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt, &clear).unwrap();

    // f(r) * g(r) * h(r) should equal the final_eval
    let f_at_r = Polynomial::new(f_evals).evaluate_and_consume(&challenges);
    let g_at_r = Polynomial::new(g_evals).evaluate_and_consume(&challenges);
    let h_at_r = Polynomial::new(h_evals).evaluate_and_consume(&challenges);
    assert_eq!(final_eval, f_at_r * g_at_r * h_at_r);
}

#[test]
fn eq_weighted_sumcheck() {
    // eq(r, x) * f(x), common pattern in Jolt (Spartan outer sumcheck).
    // eq(r, x) = prod_i (r_i * x_i + (1 - r_i)(1 - x_i))
    let num_vars = 4;
    let n = 1 << num_vars;

    let f: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 * 3 + 1)).collect();

    // Generate a random-ish point r for the eq polynomial
    let r: Vec<F> = (0..num_vars)
        .map(|i| F::from_u64(i as u64 * 7 + 13))
        .collect();

    // Compute eq(r, x) for all x in {0,1}^n
    let eq_evals = jolt_poly::EqPolynomial::evals_serial::<F>(&r, None);

    // Product: eq(r, x) * f(x)
    let product: Vec<F> = eq_evals.iter().zip(&f).map(|(&e, &fi)| e * fi).collect();

    let claimed_sum: F = product.iter().copied().sum();

    // Prove as degree-2 (eq * f, both multilinear)
    let mut pt = MockTranscript::<F>::default();
    let (proof, sum) = prove_product(&[eq_evals, f], num_vars, &mut pt);
    assert_eq!(sum, claimed_sum);

    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };

    let mut vt = MockTranscript::<F>::default();
    let clear = ClearRoundVerifier::new();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt, &clear);
    assert!(
        result.is_ok(),
        "eq-weighted verify failed: {:?}",
        result.err()
    );
}

#[test]
fn batched_heterogeneous_degrees() {
    // Batch a degree-2 claim (3 vars) with a degree-1 claim (2 vars)
    let f: Vec<F> = (1..=8).map(F::from_u64).collect();
    let g: Vec<F> = (1..=8u64).map(|i| F::from_u64(i * 2)).collect();
    let h: Vec<F> = (1..=4u64).map(|i| F::from_u64(i * 5)).collect();

    let sum_fg: F = f.iter().zip(&g).map(|(&a, &b)| a * b).sum();
    let sum_h: F = h.iter().copied().sum();

    let claims = vec![
        SumcheckClaim {
            num_vars: 3,
            degree: 2,
            claimed_sum: sum_fg,
        },
        SumcheckClaim {
            num_vars: 2,
            degree: 1,
            claimed_sum: sum_h,
        },
    ];

    // Build combined polynomial for honest proof
    let mut pt = MockTranscript::<F>::default();
    sum_fg.append_to_transcript(&mut pt);
    sum_h.append_to_transcript(&mut pt);
    let alpha: F = pt.challenge();

    let max_vars = 3;
    let n = 1 << max_vars;

    // claim_a's polynomial: f * g (degree 2 over 3 vars)
    let fg: Vec<F> = f.iter().zip(&g).map(|(&a, &b)| a * b).collect();

    // claim_b's polynomial: h extended to 3 vars (HighToLow: first var is dummy)
    // h has 4 evals = 2 vars. Extend to 8 evals = 3 vars by duplicating:
    // h_ext[i] = h_ext[i + 4] = h[i] for i in 0..4
    let h_ext: Vec<F> = h.iter().chain(h.iter()).copied().collect();

    // Combined = fg + alpha * h_ext
    let combined: Vec<F> = (0..n).map(|i| fg[i] + alpha * h_ext[i]).collect();
    let combined_sum = combined.iter().copied().sum::<F>();

    // Verify combined sum matches the batched formula
    let expected_combined = sum_fg + alpha * sum_h.mul_pow_2(max_vars - 2);
    assert_eq!(combined_sum, expected_combined);

    // Now prove the combined polynomial as degree 2
    // We need to build the round polynomials at degree max(2,1)=2 → 3 eval points
    let degree = 2;
    let mut buf = combined;
    let mut round_polys = Vec::new();

    for _round in 0..max_vars {
        let half = buf.len() / 2;
        let evals: Vec<F> = (0..=degree)
            .map(|t| {
                let ft = F::from_u64(t as u64);
                let mut sum = F::from_u64(0);
                for i in 0..half {
                    let lo = buf[i];
                    let hi = buf[i + half];
                    sum += lo + ft * (hi - lo);
                }
                sum
            })
            .collect();

        let points: Vec<(F, F)> = evals
            .iter()
            .enumerate()
            .map(|(i, &v)| (F::from_u64(i as u64), v))
            .collect();
        let round_poly = UnivariatePoly::interpolate(&points);

        for &coeff in round_poly.coefficients() {
            coeff.append_to_transcript(&mut pt);
        }
        let r: F = pt.challenge();
        round_polys.push(round_poly);

        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }

    let proof = SumcheckProof {
        round_polynomials: round_polys,
    };

    let mut vt = MockTranscript::<F>::default();
    let clear = ClearRoundVerifier::new();
    let result =
        BatchedSumcheckVerifier::verify(&claims, &proof.round_polynomials, &mut vt, &clear);
    assert!(
        result.is_ok(),
        "batched heterogeneous verify failed: {:?}",
        result.err()
    );
}

#[test]
fn large_num_vars_roundtrip() {
    // Stress test with 10 variables (1024 evaluations), degree 2
    let num_vars = 10;
    let n = 1 << num_vars;

    let f: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 + 1)).collect();
    let g: Vec<F> = (0..n).map(|i| F::from_u64((i * 7 + 3) as u64)).collect();

    let mut pt = MockTranscript::<F>::default();
    let (proof, claimed_sum) = prove_product(&[f, g], num_vars, &mut pt);

    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };

    let mut vt = MockTranscript::<F>::default();
    let clear = ClearRoundVerifier::new();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt, &clear);
    assert!(result.is_ok(), "large roundtrip failed: {:?}", result.err());
}

#[test]
fn labeled_round_verifier_roundtrip() {
    // Test the labeled round verifier path (used by jolt-verifier)
    let num_vars = 3;
    let n = 1 << num_vars;

    let f: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 + 1)).collect();
    let g: Vec<F> = (0..n).map(|i| F::from_u64((i + 5) as u64)).collect();

    let label = b"sumcheck_poly";

    // Prove with labeled absorption
    let mut pt = MockTranscript::<F>::default();
    let degree = 2;
    let mut bufs = vec![f.clone(), g.clone()];
    let claimed_sum: F = (0..n).map(|i| bufs[0][i] * bufs[1][i]).sum();
    let mut round_polys = Vec::new();

    for _round in 0..num_vars {
        let half = bufs[0].len() / 2;
        let evals: Vec<F> = (0..=degree)
            .map(|t| {
                let ft = F::from_u64(t as u64);
                let mut sum = F::from_u64(0);
                for i in 0..half {
                    let mut prod = F::from_u64(1);
                    for buf in &bufs {
                        let lo = buf[i];
                        let hi = buf[i + half];
                        prod *= lo + ft * (hi - lo);
                    }
                    sum += prod;
                }
                sum
            })
            .collect();

        let points: Vec<(F, F)> = evals
            .iter()
            .enumerate()
            .map(|(i, &v)| (F::from_u64(i as u64), v))
            .collect();
        let round_poly = UnivariatePoly::interpolate(&points);

        // Absorb WITH label
        let coeffs = round_poly.coefficients();
        pt.append(&LabelWithCount(label, coeffs.len() as u64));
        for &coeff in coeffs {
            coeff.append_to_transcript(&mut pt);
        }

        let r: F = pt.challenge();
        round_polys.push(round_poly);

        for buf in &mut bufs {
            for i in 0..half {
                buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
            }
            buf.truncate(half);
        }
    }

    let proof = SumcheckProof {
        round_polynomials: round_polys,
    };

    let claim = SumcheckClaim {
        num_vars,
        degree,
        claimed_sum,
    };

    // Verify with labeled round verifier
    let mut vt = MockTranscript::<F>::default();
    let round_verifier = ClearRoundVerifier::with_label(label);
    let result =
        SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt, &round_verifier);
    assert!(
        result.is_ok(),
        "labeled round verifier roundtrip failed: {:?}",
        result.err()
    );
}
