#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

use jolt_field::{Fr, Ring};
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::claim::{EvaluationClaim, SumcheckClaim};
use jolt_sumcheck::proof::ClearSumcheckProof;
use jolt_sumcheck::round_proof::{CompressedLabeledRoundPoly, LabeledRoundPoly, RoundMessage};
use jolt_sumcheck::{BooleanHypercube, SumcheckVerifier, SUMCHECK_ROUND_TRANSCRIPT_LABEL};
use jolt_transcript::{Blake2bTranscript, Transcript};

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
    transcript: &mut Blake2bTranscript<F>,
) -> (ClearSumcheckProof<F>, F) {
    let degree = polys.len();
    let n = 1 << num_vars;
    assert!(polys.iter().all(|p| p.len() == n));

    let claimed_sum: F = (0..n)
        .map(|i| polys.iter().map(|p| p[i]).product::<F>())
        .sum();

    let mut bufs: Vec<Vec<F>> = polys.to_vec();
    let mut round_polys = Vec::with_capacity(num_vars);

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

        // Absorb through the same path the unlabelled verifier uses.
        <UnivariatePoly<F> as RoundMessage>::append_to_transcript(&round_poly, transcript);

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
        ClearSumcheckProof {
            round_polynomials: round_polys,
        },
        claimed_sum,
    )
}

#[test]
fn degree3_final_eval_correct() {
    let num_vars = 3;
    let n = 1 << num_vars;

    let f_evals: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 + 1)).collect();
    let g_evals: Vec<F> = (0..n).map(|i| F::from_u64((i * 5 + 2) as u64)).collect();
    let h_evals: Vec<F> = (0..n).map(|i| F::from_u64((i + 7) as u64)).collect();

    let mut pt = Blake2bTranscript::new(b"sumcheck-roundtrip");
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

    let mut vt = Blake2bTranscript::new(b"sumcheck-roundtrip");
    let EvaluationClaim {
        point: challenges,
        value: final_eval,
    } = SumcheckVerifier::verify(&claim, &proof.round_polynomials, BooleanHypercube, &mut vt)
        .unwrap();

    let f_at_r = Polynomial::new(f_evals).evaluate_and_consume(&challenges);
    let g_at_r = Polynomial::new(g_evals).evaluate_and_consume(&challenges);
    let h_at_r = Polynomial::new(h_evals).evaluate_and_consume(&challenges);
    assert_eq!(final_eval, f_at_r * g_at_r * h_at_r);
}

#[test]
fn compressed_round_verifier_roundtrip() {
    // Full prover-verifier roundtrip where both the prover and the verifier
    // absorb through `CompressedLabeledRoundPoly` — the wrapper is the
    // single source of truth for the compressed wire format.
    let num_vars = 3;
    let n = 1 << num_vars;
    let label = SUMCHECK_ROUND_TRANSCRIPT_LABEL;
    let degree = 2;

    let f: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 + 1)).collect();
    let g: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 * 2 + 3)).collect();

    let mut pt = Blake2bTranscript::new(b"sumcheck-roundtrip");
    let mut bufs = vec![f.clone(), g.clone()];
    let claimed_sum: F = (0..n).map(|i| bufs[0][i] * bufs[1][i]).sum();
    let mut round_polys = Vec::with_capacity(num_vars);

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

        let compressed = CompressedLabeledRoundPoly::new(&round_poly, label);
        <CompressedLabeledRoundPoly<'_, F> as RoundMessage>::append_to_transcript(
            &compressed,
            &mut pt,
        );

        let r: F = pt.challenge();
        round_polys.push(round_poly);

        for buf in &mut bufs {
            for i in 0..half {
                buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
            }
            buf.truncate(half);
        }
    }

    let proof = ClearSumcheckProof {
        round_polynomials: round_polys,
    };
    let claim = SumcheckClaim {
        num_vars,
        degree,
        claimed_sum,
    };

    let wrapped: Vec<CompressedLabeledRoundPoly<'_, F>> = proof
        .round_polynomials
        .iter()
        .map(|p| CompressedLabeledRoundPoly::new(p, label))
        .collect();

    let mut vt = Blake2bTranscript::new(b"sumcheck-roundtrip");
    let result = SumcheckVerifier::verify(&claim, &wrapped, BooleanHypercube, &mut vt);
    assert!(
        result.is_ok(),
        "compressed round verifier roundtrip failed: {:?}",
        result.err()
    );
}

#[test]
fn labeled_round_verifier_roundtrip() {
    // Test the labeled round verifier path (used by jolt-verifier)
    let num_vars = 3;
    let n = 1 << num_vars;

    let f: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 + 1)).collect();
    let g: Vec<F> = (0..n).map(|i| F::from_u64((i + 5) as u64)).collect();

    let label = SUMCHECK_ROUND_TRANSCRIPT_LABEL;

    let mut pt = Blake2bTranscript::new(b"sumcheck-roundtrip");
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

        let labeled = LabeledRoundPoly::new(&round_poly, label);
        <LabeledRoundPoly<'_, F> as RoundMessage>::append_to_transcript(&labeled, &mut pt);

        let r: F = pt.challenge();
        round_polys.push(round_poly);

        for buf in &mut bufs {
            for i in 0..half {
                buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
            }
            buf.truncate(half);
        }
    }

    let proof = ClearSumcheckProof {
        round_polynomials: round_polys,
    };

    let claim = SumcheckClaim {
        num_vars,
        degree,
        claimed_sum,
    };

    let wrapped: Vec<LabeledRoundPoly<'_, F>> = proof
        .round_polynomials
        .iter()
        .map(|p| LabeledRoundPoly::new(p, label))
        .collect();

    let mut vt = Blake2bTranscript::new(b"sumcheck-roundtrip");
    let result = SumcheckVerifier::verify(&claim, &wrapped, BooleanHypercube, &mut vt);
    assert!(
        result.is_ok(),
        "labeled round verifier roundtrip failed: {:?}",
        result.err()
    );
}
