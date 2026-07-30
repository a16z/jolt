//! Homomorphic combination example for Dory commitments
//!
//! Demonstrates: Com(r1*P1 + r2*P2 + ... + rn*Pn) = r1*Com(P1) + r2*Com(P2) + ... + rn*Com(Pn)

use dory_pcs::backends::arkworks::{
    ArkFr, ArkG1, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
};
use dory_pcs::primitives::arithmetic::{Field, Group};
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::{prove, setup, verify, Transparent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let (prover_setup, verifier_setup) = setup::<BN254>(10);

    let nu = 4;
    let sigma = 4;
    let poly_size = 1 << (nu + sigma);
    let num_vars = nu + sigma;
    let num_polys = 5;

    let polys: Vec<ArkworksPolynomial> = (0..num_polys)
        .map(|_| {
            let coeffs: Vec<ArkFr> = (0..poly_size).map(|_| ArkFr::random()).collect();
            ArkworksPolynomial::new(coeffs)
        })
        .collect();

    let commitments: Vec<_> = polys
        .iter()
        .map(|poly| {
            poly.commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)
                .unwrap()
        })
        .collect();

    let coeffs: Vec<ArkFr> = (0..num_polys).map(|_| ArkFr::random()).collect();

    #[allow(clippy::op_ref)]
    let mut combined_tier2 = coeffs[0] * &commitments[0].0;
    for i in 1..num_polys {
        #[allow(clippy::op_ref)]
        let scaled = coeffs[i] * &commitments[i].0;
        combined_tier2 = combined_tier2 + scaled;
    }

    let num_rows = 1 << nu;
    let mut combined_tier1 = vec![ArkG1::identity(); num_rows];
    for i in 0..num_polys {
        for (row_idx, row_commitment) in commitments[i].1.iter().enumerate() {
            let scaled = coeffs[i] * row_commitment;
            combined_tier1[row_idx] = combined_tier1[row_idx] + scaled;
        }
    }

    let mut combined_coeffs = vec![ArkFr::zero(); poly_size];
    for poly_idx in 0..num_polys {
        #[allow(clippy::needless_range_loop)]
        for coeff_idx in 0..poly_size {
            let point: Vec<ArkFr> = (0..num_vars)
                .map(|bit_idx| {
                    if (coeff_idx >> bit_idx) & 1 == 1 {
                        ArkFr::one()
                    } else {
                        ArkFr::zero()
                    }
                })
                .collect();

            let eval = polys[poly_idx].evaluate(&point);
            #[allow(clippy::op_ref)]
            let scaled = coeffs[poly_idx] * eval;
            combined_coeffs[coeff_idx] = combined_coeffs[coeff_idx] + scaled;
        }
    }
    let combined_poly = ArkworksPolynomial::new(combined_coeffs);

    let point: Vec<ArkFr> = (0..num_vars).map(|_| ArkFr::random()).collect();
    let evaluation = combined_poly.evaluate(&point);

    let mut expected_eval = ArkFr::zero();
    for i in 0..num_polys {
        let poly_eval = polys[i].evaluate(&point);
        expected_eval = expected_eval + coeffs[i].mul(&poly_eval);
    }
    assert_eq!(evaluation, expected_eval);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-homomorphic-example");
    let (proof, _) = prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
        &combined_poly,
        &point,
        combined_tier1,
        ArkFr::zero(),
        nu,
        sigma,
        &prover_setup,
        &mut prover_transcript,
    )?;

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-homomorphic-example");
    verify::<_, BN254, G1Routines, G2Routines, _>(
        combined_tier2,
        evaluation,
        &point,
        &proof,
        verifier_setup,
        &mut verifier_transcript,
    )?;

    Ok(())
}
