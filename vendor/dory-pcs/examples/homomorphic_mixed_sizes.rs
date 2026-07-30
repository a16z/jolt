//! Mixed-size homomorphic combination example for Dory commitments
//!
//! Demonstrates homomorphic combination of polynomials with different matrix
//! dimensions (sizes 16 and 4, combined in a 4x4 layout).

use dory_pcs::backends::arkworks::{
    ArkFr, ArkG1, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
};
use dory_pcs::primitives::arithmetic::{Field, Group};
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::{prove, setup, verify, Transparent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let (prover_setup, verifier_setup) = setup::<BN254>(4);

    let mut coeffs_poly1 = vec![ArkFr::zero(); 16];
    let mut coeffs_poly2 = vec![ArkFr::zero(); 4];
    for coeff in coeffs_poly1.iter_mut() {
        *coeff = ArkFr::random();
    }
    for coeff in coeffs_poly2.iter_mut() {
        *coeff = ArkFr::random();
    }
    let poly1 = ArkworksPolynomial::new(coeffs_poly1.clone());
    let poly2 = ArkworksPolynomial::new(coeffs_poly2.clone());

    let commitment1 = poly1
        .commit::<BN254, Transparent, G1Routines>(2, 2, &prover_setup)
        .unwrap();
    let commitment2 = poly2
        .commit::<BN254, Transparent, G1Routines>(1, 1, &prover_setup)
        .unwrap();

    let coeff_scalars = [ArkFr::random(), ArkFr::random()];

    let combined_tier2 = coeff_scalars[0] * commitment1.0 + coeff_scalars[1] * commitment2.0;

    let mut combined_tier1 = vec![ArkG1::identity(); 4];
    for (row_idx, row_commit) in commitment1.1.iter().enumerate() {
        combined_tier1[row_idx] = combined_tier1[row_idx] + (coeff_scalars[0] * row_commit);
    }
    for (row_idx, row_commit) in commitment2.1.iter().enumerate() {
        combined_tier1[row_idx] = combined_tier1[row_idx] + (coeff_scalars[1] * row_commit);
    }

    let mut combined_coeffs = vec![ArkFr::zero(); 16];
    for idx in 0..16 {
        let term1 = coeff_scalars[0].mul(&coeffs_poly1[idx]);
        combined_coeffs[idx] = combined_coeffs[idx] + term1;
        if idx < 2 {
            let term2 = coeff_scalars[1].mul(&coeffs_poly2[idx]);
            combined_coeffs[idx] = combined_coeffs[idx] + term2;
        }
        if idx > 3 && idx < 6 {
            let term3 = coeff_scalars[1].mul(&coeffs_poly2[idx - 2]);
            combined_coeffs[idx] = combined_coeffs[idx] + term3;
        }
    }
    let combined_poly = ArkworksPolynomial::new(combined_coeffs);

    let mut padded_poly2_coefficients = vec![ArkFr::zero(); 16];
    padded_poly2_coefficients[0] = coeffs_poly2[0];
    padded_poly2_coefficients[1] = coeffs_poly2[1];
    padded_poly2_coefficients[4] = coeffs_poly2[2];
    padded_poly2_coefficients[5] = coeffs_poly2[3];
    let padded_poly2 = ArkworksPolynomial::new(padded_poly2_coefficients);

    let point: Vec<ArkFr> = (0..4).map(|_| ArkFr::random()).collect();
    let evaluation = combined_poly.evaluate(&point);

    let eval1 = poly1.evaluate(&point);
    let eval2 = padded_poly2.evaluate(&point);
    let eval3 = poly2.evaluate(&[point[0], point[2]])
        * (ArkFr::one() - point[1])
        * (ArkFr::one() - point[3]);
    assert_eq!(eval3, eval2);
    let mut expected = ArkFr::zero();
    expected = expected + coeff_scalars[0].mul(&eval1);
    expected = expected + coeff_scalars[1].mul(&eval2);
    assert_eq!(evaluation, expected);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-homomorphic-mixed");
    let (proof, _) = prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
        &combined_poly,
        &point,
        combined_tier1,
        ArkFr::zero(),
        2,
        2,
        &prover_setup,
        &mut prover_transcript,
    )?;

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-homomorphic-mixed");
    verify::<_, BN254, G1Routines, G2Routines, _>(
        combined_tier2,
        evaluation,
        &point,
        &proof,
        verifier_setup,
        &mut verifier_transcript,
    )?;

    let padded_poly_commitment = padded_poly2
        .commit::<BN254, Transparent, G1Routines>(2, 2, &prover_setup)
        .unwrap();
    assert_eq!(padded_poly_commitment.0, commitment2.0);

    Ok(())
}
