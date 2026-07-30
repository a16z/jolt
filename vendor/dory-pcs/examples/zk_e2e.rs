//! Zero-knowledge end-to-end example of Dory polynomial commitment scheme
//!
//! Demonstrates the full ZK workflow where both the commitment and the proof
//! are hiding. The `ZK` mode type parameter is used for both `commit()` and
//! `prove()`.
//!
//! Matrix dimensions: 16x16 (nu=4, sigma=4, total 256 coefficients)

use dory_pcs::backends::arkworks::{
    ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::{prove, setup, verify, ZK};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let (prover_setup, verifier_setup) = setup::<BN254>(10);

    let nu = 4;
    let sigma = 4;
    let poly_size = 1 << (nu + sigma);
    let num_vars = nu + sigma;

    let coefficients: Vec<ArkFr> = (0..poly_size).map(|_| ArkFr::random()).collect();
    let poly = ArkworksPolynomial::new(coefficients);

    let (tier_2, tier_1, commit_blind) =
        poly.commit::<BN254, ZK, G1Routines>(nu, sigma, &prover_setup)?;

    let point: Vec<ArkFr> = (0..num_vars).map(|_| ArkFr::random()).collect();
    let evaluation = poly.evaluate(&point);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-zk-example");
    let (proof, _) = prove::<_, BN254, G1Routines, G2Routines, _, _, ZK>(
        &poly,
        &point,
        tier_1,
        commit_blind,
        nu,
        sigma,
        &prover_setup,
        &mut prover_transcript,
    )?;

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-zk-example");
    verify::<_, BN254, G1Routines, G2Routines, _>(
        tier_2,
        evaluation,
        &point,
        &proof,
        verifier_setup,
        &mut verifier_transcript,
    )?;

    Ok(())
}
