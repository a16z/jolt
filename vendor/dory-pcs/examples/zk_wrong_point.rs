//! Regression check for the ZK opening-point binding soundness fix.
//!
//! A ZK evaluation proof created for a point P must NOT verify at a different
//! point P'. Prior to the fix, the ZK final check never read the verifier's
//! folded point scalars (`s1_acc`/`s2_acc`), so a proof for any point was
//! accepted at every point.
//!
//! ```sh
//! cargo run --release --features "backends zk" --example zk_wrong_point
//! ```

use dory_pcs::backends::arkworks::{
    ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::{prove, setup, verify, ZK};

fn run(nu: usize, sigma: usize) -> (bool, bool) {
    let (prover_setup, verifier_setup) = setup::<BN254>(10);
    let num_vars = nu + sigma;
    let poly_size = 1 << num_vars;

    let coefficients: Vec<ArkFr> = (0..poly_size).map(|_| ArkFr::random()).collect();
    let poly = ArkworksPolynomial::new(coefficients);

    let (tier_2, tier_1, commit_blind) = poly
        .commit::<BN254, ZK, G1Routines>(nu, sigma, &prover_setup)
        .expect("commit");

    let point: Vec<ArkFr> = (0..num_vars).map(|_| ArkFr::random()).collect();
    let evaluation = poly.evaluate(&point);

    let mut pt = Blake2bTranscript::new(b"dory-zk-wrong-point");
    let (proof, _) = prove::<_, BN254, G1Routines, G2Routines, _, _, ZK>(
        &poly,
        &point,
        tier_1,
        commit_blind,
        nu,
        sigma,
        &prover_setup,
        &mut pt,
    )
    .expect("prove");

    // Honest verification at the correct point.
    let mut vt_ok = Blake2bTranscript::new(b"dory-zk-wrong-point");
    let honest = verify::<_, BN254, G1Routines, G2Routines, _>(
        tier_2,
        evaluation,
        &point,
        &proof,
        verifier_setup.clone(),
        &mut vt_ok,
    )
    .is_ok();

    // Adversarial verification at a DIFFERENT point.
    let mut wrong_point = point.clone();
    wrong_point[0] = wrong_point[0].add(&ArkFr::one());
    let mut vt_bad = Blake2bTranscript::new(b"dory-zk-wrong-point");
    let wrong_accepted = verify::<_, BN254, G1Routines, G2Routines, _>(
        tier_2,
        evaluation,
        &wrong_point,
        &proof,
        verifier_setup,
        &mut vt_bad,
    )
    .is_ok();

    (honest, wrong_accepted)
}

fn main() {
    let mut sound = true;
    for (nu, sigma) in [(1usize, 1usize), (2, 2), (2, 3), (4, 4)] {
        let (honest, wrong_accepted) = run(nu, sigma);
        println!(
            "nu={nu} sigma={sigma}: honest_ok={honest}  wrong_point_accepted={wrong_accepted}  {}",
            if honest && !wrong_accepted {
                "OK (sound)"
            } else if wrong_accepted {
                "*** SOUNDNESS BUG: wrong point accepted ***"
            } else {
                "!! honest proof failed to verify !!"
            }
        );
        sound &= honest && !wrong_accepted;
    }
    assert!(sound, "ZK opening-point binding check failed");
    println!("all sizes sound");
}
