//! Shared fixtures: a seeded synthetic Dory opening split into `n` commitments.

#![expect(clippy::unwrap_used, reason = "test fixtures")]

use ark_bn254::{Fq12, Fr as ArkFr};
use ark_ff::{Field as ArkField, PrimeField, UniformRand, Zero};
use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_field::{Field, Fr, Ring};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_wrapper::limb_table::dory::{
    DoryChallenges, DorySetupInputs, DoryStatement, DoryWitnessInputs,
};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

pub const LABEL: &[u8] = b"limb-table-test";

pub struct Opening {
    pub statement: DoryStatement,
    pub setup: DorySetupInputs,
    pub witness: DoryWitnessInputs,
    pub verifier: ProductionVerifier,
}

/// The production Dory verifier of the opening's statement: the independent
/// oracle the table's pins are compared against.
pub struct ProductionVerifier {
    setup: DoryVerifierSetup,
    commitment: DoryCommitment,
    point: Vec<Fr>,
    evaluation: Fr,
}

impl ProductionVerifier {
    pub fn accepts(&self, proof: &DoryProof) -> bool {
        let mut transcript = Blake2bTranscript::new(LABEL);
        DoryScheme::verify(
            &self.commitment,
            &self.point,
            self.evaluation,
            proof,
            &self.setup,
            &mut transcript,
        )
        .is_ok()
    }
}

/// Splits `commitment = Π C_i^{rho^i}`: random GT offsets `ht^{e_i}` whose
/// `rho`-weighted exponents cancel (lane M1's fixture).
fn split_commitment(
    commitment: &Fq12,
    ht: &Fq12,
    rho: ArkFr,
    count: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<Fq12> {
    let mut exponents = vec![ArkFr::zero(); count];
    let mut weighted = ArkFr::zero();
    let mut power = rho;
    for exponent in &mut exponents[1..] {
        *exponent = ArkFr::rand(rng);
        weighted += *exponent * power;
        power *= rho;
    }
    exponents[0] = -weighted;
    exponents
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let offset = ht.pow(e.into_bigint());
            if i == 0 {
                *commitment * offset
            } else {
                offset
            }
        })
        .collect()
}

/// A real Dory opening of a random `2^num_vars` polynomial, its verifier
/// setup, and the statement (challenges replayed from the same transcript).
pub fn synthetic_opening(num_vars: usize, n: usize, seed: u64) -> Opening {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::verifier_setup(&prover_setup);
    // u64-valued evaluations: full-width row scalars would send every row MSM
    // of a 2^22 commitment through the arkworks fork's pool-per-chunk MSM path,
    // which spawns thousands of short-lived thread pools and exhausts the
    // process thread limit on macOS.
    let evals: Vec<Fr> = (0..1usize << num_vars)
        .map(|_| Fr::from_u64(rng.next_u64()))
        .collect();
    let poly = Polynomial::<Fr>::from(evals);
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let evaluation = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();
    let mut transcript = Blake2bTranscript::new(LABEL);
    let proof = DoryScheme::open(
        &poly,
        &point,
        evaluation,
        &prover_setup,
        Some(hint),
        &mut transcript,
    )
    .unwrap();
    let commitment_gt: Fq12 = commitment.0.into();
    let verifier = ProductionVerifier {
        setup: verifier_setup,
        commitment,
        point: point.clone(),
        evaluation,
    };
    assert!(verifier.accepts(&proof), "production Dory verifier accepts");

    let mut transcript = Blake2bTranscript::new(LABEL);
    let challenges = DoryChallenges::replay(&proof.0, &mut transcript);
    let setup = DorySetupInputs::from(&verifier.setup.0);
    let rho = ArkFr::rand(&mut rng);
    let commitments = split_commitment(&commitment_gt, &setup.ht, rho, n, &mut rng);
    Opening {
        statement: DoryStatement {
            rho,
            point: point.iter().rev().map(|x| ArkFr::from(*x)).collect(),
            evaluation: ArkFr::from(evaluation),
            challenges,
        },
        setup,
        witness: DoryWitnessInputs {
            commitments,
            proof: proof.0,
        },
        verifier,
    }
}

/// The wrapper's offset challenge `θ` of the fixture (drawn from the stream
/// transcript after phase 1a in production; any nonzero value here).
pub fn offset_challenge() -> ArkFr {
    let mut rng = ChaCha20Rng::seed_from_u64(0x000F_F5E7);
    ArkFr::rand(&mut rng)
}
