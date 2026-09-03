//! Shared fixtures: a seeded synthetic Dory opening split into `n` commitments.

#![expect(clippy::unwrap_used, clippy::expect_used, reason = "test fixtures")]

use ark_bn254::{Fq12, Fr as ArkFr};
use ark_ff::{Field as ArkField, PrimeField, UniformRand, Zero};
use jolt_dory::DoryScheme;
use jolt_field::{Field, Fr};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_wrapper::limb_table::dory::{
    DoryChallenges, DorySetupInputs, DoryStatement, DoryWitnessInputs,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

pub const LABEL: &[u8] = b"limb-table-test";

pub struct Opening {
    pub statement: DoryStatement,
    pub setup: DorySetupInputs,
    pub witness: DoryWitnessInputs,
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
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
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
    let mut transcript = Blake2bTranscript::new(LABEL);
    DoryScheme::verify(
        &commitment,
        &point,
        evaluation,
        &proof,
        &verifier_setup,
        &mut transcript,
    )
    .expect("production Dory verifier accepts");

    let mut transcript = Blake2bTranscript::new(LABEL);
    let challenges = DoryChallenges::replay(&proof.0, &mut transcript);
    let setup = DorySetupInputs::from(&verifier_setup.0);
    let rho = ArkFr::rand(&mut rng);
    let commitments = split_commitment(&commitment.0.into(), &setup.ht, rho, n, &mut rng);
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
    }
}

/// Random verifier-key constants of the right shape (for layout-shape tests).
pub fn random_setup(sigma: usize, seed: u64) -> DorySetupInputs {
    use ark_bn254::{Fq12, G1Affine, G2Affine};
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let gt = |rng: &mut ChaCha20Rng| Fq12::rand(rng);
    DorySetupInputs {
        chi: (0..=sigma).map(|_| gt(&mut rng)).collect(),
        delta_1r: (0..=sigma).map(|_| gt(&mut rng)).collect(),
        delta_2r: (0..=sigma).map(|_| gt(&mut rng)).collect(),
        ht: gt(&mut rng),
        g1_0: G1Affine::rand(&mut rng),
        g2_0: G2Affine::rand(&mut rng),
        h1: G1Affine::rand(&mut rng),
        h2: G2Affine::rand(&mut rng),
    }
}
