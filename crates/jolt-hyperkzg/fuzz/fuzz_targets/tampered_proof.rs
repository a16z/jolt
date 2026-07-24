#![no_main]

//! Tamper with one element of a fixed honest HyperKZG opening and require
//! rejection.
//!
//! The fixture (SRS, polynomial, opening point, commitment, proof) is built
//! once per process; every iteration reaches `verify` with a structured
//! mutation chosen by the fuzzer: a folded-evaluation entry, an intermediate
//! or witness commitment, the claimed evaluation (formerly the `wrong_eval`
//! target), the opening point, or the proof's shape.

use std::sync::OnceLock;

use jolt_crypto::{Bn254, JoltGroup};
use jolt_field::{Fr, RandomSampling};
use jolt_hyperkzg::{HyperKZGCommitment, HyperKZGProof, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type TestScheme = HyperKZGScheme<Bn254>;

const NUM_VARS: usize = 3;

struct Fixture {
    vk: HyperKZGVerifierSetup<Bn254>,
    commitment: HyperKZGCommitment<Bn254>,
    point: Vec<Fr>,
    eval: Fr,
    proof: HyperKZGProof<Bn254>,
}

fn fixture() -> &'static Fixture {
    static FIX: OnceLock<Fixture> = OnceLock::new();
    FIX.get_or_init(|| {
        let n = 1usize << NUM_VARS;
        let mut rng = ChaCha20Rng::seed_from_u64(0xfade);
        let pk = TestScheme::setup(&mut rng, n, Bn254::g1_generator(), Bn254::g2_generator());
        let vk = TestScheme::verifier_setup(&pk);

        let poly = Polynomial::<Fr>::random(NUM_VARS, &mut rng);
        let point: Vec<Fr> = (0..NUM_VARS).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let (commitment, ()) =
            TestScheme::commit(poly.evaluations(), &pk).expect("fixture commit");
        let mut pt = Blake2bTranscript::new(b"fuzz-tamper");
        let proof = <TestScheme as CommitmentScheme>::open(&poly, &point, eval, &pk, None, &mut pt)
            .expect("fixture open");

        let mut vt = Blake2bTranscript::new(b"fuzz-tamper");
        <TestScheme as CommitmentScheme>::verify(&commitment, &point, eval, &proof, &vk, &mut vt)
            .expect("fixture proof must verify before tampering");

        Fixture {
            vk,
            commitment,
            point,
            eval,
            proof,
        }
    })
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let fix = fixture();

    let mut proof = fix.proof.clone();
    let mut point = fix.point.clone();
    let mut eval = fix.eval;

    match data[0] % 8 {
        0 => {
            // Folded-evaluation entry (exercises folding consistency checks).
            let row = (data[1] as usize) % proof.v.len();
            let col = (data[2] as usize) % proof.v[row].len();
            let value = Fr::from_le_bytes_mod_order(&data[3..]);
            if value == proof.v[row][col] {
                return;
            }
            proof.v[row][col] = value;
        }
        1 => {
            // Intermediate commitment (exercises the pairing check).
            if proof.com.is_empty() {
                return;
            }
            let idx = (data[1] as usize) % proof.com.len();
            let scalar = Fr::from_le_bytes_mod_order(&data[2..]);
            proof.com[idx] = proof.com[idx].scalar_mul(&scalar);
            if proof.com[idx] == fix.proof.com[idx] {
                return;
            }
        }
        2 => {
            // Witness commitment (exercises the pairing check).
            let idx = (data[1] as usize) % proof.w.len();
            let scalar = Fr::from_le_bytes_mod_order(&data[2..]);
            proof.w[idx] = proof.w[idx].scalar_mul(&scalar);
            if proof.w[idx] == fix.proof.w[idx] {
                return;
            }
        }
        3 => {
            // Wrong claimed evaluation for an otherwise-honest proof.
            eval = Fr::from_le_bytes_mod_order(&data[1..]);
            if eval == fix.eval {
                return;
            }
        }
        4 => {
            // Wrong opening point for an otherwise-honest proof.
            let idx = (data[1] as usize) % point.len();
            let value = Fr::from_le_bytes_mod_order(&data[2..]);
            if value == point[idx] {
                return;
            }
            point[idx] = value;
        }
        5 => {
            // Truncated intermediate commitments (shape check).
            if proof.com.pop().is_none() {
                return;
            }
        }
        6 => {
            // Extra intermediate commitment (shape check).
            let scalar = Fr::from_le_bytes_mod_order(&data[1..]);
            proof.com.push(Bn254::g1_generator().scalar_mul(&scalar));
        }
        _ => {
            // Swapped witness commitments (transcript/pairing binding).
            if proof.w[0] == proof.w[1] {
                return;
            }
            proof.w.swap(0, 1);
        }
    }

    let mut vt = Blake2bTranscript::new(b"fuzz-tamper");
    let result = <TestScheme as CommitmentScheme>::verify(
        &fix.commitment,
        &point,
        eval,
        &proof,
        &fix.vk,
        &mut vt,
    );
    assert!(result.is_err(), "tampered proof must be rejected");
});
