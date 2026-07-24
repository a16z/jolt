#![no_main]

//! Structured tampering of an honest Dory opening must be rejected.
//!
//! The previous harness byte-decoded a full `DoryProof` from raw fuzzer
//! input, so essentially no input survived deserialization and `verify` was
//! never reached — it degenerated into a slower `deser_commitment`. Build one
//! honest proof in a process-wide fixture instead; every iteration reaches
//! the verifier with one structured mutation: a negated protocol element, a
//! dropped or swapped reduce round, or a stripped final message.

use std::sync::OnceLock;

use dory::primitives::arithmetic::Group;
use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_field::{Fr, RandomSampling};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const NUM_VARS: usize = 4;
const TRANSCRIPT_LABEL: &[u8] = b"fuzz-tampered";

struct Fixture {
    verifier_setup: DoryVerifierSetup,
    commitment: DoryCommitment,
    point: Vec<Fr>,
    eval: Fr,
    proof: DoryProof,
}

fn fixture() -> &'static Fixture {
    static FIX: OnceLock<Fixture> = OnceLock::new();
    FIX.get_or_init(|| {
        let mut rng = ChaCha20Rng::seed_from_u64(0xF0_22);
        let prover_setup = DoryScheme::setup_prover(NUM_VARS);
        let verifier_setup = DoryScheme::setup_verifier(NUM_VARS);
        let poly = Polynomial::<Fr>::random(NUM_VARS, &mut rng);
        let point: Vec<Fr> = (0..NUM_VARS).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup)
            .expect("fixture commit");

        let mut pt = Blake2bTranscript::new(TRANSCRIPT_LABEL);
        let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt)
            .expect("fixture open");

        let mut vt = Blake2bTranscript::new(TRANSCRIPT_LABEL);
        DoryScheme::verify(&commitment, &point, eval, &proof, &verifier_setup, &mut vt)
            .expect("fixture proof must verify before tampering");

        Fixture {
            verifier_setup,
            commitment,
            point,
            eval,
            proof,
        }
    })
}

/// Negates a group element in place. Returns false (skip the iteration) when
/// negation is a no-op because the element is the identity.
fn negate<G: Group + PartialEq>(element: &mut G) -> bool {
    let negated = element.neg();
    if negated == *element {
        return false;
    }
    *element = negated;
    true
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let fix = fixture();
    let mut proof = fix.proof.clone();
    let inner = &mut proof.0;

    let changed = match data[0] % 8 {
        0 => match data[1] % 3 {
            0 => negate(&mut inner.vmv_message.c),
            1 => negate(&mut inner.vmv_message.d2),
            _ => negate(&mut inner.vmv_message.e1),
        },
        1 => {
            if inner.first_messages.is_empty() {
                return;
            }
            let round = (data[1] as usize) % inner.first_messages.len();
            let message = &mut inner.first_messages[round];
            match data[2] % 6 {
                0 => negate(&mut message.d1_left),
                1 => negate(&mut message.d1_right),
                2 => negate(&mut message.d2_left),
                3 => negate(&mut message.d2_right),
                4 => negate(&mut message.e1_beta),
                _ => negate(&mut message.e2_beta),
            }
        }
        2 => {
            if inner.second_messages.is_empty() {
                return;
            }
            let round = (data[1] as usize) % inner.second_messages.len();
            let message = &mut inner.second_messages[round];
            match data[2] % 6 {
                0 => negate(&mut message.c_plus),
                1 => negate(&mut message.c_minus),
                2 => negate(&mut message.e1_plus),
                3 => negate(&mut message.e1_minus),
                4 => negate(&mut message.e2_plus),
                _ => negate(&mut message.e2_minus),
            }
        }
        3 => match &mut inner.final_message {
            Some(message) => {
                if data[1] % 2 == 0 {
                    negate(&mut message.e1)
                } else {
                    negate(&mut message.e2)
                }
            }
            None => return,
        },
        4 => inner.final_message.take().is_some(),
        5 => inner.first_messages.pop().is_some(),
        6 => inner.second_messages.pop().is_some(),
        _ => {
            // Swap two reduce rounds (Fiat-Shamir round binding).
            if inner.first_messages.len() < 2 {
                return;
            }
            if inner.first_messages[0] == inner.first_messages[1] {
                return;
            }
            inner.first_messages.swap(0, 1);
            true
        }
    };
    if !changed {
        return;
    }

    let mut transcript = Blake2bTranscript::new(TRANSCRIPT_LABEL);
    let result = DoryScheme::verify(
        &fix.commitment,
        &fix.point,
        fix.eval,
        &proof,
        &fix.verifier_setup,
        &mut transcript,
    );
    assert!(
        result.is_err(),
        "verifier accepted a tampered proof (mutation class {})",
        data[0] % 8,
    );
});
