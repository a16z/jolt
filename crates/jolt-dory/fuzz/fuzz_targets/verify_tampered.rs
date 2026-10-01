#![no_main]

//! Tampering with an honest Dory opening's argument string must be rejected.
//!
//! A process-wide fixture writes one honest transparent and one honest ZK
//! opening; every iteration picks a mode, applies one mutation to that
//! argument string, and runs the verifier plus the transcript's `finish`.
//! Raw byte edits mostly die at element decoding, so two mutations keep the
//! bytes decodable and reach dory-pcs's algebraic and Fiat-Shamir checks:
//! negating a G1/G2/GT element in place (the first offset at or after the
//! chosen one where that element type decodes), and swapping two
//! element-width chunks (reordering messages). The others flip a byte,
//! truncate, append trailing bytes, or delete an element-width range. The
//! verifier absorbs every Dory message and rereads the Σ₁ responses, so any
//! change to the bytes must fail.

use std::sync::OnceLock;

use dory::backends::arkworks::{ArkG1, ArkG2, ArkGT};
use dory::primitives::arithmetic::Group;
use dory::primitives::{DoryDeserialize, DorySerialize};
use jolt_dory::{DoryCommitment, DoryScheme, DoryVerifierSetup};
use jolt_field::{Field, Fr};
use jolt_openings::{CommitmentScheme, OpeningsError, ZkOpeningScheme};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2b512, ProtocolId, ProverTranscript, VerifierTranscript};
use libfuzzer_sys::fuzz_target;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const NUM_VARS: usize = 4;
const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-dory-fuzz/tampered");
const SESSION: &[u8] = b"fuzz-tampered";

struct Opening {
    commitment: DoryCommitment,
    narg: Vec<u8>,
}

struct Fixture {
    verifier_setup: DoryVerifierSetup,
    point: Vec<Fr>,
    eval: Fr,
    transparent: Opening,
    zk: Opening,
}

impl Fixture {
    fn opening(&self, zk: bool) -> &Opening {
        if zk {
            &self.zk
        } else {
            &self.transparent
        }
    }

    fn verify(&self, zk: bool, narg: &[u8]) -> Result<(), OpeningsError> {
        let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, narg);
        if zk {
            DoryScheme::verify_zk(
                &self.zk.commitment,
                &self.point,
                &self.verifier_setup,
                &mut transcript,
            )?;
        } else {
            DoryScheme::verify(
                &self.transparent.commitment,
                &self.point,
                self.eval,
                &self.verifier_setup,
                &mut transcript,
            )?;
        }
        Ok(transcript.finish()?)
    }
}

fn fixture() -> &'static Fixture {
    static FIX: OnceLock<Fixture> = OnceLock::new();
    FIX.get_or_init(|| {
        let mut rng = ChaCha20Rng::seed_from_u64(0xF0_22);
        let prover_setup = DoryScheme::setup_prover(NUM_VARS);
        let verifier_setup = DoryScheme::verifier_setup(&prover_setup);
        let poly = Polynomial::<Fr>::random(NUM_VARS, &mut rng);
        let point: Vec<Fr> = (0..NUM_VARS).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let (commitment, hint) =
            DoryScheme::commit(poly.evaluations(), &prover_setup).expect("fixture commit");
        let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
        DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut transcript,
        )
        .expect("fixture open");
        let transparent = Opening {
            commitment,
            narg: transcript.finish(),
        };

        let (commitment, hint) =
            DoryScheme::commit_zk(poly.evaluations(), &prover_setup).expect("fixture commit_zk");
        let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
        let (_y_com, _blind) =
            DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut transcript)
                .expect("fixture open_zk");
        let zk = Opening {
            commitment,
            narg: transcript.finish(),
        };

        let fixture = Fixture {
            verifier_setup,
            point,
            eval,
            transparent,
            zk,
        };
        for zk in [false, true] {
            fixture
                .verify(zk, &fixture.opening(zk).narg)
                .expect("fixture opening must verify before tampering");
        }
        fixture
    })
}

/// Compressed width of the element kind `kind % 3` selects.
fn element_width(kind: u8) -> usize {
    match kind % 3 {
        0 => 32,
        1 => 64,
        _ => 384,
    }
}

/// Negates the first element of type `G` that decodes at or after `from`.
/// Returns false when none decodes or the negation is a no-op.
fn negate_first<G: Group + PartialEq + DorySerialize + DoryDeserialize>(
    narg: &mut [u8],
    from: usize,
    width: usize,
) -> bool {
    for start in from..=narg.len().saturating_sub(width) {
        let window = &mut narg[start..start + width];
        let Ok(element) = G::deserialize_compressed(&*window) else {
            continue;
        };
        let negated = element.neg();
        if negated == element {
            return false;
        }
        let mut bytes = Vec::with_capacity(width);
        negated
            .serialize_compressed(&mut bytes)
            .expect("serializing into a Vec cannot fail");
        window.copy_from_slice(&bytes);
        return true;
    }
    false
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    let fix = fixture();
    let zk = data[0] & 0x80 != 0;
    let class = data[0] % 6;
    let honest = &fix.opening(zk).narg;
    let offset = usize::from(u16::from_le_bytes([data[1], data[2]])) % honest.len();
    let param = data[3];
    let other = usize::from(u16::from_le_bytes([data[4], data[5]])) % honest.len();
    let payload = &data[6..];
    let width = element_width(param);

    let mut narg = honest.clone();
    match class {
        0 => narg[offset] ^= param,
        1 => narg.truncate(offset),
        2 => narg.extend_from_slice(payload),
        3 => {
            let negated = match param % 3 {
                0 => negate_first::<ArkG1>(&mut narg, offset, width),
                1 => negate_first::<ArkG2>(&mut narg, offset, width),
                _ => negate_first::<ArkGT>(&mut narg, offset, width),
            };
            if !negated {
                return;
            }
        }
        4 => {
            if offset + width > narg.len() || other + width > narg.len() {
                return;
            }
            let first = honest[offset..offset + width].to_vec();
            let second = honest[other..other + width].to_vec();
            narg[other..other + width].copy_from_slice(&first);
            narg[offset..offset + width].copy_from_slice(&second);
        }
        _ => {
            let end = (offset + width).min(narg.len());
            narg.drain(offset..end);
        }
    }
    if narg == *honest {
        return;
    }

    assert!(
        fix.verify(zk, &narg).is_err(),
        "verifier accepted a tampered {} opening (mutation class {class})",
        if zk { "ZK" } else { "transparent" },
    );
});
