//! Pins how dory-pcs draws the challenges that `jolt-dory` shortens.
//!
//! The adapter prices `beta`, `alpha` and `d` from a loss table keyed by
//! dory-pcs's labels (`src/transcript.rs`). The table is right only while
//! each short draw follows an absorb and only `gamma` follows the last
//! `alpha` with no absorb in between. These tests record real transparent
//! and zero-knowledge proofs and check that order, so a dory-pcs release
//! that draws differently fails here.
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests unwrap successful PCS operations"
)]

use std::sync::{Arc, Mutex};

use jolt_dory::DoryScheme;
use jolt_field::{CanonicalEncoding, Field, Fr};
use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, ChallengeSetSize, GrindBitsError, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Absorb,
    Full,
    Short(u64),
    Bytes,
}

/// A Blake2b transcript that logs every absorb and draw.
#[derive(Clone, Default)]
struct Recording {
    inner: Blake2bTranscript,
    log: Arc<Mutex<Vec<Event>>>,
}

impl Recording {
    fn push(&self, event: Event) {
        self.log.lock().unwrap().push(event);
    }

    fn events(&self) -> Vec<Event> {
        self.log.lock().unwrap().clone()
    }
}

impl Transcript for Recording {
    type Challenge = Fr;
    const CHALLENGE_SET: ChallengeSetSize = <Blake2bTranscript as Transcript>::CHALLENGE_SET;

    fn new(label: &'static [u8]) -> Self {
        Self {
            inner: Blake2bTranscript::new(label),
            log: Arc::default(),
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.push(Event::Absorb);
        self.inner.append_bytes(bytes);
    }

    fn challenge(&mut self) -> Fr {
        self.push(Event::Full);
        self.inner.challenge()
    }

    fn challenge_short_scalar(&mut self, loss_factor: u64) -> Result<Fr, GrindBitsError>
    where
        Self::Challenge: CanonicalEncoding,
    {
        self.push(Event::Short(loss_factor));
        self.inner.challenge_short_scalar(loss_factor)
    }

    fn challenge_bytes(&mut self, out: &mut [u8]) {
        self.push(Event::Bytes);
        self.inner.challenge_bytes(out);
    }

    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
}

/// Checks the short draws come as `beta, alpha` per round and then `d`, each
/// right after an absorb, and that only one full draw (`gamma`) follows the
/// last `alpha` before the next absorb.
fn check_bursts(events: &[Event], rounds: usize) {
    assert!(!events.contains(&Event::Bytes), "Dory draws no raw bytes");
    let shorts: Vec<u64> = events
        .iter()
        .filter_map(|event| match event {
            Event::Short(loss) => Some(*loss),
            _ => None,
        })
        .collect();
    let mut expected = [2, 4].repeat(rounds);
    expected.push(3);
    assert_eq!(
        shorts, expected,
        "short draws: beta, alpha per round, then d"
    );

    let last_alpha = events.iter().rposition(|event| *event == Event::Short(4));
    for (index, event) in events.iter().enumerate() {
        if !matches!(event, Event::Short(_)) {
            continue;
        }
        let before = index.checked_sub(1).and_then(|prev| events.get(prev));
        assert_eq!(
            before,
            Some(&Event::Absorb),
            "the short draw at {index} shares a burst with the draw before it"
        );
        let after: Vec<Event> = events
            .iter()
            .skip(index + 1)
            .take_while(|next| **next != Event::Absorb)
            .copied()
            .collect();
        let allowed = if Some(index) == last_alpha {
            vec![Event::Full]
        } else {
            vec![]
        };
        assert_eq!(after, allowed, "draws after the short draw at {index}");
    }
}

fn instance(num_vars: usize, seed: u64) -> (Polynomial<Fr>, Vec<Fr>, Fr) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as Field>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    (poly, point, eval)
}

#[test]
fn transparent_short_draws_start_their_own_bursts() {
    for num_vars in [2, 3, 4, 5, 6] {
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::setup_verifier(num_vars);
        let (poly, point, eval) = instance(num_vars, 2100 + num_vars as u64);
        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

        let mut pt = Recording::new(b"dory-bursts");
        let proof =
            DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt).unwrap();
        let mut vt = Recording::new(b"dory-bursts");
        DoryScheme::verify(&commitment, &point, eval, &proof, &verifier_setup, &mut vt)
            .expect("transparent proof verifies");

        assert_eq!(pt.events(), vt.events(), "prover and verifier draw alike");
        check_bursts(&vt.events(), num_vars.div_ceil(2));
    }
}

#[test]
fn zk_short_draws_start_their_own_bursts() {
    for num_vars in [2, 3, 4, 5, 6] {
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::setup_verifier(num_vars);
        let (poly, point, eval) = instance(num_vars, 2200 + num_vars as u64);
        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup).unwrap();

        let mut pt = Recording::new(b"dory-zk-bursts");
        let (proof, _, _) =
            DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt).unwrap();
        let mut vt = Recording::new(b"dory-zk-bursts");
        let _eval_com =
            DoryScheme::verify_zk(&commitment, &point, &proof, &verifier_setup, &mut vt)
                .expect("zero-knowledge proof verifies");

        assert_eq!(pt.events(), vt.events(), "prover and verifier draw alike");
        check_bursts(&vt.events(), num_vars.div_ceil(2));
    }
}
