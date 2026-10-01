//! Fiat-Shamir attack constructors over a proof's argument string.
//!
//! Each constructor locates its target messages through a `logging` verifier
//! transcript's event log (site, operation, and NARG byte range per message)
//! and reads challenge values from the matching bytes of a recorded
//! [`ChallengeTape`]. The rewritten messages keep every byte length, so the
//! frozen verifier runs the same schedule over the mutated proof.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "attack constructors require a fixture with the exact audited proof shape"
)]

use std::{fmt::Debug, ops::Range};

#[cfg(not(feature = "akita"))]
use jolt_crypto::HomomorphicCommitment;

use jolt_field::{CanonicalBytes, CanonicalDecode, JoltField};
#[cfg(not(feature = "akita"))]
use jolt_field::{Fr, Ring};
use jolt_sumcheck::CenteredIntegerDomain;
use jolt_transcript::{SiteId, TranscriptEvent, TranscriptOp, VerifierTranscript};
use jolt_verifier::{
    jolt_protocol_id, sites, stages::uniskip::UniskipParams, JoltSponge, JOLT_SESSION,
};

use crate::fs_transcript::{decode_challenge, decode_small_challenge, ChallengeTape};

/// One verifier transcript operation, with the bytes it touched.
#[derive(Clone, Debug)]
pub struct LocatedEvent {
    pub site: SiteId,
    pub op: TranscriptOp,
    /// Argument-string range of a received prover message.
    pub narg: Option<Range<usize>>,
    /// Challenge-tape range of a squeeze.
    pub tape: Option<Range<usize>>,
}

/// Runs `verify` over a `logging` transcript on `narg` and returns its events
/// with each squeeze placed in the challenge stream. `verify` must succeed: an
/// attack is located on the honest proof.
pub fn locate_events<E: Debug>(
    narg: &[u8],
    verify: impl FnOnce(&mut VerifierTranscript<'_, JoltSponge>) -> Result<(), E>,
) -> Vec<LocatedEvent> {
    let mut transcript = VerifierTranscript::<JoltSponge>::new(
        &jolt_protocol_id::<JoltSponge>(),
        JOLT_SESSION,
        narg,
    );
    verify(&mut transcript).expect("honest proof rejected while locating its transcript events");
    let mut squeezed = 0;
    transcript
        .events()
        .iter()
        .map(|event: &TranscriptEvent| {
            let tape = (event.op == TranscriptOp::Challenge).then(|| {
                let range = squeezed..squeezed + event.len;
                squeezed = range.end;
                range
            });
            LocatedEvent {
                site: event.site,
                op: event.op,
                narg: event.narg.clone(),
                tape,
            }
        })
        .collect()
}

/// Cursor over one site's events.
struct SiteEvents<'a> {
    events: Box<dyn Iterator<Item = &'a LocatedEvent> + 'a>,
    tape: &'a ChallengeTape,
}

impl<'a> SiteEvents<'a> {
    fn new(events: &'a [LocatedEvent], site: SiteId, tape: &'a ChallengeTape) -> Self {
        Self {
            events: Box::new(events.iter().filter(move |event| event.site == site)),
            tape,
        }
    }

    /// The next event of kind `op`, skipping public absorbs and any event
    /// kinds in `skip`.
    fn next_skipping(
        &mut self,
        op: TranscriptOp,
        skip: &[TranscriptOp],
        what: &str,
    ) -> &'a LocatedEvent {
        loop {
            let event = self
                .events
                .next()
                .unwrap_or_else(|| panic!("{what} is missing"));
            if event.op == op {
                return event;
            }
            assert!(
                event.op == TranscriptOp::Public || skip.contains(&event.op),
                "expected {what}, found {:?}",
                event.op
            );
        }
    }

    fn next(&mut self, op: TranscriptOp, what: &str) -> &'a LocatedEvent {
        self.next_skipping(op, &[], what)
    }

    fn message(&mut self, what: &str) -> Range<usize> {
        self.next(TranscriptOp::Message, what)
            .narg
            .clone()
            .expect("received message without an argument-string range")
    }

    fn challenge_bytes(&mut self, what: &str) -> &'a [u8] {
        let range = self
            .next(TranscriptOp::Challenge, what)
            .tape
            .clone()
            .expect("squeeze without a tape range");
        self.tape
            .bytes
            .get(range)
            .expect("recorded tape is shorter than the located squeezes")
    }

    fn challenge<F: JoltField>(&mut self, what: &str) -> F {
        decode_challenge(self.challenge_bytes(what))
    }

    fn small_challenge<F: JoltField>(&mut self, what: &str) -> F {
        decode_small_challenge(self.challenge_bytes(what))
    }
}

fn read<A: CanonicalDecode>(narg: &[u8], at: usize) -> A {
    A::from_bytes_le_checked(&narg[at..at + A::NUM_BYTES]).expect("non-canonical atom in fixture")
}

fn write<A: CanonicalBytes>(narg: &mut [u8], at: usize, value: &A) {
    value.to_bytes_le(&mut narg[at..at + A::NUM_BYTES]);
}

/// Changes two homomorphic trace commitments in the kernel of the frozen
/// stage-8 batching combination: `RamInc += gamma * D` and `RdInc -= D` for
/// `D` the first instruction-RA commitment. The final-opening batch orders
/// `RamInc, RdInc` first (`final_opening_polynomial_order`), so their
/// coefficients are `1, gamma` and the joint commitment is unchanged, while
/// both individual openings become false.
#[cfg(not(feature = "akita"))]
pub fn cancel_dory_final_opening_commitments<C>(
    narg: &mut [u8],
    events: &[LocatedEvent],
    tape: &ChallengeTape,
) where
    C: CanonicalDecode + HomomorphicCommitment<Fr> + PartialEq,
{
    let mut commitments = SiteEvents::new(events, sites::COMMITMENTS, tape);
    // `ProofCommitments` send order: RdInc, RamInc, then the instruction RA family.
    let rd_inc = commitments.message("RdInc commitment");
    let ram_inc = commitments.message("RamInc commitment");
    let instruction_ra = commitments.message("first InstructionRa commitment");
    for range in [&rd_inc, &ram_inc, &instruction_ra] {
        assert_eq!(range.len(), C::NUM_BYTES, "commitment message width");
    }
    let gamma: Fr = SiteEvents::new(events, sites::STAGE8, tape)
        .challenge("stage-8 final-opening batching challenge");

    let direction: C = read(narg, instruction_ra.start);
    let original_ram_inc: C = read(narg, ram_inc.start);
    let original_rd_inc: C = read(narg, rd_inc.start);
    let ram_inc_value = C::linear_combine(&original_ram_inc, &direction, &gamma);
    let rd_inc_value = C::linear_combine(&original_rd_inc, &direction, &-Fr::from_u64(1));
    assert!(ram_inc_value != original_ram_inc && rd_inc_value != original_rd_inc);
    write(narg, ram_inc.start, &ram_inc_value);
    write(narg, rd_inc.start, &rd_inc_value);
}

/// Adds `delta` to the clear stage-1 uni-skip output claim and rewrites the
/// uni-skip round and the first remainder round so every algebraic check
/// still holds at the recorded challenges:
///
/// - the uni-skip round gains `slope * (X - mean)`, which sums to zero over the
///   centered domain and adds `delta` at the uni-skip challenge;
/// - the first compressed remainder round absorbs the batched input-claim
///   change `coefficient * delta` into its hint while keeping its value at
///   the first remainder challenge.
///
/// Every later round, and therefore every output claim, is unchanged.
pub fn equivocate_stage1_clear<F: JoltField>(
    narg: &mut [u8],
    events: &[LocatedEvent],
    tape: &ChallengeTape,
    delta: F,
) {
    let params = UniskipParams::spartan_outer();
    let mut stage1 = SiteEvents::new(events, sites::STAGE1, tape);
    // The uni-skip round is the stage's first message, after the tau draws.
    let uniskip_round = stage1
        .next_skipping(
            TranscriptOp::Message,
            &[TranscriptOp::Challenge],
            "stage-1 uni-skip round",
        )
        .narg
        .clone()
        .expect("received message without an argument-string range");
    assert_eq!(
        uniskip_round.len(),
        (params.degree() + 1) * F::NUM_BYTES,
        "uni-skip round width"
    );
    let uniskip_challenge: F = stage1.small_challenge("stage-1 uni-skip challenge");
    let output_claim = stage1.message("stage-1 uni-skip output claim");
    assert_eq!(
        output_claim.len(),
        F::NUM_BYTES,
        "uni-skip output claim width"
    );
    let batching_coefficient: F = stage1.challenge("stage-1 batching coefficient");
    let remainder_round = stage1.message("first stage-1 remainder round");
    assert_eq!(remainder_round.len() % F::NUM_BYTES, 0);
    let remainder_challenge: F = stage1.small_challenge("first stage-1 remainder challenge");

    let claim: F = read(narg, output_claim.start);
    write(narg, output_claim.start, &(claim + delta));

    let power_sums = CenteredIntegerDomain::new(params.domain_size())
        .power_sums(2)
        .expect("stage-1 uni-skip domain is invalid");
    let mean = F::from_i128(power_sums[1])
        * F::from_i128(power_sums[0])
            .inverse()
            .expect("domain size is zero in field");
    let slope = delta
        * (uniskip_challenge - mean)
            .inverse()
            .expect("uni-skip challenge equals the domain mean");
    let c0 = uniskip_round.start;
    let c1 = c0 + F::NUM_BYTES;
    write(narg, c0, &(read::<F>(narg, c0) - slope * mean));
    write(narg, c1, &(read::<F>(narg, c1) + slope));

    // A compressed round omits the linear coefficient, which the verifier
    // recovers from the running-sum hint. Shifting the hint by `h` moves the
    // value at `r` by `h * r`; the next transmitted coefficient cancels it.
    let hint_delta = batching_coefficient * delta;
    let r = remainder_challenge;
    let at = remainder_round.start;
    if remainder_round.len() >= 2 * F::NUM_BYTES {
        // coeffs_except_linear_term = [c0, c2, ...]: adjust c2.
        let c2 = at + F::NUM_BYTES;
        let correction = r
            * hint_delta
            * (r * r - r)
                .inverse()
                .expect("remainder challenge is Boolean");
        write(narg, c2, &(read::<F>(narg, c2) - correction));
    } else {
        let correction = r
            * hint_delta
            * (F::from_u64(1) - F::from_u64(2) * r)
                .inverse()
                .expect("remainder challenge is one half");
        write(narg, at, &(read::<F>(narg, at) - correction));
    }
}
