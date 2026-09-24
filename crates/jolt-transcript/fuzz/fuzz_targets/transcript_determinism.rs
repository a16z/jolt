#![no_main]

//! Determinism and absorption oracles over the digest transcripts.
//!
//! The old `transcript_no_panic` target asserted nothing on a surface with no
//! panic conditions. Instead, twin instances driven by the same fuzzer-chosen
//! op sequence must agree on every challenge and on the final `state()`, and
//! replaying the sequence with one absorbed byte flipped must reach a
//! different final state — a matching state would mean absorbed bytes are
//! ignored or collide.

use jolt_transcript::{Blake2bTranscript, KeccakTranscript, PoseidonTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

const MAX_OPS: usize = 32;

enum Op<'a> {
    Absorb(&'a [u8]),
    Challenge,
}

fn parse_ops(data: &[u8]) -> Vec<Op<'_>> {
    let mut ops = Vec::new();
    let mut cursor = 0;
    while cursor < data.len() && ops.len() < MAX_OPS {
        let tag = data[cursor];
        cursor += 1;
        if tag % 4 == 0 {
            ops.push(Op::Challenge);
        } else {
            let len = tag as usize % 33; // 0..=32 bytes per absorb
            if cursor + len > data.len() {
                break;
            }
            ops.push(Op::Absorb(&data[cursor..cursor + len]));
            cursor += len;
        }
    }
    ops
}

/// Replays `ops`, optionally XOR-ing `mask` into the absorbed byte at global
/// offset `position`. Returns the challenge stream and the final state.
fn run<T: Transcript>(ops: &[Op<'_>], flip: Option<(usize, u8)>) -> (Vec<T::Challenge>, [u8; 32]) {
    let mut transcript = T::new(b"fuzz-determinism");
    let mut challenges = Vec::new();
    let mut absorbed = 0usize;
    for op in ops {
        match op {
            Op::Absorb(chunk) => {
                match flip {
                    Some((position, mask))
                        if (absorbed..absorbed + chunk.len()).contains(&position) =>
                    {
                        let mut flipped = chunk.to_vec();
                        flipped[position - absorbed] ^= mask;
                        transcript.append_bytes(&flipped);
                    }
                    _ => transcript.append_bytes(chunk),
                }
                absorbed += chunk.len();
            }
            Op::Challenge => challenges.push(transcript.challenge()),
        }
    }
    (challenges, transcript.state())
}

fn check<T: Transcript>(ops: &[Op<'_>], total_absorbed: usize, position: usize, mask: u8)
where
    T::Challenge: PartialEq,
{
    let (challenges_a, state_a) = run::<T>(ops, None);
    let (challenges_b, state_b) = run::<T>(ops, None);
    assert!(
        challenges_a == challenges_b && state_a == state_b,
        "transcript replay is nondeterministic"
    );

    if total_absorbed > 0 && mask != 0 {
        let (_, state_flipped) = run::<T>(ops, Some((position % total_absorbed, mask)));
        assert_ne!(
            state_a, state_flipped,
            "flipping an absorbed byte did not change the transcript state"
        );
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let position = data[0] as usize;
    let mask = data[1];
    let ops = parse_ops(&data[2..]);
    let total_absorbed = ops
        .iter()
        .map(|op| match op {
            Op::Absorb(chunk) => chunk.len(),
            Op::Challenge => 0,
        })
        .sum();

    check::<Blake2bTranscript>(&ops, total_absorbed, position, mask);
    check::<KeccakTranscript>(&ops, total_absorbed, position, mask);
    check::<PoseidonTranscript>(&ops, total_absorbed, position, mask);
});
