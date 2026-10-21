#![no_main]

//! Determinism and absorption oracles over the transcript sponges.
//!
//! Twin instances driven by the same fuzzer-chosen op sequence must agree on
//! every squeeze and on the final state, observed as the next 32 squeezed
//! bytes, and replaying the sequence with one absorbed byte flipped must reach
//! a different final state — a matching state would mean absorbed bytes are
//! ignored or collide.

use jolt_transcript::{prover_transcript, Blake2b512, Keccak, PoseidonSponge, Transcript};
use libfuzzer_sys::fuzz_target;
use spongefish::DuplexSpongeInit;

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
            let len = tag as usize % 33;
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
fn run<H: DuplexSpongeInit<U = u8>>(
    ops: &[Op<'_>],
    flip: Option<(usize, u8)>,
) -> (Vec<[u8; 16]>, [u8; 32]) {
    let mut transcript = prover_transcript::<H>(b"fuzz-determinism", b"");
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
            Op::Challenge => challenges.push(transcript.verifier_messages::<u8, 16>()),
        }
    }
    (challenges, transcript.verifier_messages::<u8, 32>())
}

fn check<H: DuplexSpongeInit<U = u8>>(
    ops: &[Op<'_>],
    total_absorbed: usize,
    position: usize,
    mask: u8,
) {
    let (challenges_a, state_a) = run::<H>(ops, None);
    let (challenges_b, state_b) = run::<H>(ops, None);
    assert!(
        challenges_a == challenges_b && state_a == state_b,
        "transcript replay is nondeterministic"
    );

    if total_absorbed > 0 && mask != 0 {
        let (_, state_flipped) = run::<H>(ops, Some((position % total_absorbed, mask)));
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

    check::<Blake2b512>(&ops, total_absorbed, position, mask);
    check::<Keccak>(&ops, total_absorbed, position, mask);
    check::<PoseidonSponge>(&ops, total_absorbed, position, mask);
});
