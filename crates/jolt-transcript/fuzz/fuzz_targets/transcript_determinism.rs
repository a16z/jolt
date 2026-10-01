#![no_main]

//! Determinism and absorption oracles over the prover transcript, per sponge.
//!
//! Twin transcripts driven by the same fuzzer-chosen op sequence must agree on
//! every challenge, on the argument string, and on the final sponge state.
//! Replaying the sequence with one absorbed byte flipped (in a sent or a public
//! message) must reach a different final state; a matching state would mean
//! absorbed bytes are ignored or collide.

use jolt_transcript::{
    Blake2b512, Channel, Keccak, PoseidonSponge, ProtocolId, ProverTranscript, Sponge,
};
use libfuzzer_sys::fuzz_target;

const MAX_OPS: usize = 32;
const MAX_MESSAGE: usize = 32;

enum Op<'a> {
    Send(&'a [u8]),
    Public(&'a [u8]),
    Challenge,
}

impl Op<'_> {
    fn absorbed(&self) -> usize {
        match self {
            Op::Send(bytes) | Op::Public(bytes) => bytes.len(),
            Op::Challenge => 0,
        }
    }
}

fn parse_ops(data: &[u8]) -> Vec<Op<'_>> {
    let mut ops = Vec::new();
    let mut cursor = 0;
    while cursor < data.len() && ops.len() < MAX_OPS {
        let tag = data[cursor];
        cursor += 1;
        if tag % 3 == 0 {
            ops.push(Op::Challenge);
            continue;
        }
        let len = tag as usize % (MAX_MESSAGE + 1);
        if cursor + len > data.len() {
            break;
        }
        let bytes = &data[cursor..cursor + len];
        cursor += len;
        ops.push(if tag % 3 == 1 {
            Op::Send(bytes)
        } else {
            Op::Public(bytes)
        });
    }
    ops
}

/// The challenge stream, argument string, and final state of one replay of
/// `ops`, optionally XOR-ing `mask` into the absorbed byte at global offset
/// `position`.
fn run<H: Sponge>(
    ops: &[Op<'_>],
    flip: Option<(usize, u8)>,
) -> (Vec<[u8; 32]>, Vec<u8>, [u8; 32]) {
    let mut transcript =
        ProverTranscript::<H>::new(&ProtocolId::new::<H>("fuzz-determinism"), b"");
    let mut challenges = Vec::new();
    let mut absorbed = 0usize;
    for op in ops {
        let mut bytes = match op {
            Op::Send(bytes) | Op::Public(bytes) => bytes.to_vec(),
            Op::Challenge => {
                challenges.push(transcript.challenge_bytes());
                continue;
            }
        };
        if let Some((position, mask)) = flip {
            if (absorbed..absorbed + bytes.len()).contains(&position) {
                bytes[position - absorbed] ^= mask;
            }
        }
        absorbed += bytes.len();
        match op {
            Op::Send(_) => transcript
                .send_bounded_bytes(&bytes, MAX_MESSAGE)
                .expect("messages are within the bound"),
            Op::Public(_) => transcript.public_bytes(&bytes),
            Op::Challenge => {}
        }
    }
    let state = transcript.preview().squeeze();
    (challenges, transcript.finish(), state)
}

fn check<H: Sponge>(ops: &[Op<'_>], total_absorbed: usize, position: usize, mask: u8) {
    let first = run::<H>(ops, None);
    let second = run::<H>(ops, None);
    assert!(first == second, "transcript replay is nondeterministic");

    if total_absorbed > 0 && mask != 0 {
        let (_, _, flipped_state) = run::<H>(ops, Some((position % total_absorbed, mask)));
        assert_ne!(
            first.2, flipped_state,
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
    let total_absorbed = ops.iter().map(Op::absorbed).sum();

    check::<Blake2b512>(&ops, total_absorbed, position, mask);
    check::<Keccak>(&ops, total_absorbed, position, mask);
    check::<PoseidonSponge>(&ops, total_absorbed, position, mask);
});
