//! Akita's inline Blake2b transcript sponge against spongefish's Blake2b
//! sponge, on random absorb/squeeze/ratchet schedules.
#![expect(clippy::unwrap_used, reason = "test harness")]

use akita_transcript::TranscriptSponge;
use spongefish::instantiations::Blake2b512;
use spongefish::DuplexSpongeInterface;

#[derive(Clone, Copy, Debug)]
enum Op {
    Absorb(usize),
    Squeeze(usize),
    Ratchet,
}

fn run<S: DuplexSpongeInterface<U = u8> + Default>(ops: &[Op]) -> Vec<u8> {
    let mut sponge = S::default();
    let mut transcript = Vec::new();
    for (step, op) in ops.iter().enumerate() {
        match *op {
            Op::Absorb(len) => {
                let input: Vec<u8> = (0..len).map(|i| (i * 31 + step) as u8).collect();
                let _ = sponge.absorb(&input);
            }
            Op::Squeeze(len) => {
                let mut output = vec![0; len];
                let _ = sponge.squeeze(&mut output);
                transcript.extend(output);
            }
            Op::Ratchet => {
                let _ = sponge.ratchet();
            }
        }
    }
    transcript
}

#[test]
fn inline_sponge_matches_spongefish_blake2b512() {
    let lengths = [0, 1, 7, 8, 63, 64, 65, 127, 128, 129, 200, 300];
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = |bound: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        usize::try_from(seed % u64::try_from(bound).unwrap()).unwrap()
    };
    for _ in 0..500 {
        let ops: Vec<Op> = (0..=next(12))
            .map(|_| match next(3) {
                0 => Op::Absorb(lengths[next(lengths.len())]),
                1 => Op::Squeeze(lengths[next(lengths.len())]),
                _ => Op::Ratchet,
            })
            .collect();
        assert_eq!(
            run::<TranscriptSponge>(&ops),
            run::<Blake2b512>(&ops),
            "{ops:?}"
        );
    }
}
