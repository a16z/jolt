#![no_main]

//! Injectivity of the length-framed absorption convention: two op sequences
//! whose payloads concatenate to the same raw bytes but with different
//! chunk boundaries must reach different transcript states.
//!
//! Each chunk is absorbed the way production code absorbs variable-length
//! payloads — `append_bytes`, a length-prefixed `BytesMsg` — so a boundary
//! move changes the framing and MUST change the state. A matching state would
//! mean the framing convention fails to separate `[ab]` from `[a][b]`, the
//! classic transcript-malleability footgun. The state is observed as the next
//! 32 squeezed units: 32 bytes, or 32 `Fr` elements on Poseidon.

use std::fmt::Debug;

use jolt_transcript::{
    prover_transcript, Blake2b512, Keccak, PoseidonSponge, SpongeUnit, Transcript,
};
use libfuzzer_sys::fuzz_target;
use spongefish::DuplexSpongeInit;

const MAX_CHUNKS: usize = 16;

fn parse_chunks(data: &[u8]) -> Vec<&[u8]> {
    let mut chunks = Vec::new();
    let mut cursor = 0;
    while cursor < data.len() && chunks.len() < MAX_CHUNKS {
        let len = data[cursor] as usize % 33;
        cursor += 1;
        if cursor + len > data.len() {
            break;
        }
        chunks.push(&data[cursor..cursor + len]);
        cursor += len;
    }
    chunks
}

fn framed_state<H: DuplexSpongeInit<U: SpongeUnit + Debug>>(chunks: &[Vec<u8>]) -> Vec<H::U> {
    let mut transcript = prover_transcript::<H>(b"fuzz-injectivity", b"");
    for chunk in chunks {
        transcript.append_bytes(chunk);
    }
    transcript.verifier_message_as(32, <[H::U]>::to_vec)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let split_chunk = data[0] as usize;
    let split_at = data[1] as usize;
    let original: Vec<Vec<u8>> = parse_chunks(&data[2..])
        .into_iter()
        .map(<[u8]>::to_vec)
        .collect();
    if original.is_empty() {
        return;
    }

    // Morph one boundary: split a chunk in two. The concatenated payload
    // bytes are identical; only the chunk structure differs.
    let index = split_chunk % original.len();
    if original[index].is_empty() {
        return;
    }
    let position = 1 + split_at % original[index].len();
    if position >= original[index].len() {
        // Splitting at the end appends an extra empty-chunk frame rather than
        // moving a payload boundary; that is outside this harness's morph class.
        return;
    }
    let mut morphed = original.clone();
    let (left, right) = original[index].split_at(position);
    morphed[index] = left.to_vec();
    morphed.insert(index + 1, right.to_vec());

    debug_assert_eq!(original.concat(), morphed.concat());

    assert_ne!(
        framed_state::<Blake2b512>(&original),
        framed_state::<Blake2b512>(&morphed),
        "Blake2b framed absorption is not boundary-injective"
    );
    assert_ne!(
        framed_state::<Keccak>(&original),
        framed_state::<Keccak>(&morphed),
        "Keccak framed absorption is not boundary-injective"
    );
    assert_ne!(
        framed_state::<PoseidonSponge>(&original),
        framed_state::<PoseidonSponge>(&morphed),
        "Poseidon framed absorption is not boundary-injective"
    );
});
