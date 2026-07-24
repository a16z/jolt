#![no_main]

//! Injectivity of the length-framed absorption convention: two op sequences
//! whose payloads concatenate to the same raw bytes but with different
//! chunk boundaries must reach different transcript states.
//!
//! Each chunk is absorbed the way production code absorbs variable-length
//! payloads — a `LabelWithCount` frame followed by the payload — so a
//! boundary move changes the framing and MUST change the state. A matching
//! state would mean the framing convention fails to separate `[ab]` from
//! `[a][b]`, the classic transcript-malleability footgun.

use jolt_transcript::{
    Blake2bTranscript, KeccakTranscript, LabelWithCount, PoseidonTranscript, Transcript,
};
use libfuzzer_sys::fuzz_target;

const MAX_CHUNKS: usize = 16;

fn parse_chunks(data: &[u8]) -> Vec<&[u8]> {
    let mut chunks = Vec::new();
    let mut cursor = 0;
    while cursor < data.len() && chunks.len() < MAX_CHUNKS {
        let len = data[cursor] as usize % 33; // 0..=32 bytes
        cursor += 1;
        if cursor + len > data.len() {
            break;
        }
        chunks.push(&data[cursor..cursor + len]);
        cursor += len;
    }
    chunks
}

fn framed_state<T: Transcript>(chunks: &[Vec<u8>]) -> [u8; 32] {
    let mut transcript = T::new(b"fuzz-injectivity");
    for chunk in chunks {
        transcript.append(&LabelWithCount(b"chunk", chunk.len() as u64));
        transcript.append_bytes(chunk);
    }
    transcript.state()
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
        // Splitting at the end reproduces the original structure.
        return;
    }
    let mut morphed = original.clone();
    let (left, right) = original[index].split_at(position);
    morphed[index] = left.to_vec();
    morphed.insert(index + 1, right.to_vec());

    debug_assert_eq!(original.concat(), morphed.concat());

    assert_ne!(
        framed_state::<Blake2bTranscript>(&original),
        framed_state::<Blake2bTranscript>(&morphed),
        "Blake2b framed absorption is not boundary-injective"
    );
    assert_ne!(
        framed_state::<KeccakTranscript>(&original),
        framed_state::<KeccakTranscript>(&morphed),
        "Keccak framed absorption is not boundary-injective"
    );
    assert_ne!(
        framed_state::<PoseidonTranscript>(&original),
        framed_state::<PoseidonTranscript>(&morphed),
        "Poseidon framed absorption is not boundary-injective"
    );
});
