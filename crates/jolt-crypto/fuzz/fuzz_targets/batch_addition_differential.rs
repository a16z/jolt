#![no_main]

//! Differential check of the shared-inversion batch G1 addition against a
//! naive fold-add over the same index sets.
//!
//! Index sets are non-empty and duplicate-free by construction: the batch
//! inversion's documented precondition excludes equal or inverse points
//! within one pair, and duplicate indices over a common base set violate it
//! (the result is documented as silently garbage, not an error).

use std::sync::OnceLock;

use jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi;
use jolt_crypto::{Bn254, Bn254G1};
use libfuzzer_sys::fuzz_target;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const NUM_BASES: usize = 16;
const MAX_SETS: usize = 8;

fn bases() -> &'static Vec<Bn254G1> {
    static BASES: OnceLock<Vec<Bn254G1>> = OnceLock::new();
    BASES.get_or_init(|| {
        let mut rng = ChaCha20Rng::seed_from_u64(0xba7c);
        (0..NUM_BASES).map(|_| Bn254::random_g1(&mut rng)).collect()
    })
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let bases = bases();

    // Each set is a fuzzer-chosen subset of the bases, decoded from a
    // 16-bit membership mask so indices are unique by construction.
    let set_count = (data[0] as usize % MAX_SETS) + 1;
    if data.len() < 1 + set_count * 2 {
        return;
    }
    let mut indices_sets: Vec<Vec<usize>> = Vec::with_capacity(set_count);
    for set in 0..set_count {
        let mask = u16::from_le_bytes(data[1 + set * 2..3 + set * 2].try_into().unwrap());
        let indices: Vec<usize> = (0..NUM_BASES).filter(|i| mask & (1 << i) != 0).collect();
        if indices.is_empty() {
            return;
        }
        indices_sets.push(indices);
    }

    let batched = batch_g1_additions_multi(bases, &indices_sets);
    assert_eq!(batched.len(), indices_sets.len());

    for (set_index, (result, indices)) in batched.iter().zip(&indices_sets).enumerate() {
        let expected = indices
            .iter()
            .skip(1)
            .fold(bases[indices[0]], |acc, &i| acc + bases[i]);
        assert_eq!(
            *result, expected,
            "batch addition disagrees with fold-add for set {set_index}"
        );
    }
});
