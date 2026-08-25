use crate::{
    field::JoltField,
    utils::{index_to_field_bitvector, interleave_bits, lookup_bits::LookupBits},
    zkvm::lookup_table::{
        prefixes::{PrefixCheckpoint, Prefixes},
        suffixes::SuffixEval,
        PrefixSuffixDecomposition,
    },
};
use common::constants::XLEN;
use num::Integer;
use rand::prelude::*;
use strum::{EnumCount, IntoEnumIterator};

use super::JoltLookupTable;

pub fn lookup_table_mle_random_test<F: JoltField, T: JoltLookupTable + Default>() {
    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..1000 {
        let index = rng.gen();
        assert_eq!(
            F::from_u64(T::default().materialize_entry(index)),
            T::default().evaluate_mle::<F, F>(&index_to_field_bitvector(index, XLEN * 2)),
            "MLE did not match materialized table at index {index}",
        );
    }
}

pub fn lookup_table_mle_full_hypercube_test<F: JoltField, T: JoltLookupTable + Default>() {
    let materialized = T::default().materialize();
    for (i, entry) in materialized.iter().enumerate() {
        assert_eq!(
            F::from_u64(*entry),
            T::default().evaluate_mle::<F, F>(&index_to_field_bitvector(i as u128, 16)),
            "MLE did not match materialized table at index {i}",
        );
    }
}

/// Generates a lookup index where right operand is 111..000
pub fn gen_bitmask_lookup_index(rng: &mut StdRng) -> u128 {
    let x = rng.next_u64();
    let zeros = rng.gen_range(0..=XLEN);
    let y = (!0u64).wrapping_shl(zeros as u32);
    interleave_bits(x, y)
}

pub fn gen_bitmask_w_lookup_index(rng: &mut StdRng) -> u128 {
    let x = rng.next_u64();
    let shift = rng.gen_range(0..XLEN / 2);
    let y = ((1u128 << (XLEN / 2)) - (1u128 << shift)) as u64;
    interleave_bits(x, y)
}

pub fn prefix_suffix_test<const XLEN: usize, F: JoltField, T: PrefixSuffixDecomposition<XLEN>>() {
    prefix_suffix_test_with_phase_size::<XLEN, F, T>(16, 300);
}

/// Same check with a caller-chosen phase size. Tables whose prefixes read
/// specific low index bits use this to pin phase-boundary-agnostic behavior,
/// placing boundaries inside those bits (e.g. `rounds_per_phase = 2`).
pub fn prefix_suffix_test_with_phase_size<
    const XLEN: usize,
    F: JoltField,
    T: PrefixSuffixDecomposition<XLEN>,
>(
    rounds_per_phase: usize,
    num_runs: usize,
) {
    // Odd sizes would split an (r_x, r_y) pair across phases, which the
    // prefix machinery does not support.
    assert!(rounds_per_phase.is_multiple_of(2));
    assert!((XLEN * 2).is_multiple_of(rounds_per_phase));
    let total_phases: usize = XLEN * 2 / rounds_per_phase;
    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..num_runs {
        let mut prefix_checkpoints: Vec<PrefixCheckpoint<F>> = vec![None.into(); Prefixes::COUNT];
        let lookup_index = T::random_lookup_index(&mut rng);
        let mut j = 0;
        let mut r: Vec<F> = vec![];
        for phase in 0..total_phases {
            let suffix_len = (total_phases - 1 - phase) * rounds_per_phase;
            let (mut prefix_bits, suffix_bits) =
                LookupBits::new(lookup_index, XLEN * 2 - phase * rounds_per_phase)
                    .split(suffix_len);

            let suffix_evals: Vec<_> = T::default()
                .suffixes()
                .iter()
                .map(|suffix| SuffixEval::from(F::from_u64(suffix.suffix_mle::<XLEN>(suffix_bits))))
                .collect();

            for _ in 0..rounds_per_phase {
                let mut eval_point = r.clone();
                let c = if rng.next_u64().is_even() { 0 } else { 2 };
                eval_point.push(F::from_u32(c));
                prefix_bits.pop_msb();

                eval_point
                    .extend(index_to_field_bitvector(prefix_bits.into(), prefix_bits.len()).iter());
                eval_point
                    .extend(index_to_field_bitvector(suffix_bits.into(), suffix_bits.len()).iter());

                let mle_eval = T::default().evaluate_mle(&eval_point);

                let r_x = if j % 2 == 1 {
                    Some(*r.last().unwrap())
                } else {
                    None
                };

                let prefix_evals: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F>(&prefix_checkpoints, r_x, c, prefix_bits, j)
                    })
                    .collect();

                let combined = T::default().combine(&prefix_evals, &suffix_evals);
                if combined != mle_eval {
                    println!("Lookup index: {lookup_index}");
                    println!("{j} {prefix_bits} {suffix_bits}");
                    for (i, x) in prefix_evals.iter().enumerate() {
                        println!("prefix_evals[{i}] = {x}");
                    }
                    for (i, x) in suffix_evals.iter().enumerate() {
                        println!("suffix_evals[{i}] = {x}");
                    }
                }

                assert_eq!(combined, mle_eval);
                r.push(F::from_u64(rng.next_u64()));

                if r.len().is_multiple_of(2) {
                    Prefixes::update_checkpoints::<XLEN, F, F>(
                        &mut prefix_checkpoints,
                        r[r.len() - 2],
                        r[r.len() - 1],
                        j,
                        suffix_len,
                    );
                }

                j += 1;
            }
        }
    }
}
