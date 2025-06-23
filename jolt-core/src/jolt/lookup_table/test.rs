use crate::{
    field::JoltField,
    jolt::lookup_table::{
        prefixes::{PrefixCheckpoint, Prefixes},
        suffixes::SuffixEval,
        PrefixSuffixDecomposition,
    },
    subprotocols::sparse_dense_shout::LookupBits,
    utils::index_to_field_bitvector,
};
use num::Integer;
use rand::prelude::*;
use strum::{EnumCount, IntoEnumIterator};

use super::JoltLookupTable;

pub fn lookup_table_mle_random_test<F: JoltField, T: JoltLookupTable + Default>() {
    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..1000 {
        let index = rng.next_u64();
        assert_eq!(
            F::from_u64(T::default().materialize_entry(index)),
            T::default().evaluate_mle(&index_to_field_bitvector(index, 64)),
            "MLE did not match materialized table at index {index}",
        );
    }
}

pub fn lookup_table_mle_full_hypercube_test<F: JoltField, T: JoltLookupTable + Default>() {
    let materialized = T::default().materialize();
    for (i, entry) in materialized.iter().enumerate() {
        assert_eq!(
            F::from_u64(*entry),
            T::default().evaluate_mle(&index_to_field_bitvector(i as u64, 16)),
            "MLE did not match materialized table at index {i}",
        );
    }
}

/// Generates a lookup index where right operand is 111..000
pub fn gen_bitmask_lookup_index(rng: &mut rand::rngs::StdRng) -> u64 {
    let x = rng.gen::<u32>();

    let zeros = rng.gen_range(0, 33);
    let y = if zeros >= 32 { 0 } else { u32::MAX << zeros };

    let mut result = 0u64;
    for i in 0..32 {
        let yi = ((x >> i) & 1) as u64;
        let xi = ((y >> i) & 1) as u64;

        result |= xi << (2 * i);
        result |= yi << (2 * i + 1);
    }

    result
}

pub fn prefix_suffix_test<F: JoltField, T: PrefixSuffixDecomposition<32>>() {
    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..1000 {
        let mut prefix_checkpoints: Vec<PrefixCheckpoint<F>> = vec![None.into(); Prefixes::COUNT];
        let lookup_index = T::random_lookup_index(&mut rng);
        let mut j = 0;
        let mut r: Vec<F> = vec![];
        for phase in 0..4 {
            let suffix_len = (3 - phase) * 16;
            let (mut prefix_bits, suffix_bits) =
                LookupBits::new(lookup_index, 64 - phase * 16).split(suffix_len);

            let suffix_evals: Vec<_> = T::default()
                .suffixes()
                .iter()
                .map(|suffix| SuffixEval::from(F::from_u32(suffix.suffix_mle::<32>(suffix_bits))))
                .collect();

            for _ in 0..16 {
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
                        prefix.prefix_mle::<32, F>(&prefix_checkpoints, r_x, c, prefix_bits, j)
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

                if r.len() % 2 == 0 {
                    Prefixes::update_checkpoints::<32, F>(
                        &mut prefix_checkpoints,
                        r[r.len() - 2],
                        r[r.len() - 1],
                        j,
                    );
                }

                j += 1;
            }
        }
    }
}
