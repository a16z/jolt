#![expect(clippy::unwrap_used)]

use jolt_field::Field;
use rand::prelude::*;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::interleave::interleave_bits;
use crate::lookup_bits::LookupBits;
use crate::tables::prefixes::{PrefixCheckpoint, Prefixes, ALL_PREFIXES, NUM_PREFIXES};
use crate::tables::suffixes::SuffixEval;
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::XLEN;

pub fn index_to_field_bitvector<F: Field + ChallengeOps<F>>(value: u128, bits: usize) -> Vec<F> {
    if bits != 128 {
        assert!(value < 1u128 << bits);
    }
    let mut bitvector: Vec<F> = Vec::with_capacity(bits);
    for i in (0..bits).rev() {
        if (value >> i) & 1 == 1 {
            bitvector.push(F::one());
        } else {
            bitvector.push(F::zero());
        }
    }
    bitvector
}

pub fn gen_bitmask_lookup_index(rng: &mut StdRng) -> u128 {
    let x = rng.next_u64();
    let zeros = rng.gen_range(0..=XLEN);
    let y = (!0u64).wrapping_shl(zeros as u32);
    interleave_bits(x, y)
}

pub fn mle_random_test<F, T>()
where
    F: Field + FieldOps<F> + ChallengeOps<F>,
    T: LookupTable + Default,
{
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..1000 {
        let index: u128 = rng.gen();
        assert_eq!(
            F::from_u64(T::default().materialize_entry(index)),
            T::default().evaluate_mle::<F, F>(&index_to_field_bitvector(index, XLEN * 2)),
            "MLE did not match materialized table at index {index}",
        );
    }
}

pub fn prefix_suffix_test<F, T>()
where
    F: Field + FieldOps<F> + ChallengeOps<F>,
    T: PrefixSuffixDecomposition,
{
    const ROUNDS_PER_PHASE: usize = 16;
    let total_phases: usize = XLEN * 2 / ROUNDS_PER_PHASE;
    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..300 {
        let mut prefix_checkpoints: Vec<PrefixCheckpoint<F>> = vec![None.into(); NUM_PREFIXES];
        let lookup_index = T::random_lookup_index(&mut rng);
        let mut j = 0;
        let mut r: Vec<F> = vec![];
        for phase in 0..total_phases {
            let suffix_len = (total_phases - 1 - phase) * ROUNDS_PER_PHASE;
            let (mut prefix_bits, suffix_bits) =
                LookupBits::new(lookup_index, XLEN * 2 - phase * ROUNDS_PER_PHASE)
                    .split(suffix_len);

            let suffix_evals: Vec<_> = T::default()
                .suffixes()
                .iter()
                .map(|suffix| SuffixEval::from(F::from_u64(suffix.suffix_mle(suffix_bits))))
                .collect();

            for _ in 0..ROUNDS_PER_PHASE {
                let mut eval_point = r.clone();
                let c = if rng.next_u64().is_multiple_of(2) {
                    0
                } else {
                    2
                };
                eval_point.push(F::from_u32(c));
                let _ = prefix_bits.pop_msb();

                eval_point.extend(
                    index_to_field_bitvector::<F>(prefix_bits.into(), prefix_bits.len()).iter(),
                );
                eval_point.extend(
                    index_to_field_bitvector::<F>(suffix_bits.into(), suffix_bits.len()).iter(),
                );

                let mle_eval: F = T::default().evaluate_mle(&eval_point);

                let r_x = if j % 2 == 1 {
                    Some(*r.last().unwrap())
                } else {
                    None
                };

                let prefix_evals: Vec<_> = ALL_PREFIXES
                    .iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<F, F>(&prefix_checkpoints, r_x, c, prefix_bits, j)
                    })
                    .collect();

                let combined = T::default().combine(&prefix_evals, &suffix_evals);
                assert_eq!(
                    combined, mle_eval,
                    "prefix/suffix decomposition mismatch at round {j}, \
                     lookup_index={lookup_index}, prefix_bits={prefix_bits}, \
                     suffix_bits={suffix_bits}"
                );

                r.push(F::from_u64(rng.next_u64()));

                if r.len().is_multiple_of(2) {
                    Prefixes::update_checkpoints::<F, F>(
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
