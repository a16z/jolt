use jolt_field::Field;
use rand::prelude::*;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::interleave::interleave_bits;
use crate::lookup_bits::LookupBits;
use crate::tables::prefixes::{PrefixCheckpoint, Prefixes, NUM_PREFIXES};
use crate::tables::suffixes::SuffixEval;
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

/// Convert an integer to a vector of field elements representing its binary decomposition.
///
/// Returns MSB-first: index 0 is the highest bit.
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

/// Generate a lookup index where the right operand is a bitmask of the form `111...000`.
///
/// Used by shift/rotate tables whose inputs must have this structure.
pub fn gen_bitmask_lookup_index<const XLEN: usize>(rng: &mut StdRng) -> u128 {
    let x = rng.next_u64();
    let zeros = rng.gen_range(0..=XLEN);
    let y = (!0u64).wrapping_shl(zeros as u32);
    interleave_bits(x, y)
}

/// Verify that `evaluate_mle` matches `materialize_entry` at 1000 random points.
///
/// Uses the production XLEN (64) challenge points.
pub fn mle_random_test<const XLEN: usize, F, T>()
where
    F: Field + FieldOps<F> + ChallengeOps<F>,
    T: LookupTable<XLEN> + Default,
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

/// Verify that `evaluate_mle` matches `materialize_entry` on the full 2^16 hypercube (XLEN=8).
pub fn mle_full_hypercube_test<F, T>()
where
    F: Field + FieldOps<F> + ChallengeOps<F>,
    T: LookupTable<8> + Default,
{
    let materialized = T::default().materialize();
    for (i, entry) in materialized.iter().enumerate() {
        assert_eq!(
            F::from_u64(*entry),
            T::default().evaluate_mle::<F, F>(&index_to_field_bitvector(i as u128, 16)),
            "MLE did not match materialized table at index {i}",
        );
    }
}

/// Verify the prefix/suffix decomposition matches `evaluate_mle` across all sumcheck rounds.
///
/// For 300 random lookup indices, walks through every sumcheck round and checks that
/// `combine(prefix_evals, suffix_evals) == evaluate_mle(partially_bound_point)`.
/// This is the most comprehensive correctness test for the sparse-dense decomposition.
pub fn prefix_suffix_test<const XLEN: usize, F, T>()
where
    F: Field + FieldOps<F> + ChallengeOps<F>,
    T: PrefixSuffixDecomposition<XLEN>,
{
    const ROUNDS_PER_PHASE: usize = 16;
    let total_phases: usize = XLEN * 2 / ROUNDS_PER_PHASE;
    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..300 {
        let mut prefix_checkpoints: Vec<PrefixCheckpoint<F>> =
            vec![None.into(); NUM_PREFIXES];
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
                .map(|suffix| {
                    SuffixEval::from(F::from_u64(suffix.suffix_mle::<XLEN>(suffix_bits)))
                })
                .collect();

            for _ in 0..ROUNDS_PER_PHASE {
                let mut eval_point = r.clone();
                let c = if rng.next_u64() % 2 == 0 { 0 } else { 2 };
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

                let prefix_evals: Vec<_> = (0..NUM_PREFIXES)
                    .map(|i| {
                        // SAFETY: repr(u8) enum with NUM_PREFIXES contiguous variants
                        let prefix: Prefixes = unsafe { std::mem::transmute(i as u8) };
                        prefix.prefix_mle::<XLEN, F, F>(
                            &prefix_checkpoints,
                            r_x,
                            c,
                            prefix_bits,
                            j,
                        )
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

                if r.len() % 2 == 0 {
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
