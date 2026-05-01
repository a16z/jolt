use jolt_field::Field;
use rand::prelude::*;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::interleave::interleave_bits;
use crate::lookup_bits::LookupBits;
use crate::tables::prefixes::{PrefixEval, ALL_PREFIXES};
use crate::tables::suffixes::SuffixEval;
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

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

pub fn gen_bitmask_lookup_index<const XLEN: usize>(rng: &mut StdRng) -> u128 {
    let mask = ((1u128 << XLEN) - 1) as u64;
    let x = rng.next_u64() & mask;
    let zeros = rng.gen_range(0..=XLEN);
    let y_full = (!0u64).wrapping_shl(zeros as u32);
    let y = y_full & mask;
    interleave_bits(x, y)
}

/// Verify the MLE of `T` agrees with `materialize_entry` on every point of
/// the boolean hypercube `{0, 1}^(2*XLEN)`.
///
/// Only feasible for small `XLEN` (= 8 in the workspace), where the table
/// has `2^16` entries.
pub fn mle_full_hypercube_test<const XLEN: usize, F, T>()
where
    F: Field + FieldOps<F> + ChallengeOps<F>,
    T: LookupTable + Default,
{
    assert!(
        XLEN <= 8,
        "full hypercube test only feasible for small XLEN"
    );
    let table_bits = 2 * XLEN;
    for index in 0u128..(1u128 << table_bits) {
        assert_eq!(
            F::from_u64(T::default().materialize_entry(index)),
            T::default().evaluate_mle::<F, F>(&index_to_field_bitvector(index, table_bits)),
            "MLE did not match materialized table at index {index}",
        );
    }
}

pub fn mle_random_test<const XLEN: usize, F, T>()
where
    F: Field + FieldOps<F> + ChallengeOps<F>,
    T: LookupTable + Default,
{
    let mut rng = StdRng::seed_from_u64(12345);
    let xlen_mask = if XLEN == 64 {
        u128::MAX
    } else {
        (1u128 << (2 * XLEN)) - 1
    };
    for _ in 0..1000 {
        let raw: u128 = rng.gen();
        let index = raw & xlen_mask;
        assert_eq!(
            F::from_u64(T::default().materialize_entry(index)),
            T::default().evaluate_mle::<F, F>(&index_to_field_bitvector(index, XLEN * 2)),
            "MLE did not match materialized table at index {index}",
        );
    }
}

pub fn prefix_suffix_test<const XLEN: usize, F, T>()
where
    F: Field + FieldOps<F> + ChallengeOps<F>,
    T: PrefixSuffixDecomposition<XLEN>,
{
    const ROUNDS_PER_PHASE: usize = 16;
    let total_phases: usize = XLEN * 2 / ROUNDS_PER_PHASE;
    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..300 {
        let lookup_index = T::random_lookup_index(&mut rng);

        let mut checkpoints: Vec<PrefixEval<F>> = ALL_PREFIXES
            .iter()
            .map(|p| p.default_checkpoint::<F>())
            .collect();

        for phase in 0..total_phases {
            let suffix_len = (total_phases - 1 - phase) * ROUNDS_PER_PHASE;
            let full_bits = LookupBits::new(lookup_index, XLEN * 2);
            let (prefix_bits, suffix_bits) = full_bits.split(suffix_len);
            let (_, phase_bits) = prefix_bits.split(ROUNDS_PER_PHASE);

            let suffix_evals: Vec<_> = T::default()
                .suffixes()
                .iter()
                .map(|suffix| SuffixEval::from(F::from_u64(suffix.suffix_mle(suffix_bits))))
                .collect();

            let prefix_evals: Vec<PrefixEval<F>> = ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.evaluate(&checkpoints, phase_bits, suffix_len))
                .collect();

            let combined: F = T::default().combine(&prefix_evals, &suffix_evals);

            if phase == total_phases - 1 {
                let expected = F::from_u64(T::default().materialize_entry(lookup_index));
                assert_eq!(
                    combined, expected,
                    "prefix/suffix decomposition mismatch at final phase, \
                     lookup_index={lookup_index:#x}"
                );
            }

            checkpoints = prefix_evals;
        }
    }
}
