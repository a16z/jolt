use jolt_field::Field;
use rand::prelude::*;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::interleave::interleave_bits;
use crate::lookup_bits::LookupBits;
use crate::tables::prefixes::{PrefixEval, ALL_PREFIXES};
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
        let lookup_index = T::random_lookup_index(&mut rng);

        // Evaluate at pure binary points: iterate over phases,
        // using the actual lookup index bits for each phase.
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

            // At binary points, the full MLE should equal materialize_entry
            if phase == total_phases - 1 {
                let expected = F::from_u64(T::default().materialize_entry(lookup_index));
                assert_eq!(
                    combined, expected,
                    "prefix/suffix decomposition mismatch at final phase, \
                     lookup_index={lookup_index:#x}"
                );
            }

            // Update checkpoints: at binary points, the checkpoint for each prefix
            // becomes its evaluated value from this phase.
            checkpoints = prefix_evals;
        }
    }
}
