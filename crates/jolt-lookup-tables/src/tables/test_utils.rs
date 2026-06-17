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

/// Validate the materialization flow used by the sparse-dense sumcheck prover:
/// per phase, each prefix is materialized as a dense table of `2^RPP` binary
/// evaluations, bound HighToLow with random field challenges across the phase's
/// rounds, and its fully-bound value becomes the checkpoint for the next phase.
/// Every round checks `combine(prefixes, suffixes) == table MLE` at the
/// corresponding (partially random) evaluation point.
pub fn prefix_suffix_test<const XLEN: usize, F, T>()
where
    F: Field + FieldOps<F> + ChallengeOps<F>,
    T: PrefixSuffixDecomposition<XLEN>,
{
    prefix_suffix_materialization_test::<XLEN, F, T>(16, 3);
    prefix_suffix_materialization_test::<XLEN, F, T>(8, 6);
}

fn prefix_suffix_materialization_test<const XLEN: usize, F, T>(
    rounds_per_phase: usize,
    num_runs: usize,
) where
    F: Field + FieldOps<F> + ChallengeOps<F>,
    T: PrefixSuffixDecomposition<XLEN>,
{
    let total_bits = XLEN * 2;
    assert_eq!(total_bits % rounds_per_phase, 0);
    let total_phases = total_bits / rounds_per_phase;
    let mut rng = StdRng::seed_from_u64(12345);

    for _ in 0..num_runs {
        let lookup_index = T::random_lookup_index(&mut rng);
        let full_bits = LookupBits::new(lookup_index, total_bits);

        let mut checkpoints: Vec<PrefixEval<F>> = ALL_PREFIXES
            .iter()
            .map(|p| p.default_checkpoint::<F>())
            .collect();
        let mut r: Vec<F> = Vec::with_capacity(total_bits);

        for phase in 0..total_phases {
            let suffix_len = total_bits - (phase + 1) * rounds_per_phase;
            let (prefix_bits, suffix_bits) = full_bits.split(suffix_len);
            let (_, phase_chunk) = prefix_bits.split(rounds_per_phase);
            let chunk: usize = phase_chunk.into();

            let mut tables: Vec<Vec<F>> = ALL_PREFIXES
                .iter()
                .map(|prefix| {
                    (0..1usize << rounds_per_phase)
                        .map(|x| {
                            prefix
                                .evaluate::<F>(
                                    &checkpoints,
                                    LookupBits::new(x as u128, rounds_per_phase),
                                    suffix_len,
                                )
                                .value()
                        })
                        .collect()
                })
                .collect();

            let suffix_evals: Vec<SuffixEval<F>> = T::default()
                .suffixes()
                .iter()
                .map(|suffix| SuffixEval::from(F::from_u64(suffix.suffix_mle(suffix_bits))))
                .collect();

            for round in 0..rounds_per_phase {
                let half = 1usize << (rounds_per_phase - round - 1);
                let remaining_bits = rounds_per_phase - round - 1;
                let b = chunk & ((1usize << remaining_bits) - 1);

                for c in [0u32, 2u32] {
                    let prefix_evals: Vec<PrefixEval<F>> = tables
                        .iter()
                        .map(|tbl| {
                            let eval = if c == 0 {
                                tbl[b]
                            } else {
                                tbl[b + half] + tbl[b + half] - tbl[b]
                            };
                            PrefixEval::from(eval)
                        })
                        .collect();

                    let mut eval_point = r.clone();
                    eval_point.push(F::from_u32(c));
                    eval_point.extend(index_to_field_bitvector::<F>(b as u128, remaining_bits));
                    eval_point.extend(index_to_field_bitvector::<F>(
                        suffix_bits.into(),
                        suffix_len,
                    ));

                    let combined: F = T::default().combine(&prefix_evals, &suffix_evals);
                    let expected: F = T::default().evaluate_mle(&eval_point);
                    assert_eq!(
                        combined,
                        expected,
                        "combine != MLE: rounds_per_phase={rounds_per_phase} phase={phase} \
                         round={round} c={c} lookup_index={lookup_index:#x}\n{}",
                        format_prefix_evals(&prefix_evals),
                    );
                }

                let r_round = F::from_u64(rng.next_u64());
                r.push(r_round);
                for tbl in &mut tables {
                    let (lo, hi) = tbl.split_at_mut(half);
                    for (lo_i, hi_i) in lo.iter_mut().zip(hi.iter()) {
                        *lo_i += r_round * (*hi_i - *lo_i);
                    }
                    tbl.truncate(half);
                }
            }

            for (checkpoint, tbl) in checkpoints.iter_mut().zip(&tables) {
                *checkpoint = PrefixEval::from(tbl[0]);
            }
        }

        let suffix_evals: Vec<SuffixEval<F>> = T::default()
            .suffixes()
            .iter()
            .map(|suffix| SuffixEval::from(F::from_u64(suffix.suffix_mle(LookupBits::new(0, 0)))))
            .collect();
        let combined: F = T::default().combine(&checkpoints, &suffix_evals);
        let expected: F = T::default().evaluate_mle(&r);
        assert_eq!(
            combined,
            expected,
            "fully-bound combine != MLE: rounds_per_phase={rounds_per_phase} \
             lookup_index={lookup_index:#x}\n{}",
            format_prefix_evals(&checkpoints),
        );
    }
}

fn format_prefix_evals<F: Field>(evals: &[PrefixEval<F>]) -> String {
    use std::fmt::Write;

    let mut out = String::new();
    for (prefix, eval) in ALL_PREFIXES.iter().zip(evals) {
        let _ = writeln!(out, "  {prefix:?} = {eval}");
    }
    out
}
