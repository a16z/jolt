use super::test_helpers::inner_ajtai_multi_chunk_t_only;
use super::*;
use crate::DensePoly;
use akita_field::RandomSampling;
use akita_field::{Fp64, FpExt4, Prime128Offset275, Prime24Offset3, Prime32Offset99};
use akita_types::FlatMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn aggregate_witnesses<F: FieldCore, const D: usize>(
    witnesses: &[DecomposeFoldWitness<F, D>],
) -> DecomposeFoldWitness<F, D> {
    let mut acc = witnesses[0].clone();
    for witness in &witnesses[1..] {
        for (dst, src) in acc
            .z_folded_rings
            .iter_mut()
            .zip(witness.z_folded_rings.iter())
        {
            *dst += *src;
        }
        for (dst, src) in acc
            .centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs.iter())
        {
            for k in 0..D {
                dst[k] += src[k];
            }
        }
    }
    acc.centered_inf_norm = acc
        .centered_coeffs
        .iter()
        .flat_map(|coeffs| coeffs.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);
    acc
}

fn materialize_onehot_as_dense<F, const D: usize, I>(poly: &OneHotPoly<F, D, I>) -> DensePoly<F, D>
where
    F: FieldCore + CanonicalField,
    I: OneHotIndex,
{
    let mut coeffs = vec![CyclotomicRing::<F, D>::zero(); poly.total_ring_elems];
    for (chunk_idx, hot_idx) in poly.indices.iter().copied().enumerate() {
        let Some(raw) = hot_idx else {
            continue;
        };
        let field_pos = chunk_idx * poly.onehot_k + raw.as_usize();
        let ring_idx = field_pos / D;
        let coeff_idx = field_pos % D;
        coeffs[ring_idx].coeffs[coeff_idx] += F::one();
    }
    DensePoly::<F, D>::from_ring_coeffs(coeffs)
}

fn test_ring_scalar<F, const D: usize>(seed: u64) -> CyclotomicRing<F, D>
where
    F: CanonicalField,
{
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_canonical_u128_reduced(u128::from(seed + idx as u64 + 1))
    }))
}

// -------------------------------------------------------------------------
// Tests for the flat-storage mapping helpers and the sparse inner-Ajtai
// reference implementation. Originally in `commitment/onehot.rs`.
// -------------------------------------------------------------------------

#[test]
fn map_onehot_k_gt_d() {
    // K=16, D=4, T=2 chunks => 32 field elements => 8 ring elements
    // block_len=4 => 2 blocks of 4 ring elements each.
    let k = 16;
    let d = 4;
    let indices: Vec<Option<usize>> = vec![Some(3), Some(10)];
    let num_blocks = 2;
    let blocks =
        FlatBlocks::<SingleChunkEntry>::from_indices(k, &indices, 4, d, num_blocks).unwrap();

    assert_eq!(blocks.num_blocks(), 2);
    let total_entries: usize = (0..blocks.num_blocks())
        .map(|i| blocks.block(i).len())
        .sum();
    assert_eq!(total_entries, 2, "T=2 nonzero ring elements");

    let block0 = blocks.block(0);
    assert_eq!(block0.len(), 1);
    assert_eq!(block0[0].pos_in_block(), 0);
    assert_eq!(block0[0].coeff_idx(), 3);

    let block1 = blocks.block(1);
    assert_eq!(block1.len(), 1);
    assert_eq!(block1[0].pos_in_block(), 2);
    assert_eq!(block1[0].coeff_idx(), 2);
}

#[test]
fn map_onehot_k_eq_d() {
    // K=4, D=4, T=4 chunks => 16 field elements => 4 ring elements
    // block_len=2 => 2 blocks of 2 ring elements each.
    let k = 4;
    let d = 4;
    let indices: Vec<Option<usize>> = vec![Some(0), Some(2), Some(3), Some(1)];
    let num_blocks = 2;
    let blocks =
        FlatBlocks::<SingleChunkEntry>::from_indices(k, &indices, 2, d, num_blocks).unwrap();

    assert_eq!(blocks.num_blocks(), 2);
    let total_entries: usize = (0..blocks.num_blocks())
        .map(|i| blocks.block(i).len())
        .sum();
    assert_eq!(total_entries, 4, "K=D => every ring element is nonzero");

    let block0 = blocks.block(0);
    assert_eq!(block0.len(), 2);
    assert_eq!(block0[0].pos_in_block(), 0);
    assert_eq!(block0[0].coeff_idx(), 0);
    assert_eq!(block0[1].pos_in_block(), 1);
    assert_eq!(block0[1].coeff_idx(), 2);

    let block1 = blocks.block(1);
    assert_eq!(block1.len(), 2);
    assert_eq!(block1[0].pos_in_block(), 0);
    assert_eq!(block1[0].coeff_idx(), 3);
    assert_eq!(block1[1].pos_in_block(), 1);
    assert_eq!(block1[1].coeff_idx(), 1);
}

#[test]
fn map_onehot_k_lt_d() {
    // K=4, D=8, T=8 chunks => 32 field elements => 4 ring elements
    // block_len=2 => 2 blocks of 2 ring elements each.
    let k = 4;
    let d = 8;
    let indices: Vec<Option<usize>> = vec![
        Some(0),
        Some(2),
        Some(3),
        Some(1),
        Some(0),
        Some(0),
        Some(3),
        Some(3),
    ];
    let num_blocks = 2;
    let blocks =
        FlatBlocks::<MultiChunkEntry>::from_indices(k, &indices, 2, d, num_blocks).unwrap();

    assert_eq!(blocks.num_blocks(), 2);
    let total_entries: usize = (0..blocks.num_blocks())
        .map(|i| blocks.block(i).len())
        .sum();
    assert_eq!(total_entries, 4, "D>K => all ring elements nonzero");

    let block0 = blocks.block(0);
    assert_eq!(block0.len(), 2);
    assert_eq!(block0[0].pos_in_block(), 0);
    assert_eq!(block0[0].nonzero_coeffs(), &[0, 6]);
    assert_eq!(block0[1].pos_in_block(), 1);
    assert_eq!(block0[1].nonzero_coeffs(), &[3, 5]);

    let block1 = blocks.block(1);
    assert_eq!(block1.len(), 2);
    assert_eq!(block1[0].pos_in_block(), 0);
    assert_eq!(block1[0].nonzero_coeffs(), &[0, 4]);
    assert_eq!(block1[1].pos_in_block(), 1);
    assert_eq!(block1[1].nonzero_coeffs(), &[3, 7]);
}

#[test]
#[should_panic(expected = "FlatBlocks::block: block index 1 out of range for 1 blocks")]
fn flat_blocks_block_panics_on_out_of_range_index() {
    let blocks = super::test_helpers::from_buckets(vec![vec![1u16]]);
    let _ = blocks.block(1);
}

#[cfg(feature = "parallel")]
#[test]
fn onehot_accumulate_skips_empty_tail_partitions() {
    const D: usize = 4;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let accum = pool
        .install(|| super::accumulate::onehot_accumulate::<SingleChunkEntry, D>(&[], &[], 0, 5));

    assert_eq!(accum, vec![[0i32; D]; 5]);
}

#[test]
fn onehot_poly_rejects_non_divisible_k_d() {
    // K=3 and D=4: neither divides the other. `OneHotPoly::new` must
    // refuse to construct. The nicely-matched K/D invariant is what
    // lets `FlatBlocks::from_{single,multi}_chunk_onehot` skip their
    // own K/D check; this test pins the upstream guard that enforces
    // it.
    type F = Prime24Offset3;
    const D: usize = 4;
    let result = OneHotPoly::<F, D>::new(3, vec![Some(0usize), Some(1)]);
    assert!(result.is_err());
}

#[test]
fn tensor_column_partials_match_dense_reference() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let poly = OneHotPoly::<F, D>::new(
        8,
        vec![
            Some(0usize),
            Some(7),
            None,
            Some(3),
            Some(5),
            Some(1),
            None,
            Some(6),
        ],
    )
    .unwrap();
    let dense = materialize_onehot_as_dense(&poly);
    let point = (0..poly.num_vars())
        .map(|idx| {
            E::from_base_slice(&[
                F::from_canonical_u128_reduced(3 * idx as u128 + 2),
                F::from_canonical_u128_reduced(3 * idx as u128 + 3),
                F::from_canonical_u128_reduced(3 * idx as u128 + 5),
                F::from_canonical_u128_reduced(3 * idx as u128 + 7),
            ])
        })
        .collect::<Vec<_>>();

    let sparse_partials = poly.tensor_extension_column_partials::<E>(&point).unwrap();
    let dense_partials = dense.tensor_extension_column_partials::<E>(&point).unwrap();
    assert_eq!(sparse_partials, dense_partials);
}

#[test]
fn batched_tensor_column_partials_match_individual() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let polys = [
        OneHotPoly::<F, D>::new(
            8,
            vec![
                Some(0usize),
                Some(7),
                None,
                Some(3),
                Some(5),
                Some(1),
                None,
                Some(6),
            ],
        )
        .unwrap(),
        OneHotPoly::<F, D>::new(
            8,
            vec![
                Some(4usize),
                Some(2),
                Some(7),
                None,
                Some(1),
                None,
                Some(5),
                Some(0),
            ],
        )
        .unwrap(),
    ];
    let point = (0..polys[0].num_vars())
        .map(|idx| {
            E::from_base_slice(&[
                F::from_canonical_u128_reduced(5 * idx as u128 + 2),
                F::from_canonical_u128_reduced(5 * idx as u128 + 3),
                F::from_canonical_u128_reduced(5 * idx as u128 + 5),
                F::from_canonical_u128_reduced(5 * idx as u128 + 7),
            ])
        })
        .collect::<Vec<_>>();
    let expected = polys
        .iter()
        .map(|poly| poly.tensor_extension_column_partials::<E>(&point).unwrap())
        .collect::<Vec<_>>();
    let poly_refs = polys.iter().collect::<Vec<_>>();
    let got =
        <OneHotPoly<F, D> as AkitaPolyOps<F, D>>::tensor_extension_column_partials_batch::<E>(
            &poly_refs, &point,
        )
        .unwrap();

    assert_eq!(got, expected);
}

/// Exercises the factorized sparse fast path across *multiple outer blocks*
/// (`num_vars - low_vars` exceeds the inner-bit cap, so the high weights are
/// genuinely split into more than one outer block) and on a power-of-two
/// `onehot_k`. The batched sparse partials must be byte-identical both to the
/// dense reference and to the per-poly path.
#[test]
fn batched_tensor_column_partials_multi_block_match_dense() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;
    const ONEHOT_K: usize = 8;
    // hi_vars = NUM_VARS - log2(ONEHOT_K) = 18 - 3 = 15 > inner-bit cap, so the
    // factorization produces several outer blocks.
    const NUM_VARS: usize = 18;

    let num_chunks = (1usize << NUM_VARS) / ONEHOT_K;
    let make_indices = |seed: usize| {
        (0..num_chunks)
            .map(|c| {
                let h = c
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503));
                if h % 7 == 0 {
                    None
                } else {
                    Some(h % ONEHOT_K)
                }
            })
            .collect::<Vec<Option<usize>>>()
    };
    let polys = [
        OneHotPoly::<F, D>::new(ONEHOT_K, make_indices(1)).unwrap(),
        OneHotPoly::<F, D>::new(ONEHOT_K, make_indices(2)).unwrap(),
        OneHotPoly::<F, D>::new(ONEHOT_K, make_indices(3)).unwrap(),
    ];
    let point = (0..NUM_VARS)
        .map(|idx| {
            E::from_base_slice(&[
                F::from_canonical_u128_reduced(7 * idx as u128 + 1),
                F::from_canonical_u128_reduced(7 * idx as u128 + 2),
                F::from_canonical_u128_reduced(7 * idx as u128 + 4),
                F::from_canonical_u128_reduced(7 * idx as u128 + 6),
            ])
        })
        .collect::<Vec<_>>();

    let dense_expected = polys
        .iter()
        .map(|poly| {
            materialize_onehot_as_dense(poly)
                .tensor_extension_column_partials::<E>(&point)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let individual = polys
        .iter()
        .map(|poly| poly.tensor_extension_column_partials::<E>(&point).unwrap())
        .collect::<Vec<_>>();
    let poly_refs = polys.iter().collect::<Vec<_>>();
    let batched =
        <OneHotPoly<F, D> as AkitaPolyOps<F, D>>::tensor_extension_column_partials_batch::<E>(
            &poly_refs, &point,
        )
        .unwrap();

    assert_eq!(batched, dense_expected);
    assert_eq!(batched, individual);
}

#[test]
fn batched_tensor_column_partials_match_dense_for_fp_ext4() {
    type F = Prime32Offset99;
    type E = FpExt4<F>;
    const D: usize = 32;
    const ONEHOT_K: usize = 16;
    const NUM_VARS: usize = 10;

    let num_chunks = (1usize << NUM_VARS) / ONEHOT_K;
    let make_indices = |seed: usize| {
        (0..num_chunks)
            .map(|chunk| {
                let h = chunk
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add(seed.wrapping_mul(12_345));
                if h % 11 == 0 {
                    None
                } else {
                    Some(h % ONEHOT_K)
                }
            })
            .collect::<Vec<Option<usize>>>()
    };
    let polys = [
        OneHotPoly::<F, D>::new(ONEHOT_K, make_indices(1)).unwrap(),
        OneHotPoly::<F, D>::new(ONEHOT_K, make_indices(2)).unwrap(),
        OneHotPoly::<F, D>::new(ONEHOT_K, make_indices(3)).unwrap(),
    ];
    let point = (0..NUM_VARS)
        .map(|idx| {
            E::from_base_slice(&[
                F::from_canonical_u128_reduced(7 * idx as u128 + 1),
                F::from_canonical_u128_reduced(7 * idx as u128 + 2),
                F::from_canonical_u128_reduced(7 * idx as u128 + 4),
                F::from_canonical_u128_reduced(7 * idx as u128 + 8),
            ])
        })
        .collect::<Vec<_>>();

    let dense_expected = polys
        .iter()
        .map(|poly| {
            materialize_onehot_as_dense(poly)
                .tensor_extension_column_partials::<E>(&point)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let poly_refs = polys.iter().collect::<Vec<_>>();
    let batched =
        <OneHotPoly<F, D> as AkitaPolyOps<F, D>>::tensor_extension_column_partials_batch::<E>(
            &poly_refs, &point,
        )
        .unwrap();

    assert_eq!(batched, dense_expected);
}

#[test]
fn tensor_packed_sparse_linear_combination_matches_individual_witnesses() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let polys = [
        OneHotPoly::<F, D>::new(
            8,
            vec![
                Some(0usize),
                Some(7),
                None,
                Some(3),
                Some(5),
                Some(1),
                None,
                Some(6),
            ],
        )
        .unwrap(),
        OneHotPoly::<F, D>::new(
            8,
            vec![
                Some(4usize),
                Some(2),
                Some(7),
                None,
                Some(1),
                None,
                Some(5),
                Some(0),
            ],
        )
        .unwrap(),
    ];
    let coeffs = vec![
        E::from_base_slice(&[
            F::from_canonical_u128_reduced(3),
            F::from_canonical_u128_reduced(5),
            F::from_canonical_u128_reduced(7),
            F::from_canonical_u128_reduced(11),
        ]),
        E::from_base_slice(&[
            F::from_canonical_u128_reduced(13),
            F::from_canonical_u128_reduced(17),
            F::from_canonical_u128_reduced(19),
            F::from_canonical_u128_reduced(23),
        ]),
    ];
    let witnesses = polys
        .iter()
        .map(|poly| {
            poly.tensor_packed_extension_sparse_evals::<E>()
                .unwrap()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let expected =
        SparseExtensionOpeningWitness::linear_combination(coeffs.iter().copied().zip(&witnesses))
            .unwrap();
    let poly_refs = polys.iter().collect::<Vec<_>>();
    let got =
        <OneHotPoly<F, D> as AkitaPolyOps<F, D>>::tensor_packed_extension_sparse_linear_combination::<E>(
            &poly_refs,
            &coeffs,
        )
        .unwrap()
        .unwrap();

    assert_eq!(got.table_len(), expected.table_len());
    assert_eq!(got.entries(), expected.entries());
}

/// Diagnostic for the EOR `np = 1` plateau: dump the within-chunk hot-position
/// distribution (`raw >> lw`, equivalently `tail % stride`) read off a *real*
/// tensor-packed witness, and show the fold plateau is `log2(stride)` rounds
/// long *regardless* of how that distribution looks.
///
/// The hot positions are uniformly spread (random `raw`, exactly like
/// `examples/profile/workload.rs`: seed `0xbeef_cafe`, `gen_range(0..onehot_k)`,
/// every chunk active), yet `entries_len` is provably flat for `log2(stride)`
/// rounds. So the plateau comes from the one-entry-per-power-of-two-window
/// layout, not from any "alignment" of the hot positions.
///
/// `onehot_k = 256` and `width = [E:F] = 4` reproduce the `fp32 onehot_d32`
/// shape (`stride = 64`, expected plateau `log2(64) = 6`). The arity is
/// downscaled (2^14 chunks) purely for test speed; the per-chunk structure is
/// identical to the profiled run.
///
/// See the histogram with:
///   cargo test -p akita-prover np1_offset_distribution_and_plateau -- --nocapture
#[test]
fn np1_offset_distribution_and_plateau() {
    use rand::Rng;
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let onehot_k = 256usize;
    let log_chunks = 14usize;
    let num_chunks = 1usize << log_chunks;

    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let indices: Vec<Option<usize>> = (0..num_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k)))
        .collect();
    let poly = OneHotPoly::<F, D>::new(onehot_k, indices).unwrap();

    let witness = poly.tensor_packed_sparse_witness::<E>().unwrap();

    let (lw, width) = akita_types::tensor_opening_split::<F, E>().unwrap();
    assert!(onehot_k.is_multiple_of(width));
    let stride = onehot_k / width;
    assert!(stride.is_power_of_two() && stride >= 2);
    let s = stride.trailing_zeros() as usize;

    // (a) offset = raw >> lw, recovered from the real witness as tail % stride
    //     because tail = chunk_idx * stride + (raw >> lw).
    let tails: Vec<usize> = witness.entries().iter().map(|&(t, _)| t).collect();
    assert_eq!(
        tails.len(),
        num_chunks,
        "all chunks active => one entry each"
    );
    let mut hist = vec![0usize; stride];
    for &t in &tails {
        hist[t % stride] += 1;
    }
    let occupied = hist.iter().filter(|&&c| c > 0).count();
    let min = *hist.iter().min().unwrap();
    let max = *hist.iter().max().unwrap();
    eprintln!(
        "np=1 offset (raw>>lw) distribution: onehot_k={onehot_k} width={width} lw={lw} \
         stride={stride} entries={} occupied_buckets={occupied}/{stride} \
         per-bucket min={min} max={max} mean={}",
        tails.len(),
        tails.len() / stride,
    );
    eprintln!("  histogram[offset 0..{stride}] = {hist:?}");
    assert!(
        occupied > stride / 2,
        "hot positions are spread, not aligned (occupied {occupied}/{stride})"
    );

    // (b) entries_len after r folds == #distinct(tail >> r): flat for r=0..=s,
    //     then halves at r=s+1 — independent of the spread distribution above.
    let distinct_after = |r: usize| -> usize {
        let mut v: Vec<usize> = tails.iter().map(|&t| t >> r).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    eprintln!("plateau (expected entries_len flat for r=0..={s}):");
    for r in 0..=(s + 2) {
        eprintln!(
            "  round r={r:2}: table_len=2^{:<2} entries_len={}",
            log_chunks + s - r,
            distinct_after(r),
        );
    }
    for r in 0..=s {
        assert_eq!(
            distinct_after(r),
            num_chunks,
            "entries_len must stay flat across the log2(stride) plateau (round {r})",
        );
    }
    assert_eq!(
        distinct_after(s + 1),
        num_chunks / 2,
        "first merge halves entries_len exactly one round after the plateau",
    );
}

#[test]
fn wide_matches_reference() {
    type F = Fp64<4294967197>;
    const D: usize = 64;

    let mut rng = StdRng::seed_from_u64(0xdead_beef);
    let n_a = 3;
    let block_len = 4;
    let num_digits = 5;
    let a_matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..n_a)
        .map(|_| {
            (0..block_len * num_digits)
                .map(|_| CyclotomicRing::random(&mut rng))
                .collect()
        })
        .collect();

    let entries = vec![
        MultiChunkEntry::new(0, vec![1u16, 7, 15]),
        MultiChunkEntry::new(2, vec![0u16, 63]),
    ];

    let a_flat_elems: Vec<CyclotomicRing<F, D>> = a_matrix
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let a_flat = FlatMatrix::from_ring_slice(&a_flat_elems);
    let a_view = a_flat.ring_view::<D>(n_a, block_len * num_digits).unwrap();
    let ref_result = inner_ajtai_multi_chunk_t_only(&a_matrix, &entries, num_digits);
    let wide_result = inner_ajtai_wide_onehot(&a_view, &entries, num_digits);

    assert_eq!(ref_result.len(), wide_result.len());
    for (r, w) in ref_result.iter().zip(wide_result.iter()) {
        assert_eq!(r, w, "wide result must match reference");
    }
}

#[test]
fn wide_matches_reference_fp128() {
    type F = Prime128Offset275;
    const D: usize = 64;

    let mut rng = StdRng::seed_from_u64(0xcafe_1234);
    let n_a = 2;
    let block_len = 2;
    let num_digits = 3;
    let a_matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..n_a)
        .map(|_| {
            (0..block_len * num_digits)
                .map(|_| CyclotomicRing::random(&mut rng))
                .collect()
        })
        .collect();

    let entries = vec![
        MultiChunkEntry::new(0, vec![0u16, 5, 32, 63]),
        MultiChunkEntry::new(1, vec![10u16]),
    ];

    let a_flat_elems: Vec<CyclotomicRing<F, D>> = a_matrix
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let a_flat = FlatMatrix::from_ring_slice(&a_flat_elems);
    let a_view = a_flat.ring_view::<D>(n_a, block_len * num_digits).unwrap();
    let ref_result = inner_ajtai_multi_chunk_t_only(&a_matrix, &entries, num_digits);
    let wide_result = inner_ajtai_wide_onehot(&a_view, &entries, num_digits);

    assert_eq!(ref_result.len(), wide_result.len());
    for (r, w) in ref_result.iter().zip(wide_result.iter()) {
        assert_eq!(r, w, "wide result must match reference (Fp128)");
    }
}

// -------------------------------------------------------------------------
// Tests that exercise the column-sweep kernels and the OneHotPoly-level
// behaviour defined above.
// -------------------------------------------------------------------------

#[test]
fn single_chunk_onehot_large_block_uses_safe_accumulator_path() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let block_len = MAX_WIDE_SHIFT_ACCUMULATIONS + 1;
    let max_coeff = F::from_canonical_u128_reduced((1u128 << 24) - 4);
    let dense_ring = CyclotomicRing::from_coefficients([max_coeff; D]);
    let a_matrix = [vec![dense_ring; block_len]];
    let bucket: Vec<SingleChunkEntry> = (0..block_len)
        .map(|pos| SingleChunkEntry::new(pos as u32, (pos % D) as u16))
        .collect();
    let single_chunk_blocks = super::test_helpers::from_buckets(vec![bucket.clone()]);

    let a_flat = FlatMatrix::from_ring_slice(&a_matrix[0]);
    let a_view = a_flat.ring_view::<D>(1, block_len).unwrap();

    let single_chunk_views: Vec<&[SingleChunkEntry]> = (0..single_chunk_blocks.num_blocks())
        .map(|i| single_chunk_blocks.block(i))
        .collect();
    let got = column_sweep_ajtai_onehot::<SingleChunkEntry, F, D>(
        &a_view,
        &single_chunk_views,
        1,
        block_len,
        1,
    );
    let expected = inner_ajtai_wide_single_chunk_tiled::<F, D>(&a_view, &bucket, 1);

    assert_eq!(got.len(), 1);
    assert_eq!(got[0], expected);
}

#[test]
fn multi_chunk_onehot_large_block_uses_safe_accumulator_path() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let coeffs_per_entry: usize = D / 2;
    let num_entries: usize = MAX_WIDE_SHIFT_ACCUMULATIONS / coeffs_per_entry + 1;
    let total_shift_accumulates: usize = num_entries * coeffs_per_entry;
    assert!(total_shift_accumulates > MAX_WIDE_SHIFT_ACCUMULATIONS);

    let n_a = 1;
    let num_digits_commit = 1;
    let block_len = num_entries;

    let max_coeff = F::from_canonical_u128_reduced((1u128 << 24) - 4);
    let dense_ring = CyclotomicRing::from_coefficients([max_coeff; D]);
    let a_matrix = [vec![dense_ring; block_len * num_digits_commit]];

    let nonzero_coeffs: Vec<u16> = (0..coeffs_per_entry as u16).collect();
    let bucket: Vec<MultiChunkEntry> = (0..block_len)
        .map(|pos| MultiChunkEntry::new(pos as u32, nonzero_coeffs.clone()))
        .collect();
    let multi_chunk_blocks = super::test_helpers::from_buckets(vec![bucket.clone()]);

    let a_flat = FlatMatrix::from_ring_slice(&a_matrix[0]);
    let a_view = a_flat
        .ring_view::<D>(n_a, block_len * num_digits_commit)
        .unwrap();

    let views: Vec<&[MultiChunkEntry]> = (0..multi_chunk_blocks.num_blocks())
        .map(|i| multi_chunk_blocks.block(i))
        .collect();

    let got = column_sweep_ajtai_onehot::<MultiChunkEntry, F, D>(
        &a_view,
        &views,
        n_a,
        block_len * num_digits_commit,
        num_digits_commit,
    );
    let reference = inner_ajtai_multi_chunk_t_only::<F, D>(&a_matrix, &bucket, num_digits_commit);

    assert_eq!(got.len(), 1, "single-block test: expected one output row");
    assert_eq!(
        got[0], reference,
        "column_sweep_ajtai_onehot must agree with the non-wide \
         reference at fan-out totals above MAX_WIDE_SHIFT_ACCUMULATIONS"
    );
}

#[test]
fn multi_chunk_onehot_single_entry_overflow_splits_coeffs() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let n_a = 1;
    let num_digits_commit = 1;
    let max_coeff = F::from_canonical_u128_reduced((1u128 << 24) - 4);
    let dense_ring = CyclotomicRing::from_coefficients([max_coeff; D]);
    let a_matrix = [vec![dense_ring]];

    let coeffs = vec![0u16; MAX_WIDE_SHIFT_ACCUMULATIONS + 1];
    let bucket = vec![MultiChunkEntry::new(0, coeffs)];
    let multi_chunk_blocks = super::test_helpers::from_buckets(vec![bucket.clone()]);
    let views: Vec<&[MultiChunkEntry]> = (0..multi_chunk_blocks.num_blocks())
        .map(|i| multi_chunk_blocks.block(i))
        .collect();

    let a_flat = FlatMatrix::from_ring_slice(&a_matrix[0]);
    let a_view = a_flat.ring_view::<D>(n_a, num_digits_commit).unwrap();

    let got = column_sweep_ajtai_onehot::<MultiChunkEntry, F, D>(
        &a_view,
        &views,
        n_a,
        num_digits_commit,
        num_digits_commit,
    );
    let reference = inner_ajtai_multi_chunk_t_only::<F, D>(&a_matrix, &bucket, num_digits_commit);

    assert_eq!(got[0], reference);
}

#[test]
fn batched_single_chunk_onehot_decompose_fold_matches_individual_aggregation() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let block_len = 64;
    let mut indices0 = vec![None; 128];
    indices0[0] = Some(1usize);
    indices0[17] = Some(5usize);
    indices0[64] = Some(9usize);
    indices0[91] = Some(33usize);
    let mut indices1 = vec![None; 128];
    indices1[3] = Some(7usize);
    indices1[29] = Some(11usize);
    indices1[64] = Some(19usize);
    indices1[100] = Some(21usize);
    let polys = [
        OneHotPoly::<F, D>::new(block_len, indices0).unwrap(),
        OneHotPoly::<F, D>::new(block_len, indices1).unwrap(),
    ];
    let challenges = vec![
        SparseChallenge {
            positions: vec![0, 5],
            coeffs: vec![1, -1],
        },
        SparseChallenge {
            positions: vec![2, 7],
            coeffs: vec![1, 1],
        },
        SparseChallenge {
            positions: vec![4, 11],
            coeffs: vec![-1, 2],
        },
        SparseChallenge {
            positions: vec![8, 13],
            coeffs: vec![1, -2],
        },
    ];

    let expected = aggregate_witnesses(
        &polys
            .iter()
            .zip(challenges.chunks(2))
            .map(|(poly, poly_challenges)| poly.decompose_fold(poly_challenges, block_len, 1, 0))
            .collect::<Vec<_>>(),
    );
    let poly_refs: Vec<&OneHotPoly<F, D>> = polys.iter().collect();
    let got = <OneHotPoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_batched(
        &poly_refs,
        &challenges,
        block_len,
        1,
        0,
    )
    .expect("onehot batched path should apply");

    assert_eq!(got, expected);
}

#[test]
fn single_chunk_onehot_evaluate_and_fold_matches_factorized_eval() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let poly =
        OneHotPoly::<F, D>::new(64, vec![Some(1usize), None, Some(9usize), Some(17usize)]).unwrap();
    let block_len = 2usize;
    let fold_scalars = vec![F::from_u64(3), F::from_u64(5)];
    let eval_outer_scalars = vec![F::from_u64(7), F::from_u64(11)];

    let (eval, folded) = poly.evaluate_and_fold(&eval_outer_scalars, &fold_scalars, block_len);
    let expected_folded = poly.fold_blocks(&fold_scalars, block_len);
    assert_eq!(folded, expected_folded);

    let full_scalars: Vec<F> = eval_outer_scalars
        .iter()
        .flat_map(|outer| fold_scalars.iter().map(move |inner| *outer * *inner))
        .collect();
    let expected_eval = super::test_helpers::evaluate_ring_onehot(&poly, &full_scalars);
    assert_eq!(eval, expected_eval);
}

#[test]
fn single_chunk_onehot_ring_fold_matches_dense_materialization() {
    type F = Prime24Offset3;
    const D: usize = 8;

    let poly =
        OneHotPoly::<F, D>::new(16, vec![Some(1usize), None, Some(13usize), Some(7usize)]).unwrap();
    let dense = materialize_onehot_as_dense(&poly);
    let block_len = 4usize;
    let fold_scalars = vec![
        test_ring_scalar::<F, D>(10),
        test_ring_scalar::<F, D>(40),
        test_ring_scalar::<F, D>(90),
        test_ring_scalar::<F, D>(120),
    ];

    assert_eq!(
        poly.fold_blocks_ring(&fold_scalars, block_len),
        dense.fold_blocks_ring(&fold_scalars, block_len)
    );
}

#[test]
fn multi_chunk_onehot_evaluate_and_fold_matches_factorized_eval() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let poly = OneHotPoly::<F, D>::new(
        32,
        vec![
            Some(1usize),
            None,
            Some(7usize),
            Some(12usize),
            None,
            Some(3usize),
            None,
            Some(15usize),
        ],
    )
    .unwrap();
    let block_len = 2usize;
    let fold_scalars = vec![F::from_u64(2), F::from_u64(4)];
    let eval_outer_scalars = vec![F::from_u64(3), F::from_u64(5)];

    let (eval, folded) = poly.evaluate_and_fold(&eval_outer_scalars, &fold_scalars, block_len);
    let expected_folded = poly.fold_blocks(&fold_scalars, block_len);
    assert_eq!(folded, expected_folded);

    let full_scalars: Vec<F> = eval_outer_scalars
        .iter()
        .flat_map(|outer| fold_scalars.iter().map(move |inner| *outer * *inner))
        .collect();
    let expected_eval = super::test_helpers::evaluate_ring_onehot(&poly, &full_scalars);
    assert_eq!(eval, expected_eval);
}

#[test]
fn multi_chunk_onehot_ring_fold_matches_dense_materialization() {
    type F = Prime24Offset3;
    const D: usize = 16;

    let poly = OneHotPoly::<F, D>::new(
        4,
        vec![
            Some(0usize),
            Some(3usize),
            None,
            Some(2usize),
            Some(1usize),
            None,
            Some(3usize),
            Some(0usize),
            None,
            Some(2usize),
            Some(1usize),
            None,
            Some(3usize),
            None,
            Some(0usize),
            Some(2usize),
        ],
    )
    .unwrap();
    let dense = materialize_onehot_as_dense(&poly);
    let block_len = 2usize;
    let fold_scalars = vec![test_ring_scalar::<F, D>(7), test_ring_scalar::<F, D>(80)];

    assert_eq!(
        poly.fold_blocks_ring(&fold_scalars, block_len),
        dense.fold_blocks_ring(&fold_scalars, block_len)
    );
}
