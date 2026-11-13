use super::*;
use crate::{BindingOrder, GruenSplitEqPolynomial, Point, PointRef, TensorEqTable, LOW_TO_HIGH};
use jolt_field::{FpExt4, Fr, Prime32Offset99};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn bitrev(index: usize, bits: usize) -> usize {
    if bits == 0 {
        0
    } else {
        index.reverse_bits() >> (usize::BITS as usize - bits)
    }
}

fn assert_permuted<F: Field>(le: &[F], be: &[F], bits: usize) {
    assert_eq!(le.len(), 1 << bits);
    for (i, value) in le.iter().enumerate() {
        assert_eq!(*value, be[bitrev(i, bits)], "bits={bits}, index={i}");
    }
}

fn builders<F: Field>() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xe11);
    for n in 0..=20 {
        let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        let reversed: Vec<F> = r.iter().rev().copied().collect();
        let le = PointRef::low_to_high(&r);
        let scale = F::random(&mut rng);
        let scaling_factor = Some(scale);
        let be = EqPolynomial::<F>::evals(&r, scaling_factor);
        let expected = EqPolynomial::<F>::evals(&reversed, scaling_factor);
        let dense = EqPolynomial::<F>::evals(&le, scaling_factor);
        assert_eq!(dense, expected);
        assert_permuted(&dense, &be, n);
        for table in [
            EqPolynomial::<F>::evals_serial(&le, scaling_factor),
            EqPolynomial::<F>::evals_parallel(&le, scaling_factor),
        ] {
            assert_eq!(table, expected);
            assert_permuted(&table, &be, n);
        }

        // An MSB cache in one order selects the same coordinates as an LSB
        // cache in the other order, with the index bits reversed.
        for msb in [false, true] {
            let build = |point: PointRef<'_, LOW_TO_HIGH, F>| {
                if msb {
                    EqPolynomial::<F>::evals_cached_msb(&point, scaling_factor)
                } else {
                    EqPolynomial::<F>::evals_cached_lsb(&point, scaling_factor)
                }
            };
            let cache = build(le);
            let (opposite, reverse_cache) = if msb {
                (
                    EqPolynomial::<F>::evals_cached_lsb(&r, scaling_factor),
                    EqPolynomial::<F>::evals_cached_msb(&reversed, scaling_factor),
                )
            } else {
                (
                    EqPolynomial::<F>::evals_cached_msb(&r, scaling_factor),
                    EqPolynomial::<F>::evals_cached_lsb(&reversed, scaling_factor),
                )
            };
            assert_eq!(cache, reverse_cache);
            for j in 0..=n {
                assert_permuted(&cache[j], &opposite[j], j);
            }
        }

        let tensor = TensorEqTable::<F>::new_with_scaling_and_split(&le, scale, n - n / 2);
        for (i, value) in dense.iter().enumerate() {
            assert_eq!(tensor.evaluate_index(i), *value);
        }
        // Each block is checked at its full-domain indices, because bit
        // reversal does not preserve contiguous blocks.
        let block_size = 1 << (n / 2);
        let unscaled = EqPolynomial::<F>::evals(&r, None);
        for start in [0, dense.len() - block_size] {
            let block = EqPolynomial::<F>::evals_for_aligned_block(&le, start, block_size);
            let reverse_block =
                EqPolynomial::<F>::evals_for_aligned_block(&reversed, start, block_size);
            assert_eq!(block, reverse_block);
            let max_block = EqPolynomial::<F>::evals_for_max_aligned_block(&le, start, block_size);
            assert_eq!(max_block, (block_size, block.clone()));
            for (j, value) in block.iter().enumerate() {
                assert_eq!(*value, unscaled[bitrev(start + j, n)]);
            }
        }
    }
}

#[test]
fn builders_respect_significance_fr() {
    builders::<Fr>();
}

#[test]
fn builders_respect_significance_extension() {
    builders::<FpExt4<Prime32Offset99>>();
}

#[test]
fn scalar_queries_follow_table_order_and_preserve_msb_aliasing() {
    use num_traits::{One, Zero};
    let mut rng = ChaCha20Rng::seed_from_u64(203);
    for n in 0..=8 {
        let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let le = PointRef::low_to_high(&r);
        let table = EqPolynomial::<Fr>::evals(&le, None);
        for i in 0..=table.len() {
            assert_eq!(
                EqPolynomial::<Fr>::prefix_sum(&le, i),
                table[..i].iter().copied().sum()
            );
            assert_eq!(
                EqPolynomial::<Fr>::suffix_sum(&le, i),
                table[i..].iter().copied().sum()
            );
            assert_eq!(
                eq_index(&le, i as u128),
                table.get(i).copied().unwrap_or(Fr::zero())
            );
        }
        assert_eq!(eq_index(&le, u128::MAX), Fr::zero());
        assert_eq!(eq_index_msb(&r, (1 << n) as u128), eq_index_msb(&r, 0));
    }
    assert_eq!(eq_index(&[Fr::one(); 128], u128::MAX), Fr::one());
    let long = [Fr::one(); 129];
    assert_eq!(eq_index(&long, u128::MAX), Fr::zero());
    assert_eq!(eq_index::<Fr, _>(&[] as &[Fr], 0), Fr::one());
}

#[test]
fn gruen_constructor_uses_significance_in_both_binding_orders() {
    let mut rng = ChaCha20Rng::seed_from_u64(206);
    for n in 0..=20 {
        let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let reversed = r.iter().rev().copied().collect::<Vec<_>>();
        let le = Point::<LOW_TO_HIGH, _>::new(r);
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            let mut split = GruenSplitEqPolynomial::<Fr>::new(&le, order);
            let mut expected = GruenSplitEqPolynomial::<Fr>::new(&reversed, order);
            assert_eq!(split, expected);
            for _ in 0..n {
                let challenge = Fr::random(&mut rng);
                assert!(split
                    .round_mut()
                    .map(|round| round.bind(challenge))
                    .is_some());
                assert!(expected
                    .round_mut()
                    .map(|round| round.bind(challenge))
                    .is_some());
                assert_eq!(split, expected);
            }
        }
    }
}

fn eq_value_le<F: Field>(point: &[F], index: usize) -> F {
    point
        .iter()
        .enumerate()
        .map(|(bit, &coordinate)| {
            if (index >> bit) & 1 == 1 {
                coordinate
            } else {
                F::one() - coordinate
            }
        })
        .product()
}

#[test]
fn tensor_new_respects_owned_little_endian_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(207);
    for n in 0..=8 {
        let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let reversed: Vec<Fr> = r.iter().rev().copied().collect();
        let le = Point::<LOW_TO_HIGH, _>::new(r);
        let tensor = TensorEqTable::new(&le);
        assert_eq!(tensor, TensorEqTable::new(reversed.as_slice()));
        for index in 0..1 << n {
            assert_eq!(tensor.evaluate_index(index), eq_value_le(&le, index));
        }
    }
}
