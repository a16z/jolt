use jolt_field::{Field, FpExt4, Fr, Prime32Offset99, Ring};
use jolt_poly::{eq_index, EqPolynomial, PointLike, PointRef, TensorEqTable};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn check_builders<F: Field, P: PointLike<Coord = F> + ?Sized>(point: &P, scale: F) {
    let n = point.num_vars();
    let table = EqPolynomial::<F>::evals(point, Some(scale));
    let mut out = vec![F::from_u64(9); table.len()];
    EqPolynomial::<F>::evals_into(point, Some(scale), &mut out);
    assert_eq!(out, table);
    for i in [0, table.len() / 3, table.len() - 1] {
        assert_eq!(table[i], scale * eq_index(point, i as u128));
    }
    let mapped = EqPolynomial::<F>::evals_mapped(point, Some(scale), |x| x * x + F::one());
    EqPolynomial::<F>::evals_mapped_into(point, Some(scale), |x| x * x + F::one(), &mut out);
    assert_eq!(out, mapped);
    assert!(mapped
        .iter()
        .zip(&table)
        .all(|(a, b)| *a == *b * *b + F::one()));
    for len in [0, 1, table.len() / 3, table.len() - 1, table.len()] {
        assert_eq!(
            EqPolynomial::<F>::evals_prefix(point, Some(scale), len),
            table[..len]
        );
        EqPolynomial::<F>::evals_prefix_into(point, Some(scale), &mut out[..len]);
        assert_eq!(out[..len], table[..len]);
    }
    if n <= 8 {
        for len in 0..=table.len() {
            assert_eq!(
                EqPolynomial::<F>::prefix_sum(point, len) * scale,
                table[..len].iter().copied().sum::<F>()
            );
        }
        for split in 0..=n {
            let tensor = TensorEqTable::<F>::from_point(point, scale, split);
            for (i, value) in table.iter().enumerate() {
                assert_eq!(tensor.evaluate_index(i), *value);
            }
        }
    }
    for msb in [true, false] {
        let cache = if msb {
            EqPolynomial::<F>::evals_cached_msb(point, Some(scale))
        } else {
            EqPolynomial::<F>::evals_cached_lsb(point, Some(scale))
        };
        for (j, values) in cache.iter().enumerate() {
            let view = if msb {
                point.msb_range(0..j)
            } else {
                point.msb_range(n - j..n)
            };
            for i in [0, values.len() / 3, values.len() - 1] {
                assert_eq!(values[i], scale * eq_index(&view, i as u128));
            }
        }
    }
}

fn algebraic_contracts<F: Field>() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xe12);
    for n in 0..=20 {
        let point: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        let scale = F::random(&mut rng);
        check_builders(&point, scale);
        check_builders(&PointRef::low_to_high(&point), scale);
    }
}

#[test]
fn builders_obey_equality_and_prefix_identities_in_both_orders() {
    algebraic_contracts::<Fr>();
    algebraic_contracts::<FpExt4<Prime32Offset99>>();
}

#[test]
#[should_panic(expected = "invalid equality-table length")]
fn evals_into_rejects_short_buffers() {
    EqPolynomial::<Fr>::evals_into(&[Fr::from_u64(2)], None, &mut []);
}

#[test]
#[should_panic(expected = "invalid equality-table length")]
fn evals_into_rejects_long_buffers() {
    EqPolynomial::<Fr>::evals_into(&[] as &[Fr], None, &mut [Fr::from_u64(9); 2]);
}

#[test]
#[should_panic(expected = "invalid equality-table length")]
fn mapped_into_rejects_short_buffers() {
    EqPolynomial::<Fr>::evals_mapped_into(&[Fr::from_u64(2)], None, |x| x, &mut []);
}

#[test]
#[should_panic(expected = "invalid equality-table length")]
fn mapped_into_rejects_long_buffers() {
    EqPolynomial::<Fr>::evals_mapped_into(&[] as &[Fr], None, |x| x, &mut [Fr::from_u64(9); 2]);
}

#[test]
#[should_panic(expected = "invalid equality-table prefix")]
fn prefix_into_rejects_long_buffers() {
    EqPolynomial::<Fr>::evals_prefix_into(&[] as &[Fr], None, &mut [Fr::from_u64(9); 2]);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn tensor_evaluate_index_rejects_out_of_range() {
    let _ = TensorEqTable::<Fr>::from_point(&[] as &[Fr], Fr::from_u64(1), 0).evaluate_index(1);
}
