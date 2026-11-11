#![expect(
    clippy::unwrap_used,
    reason = "tests assert successful budgeted construction"
)]
use jolt_field::{Field, FpExt4, Fr, Prime32Offset99, Ring};
use jolt_poly::{eq_index, Budget, EqPolynomial, PointLike, PointRef, TensorEqTable};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const UNLIMITED: Budget = Budget {
    max_bytes: usize::MAX,
};

fn check_builders<F: Field, P: PointLike<Coord = F> + ?Sized>(point: &P, scale: F) {
    let n = point.num_vars();
    let table = EqPolynomial::<F>::evals_in(point, Some(scale), UNLIMITED).unwrap();

    for i in [0, table.len() / 3, table.len() - 1] {
        assert_eq!(table[i], scale * eq_index(point, i as u128));
    }
    let mapped =
        EqPolynomial::<F>::evals_mapped_in(point, Some(scale), |x| x * x + F::one(), UNLIMITED)
            .unwrap();
    assert!(mapped
        .iter()
        .zip(&table)
        .all(|(a, b)| *a == *b * *b + F::one()));
    for len in [0, 1, table.len() / 3, table.len() - 1, table.len()] {
        assert_eq!(
            EqPolynomial::<F>::evals_prefix_in(point, Some(scale), len, UNLIMITED).unwrap(),
            table[..len]
        );
    }
    if n <= 8 {
        for len in 0..=table.len() {
            assert_eq!(
                EqPolynomial::<F>::prefix_sum_in(point, len, UNLIMITED).unwrap() * scale,
                table[..len].iter().copied().sum::<F>()
            );
        }
        for split in 0..=n {
            let tensor = TensorEqTable::<F>::from_point_in(point, scale, split, UNLIMITED).unwrap();
            for (i, value) in table.iter().enumerate() {
                assert_eq!(tensor.evaluate_index_in(i, UNLIMITED).unwrap(), *value);
            }
        }
    }
    for msb in [true, false] {
        let cache = if msb {
            EqPolynomial::<F>::evals_cached_msb_in(point, Some(scale), UNLIMITED).unwrap()
        } else {
            EqPolynomial::<F>::evals_cached_lsb_in(point, Some(scale), UNLIMITED).unwrap()
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
        if n <= 5 {
            for scale in [F::zero(), F::one()] {
                check_builders(&point, scale);
                check_builders(&PointRef::low_to_high(&point), scale);
            }
        }
    }
}

#[test]
fn builders_obey_equality_and_prefix_identities_in_both_orders() {
    algebraic_contracts::<Fr>();
    algebraic_contracts::<FpExt4<Prime32Offset99>>();
}

#[test]
#[should_panic(expected = "invalid equality-table prefix")]
fn prefix_rejects_long_lengths() {
    let _ = EqPolynomial::<Fr>::evals_prefix(&[] as &[Fr], None, 2);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn tensor_evaluate_index_rejects_out_of_range() {
    let _ = TensorEqTable::<Fr>::new_with_scaling_and_split(&[] as &[Fr], Fr::from_u64(1), 0)
        .evaluate_index(1);
}
