#![expect(
    clippy::panic,
    reason = "test failures and the OOM child must fail loudly"
)]
#![expect(clippy::unwrap_used, reason = "asserting valid builder results")]

use jolt_field::{Field, FpExt4, Fr, Prime32Offset99, Ring};
use jolt_poly::{
    eq_index, EqBudget, EqPolynomial, EqTableError, PointLike, PointRef, TensorEqTable,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const UNLIMITED: EqBudget = EqBudget {
    max_bytes: usize::MAX,
};

fn check_builders<F: Field, P: PointLike<Coord = F> + ?Sized>(point: &P, scale: F) {
    let n = point.num_vars();
    let table = EqPolynomial::<F>::try_evals(point, Some(scale), UNLIMITED).unwrap();
    for i in [0, table.len() / 3, table.len() - 1] {
        assert_eq!(table[i], scale * eq_index(point, i as u128));
    }
    let mapped = EqPolynomial::<F>::evals_mapped(point, Some(scale), |x| x * x + F::one());
    let try_mapped =
        EqPolynomial::<F>::try_evals_mapped(point, Some(scale), |x| x * x + F::one(), UNLIMITED)
            .unwrap();
    assert_eq!(mapped, try_mapped);
    assert!(mapped
        .iter()
        .zip(&table)
        .all(|(a, b)| *a == *b * *b + F::one()));
    for len in [0, 1, table.len() / 3, table.len() - 1, table.len()] {
        assert_eq!(
            EqPolynomial::<F>::evals_prefix(point, Some(scale), len),
            table[..len]
        );
        assert_eq!(
            EqPolynomial::<F>::try_evals_prefix(point, Some(scale), len, UNLIMITED).unwrap(),
            table[..len]
        );
    }
    if n <= 8 {
        for len in 0..=table.len() {
            assert_eq!(
                EqPolynomial::<F>::try_prefix_sum(point, len).unwrap() * scale,
                table[..len].iter().copied().sum::<F>()
            );
        }
        for split in 0..=n {
            let tensor =
                TensorEqTable::<F>::try_from_point(point, scale, split, UNLIMITED).unwrap();
            for (i, value) in table.iter().enumerate() {
                assert_eq!(tensor.try_evaluate_index(i).unwrap(), *value);
            }
            assert!(matches!(
                tensor.try_evaluate_index(table.len()),
                Err(EqTableError::IndexOutOfRange { .. })
            ));
        }
    }
    for msb in [true, false] {
        let cache = if msb {
            EqPolynomial::<F>::try_evals_cached_msb(point, Some(scale), UNLIMITED)
        } else {
            EqPolynomial::<F>::try_evals_cached_lsb(point, Some(scale), UNLIMITED)
        }
        .unwrap();
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

fn cap_boundary<T>(bytes: usize, build: impl Fn(EqBudget) -> Result<T, EqTableError>) {
    assert!(build(EqBudget { max_bytes: bytes }).is_ok());
    assert!(build(EqBudget {
        max_bytes: bytes + 1
    })
    .is_ok());
    match build(EqBudget {
        max_bytes: bytes - 1,
    }) {
        Err(EqTableError::OverBudget {
            bytes: actual,
            max_bytes,
            at,
        }) => {
            assert_eq!(actual, bytes);
            assert_eq!(max_bytes, bytes - 1);
            assert!(at.file().ends_with("eq_tables.rs"));
        }
        _ => panic!("expected OverBudget"),
    }
}

#[test]
fn caps_dimensions_and_query_bounds_are_checked_before_allocating() {
    let point = [Fr::from_u64(2); 4];
    let bytes = size_of::<Fr>();
    cap_boundary(16 * bytes, |b| {
        EqPolynomial::<Fr>::try_evals(&point, None, b)
    });
    cap_boundary(16 * bytes, |b| {
        EqPolynomial::<Fr>::try_evals_mapped(&point, None, |x| x, b)
    });
    cap_boundary(31 * bytes, |b| {
        EqPolynomial::<Fr>::try_evals_cached_msb(&point, None, b)
    });
    cap_boundary(31 * bytes, |b| {
        EqPolynomial::<Fr>::try_evals_cached_lsb(&point, None, b)
    });
    cap_boundary(7 * bytes, |b| {
        EqPolynomial::<Fr>::try_evals_prefix(&point, None, 7, b)
    });
    cap_boundary(8 * bytes, |b| {
        TensorEqTable::<Fr>::try_from_point(&point, Fr::from_u64(1), 2, b)
    });
    let long = [Fr::from_u64(0); usize::BITS as usize];
    for n in [usize::BITS as usize, usize::BITS as usize - 1] {
        let point = &long[..n];
        assert!(matches!(
            EqPolynomial::<Fr>::try_evals(point, None, UNLIMITED),
            Err(EqTableError::DimensionOverflow { .. })
        ));
        assert!(matches!(
            EqPolynomial::<Fr>::try_evals_mapped(point, None, |x| x, UNLIMITED),
            Err(EqTableError::DimensionOverflow { .. })
        ));
        assert!(matches!(
            EqPolynomial::<Fr>::try_evals_cached_msb(point, None, UNLIMITED),
            Err(EqTableError::DimensionOverflow { .. })
        ));
        assert!(matches!(
            EqPolynomial::<Fr>::try_evals_cached_lsb(point, None, UNLIMITED),
            Err(EqTableError::DimensionOverflow { .. })
        ));
    }
    assert!(matches!(
        EqPolynomial::<Fr>::try_prefix_sum(&long, 0),
        Err(EqTableError::DimensionOverflow { .. })
    ));
    assert!(matches!(
        EqPolynomial::<Fr>::try_evals_prefix(&long, None, 0, UNLIMITED),
        Err(EqTableError::DimensionOverflow { .. })
    ));
    assert!(matches!(
        TensorEqTable::<Fr>::try_from_point(&long, Fr::from_u64(1), 32, UNLIMITED),
        Err(EqTableError::DimensionOverflow { .. })
    ));
    assert!(matches!(
        TensorEqTable::<Fr>::try_from_point(&point, Fr::from_u64(1), 5, UNLIMITED),
        Err(EqTableError::LengthMismatch {
            expected: 4,
            got: 5
        })
    ));
    assert!(matches!(
        EqPolynomial::<Fr>::try_prefix_sum(&point, 17),
        Err(EqTableError::LengthMismatch {
            expected: 16,
            got: 17
        })
    ));
    assert!(matches!(
        EqPolynomial::<Fr>::try_evals_prefix(&point, None, 17, UNLIMITED),
        Err(EqTableError::LengthMismatch {
            expected: 16,
            got: 17
        })
    ));
}
