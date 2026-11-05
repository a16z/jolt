use jolt_field::{Field, Fr, Prime128OffsetA7F7 as Fp128};
use jolt_poly::CompressedPoly;

fn compressed_spec<F: Field>() {
    // Fixed integer evaluations cover zero, linear and higher-degree cases,
    // including negative coefficients and zero leading coefficients.
    for (stored, hint, x, expected) in [
        (vec![0], 0, 2, 0),
        (vec![5], 17, 3, 26),
        (vec![1, 2, 1], 8, 3, 55),
        (vec![1, 2, 1], 8, -1, -1),
        (vec![-1, -2, -1], -8, 3, -55),
        (vec![5, 0, 0, 0], 17, 3, 26),
        (vec![7, 0, 0, 0, 0, 0, 1], 15, 2, 135),
    ] {
        let poly = CompressedPoly::new(stored.into_iter().map(F::from_i64).collect());
        assert_eq!(
            poly.eval_from_hint(&F::from_i64(hint), &F::from_i64(x)),
            F::from_i64(expected)
        );
    }
    for degree in 1..=16 {
        let stored: Vec<F> = (0..degree).map(|i| F::from_i64(i64::from(i) - 7)).collect();
        let c0 = stored[0];
        let poly = CompressedPoly::new(stored);
        for hint in [F::zero(), F::one(), -F::one(), F::from_u64(19)] {
            assert_eq!(poly.eval_from_hint(&hint, &F::zero()), c0);
            assert_eq!(poly.eval_from_hint(&hint, &F::one()), hint - c0);
        }
    }
}

#[test]
fn compressed_horner_spec_fp128() {
    compressed_spec::<Fp128>();
}

#[test]
fn compressed_horner_spec_bn254() {
    compressed_spec::<Fr>();
}

#[test]
#[should_panic(expected = "cannot evaluate an empty compressed polynomial")]
fn compressed_horner_empty_fp128() {
    use jolt_field::Ring;
    let empty = CompressedPoly::<Fp128>::new(vec![]);
    let _value = empty.eval_from_hint(&Fp128::from_u64(1), &Fp128::from_u64(2));
}
