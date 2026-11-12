//! Checks every `GruenSplitEqPolynomial` and `GruenRound` method over fields
//! that implement only `Field`, against eq products built here bit by bit and
//! the identity `s(X) = l(X) q(X)`.

use super::GruenSplitEqPolynomial;
use crate::{BindingOrder, PointRef, UnivariatePoly};
use jolt_field::{Field, FpExt4, Fr, Prime128Offset275, Prime32Offset99};

#[test]
fn rounds_and_tables_match_eq_products_over_field_only_types() {
    check_rounds::<Fr>();
    check_rounds::<Prime128Offset275>();
    check_rounds::<FpExt4<Prime32Offset99>>();
}

fn eq_bit<F: Field>(r: F, bit: bool) -> F {
    if bit {
        r
    } else {
        F::one() - r
    }
}

/// `eq(r, index)` for an MSB-first `r`: bit `m - 1 - k` of `index` pairs with `r[k]`.
fn eq_index<F: Field>(r: &[F], index: usize) -> F {
    let m = r.len();
    (0..m).fold(F::one(), |acc, k| {
        acc * eq_bit(r[k], (index >> (m - 1 - k)) & 1 == 1)
    })
}

fn eval<F: Field>(coefficients: &[F], x: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::zero(), |acc, &c| acc * x + c)
}

/// Coordinates hit `l(0) = 0` and `l(1) = 0` as well as generic values.
fn coordinate<F: Field>(k: usize) -> F {
    match k % 5 {
        1 => F::zero(),
        3 => F::one(),
        _ => F::from_u64(3 + 7 * k as u64),
    }
}

/// Which degenerate message branches the walk reached.
#[derive(Default)]
struct Branches {
    zero_scalar: bool,
    zero_at_one: bool,
    zero_at_zero: bool,
}

fn check_rounds<F: Field>() {
    let mut branches = Branches::default();
    for n in 0..=9 {
        let coords: Vec<F> = (0..n).map(coordinate).collect();
        let msb: Vec<F> = coords.iter().rev().copied().collect();
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            assert_eq!(
                GruenSplitEqPolynomial::new(&PointRef::low_to_high(&coords), order),
                GruenSplitEqPolynomial::new_with_scaling(
                    &PointRef::low_to_high(&coords),
                    order,
                    None
                ),
            );
            for scale in [F::one(), F::zero(), F::from_u64(7)] {
                let mut state = GruenSplitEqPolynomial::new_with_scaling(
                    &PointRef::low_to_high(&coords),
                    order,
                    Some(scale),
                );
                let mut scalar = scale;
                for round in 0..n {
                    let (active, unbound, remaining) = match order {
                        BindingOrder::LowToHigh => {
                            (n - 1 - round, &msb[..n - round], &msb[..n - 1 - round])
                        }
                        BindingOrder::HighToLow => (round, &msb[round..], &msb[round + 1..]),
                    };
                    assert_eq!(state.remaining_rounds(), n - round);
                    assert_eq!(state.current_scalar(), scalar);
                    for (j, &value) in state.merge().evals().iter().enumerate() {
                        assert_eq!(value, scalar * eq_index(unbound, j), "merge n={n}");
                    }
                    check_tables(&state, order, remaining);
                    if order == BindingOrder::LowToHigh {
                        check_low_to_high_weights(&state, &msb[..n - round], remaining);
                    }
                    let coordinate = msb[active];
                    assert_eq!(state.round().unwrap().coordinate(), coordinate);
                    check_messages(&state, scalar, coordinate, round, &mut branches);
                    let challenge = F::from_u64(11 + 2 * round as u64);
                    state.round_mut().unwrap().bind(challenge);
                    scalar *= eq_bit(coordinate, false) * eq_bit(challenge, false)
                        + coordinate * challenge;
                }
                assert!(state.round().is_none());
                assert_eq!(state.current_scalar(), scalar);
                assert_eq!(state.merge().evals(), &[scalar]);
            }
        }
    }
    assert!(branches.zero_scalar && branches.zero_at_one && branches.zero_at_zero);
}

/// The factored tables and the out/in fold reproduce `eq(remaining, ·)`.
fn check_tables<F: Field>(state: &GruenSplitEqPolynomial<F>, order: BindingOrder, remaining: &[F]) {
    let (inner, outer) = state.remaining_tables_after(0).unwrap();
    assert_eq!(
        (inner, outer),
        (state.e_in_current(), state.e_out_current())
    );
    assert_eq!(
        (state.e_in_current_len(), state.e_out_current_len()),
        (inner.len(), outer.len())
    );
    let len = 1usize << remaining.len();
    assert_eq!(inner.len() * outer.len(), len);
    let polynomial_index = |x_out: usize, x_in: usize| match order {
        BindingOrder::LowToHigh => x_out * inner.len() + x_in,
        BindingOrder::HighToLow => x_in * outer.len() + x_out,
    };
    for (x_out, &out_weight) in outer.iter().enumerate() {
        for (x_in, &in_weight) in inner.iter().enumerate() {
            assert_eq!(
                out_weight * in_weight,
                eq_index(remaining, polynomial_index(x_out, x_in))
            );
        }
    }
    let mut weights = state.par_fold_out_in(
        Vec::new,
        |rows: &mut Vec<(usize, usize, F)>, row, x_in, e_in| rows.push((row, x_in, e_in)),
        |x_out, e_out, rows| {
            rows.into_iter()
                .map(|(row, x_in, e_in)| {
                    assert_eq!(row, state.group_index(x_out, x_in));
                    (polynomial_index(x_out, x_in), e_out * e_in)
                })
                .collect::<Vec<_>>()
        },
        |mut a, b| {
            a.extend(b);
            a
        },
    );
    weights.sort_by_key(|&(index, _)| index);
    assert_eq!(weights.len(), len);
    for (expected, (index, weight)) in weights.into_iter().enumerate() {
        assert_eq!(index, expected);
        assert_eq!(weight, eq_index(remaining, index));
    }
}

/// Prefix folds, suffix weights and streaming windows, defined only for
/// low-to-high binding. `unbound` is the MSB-first unbound point.
fn check_low_to_high_weights<F: Field>(
    state: &GruenSplitEqPolynomial<F>,
    unbound: &[F],
    remaining: &[F],
) {
    let len = 1usize << remaining.len();
    for start in [0, 1, len / 2 + 1, len] {
        let start = start.min(len);
        let prefix = state.par_fold_out_in_prefix(
            start,
            F::zero,
            |sum, _, _, e_in| *sum += e_in,
            |_, e_out, sum| e_out * sum,
            |a, b| a + b,
        );
        let below = (0..start).fold(F::zero(), |sum, j| sum + eq_index(remaining, j));
        let above = (start..len).fold(F::zero(), |sum, j| sum + eq_index(remaining, j));
        assert_eq!(prefix, below, "prefix {start}");
        assert_eq!(state.suffix_weight(start), above, "suffix {start}");
    }
    let unbound_len = unbound.len();
    for window in 1..=unbound_len {
        let head = &unbound[..unbound_len - window];
        let (e_out, e_in) = state.e_out_in_for_window(window);
        assert_eq!(e_out.len() * e_in.len(), 1 << head.len());
        for (x_out, &out_weight) in e_out.iter().enumerate() {
            for (x_in, &in_weight) in e_in.iter().enumerate() {
                assert_eq!(
                    out_weight * in_weight,
                    eq_index(head, x_out * e_in.len() + x_in)
                );
            }
        }
        let active = state.e_active_for_window(window);
        if window == 1 {
            assert_eq!(active, vec![F::one()]);
        } else {
            let window_point = &unbound[unbound_len - window..unbound_len - 1];
            assert_eq!(active.len(), 1 << window_point.len());
            for (j, &value) in active.iter().enumerate() {
                assert_eq!(value, eq_index(window_point, j));
            }
        }
    }
}

/// The three message methods, `gruen_mul` and the prepared recovery against
/// `l(X) q(X)`, with `l` built from the coordinate and the tracked scalar.
fn check_messages<F: Field>(
    state: &GruenSplitEqPolynomial<F>,
    scalar: F,
    coordinate: F,
    round: usize,
    branches: &mut Branches,
) {
    let view = state.round().unwrap();
    let l0 = scalar * (F::one() - coordinate);
    let l1 = scalar * coordinate;
    assert_eq!(view.linear_evals(), (l0, l1));
    let l = |x: F| l0 * (F::one() - x) + l1 * x;
    let xs: Vec<F> = (0..7).map(F::from_u64).collect();
    let assert_product = |s: &UnivariatePoly<F>, q: &[F], what: &str| {
        for &x in &xs {
            assert_eq!(s.evaluate(x), l(x) * eval(q, x), "{what} round {round}");
        }
    };

    let q3 = [
        F::from_u64(5 + round as u64),
        F::from_u64(9),
        F::from_u64(4),
    ];
    let hint3 = l0 * eval(&q3, F::zero()) + l1 * eval(&q3, F::one());
    let q3_at_one = || eval(&q3, F::one());
    let s3 = view
        .gruen_poly_deg_3(q3[0], q3[2], hint3, q3_at_one)
        .unwrap();
    assert_eq!(s3.coefficients().len(), 4);
    assert_product(&s3, &q3, "degree 3");
    let wrong3 = view.gruen_poly_deg_3(q3[0], q3[2], hint3 + F::one(), q3_at_one);
    if scalar.is_zero() {
        assert_eq!(wrong3, Err(F::zero()));
        branches.zero_scalar = true;
    } else if l1.is_zero() {
        assert_eq!(wrong3, Err(hint3));
        branches.zero_at_one = true;
    } else {
        assert!(wrong3.is_ok());
    }

    let q = [
        F::from_u64(2),
        F::from_u64(3 + round as u64),
        F::from_u64(5),
        F::from_u64(11),
    ];
    let hint = l0 * eval(&q, F::zero()) + l1 * eval(&q, F::one());
    let samples = [eval(&q, F::one()), eval(&q, F::from_u64(2)), q[3]];
    let q_at_zero = || eval(&q, F::zero());
    let toom = view
        .gruen_poly_from_evals(&samples, hint, q_at_zero)
        .unwrap();
    assert_product(&toom, &q, "Toom");
    let wrong = view.gruen_poly_from_evals(&samples, hint + F::one(), q_at_zero);
    if !scalar.is_zero() && l0.is_zero() {
        assert_eq!(wrong, Err(hint));
        branches.zero_at_zero = true;
    }
    assert_product(&view.gruen_mul(&q), &q, "gruen_mul");

    let q1 = [F::from_u64(6), F::from_u64(13 + round as u64)];
    let hint2 = l0 * q1[0] + l1 * (q1[0] + q1[1]);
    let s2 = view
        .gruen_poly_deg_2(q1[0] + q1[1], hint2, || q1[0])
        .unwrap();
    assert_product(&s2, &q1, "degree 2");

    match view.prepare_linear_q_recovery() {
        Some(prepared) => {
            assert!(!l1.is_zero());
            let s = prepared.gruen_poly_deg_3(q3[0], q3[2], hint3);
            assert_eq!(s, s3);
            let mut except_linear = q.to_vec();
            let _ = except_linear.remove(1);
            let general = prepared.gruen_poly_from_coeffs_except_linear(&except_linear, hint);
            assert_product(&general, &q, "prepared");
        }
        None => assert!(l1.is_zero()),
    }
}
